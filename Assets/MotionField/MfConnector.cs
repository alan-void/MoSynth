using System;
using System.Collections.Concurrent;
using System.Threading;
using AnimationTools;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using Unity.Mathematics;
using UnityEngine;

namespace MotionField
{
/// <summary>
/// Drives the pose from a motion field running in a <em>separate Python process</em>, reached over
/// a ZeroMQ request/reply socket with JSON poses on the wire.
/// </summary>
/// <remarks>
/// The alternative to <see cref="MotionFieldStage"/>'s embedded CPython: it costs a serialization
/// round trip and buys a Python side that can be restarted and debugged without taking Unity down.
/// <para>
/// The socket lives entirely on <see cref="ClientWorker"/>'s thread, since NetMQ sockets are not
/// thread-safe. The two threads meet only at the concurrent queues and the volatile flags.
/// </para>
/// </remarks>
[Serializable]
public class MfConnector : MoSynthStage, IDisposable
{
    /// <summary>Wire format of one pose reply. Field names must match the Python server's JSON keys.</summary>
    [Serializable]
    private class PoseVectorDto
    {
        public float3[] jointLocalPositions;
        public quaternion[] jointLocalRotations;
        public float3[] jointLocalVelocities;
        public float3[] jointLocalAngularVelocities;
        public bool leftFootContact;
        public bool rightFootContact;
    }

    private Thread _clientThread;

    /// <summary>Clear to ask the worker to stop; also cleared by the worker when it dies.</summary>
    private volatile bool _isRunning;

    /// <summary>Interlocked 0/1 latch making <see cref="Dispose"/> idempotent.</summary>
    private int _disposeState;

    /// <summary>Set by the worker thread, reported and cleared on the main thread by Apply.</summary>
    private volatile string _workerError;

    private MotionSynthesisComponent _owner;

    private readonly ConcurrentQueue<float> _deltaTimeRequests = new();
    private readonly ConcurrentQueue<PoseVectorDto> _receivedPoses = new();

    [Tooltip("TCP port of the Python motion field server on localhost.")]
    [SerializeField] private int port = 5555;

    [Tooltip("Budget for draining the reply queue on the main thread, so a backlog cannot stall a frame.")]
    [SerializeField] private float dequeueTimeoutMs = 2f;

    [Tooltip("How often the worker checks for a reply. Also bounds how long shutdown waits for it.")]
    [SerializeField] [Min(1)] private int receivePollIntervalMs = 100;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _owner = motionSynthesisComponent;

        // A serialized stage can be initialized again after leaving and re-entering play mode.
        Interlocked.Exchange(ref _disposeState, 0);
        _workerError = null;
        ClearQueue(_deltaTimeRequests);
        ClearQueue(_receivedPoses);

        // Required for NetMQ to run properly in Unity
        AsyncIO.ForceDotNet.Force();

        _isRunning = true;
        _clientThread = new Thread(ClientWorker)
        {
            IsBackground = true,
            Name = nameof(MfConnector)
        };
        _clientThread.Start();
    }

    /// <summary>
    /// Queues a request for the next pose and applies the latest reply, if one has arrived. Returns
    /// false when none has, leaving the character in its previous pose rather than a half-updated one.
    /// </summary>
    /// <remarks>
    /// The drain loop returns as soon as it applies a pose, so it runs one iteration and
    /// <see cref="dequeueTimeoutMs"/> never fires. Both are here for a future version that consumes
    /// a backlog to catch up.
    /// </remarks>
    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (_workerError != null)
        {
            Debug.LogError($"MotionField client stopped: {_workerError}");
            _workerError = null;
        }

        // At most one request queued, or the queue grows every frame while the server is down.
        if (_isRunning && _deltaTimeRequests.IsEmpty)
        {
            _deltaTimeRequests.Enqueue(deltaTime);
            Debug.Log($"Requested Frame: {Time.frameCount}");
        }

        var dequeueStartTime = Time.realtimeSinceStartup;
        while (_receivedPoses.TryDequeue(out var newPose))
        {
            if ((Time.realtimeSinceStartup - dequeueStartTime) * 1000f >= dequeueTimeoutMs)
            {
                Debug.LogWarning($"Stopped dequeue loop after {dequeueTimeoutMs} ms timeout.");
                break;
            }

            Debug.Log($"Successfully received pose! Left Foot Contact: {newPose.leftFootContact}");

            var positions = pose.Positions;
            var rotations = pose.Rotations;
            var velocities = pose.Velocities;
            var angularVelocities = pose.AngularVelocities;

            var numJoints = positions.Length;

            for (var i = 0; i < numJoints; i++)
            {
                positions[i] = newPose.jointLocalPositions[i];
                rotations[i] = newPose.jointLocalRotations[i];
                velocities[i] = newPose.jointLocalVelocities[i];
                angularVelocities[i] = newPose.jointLocalAngularVelocities[i];
            }

            // The wire format carries a left/right pair, written to the first two contact slots.
            var contacts = _owner.ContactHandles;
            if (contacts.Count >= 2)
            {
                pose.SetBool(contacts[0], newPose.leftFootContact);
                pose.SetBool(contacts[1], newPose.rightFootContact);
            }

            return true;
        }

        return false;
    }

    private void ClientWorker()
    {
        try
        {
            // The socket is created, used, and disposed on this thread. NetMQ sockets are
            // not thread-safe and must not be closed from Unity's main thread.
            using var client = new RequestSocket();
            client.Connect($"tcp://localhost:{port}");

            while (_isRunning)
            {
                if (!_deltaTimeRequests.TryDequeue(out var deltaTime))
                {
                    Thread.Sleep(1);
                    continue;
                }

                client.SendFrame(JsonConvert.SerializeObject(new
                {
                    desired_dir = new[] { 0.0f, 1.0f },
                    delta_time = deltaTime
                }));

                // A REQ socket must receive its reply before sending again. Poll with a
                // short timeout so play-mode shutdown can be observed between attempts.
                while (_isRunning)
                {
                    if (!client.TryReceiveFrameString(
                            TimeSpan.FromMilliseconds(Math.Max(1, receivePollIntervalMs)),
                            out var message))
                    {
                        continue;
                    }

                    var pose = JsonConvert.DeserializeObject<PoseVectorDto>(message);
                    _receivedPoses.Enqueue(pose);
                    break;
                }
            }
        }
        catch (Exception exception)
        {
            if (_isRunning)
            {
                _workerError = exception.ToString();
            }
        }
        finally
        {
            _isRunning = false;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposeState, 1) != 0)
        {
            return;
        }

        _isRunning = false;

        // Receive polling lets this join normally finish within receivePollIntervalMs.
        if (_clientThread is { IsAlive: true })
        {
            if (!_clientThread.Join(TimeSpan.FromSeconds(2)))
            {
                Debug.LogError("MotionField client did not stop within two seconds.");
                return;
            }
        }

        _clientThread = null;
        NetMQConfig.Cleanup();
    }

    private static void ClearQueue<T>(ConcurrentQueue<T> queue)
    {
        while (queue.TryDequeue(out _))
        {
        }
    }

    public override void OnDestroy()
    {
        Dispose();
    }
}
}