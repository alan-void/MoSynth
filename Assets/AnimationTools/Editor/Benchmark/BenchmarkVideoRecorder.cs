using System;
using System.Collections.Generic;
using System.IO;
using Unity.Mathematics;
using UnityEditor;
using UnityEditor.Media;
using UnityEngine;
using UnityEngine.Splines;
using UnityEngine.UI;
using Object = UnityEngine.Object;

namespace AnimationTools.Editor
{
/// <summary>
/// Renders a live benchmark sweep to one MP4 per method, with the path and running speed/lap/error
/// metrics drawn in. One instance covers a whole sweep: a method's encoder stays open across every
/// path it runs, since <see cref="SynthesisBenchmarkDriver"/> iterates paths outermost.
/// </summary>
/// <remarks>
/// Captured on the run's own synthesis tick (<see cref="MotionSynthesisComponent.OnPoseApplied"/>),
/// one video frame per tick encoded at the sweep's synthesis frame rate. Playback therefore runs at
/// normal speed regardless of how fast <c>fixedTimestep</c> actually lets the sweep itself run.
/// </remarks>
public sealed class BenchmarkVideoRecorder : IDisposable
{
    private const float CameraFieldOfView = 45f;
    private const float FollowDistance = 4.5f;
    private const float FollowHeight = 2.2f;
    private const float PathLineWidth = 0.05f;
    private const int PathSamples = 200;

    private static readonly Color PathColor = new(1f, 0.85f, 0.1f);

    private readonly string _outputDirectory;
    private readonly int _width;
    private readonly int _height;
    private readonly float _frameRate;
    private readonly Dictionary<string, MediaEncoder> _encoders = new();
    private readonly SplineProjector _projector = new();

    private GameObject _root;
    private Camera _camera;
    private RenderTexture _target;
    private Texture2D _readback;
    private LineRenderer _pathLine;
    private Material _pathMaterial;
    private Text _label;

    private SplineContainer _currentSpline;
    private MotionSynthesisComponent _synthesizer;
    private BenchmarkLapProbe _probe;
    private string _methodName;
    private string _pathName;
    private float _targetSpeed;
    private float3 _previousRoot;
    private bool _hasPreviousRoot;
    private double _errorSum;
    private int _errorSamples;

    public BenchmarkVideoRecorder(string outputDirectory, int width, int height, float frameRate)
    {
        _outputDirectory = outputDirectory;
        _width = Mathf.Max(16, width & ~1);
        _height = Mathf.Max(16, height & ~1);
        _frameRate = frameRate;

        Directory.CreateDirectory(_outputDirectory);
        BuildResources();
    }

    /// <summary>Starts capturing one run: opens the method's encoder if this is its first run.</summary>
    public void Attach(MotionSynthesisComponent synthesizer, string methodName, string pathName,
        SplineContainer spline, float targetSpeed, BenchmarkLapProbe probe)
    {
        SetPath(spline);

        _synthesizer = synthesizer;
        _probe = probe;
        _methodName = methodName;
        _pathName = pathName;
        _targetSpeed = targetSpeed;
        _projector.Reset();
        _hasPreviousRoot = false;
        _errorSum = 0.0;
        _errorSamples = 0;

        EnsureEncoder(methodName);
        synthesizer.OnPoseApplied += HandlePoseApplied;
    }

    /// <summary>Stops capturing the run in progress. Safe to call even when nothing is attached.</summary>
    public void Detach()
    {
        if (_synthesizer != null) _synthesizer.OnPoseApplied -= HandlePoseApplied;
        _synthesizer = null;
        _probe = null;
    }

    private void SetPath(SplineContainer spline)
    {
        if (spline == _currentSpline) return;
        _currentSpline = spline;

        if (spline == null || spline.Spline == null)
        {
            _pathLine.positionCount = 0;
            return;
        }

        // One extra sample on a closed spline repeats point 0, so the line reads as a closed loop
        // without needing LineRenderer's own loop flag.
        var count = PathSamples + (spline.Spline.Closed ? 1 : 0);
        var points = new Vector3[count];
        for (var i = 0; i < count; i++)
        {
            var t = (float)i / PathSamples;
            points[i] = spline.transform.TransformPoint((Vector3)spline.EvaluatePosition(t));
        }

        _pathLine.positionCount = count;
        _pathLine.SetPositions(points);
    }

    private void HandlePoseApplied(PoseBuffer pose, float deltaTime)
    {
        var root = (float3)_synthesizer.transform.position;

        var speed = 0f;
        if (_hasPreviousRoot && deltaTime > 0f)
        {
            speed = math.length(new float2(root.x - _previousRoot.x, root.z - _previousRoot.z)) / deltaTime;
        }
        _previousRoot = root;
        _hasPreviousRoot = true;

        var pathError = 0f;
        if (_currentSpline != null && _currentSpline.Spline != null)
        {
            var local = (float3)_currentSpline.transform.InverseTransformPoint(root);
            var t = _projector.Project(_currentSpline.Spline, local);
            var nearestWorld = (float3)_currentSpline.transform.TransformPoint(
                (Vector3)_currentSpline.Spline.EvaluatePosition(t));
            pathError = math.distance(new float2(root.x, root.z), new float2(nearestWorld.x, nearestWorld.z));
            _errorSum += pathError;
            _errorSamples++;
        }

        var meanError = _errorSamples > 0 ? (float)(_errorSum / _errorSamples) : 0f;
        var laps = _probe != null ? _probe.CompletedLaps : 0f;

        _label.text = $"{_methodName} - {_pathName}\n" +
                      $"speed {speed:0.00} / {_targetSpeed:0.00} m/s\n" +
                      $"lap {laps:0.00}\n" +
                      $"path err {pathError:0.00} m (avg {meanError:0.00} m)";

        PositionCamera(root);
        RenderFrame();
    }

    private void PositionCamera(float3 root)
    {
        var forward = _synthesizer.transform.forward;
        forward.y = 0f;
        if (forward.sqrMagnitude < 1e-6f) forward = Vector3.forward;
        forward.Normalize();

        var rootWorld = new Vector3(root.x, root.y, root.z);
        var focus = rootWorld + Vector3.up * (FollowHeight * 0.5f);
        var position = rootWorld - forward * FollowDistance + Vector3.up * FollowHeight;
        var rotation = Quaternion.LookRotation((focus - position).normalized, Vector3.up);

        _camera.transform.SetPositionAndRotation(position, rotation);
    }

    private void RenderFrame()
    {
        _camera.Render();

        RenderTexture.active = _target;
        _readback.ReadPixels(new Rect(0, 0, _width, _height), 0, 0, false);
        _readback.Apply(false);
        RenderTexture.active = null;

        if (_encoders.TryGetValue(_methodName, out var encoder)) encoder.AddFrame(_readback);
    }

    private void EnsureEncoder(string methodName)
    {
        if (_encoders.ContainsKey(methodName)) return;

        var path = Path.Combine(_outputDirectory, Sanitize(methodName) + ".mp4");
        var attributes = new VideoTrackEncoderAttributes(new H264EncoderAttributes
        {
            gopSize = 25,
            numConsecutiveBFrames = 2,
            profile = VideoEncodingProfile.H264High
        })
        {
            frameRate = ToRational(_frameRate),
            width = (uint)_width,
            height = (uint)_height,
            includeAlpha = false,
            bitRateMode = VideoBitrateMode.High
        };

        _encoders[methodName] = new MediaEncoder(path, attributes);
    }

    private static MediaRational ToRational(float fps)
    {
        var rounded = Mathf.RoundToInt(fps);
        return Mathf.Approximately(rounded, fps)
            ? new MediaRational(rounded)
            : new MediaRational(Mathf.RoundToInt(fps * 1000f), 1000);
    }

    private void BuildResources()
    {
        _root = new GameObject("BenchmarkVideoRecorder") { hideFlags = HideFlags.HideAndDontSave };

        var cameraGo = new GameObject("CaptureCamera") { hideFlags = HideFlags.HideAndDontSave };
        cameraGo.transform.SetParent(_root.transform);
        _camera = cameraGo.AddComponent<Camera>();
        _camera.enabled = false;
        _camera.fieldOfView = CameraFieldOfView;
        _camera.nearClipPlane = 0.05f;
        _camera.farClipPlane = 500f;
        _camera.clearFlags = CameraClearFlags.Skybox;

        _target = new RenderTexture(_width, _height, 24, RenderTextureFormat.ARGB32)
        {
            hideFlags = HideFlags.HideAndDontSave
        };
        _target.Create();
        _camera.targetTexture = _target;

        _readback = new Texture2D(_width, _height, TextureFormat.RGBA32, false, false)
        {
            hideFlags = HideFlags.HideAndDontSave
        };

        var lineGo = new GameObject("PathLine") { hideFlags = HideFlags.HideAndDontSave };
        lineGo.transform.SetParent(_root.transform);
        _pathMaterial = new Material(Shader.Find("Universal Render Pipeline/Unlit"))
        {
            hideFlags = HideFlags.HideAndDontSave
        };
        _pathMaterial.SetColor("_BaseColor", PathColor);
        _pathLine = lineGo.AddComponent<LineRenderer>();
        _pathLine.material = _pathMaterial;
        _pathLine.widthMultiplier = PathLineWidth;
        _pathLine.useWorldSpace = true;
        _pathLine.positionCount = 0;
        _pathLine.numCapVertices = 4;

        var canvasGo = new GameObject("HudCanvas", typeof(RectTransform)) { hideFlags = HideFlags.HideAndDontSave };
        canvasGo.transform.SetParent(_root.transform);
        var canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceCamera;
        canvas.worldCamera = _camera;
        canvas.planeDistance = 1f;
        canvasGo.AddComponent<CanvasScaler>();

        var labelGo = new GameObject("Label", typeof(RectTransform)) { hideFlags = HideFlags.HideAndDontSave };
        labelGo.transform.SetParent(canvasGo.transform, false);
        _label = labelGo.AddComponent<Text>();
        _label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        _label.fontSize = 18;
        _label.color = Color.white;
        _label.alignment = TextAnchor.UpperLeft;

        var rect = _label.rectTransform;
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(1f, 1f);
        rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = new Vector2(12f, -12f);
        rect.sizeDelta = new Vector2(-24f, 100f);
    }

    /// <summary>Makes a method name safe to use as a video filename.</summary>
    private static string Sanitize(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = name.ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (Array.IndexOf(invalid, chars[i]) >= 0 || chars[i] == ' ') chars[i] = '_';
        }

        return new string(chars);
    }

    public void Dispose()
    {
        Detach();

        foreach (var encoder in _encoders.Values) encoder.Dispose();
        _encoders.Clear();

        if (_target != null)
        {
            _target.Release();
            Object.DestroyImmediate(_target);
        }

        if (_readback != null) Object.DestroyImmediate(_readback);
        if (_pathMaterial != null) Object.DestroyImmediate(_pathMaterial);
        if (_root != null) Object.DestroyImmediate(_root);

        _target = null;
        _readback = null;
        _pathMaterial = null;
        _root = null;
    }
}
}
