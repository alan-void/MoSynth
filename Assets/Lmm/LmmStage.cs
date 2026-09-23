using System;
using AnimationTools;
using MotionMatching;
using Python.Runtime;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace Lmm
{
/// <summary>
/// How much of Learned Motion Matching is running. The modes are cumulative permanent ablations:
/// each replaces one more piece of the classic matcher, so the cost of each can be measured.
/// </summary>
/// <remarks>See openwiki/motion-matching/learned-motion-matching.md.</remarks>
public enum LmmMode
{
    /// <summary>
    /// The decompressor only: the database is still searched and played, but the pose is
    /// reconstructed from a latent instead of read out of the <c>.mmpose</c>.
    /// </summary>
    DecompressorOnly,

    /// <summary>
    /// Adds the stepper, which advances the state between searches so playback no longer walks the
    /// database.
    /// </summary>
    Stepper,

    /// <summary>
    /// Adds the projector, which replaces the search: the query is answered with a state rather
    /// than looked up, so no acceleration structure is built. This is the whole method.
    /// </summary>
    Full,
}

/// <summary>
/// Synthesises motion with a trained Learned Motion Matching model (Holden et al. 2020), running
/// the networks in Python behind PythonNET.
/// </summary>
/// <remarks>
/// Searches the same database with the same query as <see cref="MotionMatchingStage"/> (see
/// <see cref="MotionMatchingQuery"/>), which is what makes the two comparable. The state is
/// <c>(X, Z)</c>: the held feature vector and its latent. In <see cref="LmmMode.DecompressorOnly"/>
/// the latent table stays in Python and the stage passes a frame index; later modes carry the
/// state here.
/// <para>
/// <b>Never splice the controller's trajectory into <c>X</c> between searches</b>: the stepper and
/// decompressor consume <c>X</c> whole, and a spliced trajectory pairs with a latent never seen in
/// training. The controller enters through the query on search ticks only.
/// </para>
/// <para>See openwiki/motion-matching/learned-motion-matching.md.</para>
/// </remarks>
[Serializable]
public class LmmStage : MoSynthStage, IDisposable, IMotionMatchingDataProvider
{
    [SerializeField]
    [Tooltip("The trained config. Its checkpoint decides which bones this stage writes.")]
    public LmmConfig config;

    [SerializeField]
    [Tooltip("How much of the method is running. Each mode replaces one more piece of the classic " +
             "matcher, so the cost of each replacement can be measured on its own.")]
    public LmmMode mode = LmmMode.DecompressorOnly;

    [SerializeField]
    [Tooltip("Re-read the Python modules on Init, so edits apply without restarting the editor. " +
             "Turn off for builds.")]
    public bool reloadPythonModules = true;

    /// <summary>
    /// How the database is searched; the same strategy type <see cref="MotionMatchingStage"/> uses,
    /// so the two methods can be held to one search.
    /// </summary>
    [SerializeReference] [SubclassSelector]
    public MotionMatchingSearch mmSearch = new BvhMotionMatchingSearch();

    [SerializeField]
    [Tooltip("Seconds between searches when the input is not changing sharply.")]
    public float searchInterval = 10.0f / 60.0f;

    /// <summary>
    /// A candidate must beat the state already held by this factor to be taken.
    /// </summary>
    /// <remarks>
    /// Stands in for <c>MotionMatchingStage.minFrameSwitchDistance</c>, since in later modes there is
    /// no held frame for a candidate to be near.
    /// </remarks>
    [SerializeField] [Range(0.5f, 1f)]
    public float acceptanceRatio = 0.95f;

    /// <summary>
    /// Whether an accepted jump raises <see cref="MotionSynthesisComponent.PoseDiscontinuity"/>.
    /// </summary>
    /// <remarks>
    /// Leave it on unless the ablation is the measurement: without it the learned matcher reads as
    /// smoother than the classic one on the same jumps. A stage field so a <c>BenchmarkOverride</c>
    /// can flip it without dirtying a ScriptableObject.
    /// </remarks>
    [SerializeField]
    public bool raisePoseDiscontinuity = true;

    [SerializeField]
    [Tooltip("Where inference runs. A single-sample forward pass is small, so CPU usually beats " +
             "the per-call transfer to a GPU.")]
    public LmmConfig.ComputeDevice inferenceDevice = LmmConfig.ComputeDevice.Cpu;

    // Static so the delegate stays rooted for as long as Python holds it; PythonNET does not
    // forward Python's stdout to Unity on its own.
    private static readonly Action<string> PythonLog = message => Debug.Log(message);

    private bool _isInitialized;
    private bool _initializationFailed;

    private dynamic _policy;
    private MotionSynthesisComponent _owner;
    private SkeletonData _skeletonData;
    private PoseSet _poseSet;
    private FeatureSet _featureSet;
    private float _databaseFrameRate;

    /// <summary>The database this drives from — see <see cref="IMotionMatchingDataProvider"/>.</summary>
    public MotionMatchingData MmData => config == null ? null : config.mmData;

    /// <summary>
    /// Current frame index in the pose/feature set. In <see cref="LmmMode.Stepper"/> this is the
    /// frame the last accepted search returned, not a playhead — nothing advances it.
    /// </summary>
    public int CurrentFrame { get; private set; }

    private float _currentFrameTime;
    private float _searchTimeLeft;
    private bool _warnedNoControlInput;

    // Search ------------------------------------------------------------------------------------
    private NativeArray<float> _featureWeights;
    private NativeArray<float> _authoredFeatureWeights;
    private NativeArray<float> _queryFeatureVector;
    private NativeArray<bool> _tagMask;

    // Packing, read off the checkpoint at Init ---------------------------------------------------
    private int[] _boneIndices;   // checkpoint slot -> skeleton bone
    private int[] _slotOfBone;    // skeleton bone -> checkpoint slot, or -1
    private PoseVectorLayout _poseLayout;

    /// <summary>
    /// Frames the search may hold, i.e. frames with a baked latent. The last frame of every clip
    /// has none, because a latent spans a frame and its successor.
    /// </summary>
    private NativeArray<bool> _hasLatent;

    /// <summary>
    /// The last frame whose latent was used, held while playback sits on a clip's latent-less last
    /// frame — where a clip crossing already raises a discontinuity.
    /// </summary>
    private int _latentFrame;

    // Per-tick working arrays, all in the character frame -----------------------------------------
    private NativeArray<float3> _positions;
    private NativeArray<quaternion> _rotations;
    private NativeArray<float3> _velocities;
    private NativeArray<float3> _angularVelocities;

    // The state, and what crosses the boundary ----------------------------------------------------

    /// <summary>The matching feature vector the character is holding.</summary>
    private float[] _x;

    /// <summary>
    /// The latent beside it. Only <see cref="LmmMode.Stepper"/> carries this on the C# side;
    /// <see cref="LmmMode.DecompressorOnly"/> names a database frame and lets Python look it up.
    /// </summary>
    private float[] _z;

    /// <summary>
    /// The query as a managed array, which is what PythonNET marshals for <see cref="LmmMode.Full"/>.
    /// </summary>
    private float[] _query;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        if (_isInitialized || _initializationFailed) return;

        _owner = motionSynthesisComponent;

        if (!TryBindDatabase()) return;

        try
        {
            PythonRuntime.EnsureInitialized();

            using (Py.GIL())
            {
                // Once, up front, so every import below comes from one generation of the source.
                if (reloadPythonModules) PythonRuntime.InvalidateProjectModules();

                dynamic runtime = PythonRuntime.Import("lmm.runtime");
                _policy = runtime.LmmPolicy(config.GetCheckpointPath(), PythonLog,
                    DeviceName(inferenceDevice));

                if (!TryBindCheckpoint()) return;
                Debug.Log($"[LMM] {config.name}: {(string)_policy.describe()}");
            }

            AllocateBuffers();
            SeedFromFirstUsableFrame();

            // LmmMode.Full never searches, and removing the structure's cost is the mode's point.
            if (mode != LmmMode.Full) mmSearch.Initialize(_featureSet, _tagMask, _featureWeights);
            _isInitialized = true;
        }
        catch (Exception e)
        {
            _initializationFailed = true;
            Debug.LogError($"[LMM] failed to start. Check Project Settings > MoSynth > Python, and " +
                           $"that '{config.GetCheckpointPath()}' exists — press Train on the config " +
                           "if it does not.");
            Debug.LogException(e);
        }
    }

    /// <summary>
    /// Resolves the database and checks it against the rig, before any of it reaches Python.
    /// </summary>
    private bool TryBindDatabase()
    {
        if (config == null)
        {
            Fail("LmmStage has no config assigned.");
            return false;
        }

        if (!config.TryValidate(out var error))
        {
            Fail($"'{config.name}' is not usable — {error}");
            return false;
        }

        _poseSet = config.mmData.GetOrImportPoseSet();
        _featureSet = config.mmData.GetOrImportFeatureSet();

        if (_poseSet == null || _featureSet == null)
        {
            Fail($"MotionMatchingData '{config.mmData.name}' produced no database. Regenerate it.");
            return false;
        }

        if (!Skeleton.StructurallyEqual(_owner.Skeleton, _poseSet.Skeleton))
        {
            Fail($"the component's Skeleton is not the rig MotionMatchingData " +
                 $"'{config.mmData.name}' was built over. Assign that rig's root bone, or " +
                 "regenerate the database.");
            return false;
        }

        _skeletonData = _owner.SkeletonData;
        _databaseFrameRate = 1f / _poseSet.FrameTime;
        return true;
    }

    /// <summary>
    /// Checks that the checkpoint describes the database and the rig this stage is about to run it
    /// against. Must be called with the GIL held.
    /// </summary>
    /// <remarks>
    /// None of these mismatches throws on its own; each only makes the character move wrongly.
    /// </remarks>
    private bool TryBindCheckpoint()
    {
        if (!HasStageFor(mode, out var missing))
        {
            Fail($"'{config.name}' has no {missing} in its checkpoint, which {mode} needs. " +
                 "Press Train on the config.");
            return false;
        }

        var frames = (int)_policy.n_frames();
        if (frames != _featureSet.NumberFeatureVectors)
        {
            Fail($"'{config.name}' was trained on {frames} frames but '{config.mmData.name}' now " +
                 $"holds {_featureSet.NumberFeatureVectors}. The latents are indexed by frame, so " +
                 "regenerating the database invalidates them — retrain.");
            return false;
        }

        var featureSize = (int)_policy.feature_size();
        if (featureSize != _featureSet.FeatureSize)
        {
            Fail($"'{config.name}' was trained on {featureSize}-float queries but " +
                 $"'{config.mmData.name}' now produces {_featureSet.FeatureSize}. Retrain.");
            return false;
        }

        if (!PredictedBoneSelection.TryResolve((string[])_policy.bone_names(), _owner.Skeleton,
                out _boneIndices, out var error))
        {
            Fail($"'{config.name}' checkpoint does not fit this character: {error}. Regenerate the " +
                 "database and retrain if the rig changed.");
            return false;
        }

        if (!PoseVectorLayout.TryBind((string[])_policy.pose_blocks(),
                (int[])_policy.pose_block_offsets(), _boneIndices.Length, out _poseLayout, out error))
        {
            Fail($"'{config.name}' checkpoint predicts a pose this code cannot read: {error}. Retrain.");
            return false;
        }

        _slotOfBone = new int[_skeletonData.BoneCount];
        for (var i = 0; i < _slotOfBone.Length; i++) _slotOfBone[i] = -1;
        for (var slot = 0; slot < _boneIndices.Length; slot++) _slotOfBone[_boneIndices[slot]] = slot;

        return TryBindFeatureWeights();
    }

    /// <summary>
    /// The metric the checkpoint was fitted under must be the metric the stage searches with.
    /// </summary>
    /// <remarks>
    /// The projector approximates a lookup under exactly these weights. Checked in every mode so a
    /// mismatch surfaces before <see cref="LmmMode.Full"/> is switched on.
    /// </remarks>
    private bool TryBindFeatureWeights()
    {
        var size = _featureSet.FeatureSize;
        var authored = new float[size];
        config.ExpandFeatureWeights(_featureSet, authored);

        var trained = (float[])_policy.feature_weights();
        for (var i = 0; i < size; i++)
        {
            if (math.abs(trained[i] - authored[i]) < 1e-5f) continue;

            Fail($"'{config.name}' was trained with different search weights than it now carries " +
                 $"(float {i}: {trained[i]} against {authored[i]}). Retrain, or put the weights back.");
            return false;
        }

        _authoredFeatureWeights = new NativeArray<float>(size, Allocator.Domain);
        _authoredFeatureWeights.CopyFrom(authored);
        _featureWeights = new NativeArray<float>(size, Allocator.Domain);
        _featureWeights.CopyFrom(authored);
        _queryFeatureVector = new NativeArray<float>(size, Allocator.Domain);
        return true;
    }

    private void AllocateBuffers()
    {
        _x = new float[_featureSet.FeatureSize];
        _query = new float[_featureSet.FeatureSize];

        var boneCount = _skeletonData.BoneCount;
        _positions = new NativeArray<float3>(boneCount, Allocator.Domain);
        _rotations = new NativeArray<quaternion>(boneCount, Allocator.Domain);
        _velocities = new NativeArray<float3>(boneCount, Allocator.Domain);
        _angularVelocities = new NativeArray<float3>(boneCount, Allocator.Domain);

        _hasLatent = new NativeArray<bool>(_featureSet.NumberFeatureVectors, Allocator.Domain);
        for (var i = 0; i < _hasLatent.Length; i++) _hasLatent[i] = true;

        int[] withoutLatent;
        using (Py.GIL()) withoutLatent = (int[])_policy.invalid_latent_frames();
        foreach (var frame in withoutLatent) _hasLatent[frame] = false;

        // Feature validity is already the search's own filter, so the mask carries only what it
        // does not know about: whether the frame has a latent to reconstruct a pose from.
        _tagMask = new NativeArray<bool>(_hasLatent.Length, Allocator.Domain);
        _tagMask.CopyFrom(_hasLatent);
    }

    /// <summary>
    /// Start on the first frame the stage could actually hold, and put its state in hand.
    /// </summary>
    /// <remarks>
    /// Seeded here because the first search scores the held state and fills the query's pose half
    /// from it; an unseeded <c>X</c> would be a pose of zeros.
    /// </remarks>
    private void SeedFromFirstUsableFrame()
    {
        for (var i = 0; i < _featureSet.NumberFeatureVectors; i++)
        {
            if (!_featureSet.IsValidFeature(i) || !_hasLatent[i]) continue;

            CurrentFrame = i;
            _latentFrame = i;
            _featureSet.GetFeatureVector(i).CopyTo(_x);
            using (Py.GIL()) _z = (float[])_policy.latent(i);
            return;
        }

        Fail($"no frame of '{config.mmData.name}' has both a valid feature vector and a latent.");
    }

    /// <summary>Whatever is steering the character, when it is a motion matching input.</summary>
    private MotionMatchingControlInput ControlInput =>
        _owner?.ControlInput as MotionMatchingControlInput;

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (!_isInitialized) return true;

        var controlInput = ControlInput;
        if (controlInput == null)
        {
            if (!_warnedNoControlInput)
            {
                Debug.LogWarning($"[LMM] no MotionMatchingControlInput is steering " +
                                 $"\"{_owner.name}\", so it cannot search. Add one and point it at " +
                                 "the character.");
                _warnedNoControlInput = true;
            }

            return true;
        }

        try
        {
            // A large input change searches now rather than waiting out the interval.
            if (controlInput.ConsumeHighInputChange()) _searchTimeLeft = 0f;

            var accepted = -1;
            if (_searchTimeLeft <= 0f)
            {
                FillQueryVector(controlInput);
                var heldDistance = HeldStateDistance();

                // Full is told the answer by the projector; the other modes look it up.
                if (mode == LmmMode.Full) ProjectBetterState(heldDistance);
                else accepted = SearchForBetterFrame(heldDistance);

                _searchTimeLeft = searchInterval;
            }
            else
            {
                _searchTimeLeft -= deltaTime;
            }

            var y = mode == LmmMode.DecompressorOnly
                ? PlayFromDatabase(deltaTime)
                : StepState(accepted, deltaTime);

            BuildCharacterSpacePose(y);
            WritePose(pose, y);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            // Stop rather than log the same failure every frame.
            _isInitialized = false;
        }

        return true;
    }

    /// <summary>
    /// What the state the character is already holding scores against the query it has just built.
    /// </summary>
    /// <remarks>
    /// Scores <c>X</c> itself, not a database frame's feature vector: in later modes the state is
    /// not a database row.
    /// </remarks>
    private float HeldStateDistance()
    {
        // A stepped or projected state is always scoreable; a played one is only scoreable while
        // the playhead is on a frame the search itself would have been allowed to return.
        var scoreable = mode != LmmMode.DecompressorOnly ||
                        (_featureSet.IsValidFeature(CurrentFrame) && _tagMask[CurrentFrame]);

        return scoreable
            ? MotionMatchingStage.SqrDistance(_queryFeatureVector, _x, _featureWeights)
            : float.MaxValue;
    }

    /// <summary>
    /// Searches the database and takes a better state, if one is enough better to be worth the jump.
    /// </summary>
    /// <returns>The database frame that was accepted, or -1 when the held state stood.</returns>
    private int SearchForBetterFrame(float heldDistance)
    {
        // Passed into the search as a bound so it can abandon candidates early.
        var bestFrame = mmSearch.FindBestFrame(_queryFeatureVector, heldDistance * acceptanceRatio);

        if (bestFrame == -1)
        {
            Debug.Assert(heldDistance < float.MaxValue,
                "Learned motion matching found no usable state and is not holding one. The database " +
                "may be empty, or every frame may be masked out.");
            return -1;
        }

        CurrentFrame = bestFrame;
        _currentFrameTime = bestFrame;
        if (raisePoseDiscontinuity) _owner.PoseDiscontinuity = true;
        return bestFrame;
    }

    /// <summary>
    /// Asks the projector for a state instead of searching for one — the
    /// <see cref="LmmMode.Full"/> search tick.
    /// </summary>
    /// <remarks>
    /// Acceptance is decided here, under the same weights and distance as
    /// <see cref="SearchForBetterFrame"/>, so both modes share one accept rule.
    /// </remarks>
    private void ProjectBetterState(float heldDistance)
    {
        _queryFeatureVector.CopyTo(_query);

        float[] candidateX, candidateZ;
        using (Py.GIL())
        {
            dynamic projected = _policy.project(_query);
            candidateX = (float[])projected[0];
            candidateZ = (float[])projected[1];
        }

        var offered = MotionMatchingStage.SqrDistance(_queryFeatureVector, candidateX, _featureWeights);
        if (offered >= heldDistance * acceptanceRatio) return;

        _x = candidateX;
        _z = candidateZ;
        if (raisePoseDiscontinuity) _owner.PoseDiscontinuity = true;
    }

    /// <summary>
    /// Advances the playhead and reads the state off the database — the
    /// <see cref="LmmMode.DecompressorOnly"/> tick.
    /// </summary>
    /// <returns>The reconstructed pose vector.</returns>
    private float[] PlayFromDatabase(float deltaTime)
    {
        AdvancePlayback(deltaTime);

        _featureSet.GetFeatureVector(CurrentFrame).CopyTo(_x);
        if (_hasLatent[CurrentFrame]) _latentFrame = CurrentFrame;

        using (Py.GIL()) return (float[])_policy.decompress_frame(_x, _latentFrame);
    }

    /// <summary>
    /// Advances the state with the stepper and reconstructs the pose it now describes — the tick
    /// of both <see cref="LmmMode.Stepper"/> and <see cref="LmmMode.Full"/>.
    /// </summary>
    /// <remarks>
    /// <see cref="LmmMode.Full"/> passes -1, having had <see cref="ProjectBetterState"/> write the
    /// state already. The state is advanced before it is decompressed, matching how
    /// <see cref="PlayFromDatabase"/> advances its playhead first.
    /// </remarks>
    /// <param name="acceptedFrame">A database frame to restart the state from, or -1.</param>
    private float[] StepState(int acceptedFrame, float deltaTime)
    {
        using (Py.GIL())
        {
            if (acceptedFrame >= 0)
            {
                _featureSet.GetFeatureVector(acceptedFrame).CopyTo(_x);
                _z = (float[])_policy.latent(acceptedFrame);
            }

            dynamic stepped = _policy.tick(_x, _z, deltaTime);
            var y = (float[])stepped[0];
            _x = (float[])stepped[1];
            _z = (float[])stepped[2];
            return y;
        }
    }

    /// <summary>
    /// The query: the trajectory the control input wants, then the pose features of the state.
    /// </summary>
    /// <remarks>
    /// The trajectory half is shared with <see cref="MotionMatchingStage"/>; the pose half is read
    /// from <c>X</c>, the "previous pose features" the paper's projector takes as state.
    /// </remarks>
    private void FillQueryVector(MotionMatchingControlInput controlInput)
    {
        var query = _queryFeatureVector.AsSpan();

        MotionMatchingQuery.FillTrajectory(config.mmData, _featureSet, controlInput,
            _owner.transform, query, _featureWeights, _authoredFeatureWeights);

        _x.AsSpan(_featureSet.PoseOffset, _featureSet.PoseFloatCount)
            .CopyTo(query.Slice(_featureSet.PoseOffset, _featureSet.PoseFloatCount));
    }

    /// <summary>
    /// Advances the playhead by real time, converted to database frames. Fractional frame time
    /// carries across ticks, so playback stays at the right speed at any synthesis rate.
    /// </summary>
    private void AdvancePlayback(float deltaTime)
    {
        var preAdvanceFrame = CurrentFrame;

        _currentFrameTime = CurrentFrame + math.frac(_currentFrameTime);
        _currentFrameTime += deltaTime * _databaseFrameRate;
        CurrentFrame = math.min((int)math.floor(_currentFrameTime), _hasLatent.Length - 1);

        // Running off the end of a clip into the next one is a pose jump, same as a search switch.
        if (CurrentFrame != preAdvanceFrame && raisePoseDiscontinuity &&
            _poseSet.GetAnimationClipIndex(CurrentFrame) !=
            _poseSet.GetAnimationClipIndex(preAdvanceFrame))
        {
            _owner.PoseDiscontinuity = true;
        }
    }

    /// <summary>
    /// Turns the decompressor's output into a full character-frame pose.
    /// </summary>
    /// <remarks>
    /// Composes the predicted joint-local rotations as <c>lmm.fk.forward_kinematics</c> does in the
    /// training loss. The root's height is the only predicted translation; its ground-plane
    /// coordinates are zero in the character frame. An unpredicted bone takes its rest local
    /// rotation, well defined because the predicted set is closed under parent.
    /// </remarks>
    private void BuildCharacterSpacePose(float[] y)
    {
        for (var bone = 0; bone < _skeletonData.BoneCount; bone++)
        {
            var slot = _slotOfBone[bone];
            var parent = _skeletonData.ParentIndices[bone];

            var local = slot >= 0
                ? RotationFrom6D(y, _poseLayout.Rotations + slot * 6)
                : _skeletonData.RestLocalRotations[bone];

            _rotations[bone] = bone == 0 ? local : math.mul(_rotations[parent], local);

            _positions[bone] = bone == 0
                ? new float3(0f, y[_poseLayout.RootHeight], 0f)
                : _positions[parent] +
                  math.rotate(_rotations[parent], _skeletonData.RestLocalPositions[bone]);

            _velocities[bone] = slot >= 0
                ? ReadFloat3(y, _poseLayout.Velocities + slot * 3)
                : float3.zero;
            _angularVelocities[bone] = slot >= 0
                ? ReadFloat3(y, _poseLayout.AngularVelocities + slot * 3)
                : float3.zero;
        }
    }

    /// <summary>Writes the reconstructed pose and the character's own motion into the pipeline.</summary>
    /// <remarks>
    /// The root's motion is already a per-second rate, so unlike <c>PfnnStage</c> it is not divided
    /// by the frame time. The pose is written in a frame at the origin: the component accumulates
    /// the travel onto its own Transform, so an absolute frame would be applied twice.
    /// </remarks>
    private void WritePose(PoseBuffer pose, float[] y)
    {
        var rootVelocity = ReadFloat3(y, _poseLayout.RootVelocity);
        var rootYawRate = y[_poseLayout.RootYawRate];

        CharacterSpacePose.Apply(pose, _skeletonData, float3.zero, quaternion.identity,
            rootVelocity, rootYawRate, _positions, _rotations, _velocities, _angularVelocities);

        pose.SetBool(_owner.LeftFootContactHandle, y[_poseLayout.Contacts] > 0.5f);
        pose.SetBool(_owner.RightFootContactHandle, y[_poseLayout.Contacts + 1] > 0.5f);
    }

    /// <summary>The training stage a mode needs, and whether the checkpoint carries it.</summary>
    private bool HasStageFor(LmmMode wanted, out string missing)
    {
        missing = wanted switch
        {
            LmmMode.Stepper => "stepper",
            LmmMode.Full => "projector",
            _ => "autoencoder"
        };

        foreach (var stage in (string[])_policy.stages_trained())
        {
            if (stage == missing) return true;
        }

        return false;
    }

    /// <summary>
    /// A rotation from the two-axis form the network regresses, by Gram-Schmidt.
    /// </summary>
    /// <remarks>
    /// The C# counterpart of <c>training.training_data.rotations_from_6d</c> (Zhou et al. 2019).
    /// </remarks>
    private static quaternion RotationFrom6D(float[] source, int offset)
    {
        var first = new float3(source[offset], source[offset + 1], source[offset + 2]);
        var second = new float3(source[offset + 3], source[offset + 4], source[offset + 5]);

        var column0 = math.normalize(first);
        var column1 = math.normalize(second - math.dot(column0, second) * column0);
        var column2 = math.cross(column0, column1);

        return new quaternion(new float3x3(column0, column1, column2));
    }

    private static float3 ReadFloat3(float[] source, int offset) =>
        new(source[offset], source[offset + 1], source[offset + 2]);

    private static string DeviceName(LmmConfig.ComputeDevice device) => device switch
    {
        LmmConfig.ComputeDevice.Cuda => "cuda",
        LmmConfig.ComputeDevice.Cpu => "cpu",
        _ => "auto"
    };

    /// <summary>Reports why the stage will not run, and makes sure it does not try again.</summary>
    private void Fail(string reason)
    {
        _initializationFailed = true;
        Debug.LogError($"[LMM] {reason}", config);
    }

    public void Dispose()
    {
        if (_policy != null)
        {
            // The GIL is held while dropping the handle because that is what decrefs the object.
            using (Py.GIL()) _policy = null;
        }

        if (_positions.IsCreated) _positions.Dispose();
        if (_rotations.IsCreated) _rotations.Dispose();
        if (_velocities.IsCreated) _velocities.Dispose();
        if (_angularVelocities.IsCreated) _angularVelocities.Dispose();

        mmSearch?.Dispose();
        _isInitialized = false;
    }

    public override void OnDestroy() => Dispose();
}
}
