using System;
using System.Collections.Generic;
using AnimationTools;
using GameplayTags;
using Python.Runtime;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace Style
{
/// <summary>
/// Restyles the pose any synthesis method produced — sad, old, proud — as per-bone rotation deltas
/// on top of it, plus bone 0's height and optionally its travel. At zero intensity it is the identity.
/// </summary>
/// <remarks>
/// Place it after <c>Inertialization</c> and before <see cref="RootFollowStage"/> and
/// <see cref="ContactLockStage"/>. Timing stays with the method: contact flags, velocity channels and
/// every bone position but bone 0's height are left alone, and bone 0's frame velocity is touched
/// only under <see cref="RootMotionMode.Stylize"/>. The model runs in Python behind one stateless
/// <c>step</c> per tick; the stage owns the phase, the history and the blends.
/// </remarks>
[Serializable]
public class StyleTransferStage : MoSynthStage, IDisposable, IContactBoneSource
{
    public enum RootMotionMode
    {
        /// <summary>The method owns the root; only bone 0's height, pitch and roll are styled.</summary>
        Preserve,

        /// <summary>The root's speed, sway and yaw wobble are styled too, through bone 0's frame velocity.</summary>
        Stylize
    }

    /// <summary>A weight for one bone's subtree, overriding any listed ancestor's.</summary>
    [Serializable]
    public class RegionWeight
    {
        public SkeletonBone bone;
        [Range(0f, 2f)] public float weight = 1f;
    }

    [Tooltip("The trained config. Its checkpoint decides which bones this stage restyles.")]
    public StyleTransferConfig config;

    [Tooltip("The style to apply, a tag under the config's style root.")]
    public GameplayTagSO style;

    [Tooltip("How much of the time-invariant posture offset to apply. Above 1 exaggerates.")]
    [Range(0f, 2f)] public float postureIntensity = 1f;

    [Tooltip("How much of the phase-dependent dynamics to apply. Above 1 exaggerates.")]
    [Range(0f, 2f)] public float dynamicsIntensity = 1f;

    [Tooltip("Preserve leaves the root's travel to the method. Stylize also restyles its speed, sway " +
             "and yaw wobble, so where the character goes then depends on the style.")]
    public RootMotionMode rootMotion = RootMotionMode.Preserve;

    [Tooltip("How much root styling to apply. Read only under Stylize.")]
    [Range(0f, 2f)] public float rootIntensity = 1f;

    [Tooltip("Per-subtree intensity, e.g. 0 on the hips for \"from the waist up\". A bone takes its " +
             "nearest listed ancestor's weight; unlisted bones take 1.")]
    public List<RegionWeight> regionWeights = new();

    [Tooltip("Half-life of the crossfade when the style or the root mode changes, in seconds.")]
    [Min(0f)] public float styleBlendHalfLife = 0.3f;

    [Tooltip("Turn the style down where the input is unlike anything the model was trained on.")]
    public bool gateEnabled = true;

    [Tooltip("Half-life the gate is smoothed with, so the style never flickers.")]
    [Min(0f)] public float gateHalfLife = 0.2f;

    [Tooltip("Re-read the Python modules on Init, so edits apply without restarting the editor. " +
             "Turn off for builds.")]
    public bool reloadPythonModules = true;

    [Tooltip("Where inference runs. A single-sample forward pass is small, so CPU usually wins.")]
    public StyleTransferConfig.ComputeDevice inferenceDevice = StyleTransferConfig.ComputeDevice.Cpu;

    // Static so the delegate stays rooted for as long as Python holds it; PythonNET does not
    // forward Python's stdout to Unity on its own.
    private static readonly Action<string> PythonLog = message => Debug.Log(message);

    private const int LeftSlot = 0;
    private const int RightSlot = 1;
    private const float BlendSettled = 1e-3f;

    private bool _isInitialized;
    private bool _initializationFailed;

    private dynamic _policy;
    private MotionSynthesisComponent _owner;
    private SkeletonData _skeletonData;
    private SimulationFrameDef _frameDef;

    // Bound off the checkpoint at Init ---------------------------------------------------------
    private int[] _styledBones;          // styled slot -> skeleton bone; slot 0 is bone 0
    private Dictionary<string, int> _styleIndexByName;
    private float[] _historyOffsets;
    private int _descriptorSize;
    private float[] _boneWeights;        // per skeleton bone, from regionWeights

    // Per-tick working state -------------------------------------------------------------------
    private NativeArray<float3> _positions;
    private NativeArray<quaternion> _rotations;
    private float[] _rotationsIn;
    private float[] _positionsIn;
    private float[] _contacts;
    private float[] _rootVelocity;
    private float[] _history;
    private int[] _oneStyle;
    private int[] _twoStyles;

    private OnlineGaitPhase _gait;
    private double _time;

    // The descriptors step returned, keyed by synthesis time, oldest first from _ringStart.
    private float[] _ringDescriptors;
    private double[] _ringTimes;
    private int _ringStart;
    private int _ringCount;

    // Blends -----------------------------------------------------------------------------------
    private GameplayTagSO _appliedStyle;
    private int _styleIndex;
    private int _previousStyleIndex;       // -1 when no crossfade is running
    private float _styleBlend;              // share of _styleIndex while crossfading
    private float _transitionHalfLife;
    private float _gate;
    private bool _hasGate;
    private float _rootBlend;
    private bool _warnedRootCancelled;
    private bool _wasStylize;

    public IReadOnlyList<string> ContactBoneNames => config == null ? null : config.ContactBoneNames;

    /// <summary>The causal gait phase the model was last fed, in radians.</summary>
    public float Phase => _gait?.Phase ?? 0f;

    /// <summary>The smoothed gate last applied, in [0, 1].</summary>
    public float Gate => _gate;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        if (_isInitialized || _initializationFailed) return;

        _owner = motionSynthesisComponent;

        if (config == null)
        {
            Fail("StyleTransferStage has no config assigned.");
            return;
        }

        if (!config.TryGetDatabaseSkeleton(out var configSkeleton) ||
            !Skeleton.StructurallyEqual(configSkeleton, _owner.Skeleton))
        {
            Fail($"the skeleton of '{config.name}' is not structurally equal to this character's.");
            return;
        }

        if (_owner.ContactHandles.Count != 2)
        {
            Fail($"the gait phase needs two contact slots, left then right; the character has " +
                 $"{_owner.ContactHandles.Count}.");
            return;
        }

        _skeletonData = _owner.SkeletonData;
        _frameDef = _owner.SimulationFrame;

        try
        {
            PythonRuntime.EnsureInitialized();

            using (Py.GIL())
            {
                // Once, up front, so every import below comes from one generation of the source.
                if (reloadPythonModules) PythonRuntime.InvalidateProjectModules();

                dynamic runtime = PythonRuntime.Import("style.runtime");
                _policy = runtime.StylePolicy(config.GetCheckpointPath(), PythonLog,
                    StyleTransferConfig.DeviceNameOf(inferenceDevice));

                if (!TryBindCheckpoint()) return;
                Debug.Log($"[Style] {config.name}: {(string)_policy.describe()}");
            }

            if (!TryGetStyleIndex(style, out _styleIndex))
            {
                Fail($"style '{(style != null ? style.TagName : "<none>")}' is not one the checkpoint " +
                     $"was trained on. It knows: {string.Join(", ", _styleIndexByName.Keys)}.");
                return;
            }

            _appliedStyle = style;
            _previousStyleIndex = -1;
            _styleBlend = 1f;
            _rootBlend = rootMotion == RootMotionMode.Stylize ? 1f : 0f;
            AllocateBuffers();
            ResolveRegionWeights();
            WarnAboutPlacement();

            _isInitialized = true;
        }
        catch (Exception e)
        {
            _initializationFailed = true;
            Debug.LogError($"[Style] failed to start. Check Project Settings > MoSynth > Python, and " +
                           $"that '{config.GetCheckpointPath()}' exists — press Train on the config " +
                           "if it does not.");
            Debug.LogException(e);
        }
    }

    private void Fail(string message)
    {
        _initializationFailed = true;
        Debug.LogError($"[Style] {message} Disabling the stage.", _owner);
    }

    /// <summary>
    /// Maps the checkpoint's bones, contacts and styles onto this character. Must be called with the
    /// GIL held.
    /// </summary>
    private bool TryBindCheckpoint()
    {
        var checkpointBones = (string[])_policy.bone_names();
        string error = null;
        if (checkpointBones.Length != _skeletonData.BoneCount ||
            !PredictedBoneSelection.TryResolve(checkpointBones, _owner.Skeleton, out _, out error))
        {
            error ??= $"it was trained on {checkpointBones.Length} bones but the rig has " +
                      $"{_skeletonData.BoneCount}";
            Fail($"'{config.name}' checkpoint does not fit this character: {error}. Regenerate the " +
                 "database and retrain if the rig changed.");
            return false;
        }

        var styledBones = (string[])_policy.styled_bone_names();
        if (!PredictedBoneSelection.TryResolve(styledBones, _owner.Skeleton, out _styledBones, out error))
        {
            Fail($"'{config.name}' checkpoint does not fit this character: {error}.");
            return false;
        }

        // Bone 0's delta is the one that needs its yaw removed, so the layout must say where it is.
        if (_styledBones.Length == 0 || _styledBones[0] != 0)
        {
            Fail($"'{config.name}' checkpoint does not style the root bone first.");
            return false;
        }

        var contactNames = PythonRuntime.ToStringArrayOrNull(_policy.contact_bone_names);
        if (!ContactBoneSources.TryMatchCheckpoint(contactNames?.Length ?? 0, contactNames,
                _owner.ContactBoneNames, out error))
        {
            Fail($"'{config.name}' checkpoint does not fit this character: {error}. Regenerate the " +
                 "database and retrain.");
            return false;
        }

        _styleIndexByName = new Dictionary<string, int>();
        var styleNames = (string[])_policy.style_names();
        for (var i = 0; i < styleNames.Length; i++) _styleIndexByName[styleNames[i]] = i;

        _historyOffsets = (float[])_policy.history_offsets();
        _descriptorSize = (int)_policy.descriptor_size();
        var frameTime = (float)_policy.frame_time();

        if (_descriptorSize != 3 + _owner.ContactHandles.Count)
        {
            Fail($"'{config.name}' checkpoint describes a tick with {_descriptorSize} values, not " +
                 $"3 + {_owner.ContactHandles.Count} contacts.");
            return false;
        }

        AllocateHistoryRing(frameTime);
        return true;
    }

    private bool TryGetStyleIndex(GameplayTagSO tag, out int index)
    {
        index = -1;
        return tag != null && _styleIndexByName != null &&
               _styleIndexByName.TryGetValue(tag.TagName, out index);
    }

    /// <summary>
    /// Enough entries to reach the oldest history offset at the fastest tick this character can run.
    /// </summary>
    private void AllocateHistoryRing(float frameTime)
    {
        var reach = 0f;
        foreach (var offset in _historyOffsets) reach = math.max(reach, offset);

        // An uncapped synthesis rate ticks at the render rate; 240 Hz bounds that generously.
        var tick = _owner.synthesisFrameRate > 1e-5f ? 1f / _owner.synthesisFrameRate : 1f / 240f;
        tick = math.min(tick, frameTime > 0f ? frameTime : tick);

        var capacity = math.clamp((int)math.ceil(reach / tick) + 2, 2, 1024);
        _ringDescriptors = new float[capacity * _descriptorSize];
        _ringTimes = new double[capacity];
        _ringStart = 0;
        _ringCount = 0;
    }

    private void AllocateBuffers()
    {
        var boneCount = _skeletonData.BoneCount;
        _positions = new NativeArray<float3>(boneCount, Allocator.Domain);
        _rotations = new NativeArray<quaternion>(boneCount, Allocator.Domain);
        _rotationsIn = new float[boneCount * 4];
        _positionsIn = new float[boneCount * 3];
        _contacts = new float[_owner.ContactHandles.Count];
        _history = new float[_historyOffsets.Length * _descriptorSize];
        _rootVelocity = new float[3];
        _oneStyle = new int[1];
        _twoStyles = new int[2];
        _gait = new OnlineGaitPhase();
        _time = 0.0;
        _boneWeights = new float[boneCount];
    }

    private void ResolveRegionWeights()
    {
        var listed = new Dictionary<int, float>();
        foreach (var region in regionWeights)
        {
            if (region?.bone == null || !region.bone.IsSet) continue;

            var index = region.bone.ResolveIndex(_owner.Skeleton);
            if (index < 0)
            {
                Debug.LogWarning($"[Style] region bone '{region.bone.Name}' is not on this rig; ignored.",
                    _owner);
                continue;
            }

            listed[index] = region.weight;
        }

        var parents = new int[_skeletonData.BoneCount];
        for (var i = 0; i < parents.Length; i++) parents[i] = i == 0 ? -1 : _skeletonData.ParentIndices[i];
        StyleComposition.ResolveRegionWeights(parents, listed, _boneWeights);
    }

    /// <summary>
    /// Warns about stage orders the design does not support: placement stages that would see the
    /// unstyled pose, and a root follower that would cancel root styling.
    /// </summary>
    private void WarnAboutPlacement()
    {
        var stages = _owner.stages;
        var self = stages.IndexOf(this);

        for (var i = 0; i < self; i++)
        {
            if (stages[i] is RootFollowStage or ContactLockStage)
            {
                Debug.LogWarning($"[Style] {stages[i].GetType().Name} runs before StyleTransferStage on " +
                                 $"\"{_owner.name}\", so it places the unstyled pose. Move it after.",
                    _owner);
            }
        }

        _wasStylize = rootMotion == RootMotionMode.Stylize;
        if (_wasStylize) WarnIfRootStyleCancelled();
    }

    private void WarnIfRootStyleCancelled()
    {
        if (_warnedRootCancelled) return;

        var stages = _owner.stages;
        for (var i = stages.IndexOf(this) + 1; i < stages.Count; i++)
        {
            if (stages[i] is not RootFollowStage { isEnabled: true, positionHalfLife: <= 0f }) continue;

            _warnedRootCancelled = true;
            Debug.LogWarning($"[Style] Root motion is Stylize, but a RootFollowStage after this one on " +
                             $"\"{_owner.name}\" has a position half-life of 0 and puts the character " +
                             "exactly on its target every tick, which cancels the root style.", _owner);
            return;
        }
    }

    /// <summary>
    /// Switch to another style, crossfading over <paramref name="blendTime"/>, a half-life in
    /// seconds; 0 switches at once. A tag the checkpoint does not know is refused with a warning.
    /// </summary>
    public void SetStyle(GameplayTagSO tag, float blendTime)
    {
        style = tag;
        _appliedStyle = tag;
        if (!_isInitialized) return;

        if (!TryGetStyleIndex(tag, out var index))
        {
            Debug.LogWarning($"[Style] '{(tag != null ? tag.TagName : "<none>")}' is not a style " +
                             $"'{config.name}' was trained on; keeping the current one.", _owner);
            return;
        }

        if (index == _styleIndex) return;

        _transitionHalfLife = math.max(0f, blendTime);
        if (_transitionHalfLife <= 0f)
        {
            _styleIndex = index;
            _previousStyleIndex = -1;
            _styleBlend = 1f;
            return;
        }

        if (_previousStyleIndex < 0)
        {
            _previousStyleIndex = _styleIndex;
            _styleIndex = index;
            _styleBlend = 0f;
        }
        else if (index == _previousStyleIndex)
        {
            // Turning back mid-fade: the same two styles, the other way, from where the mix stands.
            _previousStyleIndex = _styleIndex;
            _styleIndex = index;
            _styleBlend = 1f - _styleBlend;
        }
        else if (_styleBlend >= 0.5f)
        {
            // Only two styles run at once, so the one with the smaller share is dropped.
            _previousStyleIndex = _styleIndex;
            _styleIndex = index;
            _styleBlend = 0f;
        }
        else
        {
            _styleIndex = index;
        }
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (!_isInitialized || deltaTime <= 0f) return true;

        try
        {
            if (style != _appliedStyle) SetStyle(style, styleBlendHalfLife);

            _time += deltaTime;

            SimulationFrame.Compute(pose, _skeletonData, _frameDef, out var framePosition,
                out var frameRotation);
            SimulationFrame.ComputeVelocity(pose, _skeletonData, _frameDef, deltaTime,
                out var frameVelocity, out var frameYawRate);
            CharacterSpacePose.Extract(pose, _skeletonData, _frameDef, deltaTime, _positions, _rotations);

            PackInputs(pose, frameVelocity);
            _gait.Update(deltaTime, _contacts[LeftSlot] > 0.5f, _contacts[RightSlot] > 0.5f,
                math.length(frameVelocity.xz));
            FillHistory(frameVelocity, frameYawRate);

            var styleCount = _previousStyleIndex >= 0 ? 2 : 1;
            var styleIndices = styleCount == 2 ? _twoStyles : _oneStyle;
            if (styleCount == 2)
            {
                _twoStyles[0] = _previousStyleIndex;
                _twoStyles[1] = _styleIndex;
            }
            else
            {
                _oneStyle[0] = _styleIndex;
            }

            float[] posture, dynamics, postureHeight, dynamicsHeight, speedRatio, rootResidual, descriptor;
            float gate;

            using (Py.GIL())
            {
                dynamic result = _policy.step(_rotationsIn, _positionsIn, _contacts, _rootVelocity,
                    frameYawRate, _gait.Phase, _gait.Rate, _history, styleIndices);

                posture = (float[])result[0];
                dynamics = (float[])result[1];
                postureHeight = (float[])result[2];
                dynamicsHeight = (float[])result[3];
                speedRatio = (float[])result[4];
                rootResidual = (float[])result[5];
                gate = (float)result[6];
                descriptor = (float[])result[7];
            }

            CheckOutputSizes(styleCount, posture, dynamics, postureHeight, dynamicsHeight, speedRatio,
                rootResidual, descriptor);
            PushHistory(descriptor);
            AdvanceBlends(gate, deltaTime);

            // The fade may have finished this tick, in which case the current style is the last slot.
            var current = styleCount - 1;
            var previous = _previousStyleIndex >= 0 && styleCount == 2 ? 0 : -1;
            var g = gateEnabled ? _gate : 1f;

            WriteRotationsAndHeight(pose, posture, dynamics, postureHeight, dynamicsHeight,
                current, previous, g);

            if (_rootBlend > 0f)
            {
                RewriteRootVelocity(pose, framePosition, frameRotation, frameVelocity, frameYawRate,
                    speedRatio, rootResidual, current, previous, rootIntensity * _rootBlend * g);
            }
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            // Stop rather than log the same failure every frame.
            _isInitialized = false;
        }

        return true;
    }

    private void PackInputs(PoseBuffer pose, float3 frameVelocity)
    {
        for (var bone = 0; bone < _skeletonData.BoneCount; bone++)
        {
            var rotation = _rotations[bone].value;
            _rotationsIn[bone * 4] = rotation.x;
            _rotationsIn[bone * 4 + 1] = rotation.y;
            _rotationsIn[bone * 4 + 2] = rotation.z;
            _rotationsIn[bone * 4 + 3] = rotation.w;

            var position = _positions[bone];
            _positionsIn[bone * 3] = position.x;
            _positionsIn[bone * 3 + 1] = position.y;
            _positionsIn[bone * 3 + 2] = position.z;
        }

        var contacts = _owner.ContactHandles;
        for (var slot = 0; slot < contacts.Count; slot++)
        {
            _contacts[slot] = pose.GetBool(contacts[slot]) ? 1f : 0f;
        }

        _rootVelocity[0] = frameVelocity.x;
        _rootVelocity[1] = frameVelocity.y;
        _rootVelocity[2] = frameVelocity.z;
    }

    /// <summary>
    /// For each history offset, the stored descriptor nearest that far back — the oldest one while
    /// the ring is still filling.
    /// </summary>
    private void FillHistory(float3 frameVelocity, float frameYawRate)
    {
        if (_ringCount == 0)
        {
            SeedHistory(frameVelocity, frameYawRate);
            return;
        }

        var capacity = _ringTimes.Length;
        for (var h = 0; h < _historyOffsets.Length; h++)
        {
            var target = _time - _historyOffsets[h];
            var best = _ringStart;
            var bestDistance = double.MaxValue;

            for (var k = 0; k < _ringCount; k++)
            {
                var entry = (_ringStart + k) % capacity;
                var distance = Math.Abs(_ringTimes[entry] - target);
                if (distance >= bestDistance) continue;

                bestDistance = distance;
                best = entry;
            }

            Array.Copy(_ringDescriptors, best * _descriptorSize, _history, h * _descriptorSize,
                _descriptorSize);
        }
    }

    /// <summary>
    /// The first tick has no past, so every slot gets this tick's own descriptor, laid out as the
    /// contract defines it: forward speed, lateral speed, yaw rate, then each contact bone's height.
    /// </summary>
    private void SeedHistory(float3 frameVelocity, float frameYawRate)
    {
        var contacts = _owner.ContactHandles;
        for (var h = 0; h < _historyOffsets.Length; h++)
        {
            var offset = h * _descriptorSize;
            _history[offset] = frameVelocity.z;
            _history[offset + 1] = frameVelocity.x;
            _history[offset + 2] = frameYawRate;
            for (var slot = 0; slot < contacts.Count; slot++)
            {
                _history[offset + 3 + slot] = _positions[contacts.GetBoneIndex(slot)].y;
            }
        }
    }

    private void PushHistory(float[] descriptor)
    {
        var capacity = _ringTimes.Length;
        int entry;
        if (_ringCount < capacity)
        {
            entry = (_ringStart + _ringCount) % capacity;
            _ringCount++;
        }
        else
        {
            entry = _ringStart;
            _ringStart = (_ringStart + 1) % capacity;
        }

        _ringTimes[entry] = _time;
        Array.Copy(descriptor, 0, _ringDescriptors, entry * _descriptorSize, _descriptorSize);
    }

    private void CheckOutputSizes(int styleCount, float[] posture, float[] dynamics,
        float[] postureHeight, float[] dynamicsHeight, float[] speedRatio, float[] rootResidual,
        float[] descriptor)
    {
        var rotations = styleCount * _styledBones.Length * 4;
        if (posture.Length != rotations || dynamics.Length != rotations ||
            postureHeight.Length != styleCount || dynamicsHeight.Length != styleCount ||
            speedRatio.Length != styleCount || rootResidual.Length != styleCount * 4 ||
            descriptor.Length != _descriptorSize)
        {
            throw new InvalidOperationException(
                $"[Style] step returned arrays of the wrong size for {styleCount} style(s) over " +
                $"{_styledBones.Length} styled bones: posture {posture.Length}, dynamics " +
                $"{dynamics.Length}, heights {postureHeight.Length}/{dynamicsHeight.Length}, speed " +
                $"{speedRatio.Length}, residual {rootResidual.Length}, descriptor {descriptor.Length}.");
        }
    }

    private void AdvanceBlends(float rawGate, float deltaTime)
    {
        var gate = math.saturate(rawGate);
        _gate = _hasGate ? Damp(_gate, gate, gateHalfLife, deltaTime) : gate;
        _hasGate = true;

        if (_previousStyleIndex >= 0)
        {
            _styleBlend = Damp(_styleBlend, 1f, _transitionHalfLife, deltaTime);
            if (1f - _styleBlend < BlendSettled)
            {
                _styleBlend = 1f;
                _previousStyleIndex = -1;
            }
        }

        var stylize = rootMotion == RootMotionMode.Stylize;
        if (stylize && !_wasStylize) WarnIfRootStyleCancelled();
        _wasStylize = stylize;

        var rootTarget = stylize ? 1f : 0f;
        _rootBlend = Damp(_rootBlend, rootTarget, styleBlendHalfLife, deltaTime);
        if (math.abs(_rootBlend - rootTarget) < BlendSettled) _rootBlend = rootTarget;
    }

    /// <summary>Moves toward <paramref name="target"/>, closing half the gap every half-life.</summary>
    private static float Damp(float current, float target, float halfLife, float deltaTime) =>
        halfLife <= 0f ? target : math.lerp(target, current, math.exp(-math.LN2 * deltaTime / halfLife));

    private void WriteRotationsAndHeight(PoseBuffer pose, float[] posture, float[] dynamics,
        float[] postureHeight, float[] dynamicsHeight, int current, int previous, float gate)
    {
        var postureWeight = postureIntensity * gate;
        var dynamicsWeight = dynamicsIntensity * gate;
        var styledCount = _styledBones.Length;
        var rotations = pose.Rotations;

        for (var slot = 0; slot < styledCount; slot++)
        {
            var bone = _styledBones[slot];
            var region = _boneWeights[bone];

            var delta = StyleComposition.ComposeDelta(
                ReadQuaternion(posture, current * styledCount + slot),
                ReadQuaternion(dynamics, current * styledCount + slot),
                postureWeight * region, dynamicsWeight * region);

            if (previous >= 0)
            {
                var from = StyleComposition.ComposeDelta(
                    ReadQuaternion(posture, previous * styledCount + slot),
                    ReadQuaternion(dynamics, previous * styledCount + slot),
                    postureWeight * region, dynamicsWeight * region);
                delta = StyleComposition.Crossfade(from, delta, _styleBlend);
            }

            if (bone == 0)
            {
                // Bone 0 is stored in world space, and the character frame is read off its heading.
                var original = rotations[0];
                rotations[0] = StyleComposition.StripYaw(original, math.mul(original, delta),
                    _frameDef.ForwardAxisLocal);
            }
            else
            {
                rotations[bone] = math.mul(rotations[bone], delta);
            }
        }

        var rootRegion = _boneWeights[0];
        var height = Height(postureHeight, dynamicsHeight, current, postureWeight, dynamicsWeight);
        if (previous >= 0)
        {
            height = math.lerp(Height(postureHeight, dynamicsHeight, previous, postureWeight,
                dynamicsWeight), height, _styleBlend);
        }

        var positions = pose.Positions;
        positions[0] += new float3(0f, height * rootRegion, 0f);
    }

    private static float Height(float[] postureHeight, float[] dynamicsHeight, int style,
        float postureWeight, float dynamicsWeight) =>
        postureHeight[style] * postureWeight + dynamicsHeight[style] * dynamicsWeight;

    /// <summary>
    /// Rewrites bone 0's rates so the character frame travels at the styled velocity, by the same
    /// decompose-and-recompose route <see cref="RootFollowStage"/> takes.
    /// </summary>
    private void RewriteRootVelocity(PoseBuffer pose, float3 framePosition, quaternion frameRotation,
        float3 frameVelocity, float frameYawRate, float[] speedRatio, float[] rootResidual,
        int current, int previous, float weight)
    {
        var ratio = speedRatio[current];
        var residual = ReadFloat4(rootResidual, current);
        if (previous >= 0)
        {
            ratio = math.lerp(speedRatio[previous], ratio, _styleBlend);
            residual = math.lerp(ReadFloat4(rootResidual, previous), residual, _styleBlend);
        }

        // The frame is ground-projected, so a vertical residual has nothing to move.
        var styledVelocity = StyleComposition.StylizeRootVelocity(frameVelocity, ratio,
            new float3(residual.x, 0f, residual.z), weight);
        var styledYawRate = StyleComposition.StylizeYawRate(frameYawRate, residual.w, weight);

        var positions = pose.Positions;
        var velocities = pose.Velocities;
        var angularVelocities = pose.AngularVelocities;

        SimulationFrame.DecomposeRootVelocity(framePosition, frameRotation, frameVelocity, frameYawRate,
            positions[0], velocities[0], angularVelocities[0],
            out var rootVelocityLocal, out var rootAngularVelocityLocal);
        SimulationFrame.RecomposeRootVelocity(framePosition, frameRotation, styledVelocity,
            styledYawRate, positions[0], rootVelocityLocal, rootAngularVelocityLocal,
            out var worldRootVelocity, out var worldRootAngularVelocity);

        velocities[0] = worldRootVelocity;
        angularVelocities[0] = worldRootAngularVelocity;
    }

    private static quaternion ReadQuaternion(float[] source, int index) =>
        new(source[index * 4], source[index * 4 + 1], source[index * 4 + 2], source[index * 4 + 3]);

    private static float4 ReadFloat4(float[] source, int index) =>
        new(source[index * 4], source[index * 4 + 1], source[index * 4 + 2], source[index * 4 + 3]);

    public void Dispose()
    {
        // Cast first: comparing a dynamic Python handle with null would ask Python to compare.
        if ((object)_policy != null)
        {
            // The GIL is held while dropping the handle because that is what decrefs the object.
            using (Py.GIL())
            {
                _policy = null;
            }
        }

        if (_positions.IsCreated) _positions.Dispose();
        if (_rotations.IsCreated) _rotations.Dispose();

        _isInitialized = false;
    }

    public override void OnDestroy() => Dispose();
}
}
