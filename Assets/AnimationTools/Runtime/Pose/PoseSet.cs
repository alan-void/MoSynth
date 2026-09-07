using System.Collections.Generic;
using AnimationTools;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Stores the full pose representation of all poses for Motion Matching.
/// Poses are stored as flat <see cref="PoseBuffer"/> frames in a single
/// <see cref="PoseSequence"/> over a <see cref="AnimationTools.Skeleton"/> that is exactly the
/// clips' own skeleton: bone 0 is the rig's root, carrying clip world position and rotation.
/// Read frames with <see cref="GetPoseBuffer"/>; write new ones through
/// <see cref="BeginClip"/> or <see cref="AppendRawFrames"/>.
/// </summary>
public class PoseSet
{
    // Public ---
    public float FrameTime { get; private set; } = -1.0f;
    public int NumberPoses => _poseCount;
    public int NumberClips => _clips.Count;
    public int NumberTags => _tags.Count;

    /// <summary>The pose skeleton: bone 0 is the rig's root, bone ids = index + 1.</summary>
    public Skeleton Skeleton => _skeleton;

    /// <summary>
    /// Which bone the character frame of a stored pose is derived from, and along which of its
    /// local axes the character faces. Structural rather than stored: the skeleton root and its
    /// rest forward axis.
    /// </summary>
    public SimulationFrameDef SimulationFrame => SimulationFrameDef.Default(Skeleton);

    /// <summary>Layout of the buffers returned by <see cref="GetPoseBuffer"/>.</summary>
    public PoseLayout PoseLayout => _layout;

    /// <summary>Reads the foot-contact Bool channels of a <see cref="GetPoseBuffer"/> frame.</summary>
    public ChannelHandle LeftFootContactHandle => _leftFootContactHandle;
    public ChannelHandle RightFootContactHandle => _rightFootContactHandle;

    /// <summary>Gait phase of a stored pose, in radians in <c>[0, Tau)</c>.</summary>
    /// <remarks>
    /// Not a pose channel: every channel is bone-keyed, and nothing in C# reads phase back out of a
    /// database. It rides alongside the poses because the Python training set reads it rather than
    /// reconstructing it — one evaluation of <see cref="GaitPhase"/>, from the clips' authored
    /// footfalls, is what stops the two halves drifting apart.
    /// </remarks>
    public float GetPhase(int poseIndex) => _phase[poseIndex];

    /// <summary>
    /// Rate of the gait phase at a stored pose, in radians per second. <b>Zero marks a frame with no
    /// measurable cycle</b>, the sentinel training uses to drop it.
    /// </summary>
    public float GetPhaseRate(int poseIndex) => _phaseRate[poseIndex];

    public void SetPhase(int poseIndex, float phase, float phaseRate)
    {
        _phase[poseIndex] = phase;
        _phaseRate[poseIndex] = phaseRate;
    }

    // Private ---
    private readonly List<AnimationClip> _clips = new();
    private readonly List<AnimationTag> _tags = new();
    private readonly Dictionary<string, int> _tagNameToIndex = new();

    private Skeleton _skeleton;
    private PoseLayout _layout;
    // Capacity view over Domain-backed storage; only the first _poseCount frames hold poses.
    private PoseSequence _poseStorage;
    private int _poseCount;
    private ChannelHandle _leftFootContactHandle;
    private ChannelHandle _rightFootContactHandle;
    // Parallel to the pose storage, and grown with it. See GetPhase.
    private float[] _phase;
    private float[] _phaseRate;

    /// <summary>
    /// Adopts an already-complete skeleton and rebuilds the layout over it. Pose storage starts
    /// empty: every real caller sets the skeleton exactly once, before adding any pose.
    /// </summary>
    public void SetSkeleton(Skeleton skeleton)
    {
        Debug.Assert(_poseCount == 0, "Setting the skeleton discards previously stored poses");
        _skeleton = skeleton;
        RebuildLayout();
    }

    private void RebuildLayout()
    {
        _layout = PoseLayoutBuilder.Build(_skeleton, out var contacts);
        _leftFootContactHandle = contacts.Left;
        _rightFootContactHandle = contacts.Right;

        _poseStorage = null;
        _poseCount = 0;
        _phase = null;
        _phaseRate = null;
    }

    /// <summary>
    /// Writable view over a range of freshly appended frames. Frames are views into this
    /// set's storage — never Dispose them.
    /// </summary>
    public readonly struct PoseFrameRange
    {
        private readonly PoseSet _owner;
        public readonly int Start;
        public readonly int Count;

        internal PoseFrameRange(PoseSet owner, int start, int count)
        {
            _owner = owner;
            Start = start;
            Count = count;
        }

        public PoseBuffer this[int localIndex] => _owner.GetPoseBuffer(Start + localIndex);
    }

    /// <summary>
    /// Registers a new animation clip of <paramref name="frameCount"/> poses and returns
    /// writable frames for it (zero-initialized). The clip's range is registered up front, so
    /// tag ranges added afterwards resolve against its start offset.
    /// </summary>
    public PoseFrameRange BeginClip(int frameCount, float frameTime)
    {
        // Check if the skeleton and frameTime are compatible
        Debug.Assert(_skeleton != null, "Skeleton should be set first. Use SetSkeleton(...)");
        if (FrameTime == -1.0f) FrameTime = frameTime;
        Debug.Assert(math.abs(FrameTime - frameTime) < 0.001f, "Frame time should be the same for all clips");

        var start = _poseCount;
        _clips.Add(new AnimationClip(start, start + frameCount, frameTime));
        EnsureCapacity(_poseCount + frameCount);
        _poseCount += frameCount;
        return new PoseFrameRange(this, start, frameCount);
    }

    /// <summary>
    /// Appends <paramref name="frameCount"/> zero-initialized poses with no clip/tag
    /// bookkeeping and returns writable frames — the deserialization path, where clips and
    /// tags are registered separately.
    /// </summary>
    public PoseFrameRange AppendRawFrames(int frameCount)
    {
        Debug.Assert(_skeleton != null, "Skeleton should be set first. Use SetSkeleton(...)");
        EnsureCapacity(_poseCount + frameCount);
        var range = new PoseFrameRange(this, _poseCount, frameCount);
        _poseCount += frameCount;
        return range;
    }

    public void SetPoseCapacity(uint numPoses)
    {
        EnsureCapacity((int)numPoses);
    }

    public void SetClipCapacity(uint count)
    {
        _clips.Capacity = (int)count;
    }

    /// <summary>
    /// Grows pose storage to hold at least <paramref name="poseCapacity"/> frames. Storage is
    /// Domain-backed (never disposed, freed on domain unload); an outgrown buffer is simply
    /// abandoned to the domain after its contents are copied over.
    /// </summary>
    private void EnsureCapacity(int poseCapacity)
    {
        Debug.Assert(_layout != null, "Skeleton should be set first. Use SetSkeleton(...)");
        if (poseCapacity <= 0) return;
        if (_poseStorage != null && poseCapacity <= _poseStorage.FrameCount) return;

        var newCapacity = math.max(poseCapacity, _poseStorage == null ? 64 : _poseStorage.FrameCount * 2);
        var floatCount = _layout.FloatCount;
        var newData = new NativeArray<float>(newCapacity * floatCount, Allocator.Domain, NativeArrayOptions.ClearMemory);
        if (_poseStorage != null)
        {
            NativeArray<float>.Copy(_poseStorage.Data, newData, _poseCount * floatCount);
        }

        _poseStorage = new PoseSequence(_layout, newData);

        var newPhase = new float[newCapacity];
        var newPhaseRate = new float[newCapacity];
        if (_phase != null)
        {
            System.Array.Copy(_phase, newPhase, _poseCount);
            System.Array.Copy(_phaseRate, newPhaseRate, _poseCount);
        }

        _phase = newPhase;
        _phaseRate = newPhaseRate;
    }

    /// <summary>
    /// Add a tag to the current pose set
    /// The corresponding animation clip should be added before using AddTag(...)
    /// </summary>
    private void AddTag(int animationClip, AnnotatedAnimationClip.Tag dataTag)
    {
        // Tag Index
        if (!_tagNameToIndex.TryGetValue(dataTag.name, out int tagIndex))
        {
            tagIndex = _tags.Count;
            _tagNameToIndex[dataTag.name] = tagIndex;
            _tags.Add(new AnimationTag(dataTag.name));
        }

        // Write tag ranges
        AnimationTag animationTag = _tags[tagIndex];
        int frameOffset = _clips[animationClip].Start;
        for (int i = 0; i < dataTag.start.Length; ++i)
        {
            animationTag.AddRange(dataTag.start[i] + frameOffset, dataTag.end[i] + frameOffset);
        }
    }

    /// <summary>
    /// Add a tag to the current pose set
    /// Used when deserializing from binary format
    /// </summary>
    public void AddTagDeserialized(string name, List<int> startRangesList, List<int> endRangesList)
    {
        _tagNameToIndex[name] = _tags.Count;
        _tags.Add(new AnimationTag(name, startRangesList, endRangesList));
    }

    /// <summary>
    /// Converts all tags-related data stored in C# data structures to NativeArrays
    /// Use this function after adding all tags with AddTag(...)
    /// </summary>
    public void ConvertTagsToNativeArrays()
    {
        foreach (AnimationTag tag in _tags)
        {
            tag.ConvertToNativeArray();
        }
    }

    /// <summary>
    /// Add the animation clip to the current clips
    /// Used when deserializing from binary format
    /// </summary>
    public void AddAnimationClipDeserialized(AnimationClip clip)
    {
        Debug.Assert(math.abs(FrameTime + 1.0f) < 0.001f || math.abs(clip.FrameTime - FrameTime) < 0.001f,
            "Mixed frame rates");
        FrameTime = clip.FrameTime;
        _clips.Add(clip);
    }

    /// <summary>
    /// Whether every frame from <paramref name="framesBehind"/> before <paramref name="poseIndex"/>
    /// to <paramref name="framesAhead"/> after it belongs to the same clip. A feature vector that
    /// samples outside its own clip describes a transition that never happened, so those poses are
    /// excluded from the searchable database rather than given a bogus trajectory.
    /// </summary>
    /// <param name="framesBehind">
    /// Frames of history the features need. Non-zero only once a trajectory feature samples the
    /// past, which it does through a negative prediction frame.
    /// </param>
    public bool IsPoseValidForPrediction(int poseIndex, int framesAhead, int framesBehind = 0)
    {
        Debug.Assert(poseIndex >= 0 && poseIndex < _poseCount, "Pose index out of range");
        var clip = GetClipContaining(poseIndex);
        return poseIndex - framesBehind >= clip.Start && poseIndex + framesAhead < clip.End;
    }

    /// <summary>
    /// Returns the index of the animation clip that contains the pose at the given index.
    /// </summary>
    public int GetAnimationClipIndex(int poseIndex)
    {
        var animationClip = -1;
        for (int clipIdx = 0; clipIdx < _clips.Count; ++clipIdx)
        {
            if (poseIndex >= _clips[clipIdx].Start && poseIndex < _clips[clipIdx].End)
            {
                animationClip = clipIdx;
                break;
            }
        }

        Debug.Assert(animationClip != -1, "Clip index not found");
        return animationClip;
    }

    /// <summary>
    /// View over one stored pose — never Dispose it; it aliases this set's storage. Contacts
    /// are readable through <see cref="LeftFootContactHandle"/>/<see cref="RightFootContactHandle"/>.
    /// </summary>
    public PoseBuffer GetPoseBuffer(int poseIndex)
    {
        Debug.Assert(poseIndex >= 0 && poseIndex < _poseCount, "Pose index out of range");
        return _poseStorage.GetFrame(poseIndex);
    }

    /// <summary>
    /// Returns the tag at the given index
    /// </summary>
    public AnimationTag GetTag(int index)
    {
        return _tags[index];
    }

    /// <summary>
    /// Returns the tag with the given name
    /// </summary>
    public AnimationTag GetTag(string name)
    {
        return _tags[_tagNameToIndex[name]];
    }

    /// <summary>
    /// Returns the animation clip at the given index
    /// </summary>
    public AnimationClip GetAnimationClip(int clipIndex)
    {
        Debug.Assert(clipIndex >= 0 && clipIndex < _clips.Count, "Clip index out of range");
        return _clips[clipIndex];
    }

    /// <summary>
    /// The clip a pose belongs to. Clips are stored back to back in one array, so anything that
    /// steps between frames has to ask for this rather than treat the array as continuous.
    /// </summary>
    public AnimationClip GetClipContaining(int poseIndex) => _clips[GetAnimationClipIndex(poseIndex)];

    public void Dispose()
    {
        // Pose storage is Domain-lifetime by design: pose sets are shared (MotionMatchingData
        // caches one), so Dispose only releases the tags.
        if (_tags != null)
        {
            foreach (AnimationTag tag in _tags)
            {
                tag.Dispose();
            }
        }
    }

    public struct AnimationClip
    {
        public int Start; // Index of the first pose in the clip
        public int End; // End is exclusive
        public float FrameTime;

        public AnimationClip(int start, int end, float frameTime)
        {
            Start = start;
            End = end;
            FrameTime = frameTime;
        }
    }

    public class AnimationTag
    {
        public readonly string Name;

        private List<int> _startRangesList; // Temporal lists until they are converted to NativeArrays
        private List<int> _endRangesList;

        private NativeArray<int> _startRanges;
        private NativeArray<int> _endRanges;

        public int NumberRanges
        {
            get { return _startRanges.Length; }
        }

        public AnimationTag(string name)
        {
            Name = name;
            _startRangesList = new List<int>();
            _endRangesList = new List<int>();
        }

        public AnimationTag(string name, List<int> startRangesList, List<int> endRangesList)
        {
            Name = name;
            _startRangesList = startRangesList;
            _endRangesList = endRangesList;
        }

        public void AddRange(int start, int end)
        {
            _startRangesList.Add(start);
            _endRangesList.Add(end);
        }

        public NativeArray<int> GetStartRanges()
        {
            return _startRanges;
        }

        public NativeArray<int> GetEndRanges()
        {
            return _endRanges;
        }

        public void ConvertToNativeArray()
        {
            _startRanges = new NativeArray<int>(_startRangesList.ToArray(), Allocator.Domain);
            _endRanges = new NativeArray<int>(_endRangesList.ToArray(), Allocator.Domain);

            _startRangesList = null;
            _endRangesList = null;
        }

        public void GetRange(int rangeIndex, out int start, out int end)
        {
            Debug.Assert(_startRanges.IsCreated && _endRanges.IsCreated,
                "Call first ConvertToNativeArray() before operating over the tags.");
            start = _startRanges[rangeIndex];
            end = _endRanges[rangeIndex];
        }

        public void Dispose()
        {
            if (_startRanges.IsCreated) _startRanges.Dispose();
            if (_endRanges.IsCreated) _endRanges.Dispose();
        }
    }
}
}
