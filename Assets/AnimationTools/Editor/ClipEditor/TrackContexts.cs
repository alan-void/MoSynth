using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// One repaint of one lane: where to draw, how to map frames to pixels, and which frames are
    /// worth drawing at all.
    /// </summary>
    public readonly struct TrackDrawContext
    {
        /// <summary>The lane's rect in the timeline's GUI space, excluding the header column.</summary>
        public readonly Rect LaneRect;

        /// <summary>The timeline's shared axis. Every frame-to-pixel conversion goes through it.</summary>
        public readonly ClipTimeAxis Axis;

        /// <summary>Inclusive clip-frame bounds of what is on screen. Cull to these.</summary>
        public readonly int FirstVisibleClipFrame;
        public readonly int LastVisibleClipFrame;

        public readonly int PlayheadClipFrame;

        /// <summary>Whether this track currently owns the inspector panel.</summary>
        public readonly bool IsFocused;

        public readonly ClipEditorContext Editor;

        public TrackDrawContext(Rect laneRect, ClipTimeAxis axis, int firstVisibleClipFrame,
            int lastVisibleClipFrame, int playheadClipFrame, bool isFocused, ClipEditorContext editor)
        {
            LaneRect = laneRect;
            Axis = axis;
            FirstVisibleClipFrame = firstVisibleClipFrame;
            LastVisibleClipFrame = lastVisibleClipFrame;
            PlayheadClipFrame = playheadClipFrame;
            IsFocused = isFocused;
            Editor = editor;
        }
    }

    /// <summary>What a track is handed when it draws into the right-hand panel.</summary>
    public readonly struct TrackInspectorContext
    {
        public readonly ClipEditorContext Editor;

        /// <summary>The component's element of the clip's <c>components</c> list.</summary>
        public readonly SerializedProperty ComponentProperty;

        /// <summary>The window's preview, for a track that wants to re-frame it.</summary>
        public readonly SkeletonPreview Preview;

        public TrackInspectorContext(ClipEditorContext editor, SerializedProperty componentProperty,
            SkeletonPreview preview)
        {
            Editor = editor;
            ComponentProperty = componentProperty;
            Preview = preview;
        }
    }

    /// <summary>
    /// What a track is handed to draw 3D geometry into the preview, with the current frame's pose
    /// already resolved so an overlay never runs FK itself.
    /// </summary>
    /// <remarks>
    /// <see cref="BonePositions"/> and <see cref="BoneRotations"/> are the preview's own buffers and
    /// are overwritten on the next repaint. Read them during the call; never store them.
    /// </remarks>
    public readonly struct TrackOverlayContext
    {
        public readonly ISkeletonPreviewDrawer Draw;
        public readonly Camera Camera;

        /// <summary>The whole-clip frame being posed.</summary>
        public readonly int ClipFrame;

        /// <summary>Owned by the <see cref="Skeleton"/> and allocated for the domain - never dispose it.</summary>
        public readonly SkeletonData Skeleton;

        public readonly NativeArray<float3> BonePositions;
        public readonly NativeArray<quaternion> BoneRotations;

        public readonly ClipEditorContext Editor;

        public TrackOverlayContext(in SkeletonPreviewFrame frame, ClipEditorContext editor)
        {
            Draw = frame.Draw;
            Camera = frame.Camera;
            ClipFrame = frame.Frame;
            Skeleton = frame.Skeleton;
            BonePositions = frame.BonePositions;
            BoneRotations = frame.BoneRotations;
            Editor = editor;
        }
    }
}
