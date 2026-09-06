using System;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The state every surface of the clip editor shares: the clip and its serialized form, the two
    /// frame spaces and the conversion between them, the playhead, and the only sanctioned way to
    /// commit an edit.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Two frame spaces exist and confusing them is the easiest mistake to make here. A
    /// <em>clip</em> frame indexes the whole baked animation and is what the preview poses; a
    /// <em>slice</em> frame is relative to <c>startFrame</c> and is what
    /// <see cref="GaitPhase.Footfall"/> stores. Convert only through
    /// <see cref="SliceToClipFrame"/> and <see cref="ClipToSliceFrame"/>.
    /// </para>
    /// <para>
    /// <see cref="AnnotatedAnimationClip"/> shadows <c>FrameCount</c> with a slice-local version, so
    /// this exposes <see cref="ClipFrameCount"/> and <see cref="SliceFrameCount"/> under names that
    /// say which is which, and nothing downstream reads <c>FrameCount</c> at all.
    /// </para>
    /// </remarks>
    public sealed class ClipEditorContext
    {
        private readonly Action _repaint;

        public ClipEditorContext(AnnotatedAnimationClip clip, Action repaint)
        {
            Clip = clip;
            _repaint = repaint;

            SerializedClip = new SerializedObject(clip);
        }

        public AnnotatedAnimationClip Clip { get; }
        public SerializedObject SerializedClip { get; }

        public SerializedProperty ComponentsProperty => SerializedClip.FindProperty("components");
        public SerializedProperty StartFrameProperty => SerializedClip.FindProperty("startFrame");
        public SerializedProperty EndFrameProperty => SerializedClip.FindProperty("endFrame");

        /// <summary>Frames in the whole baked clip. The timeline axis runs over these.</summary>
        public int ClipFrameCount => Clip == null || !Clip.HasClip ? 0 : ((SkeletonAnimation)Clip).FrameCount;

        public int StartFrame => Clip == null ? 0 : Mathf.Clamp(Clip.startFrame, 0, ClipFrameCount);

        /// <summary>Exclusive, and clamped to the clip, matching what <c>OnValidate</c> enforces.</summary>
        public int EndFrame => Clip == null ? 0 : Mathf.Clamp(Clip.endFrame, StartFrame, ClipFrameCount);

        public int SliceFrameCount => Mathf.Max(0, EndFrame - StartFrame);

        public float FrameTime => Clip == null || !Clip.HasClip ? 1f / 30f : Clip.FrameTime;

        /// <summary>The frame the preview is posing, in clip frames.</summary>
        public int PlayheadClipFrame { get; private set; }

        /// <summary>
        /// Bumped whenever the asset's data may have moved, so a track can invalidate a cached
        /// texture or selection by comparing against the value it last drew at.
        /// </summary>
        public int DataVersion { get; private set; }

        /// <summary>
        /// The baked pose at a whole-clip frame. Goes through the base class deliberately, since
        /// <see cref="AnnotatedAnimationClip"/>'s own <c>GetFrame</c> offsets by the slice.
        /// </summary>
        public PoseBuffer GetClipFrame(int clipFrame) =>
            ((SkeletonAnimation)Clip).GetFrame(Mathf.Clamp(clipFrame, 0, Mathf.Max(0, ClipFrameCount - 1)));

        public int SliceToClipFrame(int sliceFrame) => StartFrame + sliceFrame;

        public int ClipToSliceFrame(int clipFrame) => clipFrame - StartFrame;

        public bool IsInsideSlice(int clipFrame) => clipFrame >= StartFrame && clipFrame < EndFrame;

        public void SeekToClipFrame(int clipFrame)
        {
            var clamped = Mathf.Clamp(clipFrame, 0, Mathf.Max(0, ClipFrameCount - 1));
            if (clamped == PlayheadClipFrame) return;

            PlayheadClipFrame = clamped;
            Repaint();
        }

        public void Repaint() => _repaint?.Invoke();

        /// <summary>
        /// The one modal operator the window allows at a time, shared by every lane.
        /// </summary>
        /// <remarks>
        /// Shared rather than one per track because Blender allows exactly one mode at a time, and
        /// because two IMGUI operators holding <c>hotControl</c> at once would deadlock the window.
        /// Claiming it ends whichever lane was mid-grab. Its owner is tracked only so a track can
        /// tell whether the running mode is its own.
        /// </remarks>
        public TimelineModalOperator Modal { get; } = new();

        public object ModalOwner { get; private set; }

        /// <summary>Ends any running mode and hands the operator to <paramref name="owner"/>.</summary>
        public TimelineModalOperator ClaimModal(object owner)
        {
            if (!ReferenceEquals(ModalOwner, owner)) Modal.Cancel();

            ModalOwner = owner;
            return Modal;
        }

        /// <summary>Whether <paramref name="owner"/> currently has a mode running.</summary>
        public bool HasModal(object owner) => Modal.IsActive && ReferenceEquals(ModalOwner, owner);

        public void CancelModal()
        {
            Modal.Cancel();
            ModalOwner = null;
        }

        /// <summary>
        /// Writes pending property changes and records one undo step. Returns whether anything
        /// actually changed.
        /// </summary>
        /// <remarks>
        /// This runs the asset's <c>OnValidate</c>, which drops the baked pose sequence, so the next
        /// preview repaint re-bakes the whole clip. Commit once when an interaction ends - never per
        /// mouse-move during a drag.
        /// </remarks>
        public bool Commit()
        {
            if (!SerializedClip.ApplyModifiedProperties()) return false;

            DataVersion++;
            return true;
        }

        /// <summary>
        /// Re-reads the asset after something mutated it outside the serialized object, and after
        /// an undo. Discards pending property changes, so commit first if any are outstanding.
        /// </summary>
        public void RefreshFromAsset()
        {
            SerializedClip.Update();
            DataVersion++;
        }

        /// <summary>Whether the clip asset still exists; a window can outlive a deleted asset.</summary>
        public bool IsValid => Clip != null;
    }
}
