using System;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Marks an <see cref="AnimationClipComponentTrack"/> as the clip editor's view of one
    /// <see cref="AnimationClipComponent"/> type, the way <c>CustomEditor</c> binds an inspector.
    /// </summary>
    /// <remarks>
    /// A component whose type is renamed or moved silently falls back to the default track; a
    /// registry test is the cheap way to catch it.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Class, Inherited = false)]
    public sealed class ClipComponentTrackAttribute : Attribute
    {
        public ClipComponentTrackAttribute(Type componentType) => ComponentType = componentType;

        public Type ComponentType { get; }
    }
}
