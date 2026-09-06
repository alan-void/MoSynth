using System;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Marks an <see cref="AnimationClipComponentTrack"/> as the clip editor's view of one
    /// <see cref="AnimationClipComponent"/> type, the way <c>CustomEditor</c> binds an inspector.
    /// </summary>
    /// <remarks>
    /// A component whose type is renamed or moved falls back to the default track silently, because
    /// nothing can tell a renamed type from one that simply has no track. That is the same identity
    /// hazard <see cref="AnimationClipComponent"/> already documents for serialized data, and a
    /// registry test is the cheap way to catch it.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Class, Inherited = false)]
    public sealed class ClipComponentTrackAttribute : Attribute
    {
        public ClipComponentTrackAttribute(Type componentType) => ComponentType = componentType;

        public Type ComponentType { get; }
    }
}
