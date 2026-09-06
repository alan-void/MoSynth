using GameplayTags;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The query the clip editor's preview panel is currently running.
    /// </summary>
    /// <remarks>
    /// A <see cref="ScriptableSingleton{T}"/> rather than a field on the track, for two reasons: a
    /// query is something you are asking, not something the clip knows, so it must not reach the
    /// asset; and drawing it with <c>PropertyField</c> - which is how it picks up the package's tag
    /// drawers - needs a <see cref="SerializedObject"/> to hang off. Surviving a domain reload is a
    /// bonus, not the point.
    /// </remarks>
    [FilePath("UserSettings/MoSynthClipTagQuery.asset", FilePathAttribute.Location.ProjectFolder)]
    public sealed class ClipTagQueryScratch : ScriptableSingleton<ClipTagQueryScratch>
    {
        [SerializeField] private GameplayTagQuery query = new();

        public GameplayTagQuery Query => query;

        public SerializedObject Serialized() => new(this);

        public void Save() => Save(true);
    }
}
