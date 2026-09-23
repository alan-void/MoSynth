using GameplayTags;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The query the clip editor's preview panel is currently running.
    /// </summary>
    /// <remarks>
    /// A <see cref="ScriptableSingleton{T}"/> rather than a field on the track: a query must not reach
    /// the clip asset, and <c>PropertyField</c> needs a <see cref="SerializedObject"/> to pick up the
    /// package's tag drawers.
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
