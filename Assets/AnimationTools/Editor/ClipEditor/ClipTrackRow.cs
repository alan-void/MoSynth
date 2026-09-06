using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>One track in the timeline, together with the view state the window remembers for it.</summary>
    public sealed class ClipTrackRow
    {
        public AnimationClipComponentTrack Track;

        /// <summary>Index into the clip's <c>components</c> list.</summary>
        public int ComponentIndex;

        public float LaneHeight;

        /// <summary>
        /// Whether <see cref="LaneHeight"/> came from the user rather than the track. Once it did,
        /// <see cref="AnimationClipComponentTrack.RequestedLaneHeight"/> stops overriding it.
        /// </summary>
        public bool HeightIsUserSet;

        /// <summary>Whether the lane is drawn. Independent of the component's own <c>isEnabled</c>.</summary>
        public bool Visible = true;
    }

    /// <summary>
    /// Per-clip lane heights and visibility, remembered across domain reloads but not stored in the
    /// asset - which lanes you happen to be looking at is not part of the animation's data.
    /// </summary>
    [Serializable]
    public sealed class ClipTrackViewState
    {
        [Serializable]
        public sealed class Entry
        {
            public string typeName;
            public float laneHeight;
            public bool heightIsUserSet;
            public bool visible = true;
        }

        public List<Entry> entries = new();

        public static ClipTrackViewState Load(string assetGuid)
        {
            var json = SessionState.GetString(KeyFor(assetGuid), string.Empty);
            if (string.IsNullOrEmpty(json)) return new ClipTrackViewState();

            try
            {
                return JsonUtility.FromJson<ClipTrackViewState>(json) ?? new ClipTrackViewState();
            }
            catch (ArgumentException)
            {
                return new ClipTrackViewState();
            }
        }

        public void Save(string assetGuid) =>
            SessionState.SetString(KeyFor(assetGuid), JsonUtility.ToJson(this));

        public void Apply(ClipTrackRow row)
        {
            var entry = Find(row);
            if (entry == null) return;

            // Only a height the user actually dragged is restored, and only that marks the row as
            // user-set. Inferring it from "a height was stored" made every row user-set after the
            // first save, which silently disabled RequestedLaneHeight for good.
            row.HeightIsUserSet = entry.heightIsUserSet;
            if (entry.heightIsUserSet && entry.laneHeight > 0f) row.LaneHeight = entry.laneHeight;

            row.Visible = entry.visible;
        }

        public void Record(ClipTrackRow row)
        {
            var entry = Find(row);
            if (entry == null)
            {
                entry = new Entry { typeName = TypeNameOf(row) };
                entries.Add(entry);
            }

            entry.laneHeight = row.LaneHeight;
            entry.heightIsUserSet = row.HeightIsUserSet;
            entry.visible = row.Visible;
        }

        private Entry Find(ClipTrackRow row)
        {
            var name = TypeNameOf(row);
            foreach (var entry in entries)
            {
                if (entry.typeName == name) return entry;
            }

            return null;
        }

        // Keyed by component type rather than list index: reordering the components list should not
        // shuffle which lanes are hidden.
        private static string TypeNameOf(ClipTrackRow row) =>
            row.Track?.Component?.GetType().FullName ?? string.Empty;

        private static string KeyFor(string assetGuid) => $"MoSynth.ClipEditor.Tracks.{assetGuid}";
    }
}
