using System.Collections;
using System.Collections.Generic;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Which keyframes are selected in a track, addressed by the row they sit in and the frame they
    /// sit on.
    /// </summary>
    /// <remarks>
    /// Keyed by frame rather than by list index, and that is the whole point. Every edit here
    /// re-sorts its list, so an index means something different afterwards - which is why the older
    /// hand-rolled selection had to be cleared after every move. A frame is unique within a row, so
    /// it is a key's identity: after a move the selected frames are simply the old ones plus the
    /// delta, and a key that collided with another and cancelled falls out of the selection on its
    /// own.
    /// </remarks>
    public sealed class TimelineKeySelection : IEnumerable<TimelineKey>
    {
        private readonly HashSet<TimelineKey> _keys = new();

        public int Count => _keys.Count;

        public bool Contains(int row, int frame) => _keys.Contains(new TimelineKey(row, frame));

        public bool Contains(in TimelineKey key) => _keys.Contains(key);

        public bool Add(int row, int frame) => _keys.Add(new TimelineKey(row, frame));

        public bool Remove(int row, int frame) => _keys.Remove(new TimelineKey(row, frame));

        /// <summary>Adds the key if it is absent, removes it if present - what shift-click does.</summary>
        public void Toggle(int row, int frame)
        {
            if (!Add(row, frame)) Remove(row, frame);
        }

        public void Clear() => _keys.Clear();

        public void SetTo(int row, int frame)
        {
            _keys.Clear();
            Add(row, frame);
        }

        /// <summary>The selected frames in one row, sorted ascending.</summary>
        public void FramesIn(int row, List<int> results)
        {
            results.Clear();
            foreach (var key in _keys)
            {
                if (key.Row == row) results.Add(key.Frame);
            }

            results.Sort();
        }

        public bool HasAnyIn(int row)
        {
            foreach (var key in _keys)
            {
                if (key.Row == row) return true;
            }

            return false;
        }

        /// <summary>
        /// Replaces the selection in one row, leaving every other row's alone - so an edit that
        /// moved one row's keys does not disturb the rest of the selection.
        /// </summary>
        public void ReplaceRow(int row, IReadOnlyList<int> frames)
        {
            _keys.RemoveWhere(key => key.Row == row);
            foreach (var frame in frames) Add(row, frame);
        }

        /// <summary>Keeps only keys that still exist, which is how a cancelled key leaves.</summary>
        public void Intersect(int row, IReadOnlyList<int> existingFrames)
        {
            var existing = new HashSet<int>(existingFrames);
            _keys.RemoveWhere(key => key.Row == row && !existing.Contains(key.Frame));
        }

        public IEnumerator<TimelineKey> GetEnumerator() => _keys.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    /// <summary>One keyframe's address: the lane row it belongs to and the frame it sits on.</summary>
    public readonly struct TimelineKey
    {
        public readonly int Row;
        public readonly int Frame;

        public TimelineKey(int row, int frame)
        {
            Row = row;
            Frame = frame;
        }

        public override bool Equals(object obj) =>
            obj is TimelineKey other && other.Row == Row && other.Frame == Frame;

        public override int GetHashCode() => Row * 397 ^ Frame;

        public override string ToString() => $"row {Row}, frame {Frame}";
    }
}
