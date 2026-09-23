// Uncomment this line to enable profiling for Motion Matching
//#define PROFILE_MOTION_MATCHING

using System.Collections.Generic;
using UnityEngine;
#if PROFILE_MOTION_MATCHING
using System.Diagnostics;
using Debug = UnityEngine.Debug;
#endif

namespace MotionMatching
{
    /// <summary>
    /// Hand-rolled timing for the motion matching hot path that accumulates min/max/rolling-average
    /// across frames, which matters for a search whose cost swings with the query.
    /// </summary>
    /// <remarks>
    /// Compiled out entirely unless PROFILE_MOTION_MATCHING is defined at the top of this file, so
    /// instrumentation can be left in permanently at zero cost. SHOUTING_CASE names are intentional:
    /// they make instrumentation stand out from the code being measured.
    /// </remarks>
    public static class PROFILE
    {
#if PROFILE_MOTION_MATCHING
        private static readonly Dictionary<string, Stopwatch> _stopwatches = new Dictionary<string, Stopwatch>();
        // Kept apart from the stopwatches because results may be queried just after a stopwatch was reset.
        private static readonly Dictionary<string, DATA> _profileData = new Dictionary<string, DATA>();
#endif
        // Bodies are empty when PROFILE_MOTION_MATCHING is undefined, so the calls compile away.
        public static void BEGIN_SAMPLE_PROFILING(string tag)
        {
#if PROFILE_MOTION_MATCHING
            if (!_stopwatches.TryGetValue(tag, out var stopwatch))
            {
                stopwatch = new Stopwatch();
                _stopwatches.Add(tag, stopwatch);
            }
            stopwatch.Reset();
            stopwatch.Start();
#endif
        }
        public static void END_SAMPLE_PROFILING(string tag)
        {
#if PROFILE_MOTION_MATCHING
            var stopwatch = _stopwatches[tag];
            stopwatch.Stop();
            if (!_profileData.TryGetValue(tag, out var data))
            {
                data = new DATA();
                _profileData.Add(tag, data);
            }
            data.AddSample((float)stopwatch.Elapsed.TotalMilliseconds, stopwatch.ElapsedTicks);
#endif
        }
        public static void END_AND_PRINT_SAMPLE_PROFILING(string tag)
        {
#if PROFILE_MOTION_MATCHING
            END_SAMPLE_PROFILING(tag);
            var stopwatch = _stopwatches[tag];
            Debug.Log("[PROFILER]" + tag + ": " + stopwatch.ElapsedMilliseconds + "ms" + "(" + stopwatch.ElapsedTicks + " ticks)");
#endif
        }
        public static DATA GET_DATA(string tag)
        {
#if PROFILE_MOTION_MATCHING
            if (_profileData.TryGetValue(tag, out var data))
            {
                return data;
            }
#endif
            return null;
        }

        /// <summary>
        /// For Editor code only; elsewhere, guard with the PROFILE_MOTION_MATCHING define instead.
        /// </summary>
        public static bool IS_PROFILING_ENABLED()
        {
#if PROFILE_MOTION_MATCHING
            return true;
#endif
#pragma warning disable 162
            return false;
        }

        public class DATA
        {
            private const int NumberSamplesToAverage = 60;

            public float MinTicks, MinMs;
            public float MaxTicks, MaxMs;
            private float _averageTicks, _averageMs;

            private readonly float[] _samplesToAverageMs;
            private readonly float[] _samplesToAverageTicks;
            private int _currentSample;
            private int _absoluteCurrentSample;

            public DATA()
            {
                MinTicks = float.MaxValue;
                MinMs = float.MaxValue;
                MaxTicks = float.MinValue;
                MaxMs = float.MinValue;
                _samplesToAverageTicks = new float[NumberSamplesToAverage];
                _samplesToAverageMs = new float[NumberSamplesToAverage];
            }

            public void AddSample(float sampleMs, float sampleTicks)
            {
                MinTicks = Mathf.Min(MinTicks, sampleTicks);
                MinMs = Mathf.Min(MinMs, sampleMs);
                MaxTicks = Mathf.Max(MaxTicks, sampleTicks);
                MaxMs = Mathf.Max(MaxMs, sampleMs);

                _samplesToAverageTicks[_currentSample] = sampleTicks;
                _samplesToAverageMs[_currentSample] = sampleMs;
                _absoluteCurrentSample += 1;
                _currentSample = _absoluteCurrentSample % NumberSamplesToAverage;

                _averageTicks = 0;
                _averageMs = 0;
                var count = Mathf.Min(_absoluteCurrentSample, NumberSamplesToAverage);
                for (var i = 0; i < count; i++)
                {
                    _averageTicks += _samplesToAverageTicks[i];
                    _averageMs += _samplesToAverageMs[i];
                }
                _averageTicks /= count;
                _averageMs /= count;
            }

            public float GetAverageTicks()
            {
                return _averageTicks;
            }
            public float GetAverageMs()
            {
                return _averageMs;
            }
        }
    }
}
