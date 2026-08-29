using System;
using System.Collections.Generic;
using System.IO;
using Unity.Collections;
using UnityEngine;

namespace MotionMatching
{
    using static AnimationTools.BinarySerializerExtensions;

    /// <summary>
    /// Reads and writes the <c>.mmfeatures</c> file: the per-frame feature vectors the search compares
    /// against, the normalization statistics needed to interpret them, and the feature schema that
    /// says what each float means.
    /// </summary>
    /// <remarks>
    /// Separate from the <c>.mmpose</c> database on purpose: features are one way of indexing poses
    /// for search, and changing them should not mean re-extracting the poses.
    /// <para>
    /// Unversioned, like the other binary formats here — staleness is caught by validating content
    /// rather than by a version byte. <see cref="Deserialize"/> checks the schema in the file against
    /// the asset's feature configuration and refuses a file that disagrees, which
    /// <c>MotionMatchingData.GetOrImportFeatureSet</c> answers by re-extracting and rewriting. So a
    /// stale file costs one extraction rather than a database of plausible numbers read off the
    /// wrong offsets.
    /// </para>
    /// <para>
    /// The schema is written so the file describes itself. The Python side has no ScriptableObject
    /// to read the feature configuration from, and anything training on this database needs to know
    /// which floats are trajectory and which are pose.
    /// </para>
    /// </remarks>
    public class FeatureSerializer
    {
        /// <summary>Name, width and repeat count of one feature, in feature-vector order.</summary>
        private readonly struct FeatureSchema
        {
            public readonly string Name;
            public readonly int FloatsPerPrediction;
            public readonly int PredictionCount;

            public FeatureSchema(string name, int floatsPerPrediction, int predictionCount)
            {
                Name = name;
                FloatsPerPrediction = floatsPerPrediction;
                PredictionCount = predictionCount;
            }

            public bool Matches(FeatureSchema other) =>
                Name == other.Name &&
                FloatsPerPrediction == other.FloatsPerPrediction &&
                PredictionCount == other.PredictionCount;

            public override string ToString() => $"\"{Name}\" ({FloatsPerPrediction}x{PredictionCount})";
        }

        /// <summary>
        /// Writes the feature database beside the pose database, as
        /// <paramref name="fileName"/><c>.mmfeatures</c>.
        /// </summary>
        public void Serialize(FeatureSet featureSet, MotionMatchingData mmData, string path, string fileName)
        {
            Directory.CreateDirectory(path); // create directory and parent directories if they don't exist

            using var stream = File.Open(Path.Combine(path, fileName + ".mmfeatures"), FileMode.Create);
            using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8);

            writer.Write((uint)featureSet.NumberFeatureVectors);
            writer.Write((uint)featureSet.FeatureSize);
            // The trajectory/pose split, so a reader can find the pose block without the asset.
            writer.Write((uint)mmData.trajectoryFeatures.Count);
            writer.Write((uint)mmData.poseFeatures.Count);

            foreach (var schema in DescribeFeatures(mmData))
            {
                writer.Write(schema.Name ?? string.Empty);
                writer.Write((uint)schema.FloatsPerPrediction);
                writer.Write((uint)schema.PredictionCount);
            }

            for (var i = 0; i < featureSet.FeatureSize; ++i)
            {
                writer.Write(featureSet.GetMean(i));
                writer.Write(featureSet.GetStandardDeviation(i));
            }

            for (var i = 0; i < featureSet.NumberFeatureVectors; ++i)
            {
                writer.Write(featureSet.IsValidFeature(i) ? 1u : 0u);

                for (var t = 0; t < mmData.trajectoryFeatures.Count; ++t)
                {
                    var trajectoryFeature = mmData.trajectoryFeatures[t];
                    var featureSize = trajectoryFeature.FloatsPerPrediction;
                    for (var p = 0; p < trajectoryFeature.PredictionCount; ++p)
                    {
                        switch (featureSize)
                        {
                            case 3:
                                WriteFloat3(writer, featureSet.Get3DTrajectoryFeature(i, t, p));
                                break;
                            case 2:
                                WriteFloat2(writer, featureSet.Get2DTrajectoryFeature(i, t, p));
                                break;
                            case 1:
                                writer.Write(featureSet.Get1DTrajectoryFeature(i, t, p));
                                break;
                            default:
                                Debug.Assert(false, "Invalid trajectory feature");
                                break;
                        }
                    }
                }

                for (var p = 0; p < mmData.poseFeatures.Count; ++p)
                {
                    WriteFloat3(writer, featureSet.GetPoseFeature(i, p));
                }
            }
        }

        /// <summary>
        /// Reads <paramref name="fileName"/><c>.mmfeatures</c> from <paramref name="path"/>. False
        /// means "do not use this file": it is missing, truncated, or was written for a different
        /// feature configuration. Every rejection says what disagreed, and the caller is expected to
        /// re-extract rather than to fail.
        /// </summary>
        public bool Deserialize(string path, string fileName, MotionMatchingData mmData, out FeatureSet featureSet)
        {
            featureSet = null;

            var featuresPath = Path.Combine(path, fileName + ".mmfeatures");
            if (!File.Exists(featuresPath)) return false;

            try
            {
                return Read(featuresPath, fileName, mmData, out featureSet);
            }
            catch (Exception e)
            {
                // A file written for a different configuration is not merely misread: string lengths
                // and array sizes come out of it, so it throws long before it runs out of bytes.
                Debug.LogError($"\"{fileName}.mmfeatures\" could not be read ({e.GetType().Name}: {e.Message}); " +
                               "treating it as stale. Regenerate the databases.");
                featureSet = null;
                return false;
            }
        }

        private static bool Read(string featuresPath, string fileName, MotionMatchingData mmData,
            out FeatureSet featureSet)
        {
            featureSet = null;

            using var stream = File.Open(featuresPath, FileMode.Open);
            using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8);

            var numberFeatureVectors = (int)reader.ReadUInt32();
            var featureSize = (int)reader.ReadUInt32();
            var trajectoryCount = (int)reader.ReadUInt32();
            var poseCount = (int)reader.ReadUInt32();

            // Before anything is allocated off the numbers just read, and before a FeatureSet is
            // built: a file written for another configuration gets its counts from the wrong offsets.
            if (!CheckSchema(reader, fileName, mmData, featureSize, trajectoryCount, poseCount))
            {
                return false;
            }

            var mean = new float[featureSize];
            var standardDeviation = new float[featureSize];
            for (var i = 0; i < featureSize; ++i)
            {
                mean[i] = reader.ReadSingle();
                standardDeviation[i] = reader.ReadSingle();
            }

            var valid = new NativeArray<bool>(numberFeatureVectors, Allocator.Domain);
            var features = new NativeArray<float>(numberFeatureVectors * featureSize, Allocator.Domain);
            for (var i = 0; i < numberFeatureVectors; ++i)
            {
                var featureIndex = i * featureSize;
                valid[i] = reader.ReadUInt32() != 0u;
                for (var f = 0; f < featureSize; ++f)
                {
                    features[featureIndex + f] = reader.ReadSingle();
                }
            }

            featureSet = new FeatureSet(mmData);
            featureSet.SetValid(valid);
            featureSet.SetFeatures(features);
            featureSet.SetMean(mean);
            featureSet.SetStandardDeviation(standardDeviation);
            return true;
        }

        /// <summary>
        /// Reads the schema block and checks it describes the feature configuration the asset holds
        /// now. This is what catches a database generated before a feature was added, renamed or
        /// re-horizoned.
        /// </summary>
        private static bool CheckSchema(BinaryReader reader, string fileName, MotionMatchingData mmData,
            int featureSize, int trajectoryCount, int poseCount)
        {
            if (trajectoryCount != mmData.trajectoryFeatures.Count || poseCount != mmData.poseFeatures.Count)
            {
                Debug.LogError(
                    $"\"{fileName}.mmfeatures\" holds {trajectoryCount} trajectory and {poseCount} pose features " +
                    $"but \"{mmData.name}\" now has {mmData.trajectoryFeatures.Count} and " +
                    $"{mmData.poseFeatures.Count} — regenerate the databases.");
                return false;
            }

            var expectedFeatureSize = ExpectedFeatureSize(mmData);
            if (featureSize != expectedFeatureSize)
            {
                Debug.LogError($"\"{fileName}.mmfeatures\" has {featureSize}-float vectors but \"{mmData.name}\" " +
                               $"now describes {expectedFeatureSize} — regenerate the databases.");
                return false;
            }

            var expected = DescribeFeatures(mmData);
            for (var i = 0; i < expected.Count; ++i)
            {
                var stored = new FeatureSchema(reader.ReadString(), (int)reader.ReadUInt32(),
                    (int)reader.ReadUInt32());
                if (stored.Matches(expected[i])) continue;

                Debug.LogError($"\"{fileName}.mmfeatures\" has {stored} as feature {i} but \"{mmData.name}\" " +
                               $"now has {expected[i]} — regenerate the databases.");
                return false;
            }

            return true;
        }

        /// <summary>
        /// The asset's features in feature-vector order: trajectory features, then pose features,
        /// which are always one three-float sample each.
        /// </summary>
        private static List<FeatureSchema> DescribeFeatures(MotionMatchingData mmData)
        {
            var schemas = new List<FeatureSchema>(mmData.trajectoryFeatures.Count + mmData.poseFeatures.Count);

            foreach (var feature in mmData.trajectoryFeatures)
            {
                schemas.Add(new FeatureSchema(feature.name, feature.FloatsPerPrediction, feature.PredictionCount));
            }

            foreach (var feature in mmData.poseFeatures)
            {
                schemas.Add(new FeatureSchema(feature.name, FeatureSet.FloatsPerPoseFeature, 1));
            }

            return schemas;
        }

        /// <summary>
        /// Feature-vector width the asset's configuration implies. Computed from the definitions
        /// rather than from a <see cref="FeatureSet"/>, so a file can be rejected before anything is
        /// built over a pose database it may not even match.
        /// </summary>
        private static int ExpectedFeatureSize(MotionMatchingData mmData)
        {
            var size = 0;
            foreach (var feature in mmData.trajectoryFeatures)
            {
                size += feature.FloatCount;
            }

            return size + mmData.poseFeatures.Count * FeatureSet.FloatsPerPoseFeature;
        }
    }
}
