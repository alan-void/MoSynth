using System.IO;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
    using static AnimationTools.BinarySerializerExtensions;

    /// <summary>
    /// Reads and writes the <c>.mmfeatures</c> file: the per-frame feature vectors the search compares
    /// against, plus the normalization statistics needed to interpret them.
    /// </summary>
    /// <remarks>
    /// Separate from the <c>.mmpose</c> database on purpose: features are one way of indexing poses
    /// for search, and changing them should not mean re-extracting the poses. Unversioned like the
    /// other binary formats — a stale file is misread, not detected — so regenerate after any change.
    /// </remarks>
    public class FeatureSerializer
    {
        /// <summary>
        /// Stores the features for Motion Matching in a binary format
        /// in the specified path with name filename and extension .mmfeatures
        /// </summary>
        public void Serialize(FeatureSet featureSet, MotionMatchingData mmData, string path, string fileName)
        {
            Directory.CreateDirectory(path); // create directory and parent directories if they don't exist

            // Write Features
            using (var stream = File.Open(Path.Combine(path, fileName + ".mmfeatures"), FileMode.Create))
            {
                using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8))
                {
                    // Serialize Number Feature Vectors
                    writer.Write((uint)featureSet.NumberFeatureVectors);
                    // Serialize Number Features Static Dimension
                    writer.Write((uint)featureSet.FeatureSize);
                    // Serialize Number Features Dimension
                    writer.Write((uint)featureSet.FeatureSize);
                    // Serialize Number Features
                    writer.Write((uint)(featureSet.NumberTrajectoryFeatures + featureSet.PoseFeatureCount));

                    // Serialize Mean & StandardDeviation
                    for (int i = 0; i < featureSet.FeatureSize; ++i)
                    {
                        writer.Write(featureSet.GetMean(i));
                        writer.Write(featureSet.GetStandardDeviation(i));
                    }

                    // Serialize Features
                    for (int t = 0; t < mmData.trajectoryFeatures.Count; ++t)
                    {
                        var trajectoryFeature = mmData.trajectoryFeatures[t];
                        writer.Write(trajectoryFeature.name);
                        writer.Write(trajectoryFeature.FloatsPerPrediction);
                        writer.Write((uint)trajectoryFeature.PredictionCount);
                    }
                    for (int p = 0; p < mmData.poseFeatures.Count; ++p)
                    {
                        var poseFeature = mmData.poseFeatures[p];
                        writer.Write(poseFeature.name);
                        writer.Write(3u);
                        writer.Write(1u);
                    }

                    // Serialize Feature Vectors
                    for (int i = 0; i < featureSet.NumberFeatureVectors; ++i)
                    {
                        writer.Write(featureSet.IsValidFeature(i) ? 1u : 0u);
                        // Trajectory
                        for (int t = 0; t < mmData.trajectoryFeatures.Count; ++t)
                        {
                            var trajectoryFeature = mmData.trajectoryFeatures[t];
                            int featureSize = trajectoryFeature.FloatsPerPrediction;
                            for (int p = 0; p < trajectoryFeature.PredictionCount; ++p)
                            {
                                if (featureSize == 3)
                                {
                                    float3 value3D = featureSet.Get3DTrajectoryFeature(i, t, p);
                                    WriteFloat3(writer, value3D);
                                }
                                else if (featureSize == 2)
                                {
                                    float2 value2D = featureSet.Get2DTrajectoryFeature(i, t, p);
                                    WriteFloat2(writer, value2D);
                                }
                                else if (featureSize == 1)
                                {
                                    float value1D = featureSet.Get1DTrajectoryFeature(i, t, p);
                                    writer.Write(value1D);
                                }
                                else
                                {
                                    Debug.Assert(false, "Invalid trajectory feature");
                                }
                            }
                        }
                        // Pose
                        for (int p = 0; p < mmData.poseFeatures.Count; ++p)
                        {
                            WriteFloat3(writer, featureSet.GetPoseFeature(i, p));
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Reads the features for Motion Matching from a binary format
        /// from the specified path with name filename and extension .mmfeatures
        /// Returns true if featureSet was successfully deserialized, false otherwise
        /// </summary>
        public bool Deserialize(string path, string fileName, MotionMatchingData mmData, out FeatureSet featureSet)
        {
            featureSet = null;

            // Read Features
            string featuresPath = Path.Combine(path, fileName + ".mmfeatures");
            if (File.Exists(featuresPath))
            {
                using (var stream = File.Open(featuresPath, FileMode.Open))
                {
                    using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8))
                    {
                        // Deserialize Number Feature Vectors
                        uint numberFeatureVectors = reader.ReadUInt32();
                        // Deserialize Number Features Static Dimension
                        uint featuresStaticDimension = reader.ReadUInt32();
                        // Deserialize Number Features Dimension
                        uint featuresDimension = reader.ReadUInt32();
                        // Deserialize Number Features
                        uint numberFeatures = reader.ReadUInt32();

                        // Deserialize Mean & StandardDeviation
                        float[] mean = new float[featuresStaticDimension];
                        float[] standardDeviation = new float[featuresStaticDimension];
                        for (int i = 0; i < featuresStaticDimension; ++i)
                        {
                            mean[i] = reader.ReadSingle();
                            standardDeviation[i] = reader.ReadSingle();
                        }

                        // Deserialize Features (basically check that everything is correct)
                        Debug.Assert(numberFeatures == (mmData.trajectoryFeatures.Count + mmData.poseFeatures.Count), "Number of features in file does not match number of features in Motion Matching Data");
                        for (int t = 0; t < mmData.trajectoryFeatures.Count; ++t)
                        {
                            var trajectoryFeature = mmData.trajectoryFeatures[t];
                            string name = reader.ReadString();
                            uint nFloatsType = reader.ReadUInt32();
                            uint nElements = reader.ReadUInt32();
                            Debug.Assert(name == trajectoryFeature.name, "Name of trajectory feature does not match");
                            Debug.Assert(nFloatsType == (uint)trajectoryFeature.FloatsPerPrediction, "Projection type of trajectory feature does not match");
                            Debug.Assert(nElements == (uint)trajectoryFeature.PredictionCount, "Number of frames prediction of trajectory feature does not match");
                        }
                        for (int p = 0; p < mmData.poseFeatures.Count; ++p)
                        {
                            var poseFeature = mmData.poseFeatures[p];
                            string name = reader.ReadString();
                            uint nFloatsType = reader.ReadUInt32();
                            uint nElements = reader.ReadUInt32();
                            Debug.Assert(name == poseFeature.name, "Name of pose feature does not match");
                            Debug.Assert(nFloatsType == 3u, "Projection type of pose feature does not match");
                            Debug.Assert(nElements == 1u, "Number of frames prediction of pose feature does not match");
                        }

                        // Deserialize Feature Vectors
                        NativeArray<bool> valid = new NativeArray<bool>((int)numberFeatureVectors, Allocator.Domain);
                        NativeArray<float> features = new NativeArray<float>((int)(numberFeatureVectors * featuresDimension), Allocator.Domain);
                        for (int i = 0; i < numberFeatureVectors; ++i)
                        {
                            int featureIndex = i * (int)featuresDimension;
                            valid[i] = reader.ReadUInt32() != 0u;
                            for (int f = 0; f < featuresDimension; ++f)
                            {
                                features[featureIndex + f] = reader.ReadSingle();
                            }
                        }

                        featureSet = new FeatureSet(mmData);
                        featureSet.SetValid(valid);
                        featureSet.SetFeatures(features);
                        featureSet.SetMean(mean);
                        featureSet.SetStandardDeviation(standardDeviation);
                    }
                }
            }
            else return false;

            return true;
        }
    }
}