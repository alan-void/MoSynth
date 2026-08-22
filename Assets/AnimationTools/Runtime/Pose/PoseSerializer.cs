using System.Collections.Generic;
using AnimationTools;
using UnityEngine;
using Unity.Mathematics;
using System.IO;
using System.Text;
using System;
using System.Runtime.InteropServices;
using NUnit.Framework;

namespace AnimationTools
{
using static AnimationTools.BinarySerializerExtensions;

public class PoseSerializer
{
    /// <summary>
    /// Format version written at the start of every .mmpose file. Bumped whenever the binary
    /// layout changes so an old file is rejected with a clear message instead of misparsed.
    /// </summary>
    private const uint MmPoseFormatVersion = 2;

    private static readonly byte[] MmPoseMagic = { (byte)'M', (byte)'M', (byte)'P', (byte)'S' };

    /// <summary>
    /// Stores the full pose representation of all poses for Motion Matching in a binary format
    /// in the specified path with name filename and extension .mmpose
    /// </summary>
    public void Serialize(PoseSet poseSet, string path, string fileName)
    {
        Directory.CreateDirectory(path); // create directory and parent directories if they don't exist

        // Write Poses
        using (var stream = File.Open(Path.Combine(path, fileName + ".mmpose"), FileMode.Create))
        {
            using (var writer = new BinaryWriter(stream, Encoding.UTF8))
            {
                // Magic + format version
                writer.Write(MmPoseMagic);
                writer.Write(MmPoseFormatVersion);

                // Serialize Number Animation Clips
                writer.Write((uint)poseSet.NumberClips);
                // Serialize Animation Clips
                for (int i = 0; i < poseSet.NumberClips; ++i)
                {
                    PoseSet.AnimationClip clip = poseSet.GetAnimationClip(i);
                    writer.Write((uint)clip.Start);
                    writer.Write((uint)clip.End);
                    writer.Write(clip.FrameTime);
                }

                // Serialize Number Poses & Number Joints & Number Tags
                writer.Write((uint)poseSet.NumberPoses);
                writer.Write((uint)poseSet.Skeleton.BoneCount);
                writer.Write((uint)poseSet.NumberTags);
                // Serialize Poses
                for (int i = 0; i < poseSet.NumberPoses; ++i)
                {
                    var frame = poseSet.GetPoseBuffer(i);
                    WriteFloat3Slice(writer, frame.Positions);
                    WriteQuaternionSlice(writer, frame.Rotations);
                    WriteFloat3Slice(writer, frame.Velocities);
                    WriteFloat3Slice(writer, frame.AngularVelocities);
                    writer.Write(frame.GetBool(poseSet.LeftFootContactHandle) ? 1u : 0u);
                    writer.Write(frame.GetBool(poseSet.RightFootContactHandle) ? 1u : 0u);
                }

                // Serialize Tags
                for (int i = 0; i < poseSet.NumberTags; ++i)
                {
                    PoseSet.AnimationTag animationTag = poseSet.GetTag(i);
                    writer.Write(animationTag.Name);
                    writer.Write((uint)animationTag.NumberRanges);
                    for (int r = 0; r < animationTag.NumberRanges; ++r)
                    {
                        animationTag.GetRange(r, out int startRange, out int endRange);
                        writer.Write((uint)startRange);
                        writer.Write((uint)endRange);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Reads the full pose representation of all poses for Motion Matching from a binary format
    /// in the specified path with name filename and extension .mmpose, over the given
    /// <paramref name="skeleton"/>. Returns true if poseSet was successfully deserialized, false
    /// otherwise.
    /// </summary>
    public bool Deserialize(string path, string fileName, Skeleton skeleton, out PoseSet poseSet)
    {
        poseSet = new PoseSet();
        if (skeleton == null || !skeleton.IsSet) return false;

        poseSet.SetSkeleton(skeleton);

        // --------------------
        // Read Pose File
        // --------------------
        string posePath = Path.Combine(path, fileName + ".mmpose");
        if (!File.Exists(posePath))
            return false;

        byte[] poseData = File.ReadAllBytes(posePath);
        using (var ms = new MemoryStream(poseData))
        {
            using (var reader = new BinaryReader(ms, Encoding.UTF8))
            {
                var magic = reader.ReadBytes(4);
                if (magic.Length != 4 || magic[0] != MmPoseMagic[0] || magic[1] != MmPoseMagic[1] ||
                    magic[2] != MmPoseMagic[2] || magic[3] != MmPoseMagic[3])
                {
                    Debug.LogError($"\"{fileName}.mmpose\" is an old or unknown format (expected MMPS v{MmPoseFormatVersion}) — regenerate the databases from the MotionMatchingData editor.");
                    return false;
                }

                uint version = reader.ReadUInt32();
                if (version != MmPoseFormatVersion)
                {
                    Debug.LogError($"\"{fileName}.mmpose\" is an old or unknown format (expected MMPS v{MmPoseFormatVersion}) — regenerate the databases from the MotionMatchingData editor.");
                    return false;
                }

                uint nClips = reader.ReadUInt32();
                poseSet.SetClipCapacity(nClips);
                for (int i = 0; i < nClips; i++)
                {
                    uint start = reader.ReadUInt32();
                    uint end = reader.ReadUInt32();
                    float frameTime = reader.ReadSingle();
                    poseSet.AddAnimationClipDeserialized(new PoseSet.AnimationClip((int)start, (int)end, frameTime));
                }

                uint nPoses = reader.ReadUInt32();
                uint nJoints = reader.ReadUInt32();
                uint nTags = reader.ReadUInt32();
                if (nJoints != skeleton.BoneCount)
                {
                    Debug.LogError($"\"{fileName}.mmpose\" has {nJoints} joints but skeleton \"{skeleton.Name}\" has {skeleton.BoneCount} — regenerate the databases from the MotionMatchingData editor.");
                    return false;
                }

                // Precompute sizes for the buffers (they remain constant across iterations)
                int float3BufferSize = (int)nJoints * 3 * sizeof(float);
                int quaternionBufferSize = (int)nJoints * 4 * sizeof(float);

                // Allocate reusable buffers once outside the loop
                byte[] float3Buffer = new byte[float3BufferSize];
                byte[] quaternionBuffer = new byte[quaternionBufferSize];

                poseSet.SetPoseCapacity(nPoses);
                var frames = poseSet.AppendRawFrames((int)nPoses);
                for (int i = 0; i < nPoses; i++)
                {
                    var frame = frames[i];
                    var positions = frame.Positions;
                    var rotations = frame.Rotations;
                    var velocities = frame.Velocities;
                    var angularVelocities = frame.AngularVelocities;

                    // --- Read JointLocalPositions ---
                    var read = reader.Read(float3Buffer, 0, float3BufferSize);
                    Assert.IsTrue(read == float3BufferSize);
                    var positionsSpan = MemoryMarshal.Cast<byte, float>(float3Buffer);
                    for (int j = 0; j < nJoints; j++)
                    {
                        positions[j] = new float3(
                            positionsSpan[j * 3],
                            positionsSpan[j * 3 + 1],
                            positionsSpan[j * 3 + 2]
                        );
                    }

                    // --- Read JointLocalRotations ---
                    read = reader.Read(quaternionBuffer, 0, quaternionBufferSize);
                    Assert.IsTrue(read == quaternionBufferSize);
                    Span<float> rotationsSpan = MemoryMarshal.Cast<byte, float>(quaternionBuffer);
                    for (int j = 0; j < nJoints; j++)
                    {
                        rotations[j] = new quaternion(
                            rotationsSpan[j * 4],
                            rotationsSpan[j * 4 + 1],
                            rotationsSpan[j * 4 + 2],
                            rotationsSpan[j * 4 + 3]
                        );
                    }

                    // --- Read JointLocalVelocities ---
                    read = reader.Read(float3Buffer, 0, float3BufferSize);
                    Assert.IsTrue(read == float3BufferSize);
                    Span<float> velocitiesSpan = MemoryMarshal.Cast<byte, float>(float3Buffer);
                    for (int j = 0; j < nJoints; j++)
                    {
                        velocities[j] = new float3(
                            velocitiesSpan[j * 3],
                            velocitiesSpan[j * 3 + 1],
                            velocitiesSpan[j * 3 + 2]
                        );
                    }

                    // --- Read JointLocalAngularVelocities ---
                    read = reader.Read(float3Buffer, 0, float3BufferSize);
                    Assert.IsTrue(read == float3BufferSize);
                    Span<float> angularVelocitiesSpan = MemoryMarshal.Cast<byte, float>(float3Buffer);
                    for (int j = 0; j < nJoints; j++)
                    {
                        angularVelocities[j] = new float3(
                            angularVelocitiesSpan[j * 3],
                            angularVelocitiesSpan[j * 3 + 1],
                            angularVelocitiesSpan[j * 3 + 2]
                        );
                    }

                    // --- Read contact flags ---
                    frame.SetBool(poseSet.LeftFootContactHandle, reader.ReadUInt32() == 1u);
                    frame.SetBool(poseSet.RightFootContactHandle, reader.ReadUInt32() == 1u);
                }

                for (int i = 0; i < nTags; i++)
                {
                    string name = reader.ReadString();
                    int nRanges = (int)reader.ReadUInt32();
                    List<int> tagStarts = new List<int>(nRanges);
                    List<int> tagEnds = new List<int>(nRanges);
                    for (int r = 0; r < nRanges; r++)
                    {
                        tagStarts.Add((int)reader.ReadUInt32());
                        tagEnds.Add((int)reader.ReadUInt32());
                    }

                    poseSet.AddTagDeserialized(name, tagStarts, tagEnds);
                }

                poseSet.ConvertTagsToNativeArrays();
            }
        }

        return true;
    }
}
}