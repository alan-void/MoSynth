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
                WriteSkeleton(writer, poseSet.Skeleton);

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
                if (!ReadAndCheckSkeleton(reader, fileName, skeleton)) return false;

                poseSet.SetSkeleton(skeleton);

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
                // The skeleton block already agreed with the asset, so this only fires on a file
                // whose two bone counts disagree with each other — a truncated or corrupt write.
                if (nJoints != skeleton.BoneCount)
                {
                    Debug.LogError($"\"{fileName}.mmpose\" says {nJoints} joints in its pose header but " +
                                   $"{skeleton.BoneCount} in its skeleton block; the file is corrupt. " +
                                   "Regenerate the databases.");
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

    // --- Skeleton block ---------------------------------------------------------------------
    //
    // The bones this database was extracted over, written ahead of the poses. C# does not need
    // them — a config carries its own Skeleton and reads rest pose live off the Transforms — but
    // the Python half has no ScriptableObject to read, and needs joint names for the bone-weight
    // table, parent indices for FK, and rest offsets for root-space positions. Keeping the block
    // in the same file as the poses is what stops the two drifting apart, which is exactly what a
    // separate .mmskeleton did. The simulation frame is not written: both halves derive it from
    // bone 0's rest rotation, which the bone entries already carry.

    private static void WriteSkeleton(BinaryWriter writer, Skeleton skeleton)
    {
        writer.Write((uint)skeleton.BoneCount);
        for (var i = 0; i < skeleton.BoneCount; ++i)
        {
            var bone = skeleton.GetBone(i);
            writer.Write(bone.Name);
            writer.Write((uint)skeleton.GetParentIndex(i)); // bone 0's -1 round-trips as 0xFFFFFFFF
            WriteFloat3(writer, bone.RestLocalPosition);
            WriteQuaternion(writer, bone.RestLocalRotation);
        }
    }

    /// <summary>
    /// Reads the skeleton block and checks it describes the same bone tree as
    /// <paramref name="skeleton"/> — matching names and parent indices, the comparison
    /// <see cref="Skeleton.MatchesFrom"/> makes. Rest transforms are read and discarded, since C#
    /// takes those live off the Transforms. This is what catches a database extracted over a
    /// different rig, or over the same rig before its bone list changed.
    /// </summary>
    private static bool ReadAndCheckSkeleton(BinaryReader reader, string fileName, Skeleton skeleton)
    {
        var boneCount = (int)reader.ReadUInt32();
        if (boneCount != skeleton.BoneCount)
        {
            Debug.LogError($"\"{fileName}.mmpose\" was built over a skeleton of {boneCount} bones but " +
                           $"\"{skeleton.Name}\" has {skeleton.BoneCount} — regenerate the database.");
            return false;
        }

        for (var i = 0; i < boneCount; ++i)
        {
            var name = reader.ReadString();
            var parentIndex = (int)reader.ReadUInt32(); // 0xFFFFFFFF back to -1
            ReadFloat3(reader);
            ReadQuaternion(reader);

            if (name != skeleton.GetBone(i).Name)
            {
                Debug.LogError($"\"{fileName}.mmpose\" has \"{name}\" at bone {i} but \"{skeleton.Name}\" " +
                               $"has \"{skeleton.GetBone(i).Name}\" — regenerate the database.");
                return false;
            }

            if (parentIndex != skeleton.GetParentIndex(i))
            {
                Debug.LogError($"\"{fileName}.mmpose\" parents bone {i} (\"{name}\") to {parentIndex} but " +
                               $"\"{skeleton.Name}\" parents it to {skeleton.GetParentIndex(i)} — " +
                               "regenerate the database.");
                return false;
            }
        }

        return true;
    }
}
}