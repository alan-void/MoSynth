using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using System.IO;
using System.Text;
using System;
using System.Runtime.InteropServices;

namespace AnimationTools
{
using static AnimationTools.BinarySerializerExtensions;

/// <summary>
/// Reads and writes a <see cref="PoseSet"/> as a <c>.mmpose</c> file, which Python reads too.
/// See openwiki/animation-tools/on-disk-formats.md.
/// </summary>
public class PoseSerializer
{
    /// <summary>Writes <paramref name="poseSet"/> to <c>path/fileName.mmpose</c>, creating the folder.</summary>
    public void Serialize(PoseSet poseSet, string path, string fileName)
    {
        Directory.CreateDirectory(path);

        using (var stream = File.Open(Path.Combine(path, fileName + ".mmpose"), FileMode.Create))
        {
            using (var writer = new BinaryWriter(stream, Encoding.UTF8))
            {
                WriteSkeleton(writer, poseSet.Skeleton);

                writer.Write((uint)poseSet.NumberClips);
                for (var i = 0; i < poseSet.NumberClips; ++i)
                {
                    var clip = poseSet.GetAnimationClip(i);
                    writer.Write((uint)clip.Start);
                    writer.Write((uint)clip.End);
                    writer.Write(clip.FrameTime);
                }

                writer.Write((uint)poseSet.NumberPoses);
                writer.Write((uint)poseSet.Skeleton.BoneCount);
                writer.Write((uint)poseSet.NumberTags);
                for (var i = 0; i < poseSet.NumberPoses; ++i)
                {
                    var frame = poseSet.GetPoseBuffer(i);
                    WriteFloat3Slice(writer, frame.Positions);
                    WriteQuaternionSlice(writer, frame.Rotations);
                    WriteFloat3Slice(writer, frame.Velocities);
                    WriteFloat3Slice(writer, frame.AngularVelocities);
                    writer.Write(frame.GetBool(poseSet.LeftFootContactHandle) ? 1u : 0u);
                    writer.Write(frame.GetBool(poseSet.RightFootContactHandle) ? 1u : 0u);
                }

                WritePhase(writer, poseSet);

                for (var i = 0; i < poseSet.NumberTags; ++i)
                {
                    var animationTag = poseSet.GetTag(i);
                    writer.Write(animationTag.Name);
                    writer.Write((uint)animationTag.NumberRanges);
                    for (var r = 0; r < animationTag.NumberRanges; ++r)
                    {
                        animationTag.GetRange(r, out var startRange, out var endRange);
                        writer.Write((uint)startRange);
                        writer.Write((uint)endRange);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Reads <c>path/fileName.mmpose</c> over <paramref name="skeleton"/>. False when the file is
    /// missing, truncated, or was extracted over a different bone tree.
    /// </summary>
    public bool Deserialize(string path, string fileName, Skeleton skeleton, out PoseSet poseSet)
    {
        poseSet = new PoseSet();
        if (skeleton == null || !skeleton.IsSet) return false;

        var posePath = Path.Combine(path, fileName + ".mmpose");
        if (!File.Exists(posePath))
            return false;

        var poseData = File.ReadAllBytes(posePath);
        using (var stream = new MemoryStream(poseData))
        {
            using (var reader = new BinaryReader(stream, Encoding.UTF8))
            {
                if (!ReadAndCheckSkeleton(reader, fileName, skeleton)) return false;

                poseSet.SetSkeleton(skeleton);

                var clipCount = reader.ReadUInt32();
                poseSet.SetClipCapacity(clipCount);
                for (var i = 0; i < clipCount; i++)
                {
                    var start = reader.ReadUInt32();
                    var end = reader.ReadUInt32();
                    var frameTime = reader.ReadSingle();
                    poseSet.AddAnimationClipDeserialized(new PoseSet.AnimationClip((int)start, (int)end, frameTime));
                }

                var poseCount = reader.ReadUInt32();
                var jointCount = reader.ReadUInt32();
                var tagCount = reader.ReadUInt32();
                // The skeleton block already agreed with the asset, so this only fires on a file
                // whose two bone counts disagree with each other — a truncated or corrupt write.
                if (jointCount != skeleton.BoneCount)
                {
                    Debug.LogError($"\"{fileName}.mmpose\" says {jointCount} joints in its pose header but " +
                                   $"{skeleton.BoneCount} in its skeleton block; the file is corrupt. " +
                                   "Regenerate the databases.");
                    return false;
                }

                var float3BufferSize = (int)jointCount * 3 * sizeof(float);
                var quaternionBufferSize = (int)jointCount * 4 * sizeof(float);
                var float3Buffer = new byte[float3BufferSize];
                var quaternionBuffer = new byte[quaternionBufferSize];

                poseSet.SetPoseCapacity(poseCount);
                var frames = poseSet.AppendRawFrames((int)poseCount);

                // The header counts already agree with the skeleton, so a short read can only be a
                // truncated file; it is reported and refused rather than asserted.
                bool ReadBlock(byte[] buffer, int byteCount, int poseIndex)
                {
                    if (reader.Read(buffer, 0, byteCount) == byteCount) return true;

                    Debug.LogError($"\"{fileName}.mmpose\" ends inside pose {poseIndex} of {poseCount}; " +
                                   "the file is truncated. Regenerate the databases.");
                    return false;
                }

                for (var i = 0; i < poseCount; i++)
                {
                    var frame = frames[i];
                    var positions = frame.Positions;
                    var rotations = frame.Rotations;
                    var velocities = frame.Velocities;
                    var angularVelocities = frame.AngularVelocities;

                    if (!ReadBlock(float3Buffer, float3BufferSize, i)) return false;
                    var positionsSpan = MemoryMarshal.Cast<byte, float>(float3Buffer);
                    for (var j = 0; j < jointCount; j++)
                    {
                        positions[j] = new float3(
                            positionsSpan[j * 3],
                            positionsSpan[j * 3 + 1],
                            positionsSpan[j * 3 + 2]
                        );
                    }

                    if (!ReadBlock(quaternionBuffer, quaternionBufferSize, i)) return false;
                    var rotationsSpan = MemoryMarshal.Cast<byte, float>(quaternionBuffer);
                    for (var j = 0; j < jointCount; j++)
                    {
                        rotations[j] = new quaternion(
                            rotationsSpan[j * 4],
                            rotationsSpan[j * 4 + 1],
                            rotationsSpan[j * 4 + 2],
                            rotationsSpan[j * 4 + 3]
                        );
                    }

                    if (!ReadBlock(float3Buffer, float3BufferSize, i)) return false;
                    var velocitiesSpan = MemoryMarshal.Cast<byte, float>(float3Buffer);
                    for (var j = 0; j < jointCount; j++)
                    {
                        velocities[j] = new float3(
                            velocitiesSpan[j * 3],
                            velocitiesSpan[j * 3 + 1],
                            velocitiesSpan[j * 3 + 2]
                        );
                    }

                    if (!ReadBlock(float3Buffer, float3BufferSize, i)) return false;
                    var angularVelocitiesSpan = MemoryMarshal.Cast<byte, float>(float3Buffer);
                    for (var j = 0; j < jointCount; j++)
                    {
                        angularVelocities[j] = new float3(
                            angularVelocitiesSpan[j * 3],
                            angularVelocitiesSpan[j * 3 + 1],
                            angularVelocitiesSpan[j * 3 + 2]
                        );
                    }

                    frame.SetBool(poseSet.LeftFootContactHandle, reader.ReadUInt32() == 1u);
                    frame.SetBool(poseSet.RightFootContactHandle, reader.ReadUInt32() == 1u);
                }

                if (!ReadPhase(reader, stream, fileName, poseSet, (int)poseCount)) return false;

                for (var i = 0; i < tagCount; i++)
                {
                    var name = reader.ReadString();
                    var rangeCount = (int)reader.ReadUInt32();
                    var tagStarts = new List<int>(rangeCount);
                    var tagEnds = new List<int>(rangeCount);
                    for (var r = 0; r < rangeCount; r++)
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

    // --- Gait phase block -------------------------------------------------------------------
    //
    // Phase and its rate, one pair per pose, evaluated in C# from the authored footfalls so Python
    // trains on what the clip editor draws. A rate of zero marks a frame with no measurable cycle.

    private static void WritePhase(BinaryWriter writer, PoseSet poseSet)
    {
        for (var i = 0; i < poseSet.NumberPoses; ++i)
        {
            writer.Write(poseSet.GetPhase(i));
            writer.Write(poseSet.GetPhaseRate(i));
        }
    }

    /// <summary>
    /// Reads the gait phase block, refusing a file that stops short of it. The format is unversioned,
    /// so a stale or truncated file is caught by checking the content is there.
    /// </summary>
    private static bool ReadPhase(BinaryReader reader, Stream stream, string fileName, PoseSet poseSet,
        int poseCount)
    {
        if (stream.Length - stream.Position < (long)poseCount * 2 * sizeof(float))
        {
            Debug.LogError($"\"{fileName}.mmpose\" ends before its gait phase block; it is truncated, or " +
                           "predates the block. Regenerate the databases.");
            return false;
        }

        for (var i = 0; i < poseCount; i++)
        {
            poseSet.SetPhase(i, reader.ReadSingle(), reader.ReadSingle());
        }

        return true;
    }

    // --- Skeleton block ---------------------------------------------------------------------
    //
    // The bones this database was extracted over, for Python, which has no Skeleton asset to read.
    // Kept in the same file as the poses so the two cannot drift apart. The simulation frame is not
    // written: both sides derive it from bone 0's rest rotation.

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