using System.Collections.Generic;
using System.Globalization;
using System.IO;
using AnimationTools;
using Unity.Mathematics;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace MotionMatching.Editor
{
    /// <summary>
    /// Imports a BVH file as a rig hierarchy (GameObject) and an AnimationClip sampled against it.
    /// </summary>
    [ScriptedImporter(6, "bvh")]
    public class BvhImporter : ScriptedImporter
    {
        public float unitScale = 0.01f;
        public bool onlyFirstFrame;
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var fileName = Path.GetFileNameWithoutExtension(ctx.assetPath);
            var (skeleton, rootPositions, boneRotations, frameTime) = Import(ctx, fileName, unitScale, onlyFirstFrame);

            var clip = BuildAnimationClip(fileName, skeleton, rootPositions, boneRotations, frameTime);
            ctx.AddObjectToAsset("clip", clip);
        }

        /// <summary>One parsed BVH joint: name, parent, and rest pose, before any GameObject exists.</summary>
        private struct BvhJoint
        {
            public string Name;
            public int ParentIndex;
            public float3 RestLocalPosition;
            public quaternion RestLocalRotation;
        }

        /// <summary>
        /// Creates one GameObject per <paramref name="joints"/> entry, parented to match the BVH
        /// joint tree (bone 0 parented under <paramref name="rigRoot"/>), with local
        /// position/rotation set from the parsed rest offsets. Returns the created Transform per
        /// joint index.
        /// </summary>
        private static Transform[] BuildJointHierarchy(List<BvhJoint> joints, Transform rigRoot)
        {
            var transforms = new Transform[joints.Count];

            for (var i = 0; i < joints.Count; i++)
            {
                var joint = joints[i];
                var boneTransform = new GameObject(joint.Name).transform;
                boneTransform.SetParent(i == 0 ? rigRoot : transforms[joint.ParentIndex], false);
                boneTransform.localPosition = joint.RestLocalPosition;
                boneTransform.localRotation = joint.RestLocalRotation;
                transforms[i] = boneTransform;
            }

            return transforms;
        }

        private static AnimationClip BuildAnimationClip(string clipName, Skeleton skeleton, Vector3[] rootPositions, Quaternion[][] boneRotations, float frameTime)
        {
            var clip = new AnimationClip { name = clipName, frameRate = 1f / frameTime };

            var boneCount = skeleton.BoneCount;
            var paths = new string[boneCount];
            for (var i = 0; i < boneCount; i++)
            {
                var bone = skeleton.GetBone(i);
                paths[i] = i == 0 ? bone.Name : paths[skeleton.GetParentIndex(i)] + "/" + bone.Name;
            }

            var frameCount = rootPositions.Length;

            var rootPosXKeys = new Keyframe[frameCount];
            var rootPosYKeys = new Keyframe[frameCount];
            var rootPosZKeys = new Keyframe[frameCount];
            for (var f = 0; f < frameCount; f++)
            {
                var time = f * frameTime;
                var position = rootPositions[f];
                rootPosXKeys[f] = new Keyframe(time, position.x);
                rootPosYKeys[f] = new Keyframe(time, position.y);
                rootPosZKeys[f] = new Keyframe(time, position.z);
            }
            clip.SetCurve(paths[0], typeof(Transform), "localPosition.x", new AnimationCurve(rootPosXKeys));
            clip.SetCurve(paths[0], typeof(Transform), "localPosition.y", new AnimationCurve(rootPosYKeys));
            clip.SetCurve(paths[0], typeof(Transform), "localPosition.z", new AnimationCurve(rootPosZKeys));

            for (var b = 0; b < boneCount; b++)
            {
                var rotations = boneRotations[b];
                var rotXKeys = new Keyframe[frameCount];
                var rotYKeys = new Keyframe[frameCount];
                var rotZKeys = new Keyframe[frameCount];
                var rotWKeys = new Keyframe[frameCount];
                for (var f = 0; f < frameCount; f++)
                {
                    var time = f * frameTime;
                    var rotation = rotations[f];
                    rotXKeys[f] = new Keyframe(time, rotation.x);
                    rotYKeys[f] = new Keyframe(time, rotation.y);
                    rotZKeys[f] = new Keyframe(time, rotation.z);
                    rotWKeys[f] = new Keyframe(time, rotation.w);
                }
                clip.SetCurve(paths[b], typeof(Transform), "localRotation.x", new AnimationCurve(rotXKeys));
                clip.SetCurve(paths[b], typeof(Transform), "localRotation.y", new AnimationCurve(rotYKeys));
                clip.SetCurve(paths[b], typeof(Transform), "localRotation.z", new AnimationCurve(rotZKeys));
                clip.SetCurve(paths[b], typeof(Transform), "localRotation.w", new AnimationCurve(rotWKeys));
            }

            // May flip quaternion key signs to remove double-cover discontinuities between keys;
            // the result samples to rotation-equivalent values, and consumers are hemisphere-corrected.
            clip.EnsureQuaternionContinuity();
            return clip;
        }

        private static (Skeleton skeleton, Vector3[] rootPositions, Quaternion[][] boneRotations, float frameTime) Import(
            AssetImportContext ctx, string fileName, float scale = 0.01f, bool onlyFirstFrame = false)
        {
            var channelAxisOrders = new List<AxisOrder>();

            var parentIndexStack = new Stack<int>();
            var whitespace = new char[] { ' ', '\t', '\r', '\n' };

            var words = File.ReadAllText(ctx.assetPath).Split(whitespace, System.StringSplitOptions.RemoveEmptyEntries);
            // string[] words = Regex.Split(bvh.text, "[\\s+|\\r*\\n+]+");
            var w = 0;
            // ROOT
            if (words[w++] != "HIERARCHY") Debug.LogError("[BVHImporter] HIERARCHY not found");
            if (words[w++] != "ROOT") Debug.LogError("[BVHImporter] ROOT not found");
            var rootName = words[w++];
            ReadLeftBracket(words, ref w);
            ReadOffset(words, ref w); // consumed but discarded: root position always comes from motion channel 0, never HIERARCHY
            ReadChannels(channelAxisOrders, words, ref w, true);
            var bones = new List<BvhJoint>
            {
                new()
                {
                    Name = rootName,
                    ParentIndex = -1,
                    RestLocalPosition = float3.zero,
                    RestLocalRotation = quaternion.identity
                }
            };
            // JOINTS
            var brackets = 1;
            var parent = 0;
            var jointIndex = 1;
            var it = 100000;
            if (ReadRightBracket(words, ref w)) brackets -= 1;
            while (brackets > 0 && --it > 0)
            {
                if (words[w++] != "JOINT") Debug.LogError("[BVHImporter] JOINT not found");
                var jointName = words[w++];
                var boneIndex = jointIndex++;
                ReadLeftBracket(words, ref w);
                parentIndexStack.Push(parent);
                var boneParentIndex = parent;
                parent = boneIndex;
                brackets += 1;
                var offset = ReadOffset(words, ref w) * scale;
                ReadChannels(channelAxisOrders, words, ref w);
                bones.Add(new BvhJoint
                {
                    Name = jointName,
                    ParentIndex = boneParentIndex,
                    RestLocalPosition = (float3)offset,
                    RestLocalRotation = quaternion.identity
                });
                if (words[w] == "End")
                {
                    w += 1;
                    if (words[w++] != "Site") Debug.LogError("[BVHImporter] End Site not found");
                    ReadLeftBracket(words, ref w);
                    ReadOffset(words, ref w);
                    if (!ReadRightBracket(words, ref w)) Debug.LogError("[BVHImporter] End Site right bracket not found");
                }
                while (words[w] == "End")
                {
                    w += 1;
                    if (words[w++] != "Site") Debug.LogError("[BVHImporter] End Site not found");
                    ReadLeftBracket(words, ref w);
                    ReadOffset(words, ref w);
                    if (!ReadRightBracket(words, ref w)) Debug.LogError("[BVHImporter] End Site right bracket not found");
                }
                while (ReadRightBracket(words, ref w))
                {
                    brackets -= 1;
                    if (parentIndexStack.Count > 0) parent = parentIndexStack.Pop();
                }
                while (words[w] == "End")
                {
                    w += 1;
                    if (words[w++] != "Site") Debug.LogError("[BVHImporter] End Site not found");
                    ReadLeftBracket(words, ref w);
                    ReadOffset(words, ref w);
                    if (!ReadRightBracket(words, ref w)) Debug.LogError("[BVHImporter] End Site right bracket not found");
                }
                while (ReadRightBracket(words, ref w))
                {
                    brackets -= 1;
                    if (parentIndexStack.Count > 0) parent = parentIndexStack.Pop();
                }
            }
            if (it <= 0) Debug.LogError("[BVHImporter] Infinite loop detected, Left and Right brackets does not match");

            // MOTION
            if (words[w++] != "MOTION") Debug.LogError("[BVHImporter] MOTION not found");
            if (words[w++] != "Frames:") Debug.LogError("[BVHImporter] Frames: not found");
            var numberFrames = int.Parse(words[w++]);
            if (words[w++] != "Frame") Debug.LogError("[BVHImporter] Frame not found");
            if (words[w++] != "Time:") Debug.LogError("[BVHImporter] Time: not found");
            var frameTime = float.Parse(words[w++], CultureInfo.InvariantCulture);

            // A .bvh imports like a small FBX: the rig is the main object and the AnimationClip a
            // sub-asset. The bone hierarchy sits under a container because Unity renames the main
            // object to the file name, which would clobber the root bone's name and break
            // name-based binding.
            var container = new GameObject(fileName);
            var boneTransforms = BuildJointHierarchy(bones, container.transform);
            var skeleton = new Skeleton(boneTransforms[0]);

            // The bone GameObjects come along as children of the container; adding one separately
            // would register it as a second root of the same asset.
            ctx.AddObjectToAsset("rig", container);
            ctx.SetMainObject(container);

            var numberChannels = channelAxisOrders.Count;
            var framesStored = onlyFirstFrame ? Mathf.Min(1, numberFrames) : numberFrames;

            var rootPositions = new Vector3[framesStored];
            var boneRotations = new Quaternion[bones.Count][];
            for (var b = 0; b < bones.Count; b++) boneRotations[b] = new Quaternion[framesStored];

            for (var f = 0; f < framesStored; f++)
            {
                for (var j = 0; j < numberChannels; ++j)
                {
                    var v1 = float.Parse(words[w++], CultureInfo.InvariantCulture);
                    var v2 = float.Parse(words[w++], CultureInfo.InvariantCulture);
                    var v3 = float.Parse(words[w++], CultureInfo.InvariantCulture);
                    var axisOrder = channelAxisOrders[j];
                    if (j == 0)
                    {
                        rootPositions[f] = BvhToUnityTranslation(v1, v2, v3, axisOrder) * scale;
                    }
                    else
                    {
                        boneRotations[j - 1][f] = BvhToUnityRotation(v1, v2, v3, axisOrder);
                    }
                }
            }

            return (skeleton, rootPositions, boneRotations, frameTime);
        }

        private static void ReadLeftBracket(string[] words, ref int w)
        {
            if (words[w++] != "{") Debug.LogError("[BVHImporter] { not found");
        }

        private static bool ReadRightBracket(string[] words, ref int w)
        {
            var isRightBracket = words[w] == "}";
            if (isRightBracket)
            {
                w += 1;
            }
            return isRightBracket;
        }

        private static Vector3 ReadOffset(string[] words, ref int w)
        {
            var offset = Vector3.zero;
            if (words[w++] != "OFFSET") Debug.LogError("[BVHImporter] OFFSET not found");
            offset.x = float.Parse(words[w++], CultureInfo.InvariantCulture);
            offset.y = float.Parse(words[w++], CultureInfo.InvariantCulture);
            offset.z = -float.Parse(words[w++], CultureInfo.InvariantCulture); // Unity is left-handed and BVH is right-handed (Z is opposite sign)
            return offset;
        }

        private static void ReadChannels(List<AxisOrder> channels, string[] words, ref int w, bool root = false)
        {
            if (words[w++] != "CHANNELS")
            {
                Debug.LogError("[BVHImporter] CHANNELS keyword not found");
                return;
            }

            var numChannels = int.Parse(words[w++]);

            if (root)
            {
                // The root **must** provide 3 translation + 3 rotation = 6 channels.
                if (numChannels != 6)
                {
                    Debug.LogError("[BVHImporter] The root joint must have exactly 6 channels");
                }

                // Remember in which order the three position axes appear (XYZ / XZY …)
                channels.Add(ReadChannelPosition(words, ref w));
            }
            else
            {
                // Non-root joints may come with 3 **or** 6 channels depending on the exporter.
                // When 6, the first three are XYZ-position that we can safely ignore.
                if (numChannels == 6)
                {
                    // Consume and discard the position axis order
                    ReadChannelPosition(words, ref w);
                }
                else if (numChannels != 3)
                {
                    Debug.LogError($"[BVHImporter] Unexpected channel count ({numChannels}) at joint — expected 3 or 6");
                }
                // If there are only 3 channels — nothing to skip, cursor is already after the count.
            }
            // Rotation channels are always 3, so we can safely read them.
            channels.Add(ReadChannelRotation(words, ref w));
        }

        private static AxisOrder ReadChannelPosition(string[] words, ref int w)
        {
            var order1 = words[w++];
            var order2 = words[w++];
            var order3 = words[w++];
            if (order1 != "Xposition" && order1 != "Yposition" && order1 != "Zposition") Debug.LogError("[BVHImporter] root position channels must be Xposition, Yposition or Zposition");
            if (order2 != "Xposition" && order2 != "Yposition" && order2 != "Zposition") Debug.LogError("[BVHImporter] root position channels must be Xposition, Yposition or Zposition");
            if (order3 != "Xposition" && order3 != "Yposition" && order3 != "Zposition") Debug.LogError("[BVHImporter] root position channels must be Xposition, Yposition or Zposition");
            if (order1 == "Xposition")
            {
                if (order2 == "Yposition")
                {
                    if (order3 == "Zposition") return AxisOrder.XYZ;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xposition, Yposition and Zposition");
                }
                else
                {
                    if (order3 == "Yposition") return AxisOrder.XZY;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xposition, Yposition and Zposition");
                }
            }
            else if (order1 == "Yposition")
            {
                if (order2 == "Xposition")
                {
                    if (order3 == "Zposition") return AxisOrder.YXZ;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xposition, Yposition and Zposition");
                }
                else
                {
                    if (order3 == "Xposition") return AxisOrder.YZX;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xposition, Yposition and Zposition");
                }
            }
            else
            {
                if (order2 == "Xposition")
                {
                    if (order3 == "Yposition") return AxisOrder.ZXY;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xposition, Yposition and Zposition");
                }
                else
                {
                    if (order3 == "Xposition") return AxisOrder.ZYX;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xposition, Yposition and Zposition");
                }
            }
            return AxisOrder.None;
        }

        private static AxisOrder ReadChannelRotation(string[] words, ref int w)
        {
            var order1 = words[w++];
            var order2 = words[w++];
            var order3 = words[w++];
            if (order1 != "Xrotation" && order1 != "Yrotation" && order1 != "Zrotation") Debug.LogError("[BVHImporter] root or joint rotation channels must be Xrotation, Yrotation or Zrotation");
            if (order2 != "Xrotation" && order2 != "Yrotation" && order2 != "Zrotation") Debug.LogError("[BVHImporter] root or joint rotation channels must be Xrotation, Yrotation or Zrotation");
            if (order3 != "Xrotation" && order3 != "Yrotation" && order3 != "Zrotation") Debug.LogError("[BVHImporter] root or joint rotation channels must be Xrotation, Yrotation or Zrotation");
            if (order1 == "Xrotation")
            {
                if (order2 == "Yrotation")
                {
                    if (order3 == "Zrotation") return AxisOrder.XYZ;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xrotation, Yrotation and Zrotation");
                }
                else
                {
                    if (order3 == "Yrotation") return AxisOrder.XZY;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xrotation, Yrotation and Zrotation");
                }
            }
            else if (order1 == "Yrotation")
            {
                if (order2 == "Xrotation")
                {
                    if (order3 == "Zrotation") return AxisOrder.YXZ;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xrotation, Yrotation and Zrotation");
                }
                else
                {
                    if (order3 == "Xrotation") return AxisOrder.YZX;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xrotation, Yrotation and Zrotation");
                }
            }
            else
            {
                if (order2 == "Xrotation")
                {
                    if (order3 == "Yrotation") return AxisOrder.ZXY;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xrotation, Yrotation and Zrotation");
                }
                else
                {
                    if (order3 == "Xrotation") return AxisOrder.ZYX;
                    else Debug.LogError("[BVHImporter] root position channels must contain Xrotation, Yrotation and Zrotation");
                }
            }
            return AxisOrder.None;
        }

        private static Quaternion BvhToUnityRotation(float v1, float v2, float v3, AxisOrder rotationOrder)
        {
            if (rotationOrder == AxisOrder.None) Debug.LogError("[BVHImporter] rotationOrder is None. There was an error while reading the channels");

            switch (rotationOrder)
            {
                // Why some are negative? Because Unity is left-handed and BVH is right-handed. See: https://stackoverflow.com/questions/31191752/right-handed-euler-angles-xyz-to-left-handed-euler-angles-xyz
                case AxisOrder.XYZ: return Quaternion.AngleAxis(-v1, Vector3.right) * Quaternion.AngleAxis(-v2, Vector3.up) * Quaternion.AngleAxis(v3, Vector3.forward); // XYZ
                case AxisOrder.XZY: return Quaternion.AngleAxis(-v1, Vector3.right) * Quaternion.AngleAxis(v2, Vector3.forward) * Quaternion.AngleAxis(-v3, Vector3.up); // XZY
                case AxisOrder.YXZ: return Quaternion.AngleAxis(-v1, Vector3.up) * Quaternion.AngleAxis(-v2, Vector3.right) * Quaternion.AngleAxis(v3, Vector3.forward); // YXZ
                case AxisOrder.YZX: return Quaternion.AngleAxis(-v1, Vector3.up) * Quaternion.AngleAxis(v2, Vector3.forward) * Quaternion.AngleAxis(-v3, Vector3.right); // YZX
                case AxisOrder.ZXY: return Quaternion.AngleAxis(v1, Vector3.forward) * Quaternion.AngleAxis(-v2, Vector3.right) * Quaternion.AngleAxis(-v3, Vector3.up); // ZXY
                case AxisOrder.ZYX: return Quaternion.AngleAxis(v1, Vector3.forward) * Quaternion.AngleAxis(-v2, Vector3.up) * Quaternion.AngleAxis(-v3, Vector3.right); // ZYX
            }

            return Quaternion.identity;
        }

        private static Vector3 BvhToUnityTranslation(float v1, float v2, float v3, AxisOrder translationOrder)
        {
            if (translationOrder == AxisOrder.None) Debug.LogError("[BVHImporter] translationOrder is None. There was an error while reading the channels");

            // BVH's z+ axis is Unity's left (z-) (Unity is left-handed BVH is right-handed)
            switch (translationOrder)
            {
                case AxisOrder.XYZ: return new Vector3(v1, v2, -v3); // XYZ
                case AxisOrder.XZY: return new Vector3(v1, v3, -v2); // XZY
                case AxisOrder.YXZ: return new Vector3(v2, v1, -v3); // YXZ
                case AxisOrder.YZX: return new Vector3(v3, v1, -v2); // YZX
                case AxisOrder.ZXY: return new Vector3(v2, v3, -v1); // ZXY
                case AxisOrder.ZYX: return new Vector3(v3, v2, -v1); // ZYX
            }
            return Vector3.zero;
        }

        private enum AxisOrder
        {
            XYZ, XZY, YXZ, YZX, ZXY, ZYX, None
        }

    }
}
