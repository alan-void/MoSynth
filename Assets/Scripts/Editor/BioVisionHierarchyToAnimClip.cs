using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Linq;

namespace AnimationTools
{
    /// <summary>
    /// Converts a selected <c>.bvh</c> file into an <see cref="AnimationClip"/> asset, from the
    /// Assets context menu.
    /// </summary>
    public static class BioVisionHierarchyToAnimClip
    {
        /// <summary>BVH is conventionally in centimetres; Unity is in metres. Use 1 for a BVH in metres.</summary>
        private const float UnitScale = 0.01f;

        [MenuItem("Assets/Convert BVH to AnimationClip")]
        public static void ConvertSelectedBvh()
        {
            foreach (var selectedObject in Selection.objects)
            {
                var path = AssetDatabase.GetAssetPath(selectedObject);

                if (string.IsNullOrEmpty(path) || (!path.EndsWith(".bvh") && !path.EndsWith(".txt")))
                {
                    Debug.LogWarning($"Skipping non-bvh file: {path}");
                    continue;
                }

                var parser = new BVHParser();
                try
                {
                    Debug.Log($"Parsing BVH: {path} with Scale {UnitScale}...");
                    var fileContent = File.ReadAllText(path);
                    var data = parser.Parse(fileContent, UnitScale);

                    var clipName = Path.GetFileNameWithoutExtension(path);
                    var clip = CreateAnimationClip(data, clipName);

                    var newPath = Path.Combine(Path.GetDirectoryName(path), clipName + ".anim");
                    AssetDatabase.CreateAsset(clip, newPath);
                    AssetDatabase.SaveAssets();

                    Debug.Log($"<color=green>Success:</color> AnimationClip created at {newPath}");
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"Failed to parse BVH: {e.Message}\n{e.StackTrace}");
                }
            }
        }

        private static AnimationClip CreateAnimationClip(BVHData data, string clipName)
        {
            var clip = new AnimationClip
            {
                name = clipName,
                frameRate = 1f / data.FrameTime,
                legacy = false
            };

            foreach (var joint in data.AllJoints)
            {
                var curvePosX = new AnimationCurve();
                var curvePosY = new AnimationCurve();
                var curvePosZ = new AnimationCurve();

                var curveRotX = new AnimationCurve();
                var curveRotY = new AnimationCurve();
                var curveRotZ = new AnimationCurve();
                var curveRotW = new AnimationCurve();

                var relativePath = GetRelativePath(joint);

                for (var frame = 0; frame < data.NumFrames; frame++)
                {
                    var time = frame * data.FrameTime;

                    var pos = joint.GetPosition(frame);
                    var rot = joint.GetRotation(frame);

                    // Right-handed BVH to left-handed Unity: mirror across X. Positions are already
                    // scaled by the parser.
                    var unityPos = new Vector3(-pos.x, pos.y, pos.z);
                    var unityRot = new Quaternion(-rot.x, rot.y, rot.z, -rot.w);

                    if (joint.HasPos)
                    {
                        curvePosX.AddKey(time, unityPos.x);
                        curvePosY.AddKey(time, unityPos.y);
                        curvePosZ.AddKey(time, unityPos.z);
                    }

                    curveRotX.AddKey(time, unityRot.x);
                    curveRotY.AddKey(time, unityRot.y);
                    curveRotZ.AddKey(time, unityRot.z);
                    curveRotW.AddKey(time, unityRot.w);
                }

                if (joint.HasPos)
                {
                    clip.SetCurve(relativePath, typeof(Transform), "localPosition.x", curvePosX);
                    clip.SetCurve(relativePath, typeof(Transform), "localPosition.y", curvePosY);
                    clip.SetCurve(relativePath, typeof(Transform), "localPosition.z", curvePosZ);
                }

                clip.SetCurve(relativePath, typeof(Transform), "localRotation.x", curveRotX);
                clip.SetCurve(relativePath, typeof(Transform), "localRotation.y", curveRotY);
                clip.SetCurve(relativePath, typeof(Transform), "localRotation.z", curveRotZ);
                clip.SetCurve(relativePath, typeof(Transform), "localRotation.w", curveRotW);
            }

            clip.EnsureQuaternionContinuity();
            return clip;
        }

        private static string GetRelativePath(BVHJoint joint)
        {
            var path = joint.Name;
            var current = joint.Parent;
            while (current != null)
            {
                path = current.Name + "/" + path;
                current = current.Parent;
            }
            return path;
        }
    }

    public class BVHData
    {
        public BVHJoint Root;
        public List<BVHJoint> AllJoints = new List<BVHJoint>();
        public int NumFrames;
        public float FrameTime;
    }

    public class BVHJoint
    {
        public string Name;
        public BVHJoint Parent;
        public Vector3 Offset;
        public List<string> Channels = new List<string>();
        public int ChannelOffsetIndex;
        public bool HasPos => Channels.Any(c => c.Contains("position"));

        public List<Vector3> PosData = new List<Vector3>();
        public List<Quaternion> RotData = new List<Quaternion>();

        public Vector3 GetPosition(int frame) => PosData.Count > frame ? PosData[frame] : Offset;
        public Quaternion GetRotation(int frame) => RotData.Count > frame ? RotData[frame] : Quaternion.identity;
    }

    public class BVHParser
    {
        private int _channelIndexCounter;
        private readonly List<float[]> _motionData = new List<float[]>();

        public BVHData Parse(string bvhText, float scaleFactor)
        {
            var data = new BVHData();
            using (var reader = new StringReader(bvhText))
            {
                var line = reader.ReadLine();
                BVHJoint currentJoint = null;

                // HIERARCHY section.
                while (line != null)
                {
                    var cleanLine = line.Trim();
                    var parts = cleanLine.Split(new[] { ' ', '\t' }, System.StringSplitOptions.RemoveEmptyEntries);

                    if (parts.Length == 0) { line = reader.ReadLine(); continue; }

                    if (parts[0] == "HIERARCHY") { }
                    else if (parts[0] == "ROOT" || parts[0] == "JOINT")
                    {
                        var joint = new BVHJoint { Name = parts[1], Parent = currentJoint };
                        data.AllJoints.Add(joint);

                        if (currentJoint == null) data.Root = joint;
                        currentJoint = joint;
                    }
                    else if (parts[0] == "End")
                    {
                        // Not added to AllJoints: an end site only carries the offset of a leaf tip.
                        currentJoint = new BVHJoint { Name = "End Site", Parent = currentJoint };
                    }
                    else if (parts[0] == "OFFSET")
                    {
                        if (currentJoint != null)
                        {
                            currentJoint.Offset = new Vector3(
                                float.Parse(parts[1], CultureInfo.InvariantCulture) * scaleFactor,
                                float.Parse(parts[2], CultureInfo.InvariantCulture) * scaleFactor,
                                float.Parse(parts[3], CultureInfo.InvariantCulture) * scaleFactor
                            );
                        }
                    }
                    else if (parts[0] == "CHANNELS")
                    {
                        if (currentJoint != null)
                        {
                            currentJoint.ChannelOffsetIndex = _channelIndexCounter;
                            var count = int.Parse(parts[1]);
                            for (var i = 0; i < count; i++)
                            {
                                currentJoint.Channels.Add(parts[2 + i]);
                            }
                            _channelIndexCounter += count;
                        }
                    }
                    else if (parts[0] == "}")
                    {
                        if (currentJoint != null)
                            currentJoint = currentJoint.Parent;
                    }
                    else if (parts[0] == "MOTION")
                    {
                        break;
                    }

                    line = reader.ReadLine();
                }

                // MOTION section.
                while (line != null)
                {
                    var cleanLine = line.Trim();
                    if (cleanLine.StartsWith("Frames:"))
                    {
                        data.NumFrames = int.Parse(cleanLine.Split(':')[1].Trim());
                    }
                    else if (cleanLine.StartsWith("Frame Time:"))
                    {
                        data.FrameTime = float.Parse(cleanLine.Split(':')[1].Trim(), CultureInfo.InvariantCulture);
                    }
                    else if (!string.IsNullOrEmpty(cleanLine) && (char.IsDigit(cleanLine[0]) || cleanLine.StartsWith("-")))
                    {
                        var values = cleanLine.Split(new[] { ' ', '\t' }, System.StringSplitOptions.RemoveEmptyEntries);
                        var floats = new float[values.Length];
                        for (var i = 0; i < values.Length; i++)
                        {
                            floats[i] = float.Parse(values[i], CultureInfo.InvariantCulture);
                        }
                        _motionData.Add(floats);
                    }
                    line = reader.ReadLine();
                }
            }

            ProcessMotionData(data, scaleFactor);
            return data;
        }

        private void ProcessMotionData(BVHData data, float scale)
        {
            foreach (var frameValues in _motionData)
            {
                foreach (var joint in data.AllJoints)
                {
                    if (joint.Name == "End Site") continue;

                    var pos = joint.Offset;
                    var dataIndex = joint.ChannelOffsetIndex;

                    float pX = 0, pY = 0, pZ = 0;
                    float rX = 0, rY = 0, rZ = 0;
                    var rotOrder = "";

                    for (var i = 0; i < joint.Channels.Count; i++)
                    {
                        var type = joint.Channels[i];
                        var val = frameValues[dataIndex + i];

                        if (type == "Xposition") pX = val;
                        if (type == "Yposition") pY = val;
                        if (type == "Zposition") pZ = val;

                        if (type == "Xrotation") { rX = val; rotOrder += "X"; }
                        if (type == "Yrotation") { rY = val; rotOrder += "Y"; }
                        if (type == "Zrotation") { rZ = val; rotOrder += "Z"; }
                    }

                    if (joint.HasPos)
                    {
                        pos = new Vector3(pX * scale, pY * scale, pZ * scale);
                    }

                    var qx = Quaternion.AngleAxis(rX, Vector3.right);
                    var qy = Quaternion.AngleAxis(rY, Vector3.up);
                    var qz = Quaternion.AngleAxis(rZ, Vector3.forward);

                    // Compose in the order the channels are listed.
                    var finalRot = Quaternion.identity;
                    foreach (var axis in rotOrder)
                    {
                        if (axis == 'Z') finalRot = finalRot * qz;
                        if (axis == 'Y') finalRot = finalRot * qy;
                        if (axis == 'X') finalRot = finalRot * qx;
                    }

                    joint.PosData.Add(pos);
                    joint.RotData.Add(finalRot);
                }
            }
        }
    }
}
