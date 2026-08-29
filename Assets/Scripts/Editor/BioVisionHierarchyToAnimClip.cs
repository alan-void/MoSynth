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
    /// <remarks>
    /// Editor-only, and in an Editor folder for that reason: it reaches for <c>AssetDatabase</c>
    /// and <c>Selection</c>, which do not exist in a player.
    /// </remarks>
    public static class BioVisionHierarchyToAnimClip
    {
        // --- CONFIGURATION ---
        // Set this to 0.01f to convert Centimeters (BVH standard) to Meters (Unity standard).
        // If your BVH is already in meters, set this to 1.0f.
        private const float UnitScale = 0.01f; 
        // ---------------------

        [MenuItem("Assets/Convert BVH to AnimationClip")]
        public static void ConvertSelectedBvh()
        {
            foreach (Object selectedObject in Selection.objects)
            {
                string path = AssetDatabase.GetAssetPath(selectedObject);

                if (string.IsNullOrEmpty(path) || (!path.EndsWith(".bvh") && !path.EndsWith(".txt")))
                {
                    Debug.LogWarning($"Skipping non-bvh file: {path}");
                    continue;
                }

                BVHParser parser = new BVHParser();
                try
                {
                    Debug.Log($"Parsing BVH: {path} with Scale {UnitScale}...");
                    string fileContent = File.ReadAllText(path);
                    BVHData data = parser.Parse(fileContent, UnitScale);

                    string clipName = Path.GetFileNameWithoutExtension(path);
                    AnimationClip clip = CreateAnimationClip(data, clipName);
                    
                    string newPath = Path.Combine(Path.GetDirectoryName(path), clipName + ".anim");
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
            AnimationClip clip = new AnimationClip();
            clip.name = clipName;
            clip.frameRate = 1f / data.FrameTime;
            clip.legacy = false; // Valid for Animator Controller

            int numFrames = data.NumFrames;

            foreach (var joint in data.AllJoints)
            {
                AnimationCurve curvePosX = new AnimationCurve();
                AnimationCurve curvePosY = new AnimationCurve();
                AnimationCurve curvePosZ = new AnimationCurve();

                AnimationCurve curveRotX = new AnimationCurve();
                AnimationCurve curveRotY = new AnimationCurve();
                AnimationCurve curveRotZ = new AnimationCurve();
                AnimationCurve curveRotW = new AnimationCurve();

                string relativePath = GetRelativePath(joint);

                for (int frame = 0; frame < numFrames; frame++)
                {
                    float time = frame * data.FrameTime;
                    
                    Vector3 pos = joint.GetPosition(frame);
                    Quaternion rot = joint.GetRotation(frame);

                    // Coordinate Conversion (Right-Handed BVH -> Left-Handed Unity)
                    // 1. Position: Flip X. (Note: Scaling is already applied in Parser)
                    Vector3 unityPos = new Vector3(-pos.x, pos.y, pos.z);

                    // 2. Rotation: Flip X and W to mirror orientation
                    Quaternion unityRot = new Quaternion(-rot.x, rot.y, rot.z, -rot.w);
                    
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
            string path = joint.Name;
            var current = joint.Parent;
            while (current != null)
            {
                path = current.Name + "/" + path;
                current = current.Parent;
            }
            return path;
        }
    }

    // ==========================================
    // DATA STRUCTURES
    // ==========================================

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

    // ==========================================
    // PARSER
    // ==========================================

    public class BVHParser
    {
        private int _channelIndexCounter = 0;
        private List<float[]> _motionData = new List<float[]>();

        public BVHData Parse(string bvhText, float scaleFactor)
        {
            BVHData data = new BVHData();
            using (StringReader reader = new StringReader(bvhText))
            {
                string line = reader.ReadLine();
                BVHJoint currentJoint = null;
                
                // --- HIERARCHY ---
                while (line != null)
                {
                    string cleanLine = line.Trim();
                    string[] parts = cleanLine.Split(new char[] { ' ', '\t' }, System.StringSplitOptions.RemoveEmptyEntries);

                    if (parts.Length == 0) { line = reader.ReadLine(); continue; }

                    if (parts[0] == "HIERARCHY") { }
                    else if (parts[0] == "ROOT" || parts[0] == "JOINT")
                    {
                        BVHJoint joint = new BVHJoint();
                        joint.Name = parts[1];
                        joint.Parent = currentJoint;
                        data.AllJoints.Add(joint);

                        if (currentJoint == null) data.Root = joint;
                        currentJoint = joint;
                    }
                    else if (parts[0] == "End") 
                    {
                        BVHJoint endSite = new BVHJoint();
                        endSite.Name = "End Site"; 
                        endSite.Parent = currentJoint;
                        currentJoint = endSite;
                    }
                    else if (parts[0] == "OFFSET")
                    {
                        if (currentJoint != null)
                        {
                            // Apply Scale to Offset immediately
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
                            int count = int.Parse(parts[1]);
                            for (int i = 0; i < count; i++)
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

                // --- MOTION ---
                while (line != null)
                {
                    string cleanLine = line.Trim();
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
                        string[] values = cleanLine.Split(new char[] { ' ', '\t' }, System.StringSplitOptions.RemoveEmptyEntries);
                        float[] floats = new float[values.Length];
                        for(int i=0; i<values.Length; i++)
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
            for (int f = 0; f < _motionData.Count; f++)
            {
                float[] frameValues = _motionData[f];

                foreach (var joint in data.AllJoints)
                {
                    if (joint.Name == "End Site") continue;

                    Vector3 pos = joint.Offset; 
                    Quaternion finalRot = Quaternion.identity;
                    
                    int dataIndex = joint.ChannelOffsetIndex;
                    
                    float pX=0, pY=0, pZ=0;
                    float rX=0, rY=0, rZ=0;
                    string rotOrder = ""; 

                    for (int i = 0; i < joint.Channels.Count; i++)
                    {
                        string type = joint.Channels[i];
                        float val = frameValues[dataIndex + i];

                        if (type == "Xposition") pX = val;
                        if (type == "Yposition") pY = val;
                        if (type == "Zposition") pZ = val;
                        
                        if (type == "Xrotation") { rX = val; rotOrder += "X"; }
                        if (type == "Yrotation") { rY = val; rotOrder += "Y"; }
                        if (type == "Zrotation") { rZ = val; rotOrder += "Z"; }
                    }

                    if (joint.HasPos)
                    {
                        // Apply Scale to Motion Positions
                        pos = new Vector3(pX * scale, pY * scale, pZ * scale);
                    }

                    Quaternion qx = Quaternion.AngleAxis(rX, Vector3.right);
                    Quaternion qy = Quaternion.AngleAxis(rY, Vector3.up);
                    Quaternion qz = Quaternion.AngleAxis(rZ, Vector3.forward);
                    
                    finalRot = Quaternion.identity;
                    foreach(char axis in rotOrder)
                    {
                        if(axis == 'Z') finalRot = finalRot * qz; 
                        if(axis == 'Y') finalRot = finalRot * qy;
                        if(axis == 'X') finalRot = finalRot * qx;
                    }

                    joint.PosData.Add(pos);
                    joint.RotData.Add(finalRot);
                }
            }
        }
    }
}