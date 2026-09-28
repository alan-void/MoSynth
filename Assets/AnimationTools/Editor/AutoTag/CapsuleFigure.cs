using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace AnimationTools.Editor
{
/// <summary>
/// A skeleton drawn as solid capsules inside a <see cref="PreviewRenderUtility"/>: one per
/// parent-to-child bone, a sphere for the head with a nose marking its front, left bones blue and
/// right bones red.
/// </summary>
/// <remarks>
/// Built for a video model to read, so it favours unambiguous shapes and side colours over likeness.
/// </remarks>
public sealed class CapsuleFigure : IDisposable
{
    public const string ShaderName = "Hidden/MoSynth/AutoTagShaded";

    private const float CapsuleRadiusFraction = 0.035f;
    private const float HeadRadiusFraction = 0.075f;
    private const float NoseRadiusFraction = 0.35f;

    // Thickens the centre line so the trunk reads as a body rather than a wire between the limbs.
    private const float TorsoRadiusScale = 1.4f;
    private const float FallbackHeight = 1.7f;

    private static readonly Color LeftColour = new(0.2f, 0.45f, 1f);
    private static readonly Color RightColour = new(1f, 0.25f, 0.2f);
    private static readonly Color CentreColour = new(0.8f, 0.8f, 0.8f);
    private static readonly Color NoseColour = new(1f, 0.85f, 0.1f);

    private readonly struct Segment
    {
        public readonly int Parent;
        public readonly int Child;
        public readonly Transform Transform;

        public Segment(int parent, int child, Transform transform)
        {
            Parent = parent;
            Child = child;
            Transform = transform;
        }
    }

    private readonly List<Segment> _segments = new();
    private readonly List<Mesh> _meshes = new();
    private readonly List<GameObject> _objects = new();
    private readonly Dictionary<BoneNameConventions.BodySide, Material> _materials = new();
    private readonly List<Material> _extraMaterials = new();

    private readonly int _headBone = -1;
    private readonly int _headChild = -1;
    private readonly Transform _head;
    private readonly Transform _nose;
    private readonly float3 _headForwardLocal;
    private readonly float _headRadius;

    /// <summary>Standing height of the rest pose, in metres; what every size here is scaled by.</summary>
    public float Height { get; }

    /// <summary>Creates the figure's objects inside <paramref name="preview"/>'s scene.</summary>
    public CapsuleFigure(PreviewRenderUtility preview, Skeleton skeleton)
    {
        if (preview == null) throw new ArgumentNullException(nameof(preview));
        if (skeleton == null || !skeleton.IsSet) throw new ArgumentException("Skeleton is not set.", nameof(skeleton));

        var data = skeleton.GetSkeletonData();
        var restPositions = RestPositions(data);
        Height = StandingHeight(restPositions);

        var capsuleRadius = Height * CapsuleRadiusFraction;
        FindHead(skeleton, data, restPositions, out _headBone, out _headChild);

        for (var i = 1; i < data.BoneCount; i++)
        {
            var parent = data.ParentIndices[i];
            var length = math.distance(restPositions[parent], restPositions[i]);
            if (length < 1e-4f) continue;

            var side = BoneNameConventions.SideOf(skeleton.GetBone(i).Name);
            var radius = side == BoneNameConventions.BodySide.Centre ? capsuleRadius * TorsoRadiusScale : capsuleRadius;
            var mesh = Track(BuildCapsuleMesh(length, radius));
            var go = CreateRenderer(preview, $"Bone {skeleton.GetBone(i).Name}", mesh, MaterialFor(side));
            _segments.Add(new Segment(parent, i, go.transform));
        }

        if (_headBone >= 0)
        {
            _headRadius = Height * HeadRadiusFraction;
            _headForwardLocal = skeleton.RestLocalAxis(_headBone, math.forward());

            var headMesh = Track(BuildCapsuleMesh(0f, _headRadius));
            _head = CreateRenderer(preview, "Head", headMesh, MaterialFor(BoneNameConventions.BodySide.Centre)).transform;

            // Without a face a sphere reads the same from every side, so forward and backward
            // walking would look alike.
            var noseMaterial = CreateMaterial(NoseColour);
            _extraMaterials.Add(noseMaterial);
            var noseMesh = Track(BuildCapsuleMesh(0f, _headRadius * NoseRadiusFraction));
            _nose = CreateRenderer(preview, "Nose", noseMesh, noseMaterial).transform;
        }
    }

    /// <summary>Places every capsule on a character-space pose, one entry per bone.</summary>
    public void Pose(NativeArray<float3> positions, NativeArray<quaternion> rotations)
    {
        foreach (var segment in _segments)
        {
            var from = positions[segment.Parent];
            var parentRotation = rotations[segment.Parent];

            // Oriented within the parent's frame so a bone pointing straight up or down cannot
            // make the look rotation degenerate.
            var localOffset = math.rotate(math.inverse(parentRotation), positions[segment.Child] - from);
            var alongBone = Quaternion.FromToRotation(Vector3.forward, localOffset);

            segment.Transform.SetPositionAndRotation(from, (Quaternion)parentRotation * alongBone);
        }

        if (_head == null) return;

        var centre = _headChild >= 0
            ? math.lerp(positions[_headBone], positions[_headChild], 0.5f)
            : positions[_headBone];
        _head.SetPositionAndRotation(centre, rotations[_headBone]);

        var forward = math.rotate(rotations[_headBone], _headForwardLocal);
        _nose.position = centre + forward * _headRadius;
    }

    private Mesh Track(Mesh mesh)
    {
        _meshes.Add(mesh);
        return mesh;
    }

    private Material MaterialFor(BoneNameConventions.BodySide side)
    {
        if (_materials.TryGetValue(side, out var material)) return material;

        var colour = side switch
        {
            BoneNameConventions.BodySide.Left => LeftColour,
            BoneNameConventions.BodySide.Right => RightColour,
            _ => CentreColour
        };

        material = CreateMaterial(colour);
        _materials[side] = material;
        return material;
    }

    /// <summary>A material on the figure's shader, or on a flat-colour fallback if it is missing.</summary>
    public static Material CreateMaterial(Color colour)
    {
        var shader = Shader.Find(ShaderName);
        if (shader == null)
        {
            Debug.LogWarning($"[AutoTag] Shader \"{ShaderName}\" not found; rendering flat colours.");
            shader = Shader.Find("Hidden/Internal-Colored");
        }

        var material = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
        material.SetColor("_Color", colour);
        return material;
    }

    private GameObject CreateRenderer(PreviewRenderUtility preview, string name, Mesh mesh, Material material)
    {
        var go = new GameObject(name) { hideFlags = HideFlags.HideAndDontSave };
        go.AddComponent<MeshFilter>().sharedMesh = mesh;

        var renderer = go.AddComponent<MeshRenderer>();
        renderer.sharedMaterial = material;
        renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        renderer.receiveShadows = false;

        preview.AddSingleGO(go);
        _objects.Add(go);
        return go;
    }

    private static float3[] RestPositions(SkeletonData data)
    {
        var positions = new float3[data.BoneCount];
        var rotations = new quaternion[data.BoneCount];
        for (var i = 0; i < data.BoneCount; i++)
        {
            var parent = i == 0 ? -1 : data.ParentIndices[i];
            if (parent < 0)
            {
                positions[i] = float3.zero;
                rotations[i] = data.RestLocalRotations[i];
                continue;
            }

            rotations[i] = math.mul(rotations[parent], data.RestLocalRotations[i]);
            positions[i] = positions[parent] + math.rotate(rotations[parent], data.RestLocalPositions[i]);
        }

        return positions;
    }

    private static float StandingHeight(float3[] restPositions)
    {
        var min = float.MaxValue;
        var max = float.MinValue;
        foreach (var position in restPositions)
        {
            min = math.min(min, position.y);
            max = math.max(max, position.y);
        }

        var height = max - min;
        return height > 0.1f ? height : FallbackHeight;
    }

    /// <summary>
    /// The shallowest bone named "head", else the highest leaf; and that bone's highest child, which
    /// the head sphere is centred towards.
    /// </summary>
    private static void FindHead(Skeleton skeleton, SkeletonData data, float3[] restPositions,
        out int head, out int headChild)
    {
        head = -1;
        var bestDepth = int.MaxValue;
        for (var i = 0; i < data.BoneCount; i++)
        {
            if (skeleton.GetBone(i).Name.IndexOf("head", StringComparison.OrdinalIgnoreCase) < 0) continue;

            var depth = Depth(data, i);
            if (depth >= bestDepth) continue;

            bestDepth = depth;
            head = i;
        }

        if (head < 0)
        {
            var hasChild = new bool[data.BoneCount];
            for (var i = 1; i < data.BoneCount; i++) hasChild[data.ParentIndices[i]] = true;

            for (var i = 0; i < data.BoneCount; i++)
            {
                if (hasChild[i]) continue;
                if (head < 0 || restPositions[i].y > restPositions[head].y) head = i;
            }
        }

        headChild = -1;
        if (head < 0) return;

        for (var i = head + 1; i < data.BoneCount; i++)
        {
            if (data.ParentIndices[i] != head) continue;
            if (headChild < 0 || restPositions[i].y > restPositions[headChild].y) headChild = i;
        }
    }

    private static int Depth(SkeletonData data, int bone)
    {
        var depth = 0;
        for (var i = bone; i > 0; i = data.ParentIndices[i]) depth++;
        return depth;
    }

    /// <summary>
    /// A capsule whose axis runs along +Z from 0 to <paramref name="length"/>, hemispheres centred
    /// on both ends; a length of zero makes a sphere.
    /// </summary>
    public static Mesh BuildCapsuleMesh(float length, float radius, int segments = 16, int hemisphereRings = 6)
    {
        var vertices = new List<Vector3>();
        var normals = new List<Vector3>();
        var indices = new List<int>();

        var ringCount = 2 * (hemisphereRings + 1);
        for (var ring = 0; ring < ringCount; ring++)
        {
            // The first half of the rings sweeps the back hemisphere from its pole to the equator,
            // the second half the front one from the equator to its pole.
            var front = ring > hemisphereRings;
            var step = front ? ring - hemisphereRings - 1 : ring;
            var latitude = (front ? step : step - hemisphereRings) * (0.5f * Mathf.PI / hemisphereRings);
            var centreZ = front ? length : 0f;

            var ringRadius = Mathf.Cos(latitude);
            var axial = Mathf.Sin(latitude);

            for (var s = 0; s <= segments; s++)
            {
                var angle = s * 2f * Mathf.PI / segments;
                var normal = new Vector3(ringRadius * Mathf.Cos(angle), ringRadius * Mathf.Sin(angle), axial);
                normals.Add(normal);
                vertices.Add(normal * radius + new Vector3(0f, 0f, centreZ));
            }
        }

        var stride = segments + 1;
        for (var ring = 0; ring < ringCount - 1; ring++)
        {
            for (var s = 0; s < segments; s++)
            {
                var a = ring * stride + s;
                var b = a + stride;

                indices.Add(a);
                indices.Add(b);
                indices.Add(a + 1);

                indices.Add(a + 1);
                indices.Add(b);
                indices.Add(b + 1);
            }
        }

        var mesh = new Mesh { hideFlags = HideFlags.HideAndDontSave };
        mesh.SetVertices(vertices);
        mesh.SetNormals(normals);
        mesh.SetTriangles(indices, 0);
        mesh.RecalculateBounds();
        return mesh;
    }

    /// <summary>Destroys the figure's objects, meshes and materials; the preview itself is left alone.</summary>
    public void Dispose()
    {
        foreach (var go in _objects)
        {
            if (go != null) Object.DestroyImmediate(go);
        }

        foreach (var mesh in _meshes)
        {
            if (mesh != null) Object.DestroyImmediate(mesh);
        }

        foreach (var material in _materials.Values.Concat(_extraMaterials))
        {
            if (material != null) Object.DestroyImmediate(material);
        }

        _objects.Clear();
        _meshes.Clear();
        _materials.Clear();
        _extraMaterials.Clear();
        _segments.Clear();
    }
}
}
