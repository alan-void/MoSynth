using System.Collections.Generic;
using System.Linq;
using AnimationTools;
using UnityEngine;
using UnityEngine.Rendering;

namespace MotionField
{
/// <summary>
/// Draws the motion field as a 3D UMAP point cloud with the live policy overlaid, so a misbehaving
/// policy can be watched rather than inferred.
/// </summary>
/// <remarks>
/// Under a position-only projection the cloud is a pose manifold and the frame-adjacency edges are
/// velocity. Overlay: yellow is the live pose, cyan its nearest state, magenta the tug target, green
/// the rest of the neighbourhood. Add it beside a <see cref="MotionSynthesisComponent"/> running a
/// <see cref="MotionFieldStage"/> with <c>collectDebugData</c> on. See
/// openwiki/motion-field/pose-manifold-embedding.md.
/// </remarks>
[AddComponentMenu("MotionField/Motion Field Visualizer")]
public class MotionFieldVisualizer : MonoBehaviour
{
    public MotionSynthesisComponent synthesisComponent;

    public enum ColorMode
    {
        /// <summary>One flat colour; leaves the overlay to carry all the signal.</summary>
        Uniform,

        /// <summary>Root speed, red (stationary) to cyan (fastest). Makes dead zones obvious.</summary>
        Speed,

        /// <summary>Position in the database, hue cycling with the frame index.</summary>
        StateIndex
    }

    [Header("Placement")]
    [Tooltip("Largest dimension of the cloud in metres. The raw UMAP extent is fitted into this.")]
    [SerializeField]
    [Min(0.01f)]
    private float size = 5f;

    [Header("Appearance")]
    [Tooltip("Point cube edge length, as a fraction of the cloud size.")]
    [SerializeField]
    [Range(0.0005f, 0.05f)]
    private float pointScale = 0.004f;

    [SerializeField] private ColorMode colorMode = ColorMode.Speed;

    [Tooltip("Optional override. Left empty, a material is built from the MotionField/Points shader.")] [SerializeField]
    private Material materialOverride;

    [Header("What To Draw")] [SerializeField]
    private bool showCloud = true;

    [Tooltip("Links each state to the next frame of the same clip, which under a position-only " +
             "projection is that state's velocity. The line darkens at its start and brightens at " +
             "its end, so the gradient reads as the direction of travel.")]
    [SerializeField]
    private bool showEdges = true;

    [Tooltip("The live pose, its k nearest neighbours, the tug target, the trail, and the live " +
             "pose's own velocity edge.")]
    [SerializeField]
    private bool showOverlay = true;

    [Tooltip("Lines from the live pose to each of its neighbours. Off by default: on a dense cloud " +
             "they hide the markers they are meant to explain. Does not affect the trail or the " +
             "live pose's velocity edge, which are always drawn with the overlay.")]
    [SerializeField]
    private bool showLinks;

    [Tooltip("Frames of history behind the live pose. A stalled policy piles this into one spot.")]
    [SerializeField]
    [Range(0, 600)]
    private int trailLength = 120;

    [Tooltip("How many frames ahead the live pose's velocity edge reaches. 1 is the true one-frame " +
             "velocity and the default. Raise it if the edge is too short to read -- at a large " +
             "pointScale one frame of motion is about the width of the marker it leaves. The same " +
             "count applies to every pose, so a longer edge never means a faster one.")]
    [SerializeField]
    [Range(1, 30)]
    private int velocityLookahead = 1;

    private MotionFieldStage _stage;
    private Material _material;

    private Vector3[] _points; // embedding fitted into the size box, anchor-local

    /// <summary>
    /// State one frame later in the same clip, or -1 at a clip's last frame. Derived from the edge
    /// list rather than assumed to be i+1, so clip boundaries are respected for free.
    /// </summary>
    private int[] _successor;

    private Mesh _cloudMesh;
    private Mesh _edgeMesh;
    private Mesh _overlayTriangles;
    private Mesh _overlayLines;

    private readonly MeshBuffer _overlayTriangleBuffer = new();
    private readonly MeshBuffer _overlayLineBuffer = new();
    private readonly Queue<Vector3> _trail = new();

    private static readonly Color CurrentColor = new(1f, 0.95f, 0.2f);
    private static readonly Color TugColor = new(1f, 0.2f, 0.85f);
    private static readonly Color NearestColor = new(0.25f, 0.8f, 1f);
    private static readonly Color NeighborColor = new(0.2f, 1f, 0.45f);
    private static readonly Color LinkColor = new(1f, 1f, 1f, 1f);

    private void OnEnable()
    {
        if (synthesisComponent == null)
        {
            synthesisComponent = GetComponent<MotionSynthesisComponent>();
        }

        _stage = synthesisComponent.stages?.OfType<MotionFieldStage>().FirstOrDefault();

        if (_stage == null)
        {
            Debug.LogWarning("[MotionField] MotionFieldVisualizer found no MotionFieldStage in the " +
                             "MotionSynthesisComponent stage list. Nothing to draw.", this);
            enabled = false;
            return;
        }

        if (!_stage.collectDebugData)
        {
            Debug.LogWarning("[MotionField] MotionFieldStage.collectDebugData is off, so the " +
                             "visualizer has no data to draw. Enable it on the stage.", this);
            enabled = false;
        }

#if UNITY_EDITOR
        UnityEditor.EditorApplication.update += EditorUpdate;
#endif
    }

    private void OnDisable()
    {
        _trail.Clear();

#if UNITY_EDITOR
        UnityEditor.EditorApplication.update -= EditorUpdate;
#endif
    }

#if UNITY_EDITOR
    private void EditorUpdate()
    {
        // Keeps the overlay drawing while play mode is paused.
        if (UnityEditor.EditorApplication.isPaused)
        {
            LateUpdate();
        }
    }
#endif

    private void OnDestroy()
    {
        ClearResources();
    }

    private void ClearResources()
    {
        if (_cloudMesh != null) Destroy(_cloudMesh);
        if (_edgeMesh != null) Destroy(_edgeMesh);
        if (_overlayTriangles != null) Destroy(_overlayTriangles);
        if (_overlayLines != null) Destroy(_overlayLines);
        if (_material != null && materialOverride == null) Destroy(_material);

        _cloudMesh = null;
        _edgeMesh = null;
        _overlayTriangles = null;
        _overlayLines = null;
        _material = null;

        // Dropped too, so that editing `size` in play mode refits the cloud instead of rebuilding
        // the meshes from the previous fit. `_successor` is size-independent, so it survives.
        _points = null;
    }

    private void LateUpdate()
    {
        // The stage builds this during Init, which runs after this component may first have woken.
        if (_stage is not { HasEmbedding: true }) return;
        if (!EnsureResources()) return;

        var toWorld = Matrix4x4.TRS(transform.position, Quaternion.identity, Vector3.one);

        var parameters = new RenderParams(_material)
        {
            worldBounds = new Bounds(toWorld.GetColumn(3), Vector3.one * (size * 4f)),
            shadowCastingMode = ShadowCastingMode.Off,
            receiveShadows = false
        };

        if (showCloud && _cloudMesh != null) Graphics.RenderMesh(parameters, _cloudMesh, 0, toWorld);
        if (showEdges && _edgeMesh != null) Graphics.RenderMesh(parameters, _edgeMesh, 0, toWorld);

        if (!showOverlay) return;

        BuildOverlay();
        if (_overlayTriangles.vertexCount > 0) Graphics.RenderMesh(parameters, _overlayTriangles, 0, toWorld);

        // Not gated on showLinks: the trail and velocity edge share this buffer; showLinks only
        // decides whether BuildOverlay adds the per-neighbour lines.
        if (_overlayLines.vertexCount > 0) Graphics.RenderMesh(parameters, _overlayLines, 0, toWorld);
    }

    private void OnValidate()
    {
        if (Application.isPlaying)
        {
            ClearResources();
        }
    }

    /// <summary>Build the static meshes once the embedding has arrived. False if it cannot draw.</summary>
    private bool EnsureResources()
    {
        if (_material == null)
        {
            if (materialOverride != null)
            {
                _material = materialOverride;
            }
            else
            {
                var shader = Shader.Find("MotionField/Points");
                if (shader == null)
                {
                    Debug.LogError("[MotionField] Shader 'MotionField/Points' not found. Assign a " +
                                   "material override, or add the shader to Always Included Shaders " +
                                   "for a player build.", this);
                    enabled = false;
                    return false;
                }

                _material = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
            }
        }

        _points ??= FitToBox(_stage.GetEmbedding(), size);
        _successor ??= BuildSuccessors(_stage.GetEmbeddingEdges(), _points.Length);

        _cloudMesh ??= BuildCloudMesh();
        _edgeMesh ??= BuildEdgeMesh();
        _overlayTriangles ??= NewMesh("MotionField Overlay Tris");
        _overlayLines ??= NewMesh("MotionField Overlay Lines");

        return true;
    }

    /// <summary>
    /// Centre the raw UMAP coordinates, which have no meaningful units or origin, and scale them
    /// into a box of the requested size.
    /// </summary>
    private static Vector3[] FitToBox(IReadOnlyList<Vector3> raw, float size)
    {
        var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        for (var i = 0; i < raw.Count; i++)
        {
            min = Vector3.Min(min, raw[i]);
            max = Vector3.Max(max, raw[i]);
        }

        var centre = (min + max) * 0.5f;
        var extent = max - min;
        var longest = Mathf.Max(extent.x, Mathf.Max(extent.y, extent.z));
        var scale = longest > 1e-6f ? size / longest : 1f;

        var fitted = new Vector3[raw.Count];
        for (var i = 0; i < raw.Count; i++) fitted[i] = (raw[i] - centre) * scale;
        return fitted;
    }

    /// <summary>
    /// Invert the flat (from, to) edge pairs into a per-state successor lookup, so the overlay can
    /// ask "where does this state go next" without scanning the edge list every frame.
    /// </summary>
    private static int[] BuildSuccessors(IReadOnlyList<int> edges, int stateCount)
    {
        var successor = new int[stateCount];
        for (var i = 0; i < stateCount; i++) successor[i] = -1;

        if (edges == null) return successor;

        for (var e = 0; e + 1 < edges.Count; e += 2)
        {
            var from = edges[e];
            var to = edges[e + 1];
            if (from >= 0 && from < stateCount && to >= 0 && to < stateCount) successor[from] = to;
        }

        return successor;
    }

    private Mesh BuildCloudMesh()
    {
        var buffer = new MeshBuffer();
        var half = size * pointScale * 0.5f;

        for (var i = 0; i < _points.Length; i++)
        {
            buffer.AddCube(_points[i], half, ColorForState(i));
        }

        var mesh = NewMesh("MotionField Cloud");
        buffer.Fill(mesh, MeshTopology.Triangles);
        return mesh;
    }

    /// <summary>
    /// One line per consecutive-frame transition, darkened at the tail and brightened at the head;
    /// unlike a flat arrow, a gradient never disappears edge-on.
    /// </summary>
    private Mesh BuildEdgeMesh()
    {
        var edges = _stage.GetEmbeddingEdges();
        var mesh = NewMesh("MotionField Edges");
        if (edges == null || edges.Length < 2) return mesh;

        var buffer = new MeshBuffer();
        for (var e = 0; e + 1 < edges.Length; e += 2)
        {
            var from = edges[e];
            var to = edges[e + 1];
            if (!InRange(from) || !InRange(to)) continue;

            var head = ColorForState(to);
            buffer.AddLine(_points[from], _points[to], head * 0.15f, head);
        }

        buffer.Fill(mesh, MeshTopology.Lines);
        return mesh;
    }

    private void BuildOverlay()
    {
        _overlayTriangleBuffer.Clear();
        _overlayLineBuffer.Clear();

        var neighbors = _stage.LastNeighbors;
        var weights = _stage.LastNeighborWeights;
        var chosen = _stage.LastChosenSlot;

        var unit = size * pointScale;
        var haveNeighbors = neighbors is { Length: > 0 } && weights != null &&
                            weights.Length == neighbors.Length;

        if (haveNeighbors)
        {
            // UMAP fitted from a precomputed k-NN graph cannot project new points, so the live pose
            // is placed at its neighbours' weighted average — the field's own interpolation.
            var current = Vector3.zero;
            var total = 0f;
            for (var i = 0; i < neighbors.Length; i++)
            {
                if (!InRange(neighbors[i])) continue;
                current += _points[neighbors[i]] * weights[i];
                total += weights[i];
            }

            if (total > 1e-6f)
            {
                current /= total;
                PushTrail(current);

                var heaviest = weights.Max();
                for (var i = 0; i < neighbors.Length; i++)
                {
                    if (!InRange(neighbors[i])) continue;

                    var point = _points[neighbors[i]];
                    var relative = heaviest > 1e-6f ? weights[i] / heaviest : 0f;

                    // Slot 0 is the nearest state (get_knn sorts nearest first); the tug is chosen
                    // on value, so usually differs. When they coincide the tug colour wins, at the
                    // nearest marker's larger size.
                    var isTug = i == chosen;
                    var isNearest = i == 0;

                    var colour = isTug ? TugColor
                        : isNearest ? NearestColor
                        : NeighborColor * Mathf.Lerp(0.25f, 1f, relative);
                    colour.a = 1f;

                    var scale = isNearest ? 2.2f : Mathf.Lerp(0.9f, 1.8f, relative);

                    _overlayTriangleBuffer.AddCube(point, unit * scale, colour);
                    if (showLinks) _overlayLineBuffer.AddLine(current, point, LinkColor * 0.1f, colour);
                }

                _overlayTriangleBuffer.AddCube(current, unit * 3f, CurrentColor);
                BuildVelocityEdge(neighbors, weights, current, unit);
            }
        }

        BuildTrail();

        _overlayTriangleBuffer.Fill(_overlayTriangles, MeshTopology.Triangles);
        _overlayLineBuffer.Fill(_overlayLines, MeshTopology.Lines);
    }

    /// <summary>
    /// The live pose's velocity, as an edge to where it is going.
    /// </summary>
    /// <remarks>
    /// The live pose has no successor of its own, so the head is its neighbours' successors averaged
    /// under the weights that placed it, renormalised by the surviving weight as <c>current</c> is.
    /// See openwiki/motion-field/pose-manifold-embedding.md.
    /// </remarks>
    private void BuildVelocityEdge(IReadOnlyList<int> neighbors, IReadOnlyList<float> weights,
                                   Vector3 current, float unit)
    {
        var ahead = Vector3.zero;
        var total = 0f;

        for (var i = 0; i < neighbors.Count; i++)
        {
            if (!InRange(neighbors[i])) continue;

            var next = Advance(neighbors[i], velocityLookahead);
            if (next < 0) continue; // clip ends on this neighbour's own frame

            ahead += _points[next] * weights[i];
            total += weights[i];
        }

        if (total <= 1e-6f) return;

        ahead /= total;
        _overlayLineBuffer.AddLine(current, ahead, CurrentColor * 0.15f, CurrentColor);

        // A head marker as well, since a line seen end-on shows no direction.
        _overlayTriangleBuffer.AddCube(ahead, unit * 1.2f, CurrentColor);
    }

    /// <summary>
    /// Walk <paramref name="steps"/> frames along a state's clip, stopping at the clip's last
    /// frame rather than running off it. -1 only when the start state has no successor at all.
    /// </summary>
    private int Advance(int state, int steps)
    {
        var at = _successor[state];
        for (var i = 1; i < steps && at >= 0; i++)
        {
            var next = _successor[at];
            if (next < 0) break; // end of the clip: stop here rather than losing the edge entirely
            at = next;
        }

        return at;
    }

    private void PushTrail(Vector3 point)
    {
        if (trailLength <= 0)
        {
            _trail.Clear();
            return;
        }

        _trail.Enqueue(point);
        while (_trail.Count > trailLength) _trail.Dequeue();
    }

    /// <summary>Fade the trail from dark (oldest) to bright (newest) so recent motion stands out.</summary>
    private void BuildTrail()
    {
        if (_trail.Count < 2) return;

        var index = 0;
        var last = _trail.Count - 1;
        var previous = Vector3.zero;

        foreach (var point in _trail)
        {
            if (index > 0)
            {
                var from = (index - 1) / (float)last;
                var to = index / (float)last;
                _overlayLineBuffer.AddLine(previous, point, CurrentColor * from, CurrentColor * to);
            }

            previous = point;
            index++;
        }
    }

    private bool InRange(int state) => state >= 0 && state < _points.Length;

    private Color ColorForState(int state)
    {
        switch (colorMode)
        {
            case ColorMode.Speed:
            {
                var speeds = _stage.GetStateSpeeds();
                if (speeds == null || state >= speeds.Length) return Color.gray;
                // 1.5 m/s covers a brisk walk; anything above saturates.
                var t = Mathf.Clamp01(speeds[state] / 1.5f);
                return Color.Lerp(new Color(1f, 0.15f, 0.1f), new Color(0.2f, 0.9f, 1f), t);
            }
            case ColorMode.StateIndex:
            {
                var t = _points.Length > 1 ? state / (float)(_points.Length - 1) : 0f;
                return Color.HSVToRGB(t * 0.85f, 0.65f, 0.95f);
            }
            default:
                return new Color(0.35f, 0.35f, 0.42f);
        }
    }

    private static Mesh NewMesh(string name) => new()
    {
        name = name,
        indexFormat = IndexFormat.UInt32, // the cloud alone is ~97k indices
        hideFlags = HideFlags.HideAndDontSave
    };

    /// <summary>
    /// Accumulates coloured geometry into reusable lists. One mesh per topology, because a cloud of
    /// cubes and a set of lines cannot share a submesh.
    /// </summary>
    private sealed class MeshBuffer
    {
        // Corner order the triangle table below is written against. Explicit rather than derived
        // from the index bits, so the two cannot drift apart into folded faces.
        private static readonly Vector3[] CubeCorners =
        {
            new(-1f, -1f, -1f), new(1f, -1f, -1f), new(1f, 1f, -1f), new(-1f, 1f, -1f),
            new(-1f, -1f, 1f), new(1f, -1f, 1f), new(1f, 1f, 1f), new(-1f, 1f, 1f)
        };

        private static readonly int[] CubeTriangles =
        {
            0, 2, 1, 0, 3, 2, // -z
            4, 5, 6, 4, 6, 7, // +z
            0, 4, 7, 0, 7, 3, // -x
            1, 2, 6, 1, 6, 5, // +x
            0, 1, 5, 0, 5, 4, // -y
            3, 7, 6, 3, 6, 2 // +y
        };

        private readonly List<Vector3> _vertices = new();
        private readonly List<Color> _colors = new();
        private readonly List<int> _indices = new();

        public void Clear()
        {
            _vertices.Clear();
            _colors.Clear();
            _indices.Clear();
        }

        public void AddCube(Vector3 centre, float halfExtent, Color color)
        {
            var origin = _vertices.Count;

            for (var corner = 0; corner < CubeCorners.Length; corner++)
            {
                _vertices.Add(centre + CubeCorners[corner] * halfExtent);
                _colors.Add(color);
            }

            for (var i = 0; i < CubeTriangles.Length; i++) _indices.Add(origin + CubeTriangles[i]);
        }

        public void AddLine(Vector3 from, Vector3 to, Color fromColor, Color toColor)
        {
            var origin = _vertices.Count;
            _vertices.Add(from);
            _colors.Add(fromColor);
            _vertices.Add(to);
            _colors.Add(toColor);
            _indices.Add(origin);
            _indices.Add(origin + 1);
        }

        public void Fill(Mesh mesh, MeshTopology topology)
        {
            mesh.Clear();
            if (_vertices.Count == 0) return;

            mesh.SetVertices(_vertices);
            mesh.SetColors(_colors);
            mesh.SetIndices(_indices, topology, 0, calculateBounds: true);
        }
    }
}
}