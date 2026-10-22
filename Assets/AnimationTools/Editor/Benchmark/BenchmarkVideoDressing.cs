using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Splines;
using Object = UnityEngine.Object;

namespace AnimationTools.Editor
{
/// <summary>
/// What a benchmark video draws around the character while it records: a grid ground, a shadowing
/// light, the path, the steering target, a fading root trail and footprints.
/// </summary>
/// <remarks>
/// Nothing here has a collider and nothing writes to the character, so the simulation and the
/// metrics are the same with or without it. The scene lighting it overrides is restored on
/// <see cref="Dispose"/>.
/// </remarks>
internal sealed class BenchmarkVideoDressing
{
    public static readonly Color BackgroundColor = new(0.80f, 0.84f, 0.88f);

    private const float GroundSize = 400f;
    private const int GridTexturePixels = 256;

    /// <summary>Lift above the ground for flat markers, so they never z-fight it.</summary>
    private const float MarkerLift = 0.004f;

    private const int PathSamples = 400;
    private const float PathLineWidth = 0.05f;
    private const float TargetRadius = 0.11f;
    private const float TargetLineWidth = 0.015f;
    private const int TrailCapacity = 150;
    private const float TrailWidth = 0.035f;
    private const int FootprintCapacity = 48;
    private const float FootprintLifetime = 6f;
    private static readonly Vector3 FootprintScale = new(0.05f, 1f, 0.11f);

    /// <summary>Ticks between re-estimates of each toe's contact height, a sort of the run's history.</summary>
    private const int ContactHeightRefreshTicks = 15;

    private static readonly Color PathColor = new(0.95f, 0.70f, 0.10f);
    private static readonly Color TargetColor = new(0.85f, 0.15f, 0.55f);
    private static readonly Color TrailColor = new(0.15f, 0.45f, 0.85f);
    private static readonly Color[] FootprintColors = { new(0.20f, 0.35f, 0.80f), new(0.85f, 0.35f, 0.15f) };

    private static readonly int ColorId = Shader.PropertyToID("_Color");

    private readonly GameObject _root;
    private readonly List<Object> _owned = new();
    private readonly SceneLightingState _savedLighting;
    private readonly List<Light> _disabledLights = new();

    private readonly Transform _ground;
    private readonly LineRenderer _pathLine;
    private readonly LineRenderer _targetLine;
    private readonly Transform _targetMarker;
    private readonly LineRenderer _trailLine;
    private readonly Vector3[] _trailRing = new Vector3[TrailCapacity];
    private readonly Vector3[] _trailOrdered = new Vector3[TrailCapacity];
    private readonly Vector3[] _targetPoints = new Vector3[2];
    private readonly Footprint[] _footprints = new Footprint[FootprintCapacity];
    private readonly MaterialPropertyBlock _block = new();

    private int _trailStart;
    private int _trailCount;
    private int _nextFootprint;
    private float _groundHeight;

    private MotionSynthesisComponent _synthesizer;
    private ISteeringTargetSource _steering;
    private GroundContactSettings _contactSettings;
    private ToeContact[] _toes = Array.Empty<ToeContact>();

    public BenchmarkVideoDressing(Transform parent)
    {
        _root = new GameObject("Dressing") { hideFlags = HideFlags.HideAndDontSave };
        _root.transform.SetParent(parent, false);

        _savedLighting = new SceneLightingState();
        ApplyLighting();
        CreateSunLight();

        var unlit = Shader.Find("Sprites/Default");

        _ground = CreateGround();
        _pathLine = CreateGroundLine("Path", unlit, PathColor, PathLineWidth);
        _targetLine = CreateGroundLine("TargetLine", unlit, TargetColor, TargetLineWidth);
        _targetMarker = CreateFlatMarker("Target", Disc(24), unlit, TargetColor).transform;
        _targetMarker.localScale = new Vector3(TargetRadius, 1f, TargetRadius);

        _trailLine = CreateGroundLine("Trail", unlit, Color.white, TrailWidth);
        _trailLine.colorGradient = FadingGradient(TrailColor);

        var footprintMesh = Disc(16);
        for (var i = 0; i < FootprintCapacity; i++)
        {
            var renderer = CreateFlatMarker("Footprint", footprintMesh, unlit, Color.white);
            renderer.gameObject.SetActive(false);
            _footprints[i] = new Footprint { Renderer = renderer };
        }
    }

    /// <summary>World height the path, and so the ground, sits at.</summary>
    public float GroundHeight => _groundHeight;

    /// <summary>Lays the path on the ground and moves the ground under it.</summary>
    /// <param name="center">World centre of the path's ground-plane bounds.</param>
    /// <param name="radius">Ground-plane radius around <paramref name="center"/> enclosing the path.</param>
    public void SetPath(SplineContainer spline, out Vector3 center, out float radius)
    {
        center = Vector3.zero;
        radius = 1f;
        _pathLine.positionCount = 0;
        if (spline == null || spline.Spline == null) return;

        // t = 1 is sampled too: it ends an open path, and repeats point 0 to close a closed one.
        var count = PathSamples + 1;
        var points = new Vector3[count];
        var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        for (var i = 0; i < count; i++)
        {
            var t = (float)i / PathSamples;
            points[i] = spline.transform.TransformPoint((Vector3)spline.EvaluatePosition(t));
            min = Vector3.Min(min, points[i]);
            max = Vector3.Max(max, points[i]);
        }

        _groundHeight = points[0].y;
        for (var i = 0; i < count; i++) points[i].y = _groundHeight + MarkerLift;

        _pathLine.positionCount = count;
        _pathLine.SetPositions(points);

        center = new Vector3(0.5f * (min.x + max.x), _groundHeight, 0.5f * (min.z + max.z));
        radius = 0.5f * new Vector2(max.x - min.x, max.z - min.z).magnitude;
        _ground.position = new Vector3(center.x, _groundHeight, center.z);
    }

    /// <summary>Distance fog matched to how far the camera stands, so the ground fades into the background.</summary>
    public void SetFogRange(float start, float end)
    {
        RenderSettings.fogStartDistance = start;
        RenderSettings.fogEndDistance = end;
    }

    /// <summary>Starts drawing one run: clears the previous run's trail and footprints.</summary>
    public void BeginRun(MotionSynthesisComponent synthesizer, ISteeringTargetSource steering,
        GroundContactSettings contactSettings)
    {
        _synthesizer = synthesizer;
        _steering = steering;
        _contactSettings = contactSettings;
        _trailStart = 0;
        _trailCount = 0;
        _trailLine.positionCount = 0;
        _targetLine.positionCount = 0;
        _targetMarker.gameObject.SetActive(false);

        for (var i = 0; i < FootprintCapacity; i++)
        {
            _footprints[i].Renderer.gameObject.SetActive(false);
            _footprints[i].Age = float.PositiveInfinity;
        }
        _nextFootprint = 0;

        var contacts = synthesizer.ContactHandles;
        _toes = new ToeContact[contacts.Count];
        for (var slot = 0; slot < _toes.Length; slot++) _toes[slot] = new ToeContact(contacts.GetBoneIndex(slot));
    }

    /// <summary>Stops drawing the run; the next <see cref="BeginRun"/> starts clean.</summary>
    public void EndRun()
    {
        _synthesizer = null;
        _steering = null;
    }

    /// <summary>Advances every marker one synthesis tick. Call after the pose has been written.</summary>
    public void Tick(float deltaTime)
    {
        if (_synthesizer == null) return;

        var root = _synthesizer.transform.position;
        var rootOnGround = new Vector3(root.x, _groundHeight + MarkerLift * 2f, root.z);

        UpdateTrail(rootOnGround);
        UpdateTarget(rootOnGround);
        UpdateFootprints(deltaTime);
    }

    private void UpdateTrail(Vector3 rootOnGround)
    {
        var slot = (_trailStart + _trailCount) % TrailCapacity;
        _trailRing[slot] = rootOnGround;
        if (_trailCount < TrailCapacity) _trailCount++;
        else _trailStart = (_trailStart + 1) % TrailCapacity;

        for (var i = 0; i < _trailCount; i++) _trailOrdered[i] = _trailRing[(_trailStart + i) % TrailCapacity];
        _trailLine.positionCount = _trailCount;
        _trailLine.SetPositions(_trailOrdered);
    }

    private void UpdateTarget(Vector3 rootOnGround)
    {
        if (_steering == null || !_steering.TryGetSteeringTarget(out var target))
        {
            _targetMarker.gameObject.SetActive(false);
            _targetLine.positionCount = 0;
            return;
        }

        var onGround = new Vector3(target.x, _groundHeight + MarkerLift * 3f, target.z);
        _targetMarker.gameObject.SetActive(true);
        _targetMarker.position = onGround;

        _targetPoints[0] = rootOnGround;
        _targetPoints[1] = onGround;
        _targetLine.positionCount = 2;
        _targetLine.SetPositions(_targetPoints);
    }

    /// <summary>
    /// Drops a footprint where a toe lands, by <see cref="GroundContactSettings"/>' geometric rule rather than
    /// the method's own flags, and fades the old ones.
    /// </summary>
    private void UpdateFootprints(float deltaTime)
    {
        var transforms = _synthesizer.SkeletonTransforms;
        var facing = _synthesizer.transform.forward;
        var yaw = math.degrees(math.atan2(facing.x, facing.z));

        for (var slot = 0; slot < _toes.Length; slot++)
        {
            var toe = _toes[slot];
            if (toe.Sample(transforms[toe.BoneIndex].position, deltaTime, _contactSettings, out var landedAt))
                PlaceFootprint(landedAt, yaw, FootprintColors[slot % FootprintColors.Length]);
        }

        for (var i = 0; i < FootprintCapacity; i++)
        {
            ref var footprint = ref _footprints[i];
            if (!footprint.Renderer.gameObject.activeSelf) continue;

            footprint.Age += deltaTime;
            if (footprint.Age >= FootprintLifetime)
            {
                footprint.Renderer.gameObject.SetActive(false);
                continue;
            }

            var color = footprint.Color;
            color.a = 0.85f * (1f - footprint.Age / FootprintLifetime);
            _block.SetColor(ColorId, color);
            footprint.Renderer.SetPropertyBlock(_block);
        }
    }

    private void PlaceFootprint(Vector3 position, float yawDegrees, Color color)
    {
        ref var footprint = ref _footprints[_nextFootprint];
        _nextFootprint = (_nextFootprint + 1) % FootprintCapacity;

        footprint.Age = 0f;
        footprint.Color = color;
        var transform = footprint.Renderer.transform;
        transform.SetPositionAndRotation(new Vector3(position.x, _groundHeight + MarkerLift * 1.5f, position.z),
            Quaternion.Euler(0f, yawDegrees, 0f));
        footprint.Renderer.gameObject.SetActive(true);
    }

    private void ApplyLighting()
    {
        RenderSettings.ambientMode = AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = new Color(0.62f, 0.66f, 0.72f);
        RenderSettings.ambientEquatorColor = new Color(0.50f, 0.52f, 0.55f);
        RenderSettings.ambientGroundColor = new Color(0.30f, 0.30f, 0.30f);
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Linear;
        RenderSettings.fogColor = BackgroundColor;
        RenderSettings.fogStartDistance = 25f;
        RenderSettings.fogEndDistance = 80f;

        // The default scene's own sun would double the lighting and the shadows.
        foreach (var light in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
        {
            if (light.type != LightType.Directional || !light.enabled) continue;
            light.enabled = false;
            _disabledLights.Add(light);
        }
    }

    private void CreateSunLight()
    {
        var go = new GameObject("Sun") { hideFlags = HideFlags.HideAndDontSave };
        go.transform.SetParent(_root.transform, false);
        go.transform.rotation = Quaternion.Euler(55f, -35f, 0f);

        var light = go.AddComponent<Light>();
        light.type = LightType.Directional;
        light.color = new Color(1f, 0.97f, 0.92f);
        light.intensity = 1.2f;
        light.shadows = LightShadows.Soft;
        light.shadowStrength = 0.75f;
    }

    private Transform CreateGround()
    {
        var mesh = Track(new Mesh { name = "BenchmarkGround" });
        var half = GroundSize * 0.5f;
        mesh.vertices = new[]
        {
            new Vector3(-half, 0f, -half), new Vector3(-half, 0f, half),
            new Vector3(half, 0f, half), new Vector3(half, 0f, -half)
        };
        mesh.normals = new[] { Vector3.up, Vector3.up, Vector3.up, Vector3.up };
        mesh.uv = new[] { Vector2.zero, new Vector2(0f, 1f), Vector2.one, new Vector2(1f, 0f) };
        mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
        mesh.RecalculateBounds();

        var material = Track(new Material(FindLitShader()) { name = "BenchmarkGround" });
        var grid = GridTexture();
        material.SetTexture("_BaseMap", grid);
        material.SetTextureScale("_BaseMap", new Vector2(GroundSize, GroundSize));
        material.SetTexture("_MainTex", grid);
        material.SetTextureScale("_MainTex", new Vector2(GroundSize, GroundSize));
        material.SetColor("_BaseColor", Color.white);
        material.SetFloat("_Smoothness", 0.05f);

        var go = new GameObject("Ground") { hideFlags = HideFlags.HideAndDontSave };
        go.transform.SetParent(_root.transform, false);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        var renderer = go.AddComponent<MeshRenderer>();
        renderer.sharedMaterial = material;
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = true;
        return go.transform;
    }

    /// <summary>One 1 m grid cell: a light fill with darker lines on two edges, tiled across the ground.</summary>
    private Texture2D GridTexture()
    {
        const int lineWidth = 3;
        var fill = new Color32(196, 199, 202, 255);
        var line = new Color32(150, 154, 160, 255);

        var texture = Track(new Texture2D(GridTexturePixels, GridTexturePixels, TextureFormat.RGBA32, true)
        {
            name = "BenchmarkGrid",
            wrapMode = TextureWrapMode.Repeat,
            filterMode = FilterMode.Trilinear,
            anisoLevel = 16
        });

        var pixels = new Color32[GridTexturePixels * GridTexturePixels];
        for (var y = 0; y < GridTexturePixels; y++)
        for (var x = 0; x < GridTexturePixels; x++)
        {
            var onLine = x < lineWidth || y < lineWidth;
            pixels[y * GridTexturePixels + x] = onLine ? line : fill;
        }

        texture.SetPixels32(pixels);
        texture.Apply(true, true);
        return texture;
    }

    private LineRenderer CreateGroundLine(string name, Shader shader, Color color, float width)
    {
        var go = new GameObject(name) { hideFlags = HideFlags.HideAndDontSave };
        go.transform.SetParent(_root.transform, false);
        // TransformZ alignment with Z pointing down lays the ribbon flat on the ground.
        go.transform.rotation = Quaternion.Euler(90f, 0f, 0f);

        var line = go.AddComponent<LineRenderer>();
        line.sharedMaterial = Track(new Material(shader) { name = name });
        line.startColor = color;
        line.endColor = color;
        line.widthMultiplier = width;
        line.useWorldSpace = true;
        line.alignment = LineAlignment.TransformZ;
        line.positionCount = 0;
        line.numCapVertices = 4;
        line.numCornerVertices = 2;
        line.shadowCastingMode = ShadowCastingMode.Off;
        line.receiveShadows = false;
        return line;
    }

    private MeshRenderer CreateFlatMarker(string name, Mesh mesh, Shader shader, Color color)
    {
        var go = new GameObject(name) { hideFlags = HideFlags.HideAndDontSave };
        go.transform.SetParent(_root.transform, false);
        go.transform.localScale = FootprintScale;
        go.AddComponent<MeshFilter>().sharedMesh = mesh;

        var renderer = go.AddComponent<MeshRenderer>();
        renderer.sharedMaterial = Track(new Material(shader) { name = name, color = color });
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        return renderer;
    }

    /// <summary>A unit-radius disc in the XZ plane, facing up.</summary>
    private Mesh Disc(int segments)
    {
        var vertices = new Vector3[segments + 1];
        var triangles = new int[segments * 3];
        vertices[0] = Vector3.zero;
        for (var i = 0; i < segments; i++)
        {
            var angle = 2f * math.PI * i / segments;
            vertices[i + 1] = new Vector3(math.sin(angle), 0f, math.cos(angle));
            triangles[i * 3] = 0;
            triangles[i * 3 + 1] = i + 1;
            triangles[i * 3 + 2] = (i + 1) % segments + 1;
        }

        var mesh = Track(new Mesh { name = "BenchmarkDisc", vertices = vertices, triangles = triangles });
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private static Gradient FadingGradient(Color color)
    {
        var gradient = new Gradient();
        gradient.SetKeys(
            new[] { new GradientColorKey(color, 0f), new GradientColorKey(color, 1f) },
            new[] { new GradientAlphaKey(0f, 0f), new GradientAlphaKey(0.9f, 1f) });
        return gradient;
    }

    private static Shader FindLitShader()
    {
        var shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader != null) return shader;

        Debug.LogWarning("[Benchmark] URP Lit shader not found; the video ground falls back to Standard.");
        return Shader.Find("Standard");
    }

    private T Track<T>(T asset) where T : Object
    {
        asset.hideFlags = HideFlags.HideAndDontSave;
        _owned.Add(asset);
        return asset;
    }

    public void Dispose()
    {
        EndRun();
        _savedLighting.Restore();
        foreach (var light in _disabledLights)
        {
            if (light != null) light.enabled = true;
        }
        _disabledLights.Clear();

        foreach (var asset in _owned)
        {
            if (asset != null) Object.DestroyImmediate(asset);
        }
        _owned.Clear();

        if (_root != null) Object.DestroyImmediate(_root);
    }

    private struct Footprint
    {
        public MeshRenderer Renderer;
        public Color Color;
        public float Age;
    }

    /// <summary>
    /// One toe judged live by <see cref="GroundContactSettings"/>' rule: height above its contact height
    /// under H, horizontal speed under V.
    /// </summary>
    /// <remarks>
    /// Judged one tick late, so the speed can be a central difference as in
    /// <see cref="MotionQualityMetricsCalculator"/>. The contact height is a percentile of the run so far
    /// rather than of the whole run, which only the recording afterwards can know.
    /// </remarks>
    private sealed class ToeContact
    {
        public readonly int BoneIndex;

        private float[] _heights = new float[1024];
        private float[] _sorted = new float[1024];
        private float _contactHeight = float.PositiveInfinity;
        private Vector3 _beforePrevious;
        private Vector3 _previous;
        private float _previousDeltaTime;
        private int _samples;
        private bool _wasInContact;

        public ToeContact(int boneIndex)
        {
            BoneIndex = boneIndex;
        }

        /// <summary>Takes this tick's position; true when the previous tick was a landing.</summary>
        /// <param name="landedAt">Where the toe was on the tick it landed.</param>
        public bool Sample(Vector3 position, float deltaTime, GroundContactSettings settings, out Vector3 landedAt)
        {
            landedAt = _previous;

            if (_samples == _heights.Length)
            {
                Array.Resize(ref _heights, _heights.Length * 2);
                _sorted = new float[_heights.Length];
            }
            _heights[_samples] = position.y;

            if (_samples < 2 * ContactHeightRefreshTicks) _contactHeight = math.min(_contactHeight, position.y);
            else if (_samples % ContactHeightRefreshTicks == 0)
                _contactHeight = Percentile(_samples + 1, settings.contactHeightPercentile);

            var landed = false;
            if (_samples >= 2)
            {
                var span = _previousDeltaTime + deltaTime;
                var travel = new Vector2(position.x - _beforePrevious.x, position.z - _beforePrevious.z).magnitude;
                var speed = span > 0f ? travel / span : 0f;
                var inContact = _previous.y - _contactHeight < settings.heightThreshold &&
                                speed < settings.speedThreshold;

                landed = inContact && !_wasInContact;
                _wasInContact = inContact;
            }

            _beforePrevious = _previous;
            _previous = position;
            _previousDeltaTime = deltaTime;
            _samples++;
            return landed;
        }

        /// <summary>
        /// Nearest-rank percentile of the first <paramref name="count"/> heights, as
        /// <see cref="MetricsMath.Percentile"/> computes it, sorted in a reused buffer so the
        /// recorder adds nothing to the sweep's GC-per-tick metric.
        /// </summary>
        private float Percentile(int count, float quantile)
        {
            Array.Copy(_heights, _sorted, count);
            Array.Sort(_sorted, 0, count);
            var rank = (int)math.ceil(math.clamp(quantile, 0f, 1f) * count) - 1;
            return _sorted[math.clamp(rank, 0, count - 1)];
        }
    }

    /// <summary>The scene-wide lighting the dressing overrides, so it can be put back.</summary>
    private sealed class SceneLightingState
    {
        private readonly AmbientMode _ambientMode;
        private readonly Color _ambientSky;
        private readonly Color _ambientEquator;
        private readonly Color _ambientGround;
        private readonly bool _fog;
        private readonly FogMode _fogMode;
        private readonly Color _fogColor;
        private readonly float _fogStart;
        private readonly float _fogEnd;

        public SceneLightingState()
        {
            _ambientMode = RenderSettings.ambientMode;
            _ambientSky = RenderSettings.ambientSkyColor;
            _ambientEquator = RenderSettings.ambientEquatorColor;
            _ambientGround = RenderSettings.ambientGroundColor;
            _fog = RenderSettings.fog;
            _fogMode = RenderSettings.fogMode;
            _fogColor = RenderSettings.fogColor;
            _fogStart = RenderSettings.fogStartDistance;
            _fogEnd = RenderSettings.fogEndDistance;
        }

        public void Restore()
        {
            RenderSettings.ambientMode = _ambientMode;
            RenderSettings.ambientSkyColor = _ambientSky;
            RenderSettings.ambientEquatorColor = _ambientEquator;
            RenderSettings.ambientGroundColor = _ambientGround;
            RenderSettings.fog = _fog;
            RenderSettings.fogMode = _fogMode;
            RenderSettings.fogColor = _fogColor;
            RenderSettings.fogStartDistance = _fogStart;
            RenderSettings.fogEndDistance = _fogEnd;
        }
    }
}
}
