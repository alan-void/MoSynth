using System;
using System.IO;
using Unity.Mathematics;
using UnityEditor.Media;
using UnityEngine;
using UnityEngine.Splines;
using UnityEngine.UI;
using Object = UnityEngine.Object;

namespace AnimationTools.Editor
{
/// <summary>
/// Renders each run of a live benchmark sweep to its own MP4 under <c>videos/</c>, with a label
/// overlay and, optionally, the scene dressing of <see cref="BenchmarkVideoDressing"/> and the
/// method's consumed inputs from <see cref="BenchmarkInputOverlay"/>. Beside each
/// video goes a JSON sidecar (<see cref="BenchmarkVideoSidecar"/>) that
/// <c>python -m benchmark.compose_videos</c> reads to tile the methods of one path side by side.
/// </summary>
/// <remarks>
/// Captured on the run's own synthesis tick (<see cref="MotionSynthesisComponent.OnPoseApplied"/>),
/// one video frame per tick encoded at the sweep's synthesis frame rate, so playback runs at normal
/// speed however fast <c>fixedTimestep</c> lets the sweep itself run. Frames are rendered at twice the
/// output size and box-filtered down, because the project's URP asset has MSAA off and URP ignores a
/// target texture's own sample count in that case.
/// </remarks>
public sealed class BenchmarkVideoRecorder : IDisposable
{
    private const int Supersampling = 2;

    /// <summary>H.264 bits per output pixel per frame; ~15 Mbit/s at 1080p30.</summary>
    private const float BitsPerPixel = 0.25f;

    private static readonly Color PanelColor = new(0.05f, 0.06f, 0.08f, 0.88f);
    private static readonly Color SecondaryTextColor = new(0.80f, 0.83f, 0.87f);
    private static readonly Color SettlingColor = new(1f, 0.75f, 0.25f);

    private readonly string _outputDirectory;
    private readonly int _width;
    private readonly int _height;
    private readonly float _frameRate;
    private readonly BenchmarkVideoCamera _cameraMode;
    private readonly GroundContactSettings _groundContact;
    private readonly ContactFloor _contactFloor;
    private readonly SplineProjector _projector = new();

    private GameObject _root;
    private Camera _camera;
    private BenchmarkCameraRig _rig;
    private BenchmarkVideoDressing _dressing;
    private BenchmarkInputOverlay _inputOverlay;
    private ISynthesisInputSource _inputSource;
    private RenderTexture _renderTarget;
    private RenderTexture _outputTarget;
    private Texture2D _readback;
    private Text _title;
    private Text _subtitle;
    private Text _clock;
    private Text _metrics;

    private MediaEncoder _encoder;
    private BenchmarkVideoSidecar _sidecar;
    private string _sidecarPath;

    private SplineContainer _currentSpline;
    private MotionSynthesisComponent _synthesizer;
    private BenchmarkLapProbe _probe;
    private float _targetSpeed;
    private float3 _previousRoot;
    private bool _hasPreviousRoot;
    private double _errorSum;
    private int _errorSamples;

    /// <param name="contactFloor">Each contact bone's floor height, where footprints are judged from;
    /// null places none.</param>
    public BenchmarkVideoRecorder(string outputDirectory, SynthesisBenchmarkConfig config, ContactFloor contactFloor)
    {
        _outputDirectory = outputDirectory;
        _width = Mathf.Max(16, config.videoWidth & ~1);
        _height = Mathf.Max(16, config.videoHeight & ~1);
        _frameRate = config.synthesisFrameRate;
        _cameraMode = config.videoCamera;
        _groundContact = config.groundContact;
        _contactFloor = contactFloor;

        Directory.CreateDirectory(_outputDirectory);
        BuildResources(config.videoSceneDressing, config.videoShowMethodInputs);
    }

    /// <summary>Starts capturing one run into a video of its own.</summary>
    /// <param name="runIndex">The run's position in the sweep's method list, which orders the comparison tiles.</param>
    public void Attach(MotionSynthesisComponent synthesizer, BenchmarkMethodRun run, int runIndex, string pathName,
        SplineContainer spline, IMotionSynthesisSplineControlInput input, BenchmarkLapProbe probe)
    {
        Detach();
        SetPath(spline);

        _synthesizer = synthesizer;
        _probe = probe;
        _targetSpeed = input.TargetSpeed;
        _projector.Reset();
        _rig.Reset();
        _hasPreviousRoot = false;
        _errorSum = 0.0;
        _errorSamples = 0;

        _title.text = run.Method.name;
        _subtitle.text = run.Seed.HasValue ? $"{pathName}   ·   seed {run.Seed.Value}" : pathName;

        _dressing?.BeginRun(synthesizer, input as ISteeringTargetSource, _groundContact, _contactFloor);
        _inputSource = SynthesisInputs.FindSource(synthesizer);
        _inputOverlay?.BeginRun();

        var fileName = Sanitize($"{run.Label}_{pathName}");
        _sidecarPath = Path.Combine(_outputDirectory, fileName + ".json");
        _sidecar = new BenchmarkVideoSidecar
        {
            video = fileName + ".mp4",
            method = run.Method.name,
            label = run.Label,
            path = pathName,
            seeded = run.Seed.HasValue,
            seed = run.Seed ?? 0,
            runIndex = runIndex,
            frameRate = _frameRate,
            width = _width,
            height = _height,
            camera = _cameraMode.ToString(),
            measureStartFrame = -1
        };
        _encoder = new MediaEncoder(Path.Combine(_outputDirectory, _sidecar.video), EncoderAttributes());

        synthesizer.OnPoseApplied += HandlePoseApplied;
    }

    /// <summary>Stops capturing the run in progress and closes its video. Safe to call when nothing is attached.</summary>
    public void Detach()
    {
        if (_synthesizer != null) _synthesizer.OnPoseApplied -= HandlePoseApplied;
        _synthesizer = null;
        _probe = null;
        _inputSource = null;
        _dressing?.EndRun();
        _inputOverlay?.EndRun();

        if (_encoder == null) return;
        _encoder.Dispose();
        _encoder = null;

        try
        {
            File.WriteAllText(_sidecarPath, JsonUtility.ToJson(_sidecar, true));
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Benchmark] Could not write video sidecar \"{_sidecarPath}\": {e.Message}");
        }

        _sidecar = null;
    }

    private void SetPath(SplineContainer spline)
    {
        if (spline == _currentSpline) return;
        _currentSpline = spline;

        Vector3 center;
        float radius;
        if (_dressing != null) _dressing.SetPath(spline, out center, out radius);
        else PathBounds(spline, out center, out radius);
        _rig.FramePath(center, radius);

        if (_dressing == null) return;
        if (_cameraMode == BenchmarkVideoCamera.Overview)
            _dressing.SetFogRange(_rig.OverviewDistance * 1.5f, _rig.OverviewDistance * 3f);
        else
            _dressing.SetFogRange(25f, 80f);
    }

    private void HandlePoseApplied(PoseBuffer pose, float deltaTime)
    {
        var root = (float3)_synthesizer.transform.position;

        var speed = 0f;
        if (_hasPreviousRoot && deltaTime > 0f)
        {
            speed = math.length(new float2(root.x - _previousRoot.x, root.z - _previousRoot.z)) / deltaTime;
        }
        _previousRoot = root;
        _hasPreviousRoot = true;

        var pathError = 0f;
        if (_currentSpline != null && _currentSpline.Spline != null)
        {
            var local = (float3)_currentSpline.transform.InverseTransformPoint(root);
            var t = _projector.Project(_currentSpline.Spline, local);
            var nearestWorld = (float3)_currentSpline.transform.TransformPoint(
                (Vector3)_currentSpline.Spline.EvaluatePosition(t));
            pathError = math.distance(new float2(root.x, root.z), new float2(nearestWorld.x, nearestWorld.z));
            _errorSum += pathError;
            _errorSamples++;
        }

        var meanError = _errorSamples > 0 ? (float)(_errorSum / _errorSamples) : 0f;
        var laps = _probe != null ? _probe.CompletedLaps : 0f;
        var elapsed = _probe != null ? _probe.ElapsedSeconds : 0f;
        var settling = _probe != null && elapsed < _probe.settleTime;
        if (!settling && _sidecar.measureStartFrame < 0) _sidecar.measureStartFrame = _sidecar.frameCount;

        _clock.text = settling ? $"{elapsed:0.0} s   settling" : $"{elapsed:0.0} s   lap {laps:0.00}";
        _clock.color = settling ? SettlingColor : Color.white;
        _metrics.text = $"speed {speed:0.00} / {_targetSpeed:0.00} m/s   ·   " +
                        $"path error {pathError:0.00} m (mean {meanError:0.00})";

        _dressing?.Tick(deltaTime);
        _inputOverlay?.Tick(_inputSource?.ConsumedInputs, deltaTime);

        var groundHeight = _dressing?.GroundHeight ?? root.y;
        _rig.Follow(root, _synthesizer.transform.forward, groundHeight, deltaTime);

        RenderFrame();
    }

    private void RenderFrame()
    {
        _camera.Render();
        Graphics.Blit(_renderTarget, _outputTarget);

        var previous = RenderTexture.active;
        RenderTexture.active = _outputTarget;
        _readback.ReadPixels(new Rect(0, 0, _width, _height), 0, 0, false);
        RenderTexture.active = previous;

        // The raw-buffer overload, because AddFrame(Texture2D) allocates a managed copy per frame and
        // the sweep's GC-per-tick metric counts every allocation in the frame, the recorder's included.
        _encoder.AddFrame(_width, _height, _width * 4, TextureFormat.RGBA32, _readback.GetRawTextureData<byte>());
        _sidecar.frameCount++;
    }

    private VideoTrackEncoderAttributes EncoderAttributes()
    {
        return new VideoTrackEncoderAttributes(new H264EncoderAttributes
        {
            gopSize = (uint)Mathf.Max(1, Mathf.RoundToInt(_frameRate)),
            numConsecutiveBFrames = 2,
            profile = UnityEditor.VideoEncodingProfile.H264High
        })
        {
            frameRate = ToRational(_frameRate),
            width = (uint)_width,
            height = (uint)_height,
            includeAlpha = false,
            bitRateMode = UnityEditor.VideoBitrateMode.High,
            targetBitRate = (uint)(BitsPerPixel * _width * _height * math.max(1f, _frameRate))
        };
    }

    private static MediaRational ToRational(float fps)
    {
        var rounded = Mathf.RoundToInt(fps);
        return Mathf.Approximately(rounded, fps)
            ? new MediaRational(rounded)
            : new MediaRational(Mathf.RoundToInt(fps * 1000f), 1000);
    }

    /// <summary>Ground-plane bounds of a path, for framing it when no dressing computed them.</summary>
    private static void PathBounds(SplineContainer spline, out Vector3 center, out float radius)
    {
        center = Vector3.zero;
        radius = 1f;
        if (spline == null || spline.Spline == null) return;

        const int samples = 200;
        var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        for (var i = 0; i <= samples; i++)
        {
            var point = spline.transform.TransformPoint((Vector3)spline.EvaluatePosition((float)i / samples));
            min = Vector3.Min(min, point);
            max = Vector3.Max(max, point);
        }

        center = 0.5f * (min + max);
        radius = 0.5f * new Vector2(max.x - min.x, max.z - min.z).magnitude;
    }

    private void BuildResources(bool sceneDressing, bool showMethodInputs)
    {
        _root = new GameObject("BenchmarkVideoRecorder") { hideFlags = HideFlags.HideAndDontSave };

        var cameraGo = new GameObject("CaptureCamera") { hideFlags = HideFlags.HideAndDontSave };
        cameraGo.transform.SetParent(_root.transform);
        _camera = cameraGo.AddComponent<Camera>();
        _camera.enabled = false;
        _camera.nearClipPlane = 0.05f;
        _camera.farClipPlane = 500f;

        if (sceneDressing)
        {
            _camera.clearFlags = CameraClearFlags.SolidColor;
            _camera.backgroundColor = BenchmarkVideoDressing.BackgroundColor;
        }
        else
        {
            _camera.clearFlags = CameraClearFlags.Skybox;
        }

        _renderTarget = new RenderTexture(_width * Supersampling, _height * Supersampling, 24,
            RenderTextureFormat.ARGB32)
        {
            hideFlags = HideFlags.HideAndDontSave,
            filterMode = FilterMode.Bilinear
        };
        _renderTarget.Create();
        _camera.targetTexture = _renderTarget;

        // Exactly half size with bilinear sampling, so each output pixel averages a 2x2 block.
        _outputTarget = new RenderTexture(_width, _height, 0, RenderTextureFormat.ARGB32)
        {
            hideFlags = HideFlags.HideAndDontSave
        };
        _outputTarget.Create();

        _readback = new Texture2D(_width, _height, TextureFormat.RGBA32, false, false)
        {
            hideFlags = HideFlags.HideAndDontSave
        };

        _rig = new BenchmarkCameraRig(_camera, _cameraMode);
        if (sceneDressing) _dressing = new BenchmarkVideoDressing(_root.transform);

        var canvas = BuildOverlay(out var font);
        if (showMethodInputs) _inputOverlay = new BenchmarkInputOverlay(_root.transform, canvas, font);
    }

    /// <summary>
    /// A translucent panel top-left: method in large type, then path and seed, the clock, and the
    /// live path-following figures. Laid out at 1080p and scaled with the frame.
    /// </summary>
    /// <returns>The canvas, for the method-input panel to share.</returns>
    private RectTransform BuildOverlay(out Font font)
    {
        var canvasGo = new GameObject("HudCanvas", typeof(RectTransform)) { hideFlags = HideFlags.HideAndDontSave };
        canvasGo.transform.SetParent(_root.transform);
        var canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceCamera;
        canvas.worldCamera = _camera;
        canvas.planeDistance = 1f;
        var scaler = canvasGo.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920f, 1080f);
        scaler.matchWidthOrHeight = 1f;

        var panelGo = new GameObject("Panel", typeof(RectTransform)) { hideFlags = HideFlags.HideAndDontSave };
        panelGo.transform.SetParent(canvasGo.transform, false);
        panelGo.AddComponent<Image>().color = PanelColor;
        var panel = (RectTransform)panelGo.transform;
        panel.anchorMin = panel.anchorMax = panel.pivot = new Vector2(0f, 1f);
        panel.anchoredPosition = new Vector2(32f, -32f);
        panel.sizeDelta = new Vector2(760f, 196f);

        font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        _title = AddLine(panel, font, 0f, 54, FontStyle.Bold, Color.white);
        _subtitle = AddLine(panel, font, 64f, 30, FontStyle.Normal, SecondaryTextColor);
        _clock = AddLine(panel, font, 104f, 30, FontStyle.Normal, Color.white);
        _metrics = AddLine(panel, font, 146f, 24, FontStyle.Normal, SecondaryTextColor);
        return (RectTransform)canvasGo.transform;
    }

    private static Text AddLine(RectTransform panel, Font font, float top, int size, FontStyle style, Color color)
    {
        const float padding = 22f;

        var go = new GameObject("Line", typeof(RectTransform)) { hideFlags = HideFlags.HideAndDontSave };
        go.transform.SetParent(panel, false);
        var text = go.AddComponent<Text>();
        text.font = font;
        text.fontSize = size;
        text.fontStyle = style;
        text.color = color;
        text.alignment = TextAnchor.UpperLeft;
        text.horizontalOverflow = HorizontalWrapMode.Overflow;
        text.verticalOverflow = VerticalWrapMode.Overflow;

        var rect = text.rectTransform;
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(1f, 1f);
        rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = new Vector2(padding, -(padding * 0.6f + top));
        rect.sizeDelta = new Vector2(-2f * padding, size * 1.3f);
        return text;
    }

    /// <summary>Makes a run label safe to use as a video filename.</summary>
    private static string Sanitize(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = name.ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (Array.IndexOf(invalid, chars[i]) >= 0 || chars[i] == ' ') chars[i] = '_';
        }

        return new string(chars);
    }

    public void Dispose()
    {
        Detach();

        _dressing?.Dispose();
        _dressing = null;
        _inputOverlay?.Dispose();
        _inputOverlay = null;

        if (_renderTarget != null)
        {
            _renderTarget.Release();
            Object.DestroyImmediate(_renderTarget);
        }

        if (_outputTarget != null)
        {
            _outputTarget.Release();
            Object.DestroyImmediate(_outputTarget);
        }

        if (_readback != null) Object.DestroyImmediate(_readback);
        if (_root != null) Object.DestroyImmediate(_root);

        _renderTarget = null;
        _outputTarget = null;
        _readback = null;
        _root = null;
    }
}

/// <summary>
/// What a benchmark video's JSON sidecar records: which run it shows, and the frame measurement starts
/// at, so a composer can line several methods up at the end of their settle time.
/// </summary>
[Serializable]
public class BenchmarkVideoSidecar
{
    public string video;
    public string method;
    public string label;
    public string path;
    public bool seeded;
    public int seed;
    public int runIndex;
    public float frameRate;
    public int width;
    public int height;
    public string camera;
    public int frameCount;

    /// <summary>First frame after the settle time; -1 when the run ended before it.</summary>
    public int measureStartFrame;
}
}
