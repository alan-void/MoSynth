using System;
using System.IO;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEditor.Media;
using UnityEngine;
using Object = UnityEngine.Object;

namespace AnimationTools.Editor
{
/// <summary>
/// Renders an <see cref="AnnotatedAnimationClip"/>'s slice as a capsule figure on a checker floor,
/// to an H.264 MP4 with each frame's clip-local frame number burned in.
/// </summary>
/// <remarks>
/// The camera follows the root over the ground but never turns, so travel and turning show up as
/// motion against the floor rather than being cancelled by the view. One instance can render any
/// number of clips; it rebuilds the figure when the skeleton changes.
/// </remarks>
public sealed class ClipVideoRenderer : IDisposable
{
    private const float CheckerSize = 0.5f;
    private const float GroundSize = 80f;
    private const float CameraPitch = 22f;
    private const float CameraYawOffset = 35f;
    private const float FieldOfView = 35f;

    // Half the figure's height of margin above and below it.
    private const float FramedHeights = 1f;

    // Rendered at this multiple and box-filtered down: MSAA targets fail to resolve in URP's
    // preview path, and aliased limbs are harder for a video model to read.
    private const int Supersampling = 2;
    // Image rows per font pixel, so the label keeps its share of the frame: scale 4 at 480 rows.
    private const int RowsPerLabelPixel = 110;

    private static readonly Color Background = new(0.12f, 0.13f, 0.15f);

    private readonly int _width;
    private readonly int _height;

    private PreviewRenderUtility _preview;
    private RenderTexture _target;
    private RenderTexture _downsampled;
    private Texture2D _readback;
    private Material _groundMaterial;
    private Mesh _groundMesh;
    private Transform _ground;

    private Skeleton _skeleton;
    private int _skeletonContentHash;
    private SkeletonData _skeletonData;
    private CapsuleFigure _figure;
    private NativeArray<float3> _positions;
    private NativeArray<quaternion> _rotations;

    private float _cameraYaw;

    public enum Outcome
    {
        Rendered,
        Failed,
        Cancelled
    }

    public ClipVideoRenderer(int width = 640, int height = 480)
    {
        // H.264 needs even dimensions.
        _width = Mathf.Max(16, width & ~1);
        _height = Mathf.Max(16, height & ~1);
    }

    /// <summary>Clip frames per video frame: the nearest whole step to the requested rate, at least 1.</summary>
    public static int FrameStep(float clipFrameRate, float renderFps)
    {
        if (clipFrameRate <= 0f || renderFps <= 0f) return 1;
        return Math.Max(1, (int)Math.Round(clipFrameRate / renderFps));
    }

    /// <summary>How many video frames a slice of <paramref name="sliceFrames"/> frames renders to.</summary>
    public static int VideoFrameCount(int sliceFrames, int frameStep) =>
        sliceFrames <= 0 ? 0 : (sliceFrames + frameStep - 1) / Math.Max(1, frameStep);

    /// <summary>
    /// Encodes clip frames <c>startFrame + k * frameStep</c> within the slice to
    /// <paramref name="mp4Path"/>. <paramref name="progress"/> is given the fraction done and
    /// returns true to cancel. <paramref name="error"/> is set when the outcome is a failure.
    /// </summary>
    public Outcome Render(AnnotatedAnimationClip clip, string mp4Path, float renderFps, out string error,
        Func<float, bool> progress = null)
    {
        if (!TryBegin(clip, out var source, out error)) return Outcome.Failed;

        var start = clip.startFrame;
        var end = SliceEnd(clip);
        var step = FrameStep(clip.Clip.frameRate, renderFps);
        var frameCount = VideoFrameCount(end - start, step);
        if (frameCount == 0)
        {
            error = "The clip's slice is empty.";
            return Outcome.Failed;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(mp4Path)) ?? ".");

        var attributes = new VideoTrackEncoderAttributes(new H264EncoderAttributes
        {
            gopSize = 25,
            numConsecutiveBFrames = 2,
            profile = VideoEncodingProfile.H264High
        })
        {
            frameRate = ToRational(renderFps),
            width = (uint)_width,
            height = (uint)_height,
            includeAlpha = false,
            bitRateMode = VideoBitrateMode.High
        };

        using (var encoder = new MediaEncoder(mp4Path, attributes))
        {
            for (var k = 0; k < frameCount; k++)
            {
                if (progress != null && progress((float)k / frameCount)) return Outcome.Cancelled;

                if (!TryRenderFrame(source, start + k * step, out error)) return Outcome.Failed;

                encoder.AddFrame(_readback);
            }
        }

        return Outcome.Rendered;
    }

    /// <summary>Renders one clip-local frame, framed as a video of the clip would be, to a PNG.</summary>
    public bool TrySaveFramePng(AnnotatedAnimationClip clip, int clipFrame, string pngPath, out string error)
    {
        if (!TryBegin(clip, out var source, out error)) return false;

        var frame = Mathf.Clamp(clipFrame, 0, source.FrameCount - 1);
        if (!TryRenderFrame(source, frame, out error)) return false;

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(pngPath)) ?? ".");
        File.WriteAllBytes(pngPath, _readback.EncodeToPNG());
        return true;
    }

    /// <summary>The slice's end, clamped to the frames the clip actually has.</summary>
    public static int SliceEnd(AnnotatedAnimationClip clip) =>
        Math.Min(clip.endFrame, ((SkeletonAnimation)clip).FrameCount);

    private static MediaRational ToRational(float fps)
    {
        var rounded = Mathf.RoundToInt(fps);
        return Mathf.Approximately(rounded, fps)
            ? new MediaRational(rounded)
            : new MediaRational(Mathf.RoundToInt(fps * 1000f), 1000);
    }

    /// <summary>
    /// Validates the clip, prepares resources for its skeleton and fixes the camera's heading for it.
    /// </summary>
    /// <remarks>
    /// Returns the clip typed as <see cref="SkeletonAnimation"/>, whose <c>GetFrame</c> and
    /// <c>FrameCount</c> are whole-clip; <see cref="AnnotatedAnimationClip"/> shadows both with
    /// slice-local versions.
    /// </remarks>
    private bool TryBegin(AnnotatedAnimationClip clip, out SkeletonAnimation source, out string error)
    {
        source = clip;
        if (clip == null)
        {
            error = "No clip.";
            return false;
        }

        if (!clip.TryValidate(out error)) return false;

        if (clip.PoseSequence == null || source.FrameCount == 0)
        {
            error = "The clip did not bake.";
            return false;
        }

        EnsureResources();
        RefreshSkeleton(clip.Skeleton);

        var first = source.GetFrame(Mathf.Clamp(clip.startFrame, 0, source.FrameCount - 1));
        if (first.Layout.RotationCount != _skeletonData.BoneCount)
        {
            error = "The baked pose does not match the skeleton's bone count.";
            return false;
        }

        // Faces the camera three-quarters on to where the character is heading at the slice's start.
        SimulationFrame.Compute(first, _skeletonData, SimulationFrameDef.Default(clip.Skeleton), out _,
            out var facing);
        _cameraYaw = math.degrees(SimulationFrame.Yaw(facing)) + 180f + CameraYawOffset;

        error = null;
        return true;
    }

    private bool TryRenderFrame(SkeletonAnimation source, int clipFrame, out string error)
    {
        var pose = source.GetFrame(clipFrame);
        if (pose.Layout.RotationCount != _skeletonData.BoneCount)
        {
            error = "The baked pose does not match the skeleton's bone count.";
            return false;
        }

        _skeletonData.LocalSpaceToCharacterSpace(pose, _positions, _rotations);
        _figure.Pose(_positions, _rotations);

        var root = _positions[0];
        var groundTarget = new Vector3(root.x, 0f, root.z);
        _ground.position = groundTarget;

        var height = _figure.Height;
        var lookAt = groundTarget + Vector3.up * (height * 0.5f);
        var rotation = Quaternion.Euler(CameraPitch, _cameraYaw, 0f);
        var distance = height * FramedHeights / Mathf.Tan(0.5f * FieldOfView * Mathf.Deg2Rad);

        var camera = _preview.camera;
        camera.transform.SetPositionAndRotation(lookAt - rotation * Vector3.forward * distance, rotation);

        var previousTarget = camera.targetTexture;
        var previousActive = RenderTexture.active;
        try
        {
            camera.targetTexture = _target;
            camera.Render();
            Graphics.Blit(_target, _downsampled);

            RenderTexture.active = _downsampled;
            _readback.ReadPixels(new Rect(0, 0, _width, _height), 0, 0, false);
        }
        finally
        {
            RenderTexture.active = previousActive;
            camera.targetTexture = previousTarget;
        }

        var pixels = _readback.GetPixelData<Color32>(0);
        BurnInLabel.Draw(pixels, _width, _height, $"frame {clipFrame}", Mathf.Max(2, _height / RowsPerLabelPixel));
        _readback.Apply(false);

        error = null;
        return true;
    }

    private void EnsureResources()
    {
        if (_preview == null)
        {
            _preview = new PreviewRenderUtility();
            var camera = _preview.camera;
            camera.fieldOfView = FieldOfView;
            camera.nearClipPlane = 0.05f;
            camera.farClipPlane = 500f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Background;
        }

        if (_target == null)
        {
            _target = CreateTarget(_width * Supersampling, _height * Supersampling, 24);
            _downsampled = CreateTarget(_width, _height, 0);
        }

        if (_readback == null)
        {
            _readback = new Texture2D(_width, _height, TextureFormat.RGBA32, false, false)
            {
                hideFlags = HideFlags.HideAndDontSave
            };
        }

        if (_ground == null)
        {
            _groundMaterial = CapsuleFigure.CreateMaterial(new Color(0.55f, 0.55f, 0.55f));
            _groundMaterial.SetColor("_CheckerColor", new Color(0.32f, 0.32f, 0.32f));
            _groundMaterial.SetFloat("_CheckerSize", CheckerSize);
            _groundMesh = BuildGroundMesh(GroundSize);

            var go = new GameObject("Ground") { hideFlags = HideFlags.HideAndDontSave };
            go.AddComponent<MeshFilter>().sharedMesh = _groundMesh;
            go.AddComponent<MeshRenderer>().sharedMaterial = _groundMaterial;
            _preview.AddSingleGO(go);
            _ground = go.transform;
        }
    }

    private static RenderTexture CreateTarget(int width, int height, int depth)
    {
        var target = new RenderTexture(width, height, depth, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB)
        {
            hideFlags = HideFlags.HideAndDontSave,
            filterMode = FilterMode.Bilinear
        };
        target.Create();
        return target;
    }

    private void RefreshSkeleton(Skeleton skeleton)
    {
        var contentHash = skeleton.ContentHash;
        if (_figure != null && skeleton.Root == _skeleton?.Root && contentHash == _skeletonContentHash) return;

        DisposeSkeleton();

        _skeleton = skeleton;
        _skeletonContentHash = contentHash;
        _skeletonData = skeleton.GetSkeletonData();
        _positions = new NativeArray<float3>(skeleton.BoneCount, Allocator.Persistent);
        _rotations = new NativeArray<quaternion>(skeleton.BoneCount, Allocator.Persistent);
        _figure = new CapsuleFigure(_preview, skeleton);
    }

    private static Mesh BuildGroundMesh(float size)
    {
        var half = size * 0.5f;
        var mesh = new Mesh { hideFlags = HideFlags.HideAndDontSave };
        mesh.SetVertices(new[]
        {
            new Vector3(-half, 0f, -half), new Vector3(-half, 0f, half),
            new Vector3(half, 0f, half), new Vector3(half, 0f, -half)
        });
        mesh.SetNormals(new[] { Vector3.up, Vector3.up, Vector3.up, Vector3.up });
        mesh.SetTriangles(new[] { 0, 1, 2, 0, 2, 3 }, 0);
        mesh.RecalculateBounds();
        return mesh;
    }

    private void DisposeSkeleton()
    {
        if (_figure != null)
        {
            _figure.Dispose();
            _figure = null;
        }

        if (_positions.IsCreated) _positions.Dispose();
        if (_rotations.IsCreated) _rotations.Dispose();

        _skeleton = null;
        _skeletonContentHash = 0;
        _skeletonData = default;
    }

    private static void DestroyTarget(RenderTexture target)
    {
        if (target == null) return;

        target.Release();
        Object.DestroyImmediate(target);
    }

    public void Dispose()
    {
        DisposeSkeleton();

        if (_preview != null)
        {
            _preview.Cleanup();
            _preview = null;
        }

        DestroyTarget(_target);
        DestroyTarget(_downsampled);

        if (_readback != null) Object.DestroyImmediate(_readback);
        if (_groundMaterial != null) Object.DestroyImmediate(_groundMaterial);
        if (_groundMesh != null) Object.DestroyImmediate(_groundMesh);

        _target = null;
        _downsampled = null;
        _readback = null;
        _groundMaterial = null;
        _groundMesh = null;
        _ground = null;
    }
}
}
