using Unity.Collections;
using Unity.Mathematics;
using Debug = UnityEngine.Debug;

namespace AnimationTools
{
/// <summary>
/// A <see cref="StateBuffer"/> holding one pose, plus the named-section map
/// (<see cref="PoseLayoutData"/>) that gives positions/rotations/etc. typed slice access.
/// </summary>
/// <remarks>
/// A view struct with <see cref="StateBuffer"/>'s aliasing and ownership rules: copies share
/// <see cref="Data"/>, and whoever calls <see cref="Allocate"/> calls <see cref="Dispose"/>.
/// Converts implicitly to <see cref="StateBuffer"/> for handle-addressed access.
/// </remarks>
public struct PoseBuffer
{
    public StateBuffer State;
    public PoseLayoutData Layout;

    public NativeArray<float> Data => State.Data;

    public bool IsCreated => State.IsCreated;
    public int FloatCount => State.Layout.FloatCount;
    public int LayoutHash => State.Layout.LayoutHash;

    public static implicit operator StateBuffer(PoseBuffer pose) => pose.State;

    public NativeSlice<float3> Positions =>
        new NativeSlice<float>(State.Data, Layout.PositionStart, Layout.PositionCount * 3).SliceConvert<float3>();

    public NativeSlice<quaternion> Rotations
    {
        get
        {
            Debug.Assert(Layout.RotationStride == 4, "Rotations slice requires a Quaternion (stride 4) representation.");
            return new NativeSlice<float>(State.Data, Layout.RotationStart, Layout.RotationCount * 4).SliceConvert<quaternion>();
        }
    }

    public NativeSlice<float3> Scales =>
        new NativeSlice<float>(State.Data, Layout.ScaleStart, Layout.ScaleCount * 3).SliceConvert<float3>();

    public NativeSlice<float3> Velocities =>
        new NativeSlice<float>(State.Data, Layout.VelocityStart, Layout.VelocityCount * 3).SliceConvert<float3>();

    public NativeSlice<float3> AngularVelocities =>
        new NativeSlice<float>(State.Data, Layout.AngularVelocityStart, Layout.AngularVelocityCount * 3).SliceConvert<float3>();

    public float GetFloat(ChannelHandle h, int offset = 0) => State.GetFloat(h, offset);
    public void SetFloat(ChannelHandle h, int offset, float value) => State.SetFloat(h, offset, value);
    public float2 GetFloat2(ChannelHandle h) => State.GetFloat2(h);
    public void SetFloat2(ChannelHandle h, float2 value) => State.SetFloat2(h, value);
    public float3 GetFloat3(ChannelHandle h) => State.GetFloat3(h);
    public void SetFloat3(ChannelHandle h, float3 value) => State.SetFloat3(h, value);
    public quaternion GetQuaternion(ChannelHandle h) => State.GetQuaternion(h);
    public void SetQuaternion(ChannelHandle h, quaternion value) => State.SetQuaternion(h, value);
    public bool GetBool(ChannelHandle h) => State.GetBool(h);
    public void SetBool(ChannelHandle h, bool value) => State.SetBool(h, value);
    public NativeSlice<float> GetSlice(ChannelHandle h) => State.GetSlice(h);

    public void CopyFrom(in PoseBuffer other) => State.CopyFrom(other.State);

    public static void Lerp(in PoseBuffer a, in PoseBuffer b, float t, PoseBuffer destination)
    {
        Debug.Assert(a.LayoutHash == b.LayoutHash && a.LayoutHash == destination.LayoutHash,
            "Lerp requires all three buffers to share the same layout.");

        var layout = a.Layout;
        var srcA = a.Data;
        var srcB = b.Data;
        var dst = destination.Data;

        for (var i = 0; i < layout.PositionCount * 3; i++)
        {
            var idx = layout.PositionStart + i;
            dst[idx] = math.lerp(srcA[idx], srcB[idx], t);
        }

        if (layout.RotationStride == 4)
        {
            for (var e = 0; e < layout.RotationCount; e++)
            {
                var idx = layout.RotationStart + e * 4;
                var qa = new float4(srcA[idx], srcA[idx + 1], srcA[idx + 2], srcA[idx + 3]);
                var qb = new float4(srcB[idx], srcB[idx + 1], srcB[idx + 2], srcB[idx + 3]);
                if (math.dot(qa, qb) < 0f) qb = -qb;
                var lerped = math.normalizesafe(math.lerp(qa, qb, t));
                dst[idx] = lerped.x;
                dst[idx + 1] = lerped.y;
                dst[idx + 2] = lerped.z;
                dst[idx + 3] = lerped.w;
            }
        }
        else
        {
            for (var i = 0; i < layout.RotationCount * layout.RotationStride; i++)
            {
                var idx = layout.RotationStart + i;
                dst[idx] = math.lerp(srcA[idx], srcB[idx], t);
            }
        }

        for (var i = 0; i < layout.ScaleCount * 3; i++)
        {
            var idx = layout.ScaleStart + i;
            dst[idx] = math.lerp(srcA[idx], srcB[idx], t);
        }

        for (var i = 0; i < layout.VelocityCount * 3; i++)
        {
            var idx = layout.VelocityStart + i;
            dst[idx] = math.lerp(srcA[idx], srcB[idx], t);
        }

        for (var i = 0; i < layout.AngularVelocityCount * 3; i++)
        {
            var idx = layout.AngularVelocityStart + i;
            dst[idx] = math.lerp(srcA[idx], srcB[idx], t);
        }

        for (var e = 0; e < layout.BoolCount; e++)
        {
            var idx = layout.BoolStart + e;
            dst[idx] = t < 0.5f ? srcA[idx] : srcB[idx];
        }

        // Sections past the built-in six have no per-kind blend rule, so they interpolate linearly.
        for (var i = 0; i < layout.ExtraCount; i++)
        {
            var idx = layout.ExtraStart + i;
            dst[idx] = math.lerp(srcA[idx], srcB[idx], t);
        }
    }

    public static PoseBuffer Allocate(PoseLayout layout, Allocator allocator)
    {
        return new PoseBuffer
        {
            State = StateBuffer.Allocate(layout, allocator),
            Layout = layout.Data
        };
    }

    public void Dispose()
    {
        State.Dispose();
    }
}
}
