using Unity.Mathematics;

namespace AnimationTools
{
// The six channel kinds a PoseLayout gives a named section to. Grouped in one file because they
// differ only in width and section, and reading them side by side is how you see that.

public sealed class PositionChannel : SpacedBoneChannelDescriptor
{
    public PositionChannel(int boneId, ChannelSpace space = ChannelSpace.ParentLocal,
        ChannelUsage usage = ChannelUsage.Default)
        : base(boneId, space, usage)
    {
    }

    public override int FloatCount => 3;
    public override int SectionKey => ChannelSections.Position;

    public void Set(StateBuffer buffer, StateBufferLayout layout, float3 position)
    {
        var handle = layout.BindChannel(this);
        buffer.SetFloat3(handle, position);
    }

    public float3 Get(StateBuffer buffer, StateBufferLayout layout)
    {
        var handle = layout.BindChannel(this);
        return buffer.GetFloat3(handle);
    }
}

public sealed class RotationChannel : SpacedBoneChannelDescriptor
{
    public RotationRepresentation Representation { get; }

    public RotationChannel(int boneId, RotationRepresentation representation = RotationRepresentation.Quaternion,
        ChannelSpace space = ChannelSpace.ParentLocal, ChannelUsage usage = ChannelUsage.Default)
        : base(boneId, space, usage)
    {
        Representation = representation;
    }

    public override int FloatCount => Representation == RotationRepresentation.Quaternion ? 4 : 3;
    public override int SectionKey => ChannelSections.Rotation;

    public override int GetContentHash()
    {
        unchecked
        {
            return base.GetContentHash() * 31 + (int)Representation;
        }
    }
}

public sealed class ScaleChannel : SpacedBoneChannelDescriptor
{
    public ScaleChannel(int boneId, ChannelSpace space = ChannelSpace.ParentLocal,
        ChannelUsage usage = ChannelUsage.Default)
        : base(boneId, space, usage)
    {
    }

    public override int FloatCount => 3;
    public override int SectionKey => ChannelSections.Scale;
}

public sealed class VelocityChannel : SpacedBoneChannelDescriptor
{
    public VelocityChannel(int boneId, ChannelSpace space = ChannelSpace.ParentLocal,
        ChannelUsage usage = ChannelUsage.Default)
        : base(boneId, space, usage)
    {
    }

    public override int FloatCount => 3;
    public override int SectionKey => ChannelSections.Velocity;
}

public sealed class AngularVelocityChannel : SpacedBoneChannelDescriptor
{
    /// <summary>Always RotationVector; a quaternion has no meaningful rate form.</summary>
    public RotationRepresentation Representation => RotationRepresentation.RotationVector;

    public AngularVelocityChannel(int boneId, ChannelSpace space = ChannelSpace.ParentLocal,
        ChannelUsage usage = ChannelUsage.Default)
        : base(boneId, space, usage)
    {
    }

    public override int FloatCount => 3;
    public override int SectionKey => ChannelSections.AngularVelocity;
}

/// <summary>A flag stored as one float, 0f or 1f. Has no reference frame.</summary>
public sealed class BoolChannel : BoneChannelDescriptor
{
    public BoolChannel(int boneId, ChannelUsage usage = ChannelUsage.Default)
        : base(boneId, usage)
    {
    }

    public override int FloatCount => 1;
    public override int SectionKey => ChannelSections.Bool;
}
}
