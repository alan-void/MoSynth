namespace AnimationTools
{
/// <summary>
/// Implemented by components that own the skeleton binding for their serialized object. Exposes
/// the edit-time-resolvable rig used by bone-picker drawers to populate a bone-name dropdown.
/// </summary>
public interface ISkeletonProvider
{
    SkeletonBoneOverrides CharacterRig { get; }
}
}
