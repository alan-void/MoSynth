using System;

namespace AnimationTools
{
/// <summary>
/// Name heuristics for the shipped BVH naming convention (Hips, LeftUpLeg, LeftFoot, LeftToe,
/// LeftShoulder, LeftArm, LeftForeArm, LeftHand, ...). These are best-effort fallbacks for rigs
/// that follow this convention; they are not a substitute for an explicit
/// <see cref="SkeletonBone"/> reference where one is available.
/// </summary>
public static class BoneNameConventions
{
    /// <summary>
    /// Finds the foot-contact bone for a side: prefers a bone whose name contains "LeftToe" /
    /// "RightToe", falling back to one containing "LeftFoot" / "RightFoot". False when neither
    /// is found.
    /// </summary>
    public static bool TryFindContactBone(Skeleton skeleton, bool left, out int index)
    {
        var toeToken = left ? "LeftToe" : "RightToe";
        var footToken = left ? "LeftFoot" : "RightFoot";

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            if (skeleton.GetBone(i).Name.IndexOf(toeToken, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                index = i;
                return true;
            }
        }

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            if (skeleton.GetBone(i).Name.IndexOf(footToken, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                index = i;
                return true;
            }
        }

        index = -1;
        return false;
    }

    public static bool IsLeftArmBone(string boneName)
    {
        return IsSideArmBone(boneName, "Left");
    }

    public static bool IsRightArmBone(string boneName)
    {
        return IsSideArmBone(boneName, "Right");
    }

    private static readonly string[] ArmTokens =
    {
        "Shoulder", "Arm", "Hand", "Thumb", "Index", "Middle", "Ring", "Pinky"
    };

    // "ForeArm" matches via the "Arm" token; that is intended.
    private static bool IsSideArmBone(string boneName, string side)
    {
        if (string.IsNullOrEmpty(boneName)) return false;
        if (boneName.IndexOf(side, StringComparison.OrdinalIgnoreCase) < 0) return false;

        foreach (var token in ArmTokens)
        {
            if (boneName.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0) return true;
        }

        return false;
    }
}
}
