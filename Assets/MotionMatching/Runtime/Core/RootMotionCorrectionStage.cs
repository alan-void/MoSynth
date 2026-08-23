using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Turns the animation database's absolute root track into continuous world movement, so the
/// character keeps going from where it is instead of teleporting to wherever the newly selected
/// clip happens to start.
/// </summary>
/// <remarks>
/// Upstream stages write the root in the database's own "animation space", which restarts somewhere
/// arbitrary each time the search jumps clips. This stage remembers the animation-space root and the
/// world transform at one moment, then rewrites the root as that world transform plus the motion
/// accumulated since, cancelling the offset between the two spaces.
/// <para>
/// Incomplete: <see cref="_hasRootJumped"/> is cleared on the first tick and never set again, so the
/// anchor is taken once at startup. Re-taking it on
/// <see cref="MotionSynthesisComponent.PoseDiscontinuity"/> is what would make jumps seamless.
/// </para>
/// <para>Place after whatever produces the root track, before anything consuming world root motion.</para>
/// </remarks>
[Serializable]
public class RootMotionCorrectionStage : MoSynthStage
{
    Transform _root;

    MotionSynthesisComponent _owner;

    float3 _rootPosition;
    quaternion _rootRotation;

    /// <summary>Anchor is stale and must be re-taken before motion is replayed against it.</summary>
    private bool _hasRootJumped = true;

    // The anchor: the animation-space root pose and the world transform that were current at the
    // last jump. Everything since is expressed as a delta from the former, applied to the latter.
    private float3 _animSpacePos;
    private quaternion _animSpaceRot;
    private float3 _transformPosAtLastJump;
    private quaternion _transformRotAtLastJump;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _owner = motionSynthesisComponent;
        _root = motionSynthesisComponent.transform;
        _rootPosition = _root.position;
        _rootRotation = _root.rotation;
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;

        if (_hasRootJumped)
        {
            _animSpacePos = positions[0];
            _animSpaceRot = rotations[0];
            _transformPosAtLastJump = _owner.transform.position;
            _transformRotAtLastJump = _owner.transform.rotation;

            _hasRootJumped = false;
        }
        else
        {
            var newAnimSpacePos = positions[0];
            var newAnimSpaceRot = rotations[0];
            var posWrtLastJump = math.mul(math.inverse(_animSpaceRot),
                                   (newAnimSpacePos - _animSpacePos));
            var rotWrtLastJump = math.mul(math.inverse(_animSpaceRot), newAnimSpaceRot);

            positions[0] = _transformPosAtLastJump + math.mul(_transformRotAtLastJump, posWrtLastJump);
            rotations[0] = math.mul(_transformRotAtLastJump, rotWrtLastJump);
        }
        return true;
    }
}
}