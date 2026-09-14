"""Converts the Edinburgh Locomotion MOCAP Database from .npz point clouds to BVH.

    python Tools/Retargeting/edinburgh_npz_to_bvh.py \
        --npz External/LFS/edinburgh_locomotion_mocap_dataset/data/edinburgh_locomotion_test.npz \
        --out-dir Assets/LFS/Animation/Edinburgh/bvh --name edin_test --verify

The release holds no motion files: each sequence is 240 frames of 21 joint *positions* plus
three root velocity scalars, with no rotations, no hierarchy and no skeleton. Every retarget
setup in this repo consumes BVH, so the rotations have to be solved here.

Because this script authors the BVH, it also chooses the rest pose and the joint names, and it
chooses both to suit the retarget that follows: the rest is a genuine T-pose and the names are
the target rig's, so the Rokoko mapping is an identity and stage one of the retarget has nothing
left to clean. See openwiki/animation-tools/retargeting-pipeline.md for why that matters.

Needs numpy and nothing else; it does not run inside Blender or Unity.
"""

import argparse
import collections
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SOURCE_FPS = 60
OUTPUT_FPS = 30

# Holden's preprocessing scaled the captures this dataset derives from by a constant, and
# recovering it is what puts the actor at human size. At 5.644 cm per unit the thighs measure
# 40.4 cm and the hips stand 96.0 cm, against LAFAN's 43.5 cm and 91.9 cm for the same bones;
# report_scale prints both so a wrong factor is obvious rather than merely small.
DEFAULT_SCALE_CM = 5.644

# The sine of the joint angle at which a limb's bend plane is trusted completely. Below it the
# plane is increasingly meaningless -- at zero bend there is none at all -- so the roll reference
# fades to the limb's zero-twist pose instead.
BEND_PLANE_TRUST_SINE = math.sin(math.radians(30.0))

# Frames of box filtering over the roll reference. Roll is never determined by a joint position, so
# smoothing it costs no accuracy at all -- the trade is only against genuine fast roll, of which
# locomotion has almost none. Together with the trust angle these sit on a plateau: 20/5 leaves a
# worst step of 16.3 degrees, 30/9 and 45/9 both reach 13.5, and the remainder is real motion.
BEND_PLANE_SMOOTHING = 9

# How far the head may sit from directly over the ankles for a frame to count as the actor standing
# upright. Those frames, and not the whole dataset, are what the root's rest pose is averaged from:
# this actor genuinely leans forward while he walks -- his head moves out over his feet as he does
# -- so a dataset-wide mean bakes that lean into the rest and the retargeted character then spends
# every upright moment leaning backwards instead.
UPRIGHT_LEAN_DEGREES = 5.0

# A toe tip and the crown of the head are not in the data. Their End Sites continue the parent
# bone by this fraction of its length, which beats the zero-length End Sites LAFAN ships -- those
# get silently nudged into a 1 mm stub by Blender's importer.
END_SITE_FRACTION = 0.5

# Upstream's joint order (README channel table) and parent array (AnimationPlotLines.py).
SOURCE_PARENTS = (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 11, 13, 14, 15, 11, 17, 18, 19)

HIP, L_HIP, R_HIP = 0, 1, 5
L_KNEE, R_KNEE = 2, 6
L_HEEL, L_TOE, R_HEEL, R_TOE = 3, 4, 7, 8
LOWER_BACK, MID_SPINE, NECK, HEAD = 9, 10, 11, 12
L_SHOULDER, L_ELBOW, L_HAND, L_FINGER = 13, 14, 15, 16
R_SHOULDER, R_ELBOW, R_HAND, R_FINGER = 17, 18, 19, 20

# How a joint's twist is measured. A bone direction alone leaves the roll about that bone free,
# and roll is the one thing positions cannot tell us; each of these is the second, independent
# direction that pins it down.
HIP_LINE, SHOULDER_LINE = "hip-line", "shoulder-line"
PELVIS_UP, CHEST_UP = "pelvis-up", "chest-up"
L_SHIN, R_SHIN = "left-shin", "right-shin"
# Not a measurement at all: the bone takes its parent's roll, so it cannot twist against it.
PARENT_ROLL = "parent-roll"
L_KNEE_PLANE, R_KNEE_PLANE = "left-knee-plane", "right-knee-plane"
L_ELBOW_PLANE, R_ELBOW_PLANE = "left-elbow-plane", "right-elbow-plane"

# The three joints each bend plane is measured across.
BEND_PLANE_JOINTS = {}    # filled in below, once the joint indices above are all named

# The character's own anatomy written in BVH axes. Blender's importer, with the axis_forward="Z"
# and axis_up="Y" flags batch_retarget uses, lands BVH (x, y, z) at Blender (-x, z, y) -- measured
# against a real import, not read off the flag names. The target rig rests with its left on Blender
# +X and its face on Blender -Y, so matching it puts the character's left along BVH -X and its face
# along BVH -Z. The 180 degrees this invites is worth getting right: Rokoko takes orientation from
# each rig's own rest pose while the root translation is copied straight through, so a rest facing
# the wrong way leaves the character walking backwards along a correct path.
LEFT, RIGHT = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
UP, DOWN = (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)
FORWARD = (0.0, 0.0, -1.0)

# One row per bone of the BVH we write, in the depth-first order BVH declares them.
#   source     the Edinburgh joint this bone's head sits on
#   parent     index into this table, -1 for the root
#   aim        the Edinburgh joint whose position gives this bone its direction; None for a leaf,
#              which inherits its parent's frame and so keeps an identity local rotation
#   rest_dir   the direction `aim` lies in once the character is in the rest T-pose, or None to
#              take the direction the actor actually holds the bone at, averaged over the dataset.
#              Limbs are canonical, because arms out and legs down is what a T-pose *means* and
#              the target rig agrees. A spine is not: forcing its segments straight makes a rest
#              no human stands in, and a rotation retarget then adds the whole difference between
#              that and the target's own spinal curve on top of every pose. LAFAN's rig carries
#              about 15 degrees of curve, which arrived as a hunched chest and a jutting head
#   reference  which measurement pins the roll
#   rest_ref   the direction that measurement points in once the character is in the rest T-pose;
#              the bend-plane entries are anatomy, and validate_joint_bends is what checks them
#   end_site   the Edinburgh joint that terminates the chain, or None to synthesise a tip
Bone = collections.namedtuple(
    "Bone", "name source parent aim rest_dir reference rest_ref end_site")

# The hip joints and the clavicles are rotating segments in the source, not fixed offsets: the
# distance across the hip joints swings by 7.9 cm and across the shoulders by 3.9 cm, while every
# parent-child bone length is constant to the last digit. A joint whose children move
# independently cannot be one BVH joint, so each of those four gets a bone of its own, sitting on
# its parent at a zero offset and carrying the rotation that aims it. Without them the whole leg
# hangs off a fixed pelvis offset and trails the source by up to 4 cm.
SKELETON = (
    Bone("Hips",          HIP,        -1, LOWER_BACK, None,    HIP_LINE,      LEFT,  None),
    Bone("LeftHipJoint",  HIP,         0, L_HIP,      None,    PARENT_ROLL,   None,  None),
    Bone("LeftUpLeg",     L_HIP,       1, L_KNEE,     DOWN,    L_KNEE_PLANE,  LEFT,  None),
    Bone("LeftLeg",       L_KNEE,      2, L_HEEL,     DOWN,    L_KNEE_PLANE,  LEFT,  None),
    Bone("LeftFoot",      L_HEEL,      3, L_TOE,      FORWARD, L_SHIN,        DOWN,  None),
    Bone("LeftToe",       L_TOE,       4, None,       None,    None,          None,  None),
    Bone("RightHipJoint", HIP,         0, R_HIP,      None,    PARENT_ROLL,   None,  None),
    Bone("RightUpLeg",    R_HIP,       6, R_KNEE,     DOWN,    R_KNEE_PLANE,  LEFT,  None),
    Bone("RightLeg",      R_KNEE,      7, R_HEEL,     DOWN,    R_KNEE_PLANE,  LEFT,  None),
    Bone("RightFoot",     R_HEEL,      8, R_TOE,      FORWARD, R_SHIN,        DOWN,  None),
    Bone("RightToe",      R_TOE,       9, None,       None,    None,          None,  None),
    Bone("Spine",         LOWER_BACK,  0, MID_SPINE,  None,    HIP_LINE,      LEFT,  None),
    Bone("Spine1",        MID_SPINE,  11, NECK,       None,    SHOULDER_LINE, LEFT,  None),
    Bone("Neck",          NECK,       12, HEAD,       None,    SHOULDER_LINE, LEFT,  None),
    Bone("Head",          HEAD,       13, None,       None,    None,          None,  None),
    Bone("LeftShoulder",  NECK,       13, L_SHOULDER, None,    PARENT_ROLL,   None,  None),
    Bone("LeftArm",       L_SHOULDER, 15, L_ELBOW,    LEFT,    L_ELBOW_PLANE, DOWN,  None),
    Bone("LeftForeArm",   L_ELBOW,    16, L_HAND,     LEFT,    L_ELBOW_PLANE, DOWN,  None),
    Bone("LeftHand",      L_HAND,     17, L_FINGER,   LEFT,    L_ELBOW_PLANE, DOWN,  L_FINGER),
    Bone("RightShoulder", NECK,       13, R_SHOULDER, None,    PARENT_ROLL,   None,  None),
    Bone("RightArm",      R_SHOULDER, 19, R_ELBOW,    RIGHT,   R_ELBOW_PLANE, UP,    None),
    Bone("RightForeArm",  R_ELBOW,    20, R_HAND,     RIGHT,   R_ELBOW_PLANE, UP,    None),
    Bone("RightHand",     R_HAND,     21, R_FINGER,   RIGHT,   R_ELBOW_PLANE, UP,    R_FINGER),
)

BONE_INDEX = {bone.name: i for i, bone in enumerate(SKELETON)}

BEND_PLANE_JOINTS.update({
    L_KNEE_PLANE: (L_HIP, L_KNEE, L_HEEL),
    R_KNEE_PLANE: (R_HIP, R_KNEE, R_HEEL),
    L_ELBOW_PLANE: (L_SHOULDER, L_ELBOW, L_HAND),
    R_ELBOW_PLANE: (R_SHOULDER, R_ELBOW, R_HAND),
})

# What the two rigs cannot agree on, recorded here because this is where the shortfall originates;
# make_edinburgh_setup.py holds the copy Rokoko is configured against. The target's Spine2 rides at
# rest because the source bends its back in two places, not three -- the same compromise the Bandai
# setup makes with Spine1. The hip-joint segments have no counterpart at all: the target rig hangs
# its thighs off a rigid pelvis, so their rotation is simply dropped.
UNMAPPED_TARGET_BONES = ("Spine2",)
UNMAPPED_SOURCE_BONES = ("LeftHipJoint", "RightHipJoint")


class ConversionError(Exception):
    """The data is not shaped the way this converter expects."""


def log(message):
    print("[edinburgh] " + message, flush=True)


def resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


# --- vector helpers -------------------------------------------------------------------


def normalise(v, axis=-1):
    length = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(length, 1e-12)


def orthonormal_frames(primary, reference):
    """Columns [u, v, u x v], with `reference` orthogonalised against `u`.

    Two measured directions make a rotation: the first fixes where the bone points, the second
    fixes the roll about it, which is the part joint positions alone never determine.
    """
    u = normalise(primary)
    v = normalise(reference - u * np.sum(reference * u, axis=-1, keepdims=True))
    return np.stack((u, v, np.cross(u, v)), axis=-1)


def matrices_to_zyx_degrees(matrices):
    """Decomposes R = Rz @ Ry @ Rx, which is how Blender reads `Zrotation Yrotation Xrotation`."""
    ry = np.arcsin(np.clip(-matrices[..., 2, 0], -1.0, 1.0))
    rx = np.arctan2(matrices[..., 2, 1], matrices[..., 2, 2])
    rz = np.arctan2(matrices[..., 1, 0], matrices[..., 0, 0])
    return np.degrees(np.stack((rx, ry, rz), axis=-1))


def zyx_degrees_to_matrices(angles):
    rx, ry, rz = (np.radians(angles[..., i]) for i in range(3))
    zero, one = np.zeros_like(rx), np.ones_like(rx)

    def stack(rows):
        return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)

    mx = stack(((one, zero, zero),
                (zero, np.cos(rx), -np.sin(rx)),
                (zero, np.sin(rx), np.cos(rx))))
    my = stack(((np.cos(ry), zero, np.sin(ry)),
                (zero, one, zero),
                (-np.sin(ry), zero, np.cos(ry))))
    mz = stack(((np.cos(rz), -np.sin(rz), zero),
                (np.sin(rz), np.cos(rz), zero),
                (zero, zero, one)))
    return mz @ my @ mx


# --- loading and world reconstruction -------------------------------------------------


def load_clips(npz_path):
    with np.load(npz_path) as data:
        if "clips" not in data:
            raise ConversionError("{} has no 'clips' array; found {}."
                                  .format(npz_path, ", ".join(sorted(data.files))))
        clips = data["clips"]
    if clips.ndim != 3:
        raise ConversionError("Expected a 3D clips array; got {}.".format(clips.shape))
    # On disk the layout is already (clip, frame, channel). Upstream reaches the same place by
    # swapping the last two axes in animate_locomotion_data.py and swapping them back inside
    # AnimationPlotLines.animation_plot, so neither call says anything about the file.
    if clips.shape[2] != 66:
        clips = np.swapaxes(clips, 1, 2)
    if clips.shape[2] != 66:
        raise ConversionError("Expected 66 channels on one axis; got {}.".format(clips.shape))
    return clips.astype(np.float64)


def reconstruct_world(clip):
    """Integrates the three root channels back into world-space joint positions.

    The joint block is expressed in the character's own frame with ground translation and yaw
    removed, and the last three channels are the per-frame deltas that were taken out. The update
    order is AnimationPlotLines.animation_plot's, which is the only authority on it: a frame is
    placed with the pose accumulated so far, and only then does the root advance.

    Take the channel assignment from upstream's code and not from its README, which labels channel
    63 "Forward Velocity" while the code spends it on the lateral axis. The code is right, and the
    planted foot is what says so: reconstructed this way a stance toe slides 5.6 cm/s, against
    72 cm/s with no root motion at all and 111 cm/s with the two channels swapped to match the
    README. Do not "fix" this against the documentation.
    """
    frames = clip.shape[0]
    local = clip[:, :63].reshape(frames, 21, 3)
    root_x, root_z, root_r = clip[:, 63], clip[:, 64], clip[:, 65]

    world = np.empty_like(local)
    angle = 0.0
    offset = np.zeros(2)
    for i in range(frames):
        cos, sin = math.cos(angle), math.sin(angle)
        world[i] = local[i] @ np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]]).T
        world[i, :, 0] += offset[0]
        world[i, :, 2] += offset[1]

        angle -= root_r[i]
        cos, sin = math.cos(angle), math.sin(angle)
        offset += np.array([cos * root_x[i] + sin * root_z[i],
                            cos * root_z[i] - sin * root_x[i]])
    return world


def detect_orientation(local_positions):
    """Finds the yaw that lays the character out along the LEFT and FORWARD axes above.

    The release documents neither of its horizontal axes, and the whole rest pose is built on
    them: with the lateral axis guessed wrong the T-pose comes out with its arms crossed. Only the
    four quarter turns are considered -- a reflection would mirror the motion rather than re-label
    it, and would turn every left limb into a right one.
    """
    lateral = np.mean(local_positions[:, L_HIP] - local_positions[:, R_HIP], axis=0)
    facing = np.mean((local_positions[:, L_TOE] - local_positions[:, L_HEEL])
                     + (local_positions[:, R_TOE] - local_positions[:, R_HEEL]), axis=0)

    best, best_score, best_quarter = np.eye(3), -np.inf, 0
    for quarter in range(4):
        cos, sin = math.cos(quarter * math.pi / 2), math.sin(quarter * math.pi / 2)
        yaw = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
        score = float(np.dot(normalise(yaw @ lateral), LEFT))
        if score > best_score:
            best, best_score, best_quarter = yaw, score, quarter

    forward = float(np.dot(normalise(best @ facing), FORWARD))
    if forward < 0:
        raise ConversionError(
            "With the character's left on the lateral axis its feet point backwards ({:.2f}), so "
            "the data is mirrored relative to the target rig and no yaw can fix it.".format(forward))
    log("orientation: a {} degree turn puts left {:.2f} along the lateral axis and the feet {:.2f} "
        "along forward".format(best_quarter * 90, best_score, forward))
    return best


# --- skeleton fitting -----------------------------------------------------------------


def bend_plane(positions, root, middle, tip):
    """The plane a two-bone chain bends in: its normal, and how far that normal can be trusted.

    A straight limb has no bend plane, and the cross product there is noise, so the trust fades to
    zero as the joint straightens. Holding the last good normal instead is worse than it looks: it
    freezes a world-space direction while the body keeps turning, so the longer the limb stays
    straight the further the held vector drifts, and the whole difference then lands in the single
    frame the measurement resumes on -- 18 degrees of knee roll in one frame, on a leg that was
    merely below a threshold for eight.
    """
    upper = normalise(positions[:, middle] - positions[:, root])
    lower = normalise(positions[:, tip] - positions[:, middle])
    normals = np.cross(upper, lower)
    trust = np.clip(np.linalg.norm(normals, axis=-1, keepdims=True) / BEND_PLANE_TRUST_SINE,
                    0.0, 1.0)
    return normalise(normals), trust * trust * (3.0 - 2.0 * trust)


def slerp_directions(start, end, amount):
    """Unit directions interpolated along the shorter arc between them."""
    cosine = np.clip(np.sum(start * end, axis=-1, keepdims=True), -1.0, 1.0)
    angle = np.arccos(cosine)
    sine = np.sin(angle)
    arc = (np.sin((1.0 - amount) * angle) * start + np.sin(amount * angle) * end) / np.maximum(sine, 1e-9)
    # Where the two nearly coincide the arc is ill-conditioned and a straight blend is exact enough.
    return normalise(np.where(sine < 1e-6, start + amount * (end - start), arc))


def smooth_directions(directions):
    """A short box filter over a direction field, renormalised.

    Femoral and humeral roll change slowly; what does not is the estimate of them, when a limb is
    near enough straight that its bend plane is mostly noise. Averaging over BEND_PLANE_SMOOTHING
    frames removes that without touching sustained roll -- and it cannot move a joint, because roll
    about a bone leaves every joint position exactly where it was.
    """
    if BEND_PLANE_SMOOTHING < 2:
        return directions
    pad = BEND_PLANE_SMOOTHING // 2
    padded = np.concatenate([directions[:1]] * pad + [directions] + [directions[-1:]] * pad)
    window = [padded[i:i + len(directions)] for i in range(BEND_PLANE_SMOOTHING)]
    return normalise(np.mean(window, axis=0))


def rotate_between(source, targets, vector):
    """`vector` turned by the shortest rotation carrying unit `source` onto each unit target.

    Every argument broadcasts, so `source` may vary per frame as well -- which is what carrying a
    parent bone's own axis onto its child needs.
    """
    source, targets = np.broadcast_arrays(np.asarray(source, dtype=float), targets)
    axis = np.cross(source, targets)
    sine = np.linalg.norm(axis, axis=-1, keepdims=True)
    cosine = np.sum(source * targets, axis=-1, keepdims=True)
    axis = np.where(sine > 1e-9, axis / np.maximum(sine, 1e-9), source)
    return (vector * cosine + np.cross(axis, vector) * sine
            + axis * np.sum(axis * vector, axis=-1, keepdims=True) * (1.0 - cosine))


def carried_parent_roll(parent_frames, directions):
    """The parent's own roll axis, carried onto a child pointing `directions`.

    A bone built against this has zero twist relative to its parent, by construction. That is the
    honest convention for a segment whose roll nothing measures -- a clavicle reaching one shoulder
    marker, or a hip joint reaching one hip marker. Borrowing a neighbour's direction instead lets
    the segment twist against the very body part it is rigidly attached to: measured on the source,
    the clavicles drifted 28 to 35 degrees against the chest over a single clip, and the hip joints
    38 against the pelvis, symmetric and opposite left to right.
    """
    return rotate_between(parent_frames[..., 0], directions, parent_frames[..., 1])


def zero_twist_reference(directions, bone, body):
    """Where a bone's roll reference sits when the bone carries no roll of its own.

    The rest reference, carried onto the bone's current direction by the shortest rotation that
    gets there, so it follows the limb. A body-fixed axis is only a good guess while a bone lies
    near its rest direction: a thigh does, a swinging arm does not -- during a stride the arms hang
    about 90 degrees from the T-pose, and a body-fixed fallback drags their roll a quarter turn off.
    """
    local = np.einsum("fba,fb->fa", body, directions)
    carried = rotate_between(np.array(bone.rest_dir), local, np.array(bone.rest_ref))
    return np.einsum("fab,fb->fa", body, carried)


def root_rest_frame(positions):
    """The root's rest frame: the pelvis the actor actually stands with, not a vertical line.

    Every other bone derives its rest direction against its parent, and the root has none -- which
    is why this one was left canonical at first, and why the whole character came out pitched
    forward. The actor's pelvis-up axis sits about 17 degrees ahead of vertical, so calling
    vertical its rest made that 17 degrees a permanent forward lean on the retargeted character.

    Measuring it against the world vertical and the hip line gives a frame that does not depend on
    which way the character happens to be facing, and so a mean that means something.
    """
    root = SKELETON[0]
    if root.rest_dir is not None:
        return orthonormal_frames(np.array(root.rest_dir), np.array(root.rest_ref))

    hip_line = normalise(positions[:, L_HIP] - positions[:, R_HIP])
    pelvis_up = normalise(positions[:, root.aim] - positions[:, root.source])
    heading = orthonormal_frames(np.broadcast_to(np.array(UP), hip_line.shape), hip_line)
    local = np.einsum("fba,fb->fa", heading, pelvis_up)

    upright = standing_upright(positions)
    if np.count_nonzero(upright) > len(positions) // 100:
        local = local[upright]
    mean = np.mean(local, axis=0)
    canonical = orthonormal_frames(np.array(UP), np.array(LEFT))
    return orthonormal_frames(normalise(canonical @ mean), np.array(root.rest_ref))


def standing_upright(positions):
    """Frames where the actor's head sits over his ankles, whatever his pelvis markers say.

    The pelvis-to-lower-back axis reads about 2.4 times the actor's real lean -- the two markers
    are 11 cm apart and sample the spine differently from the target rig -- so it cannot say by
    itself when he is upright. Where his head is relative to his feet can.
    """
    ankles = (positions[:, L_HEEL] + positions[:, R_HEEL]) / 2.0
    lean = positions[:, HEAD] - ankles
    height = lean @ np.array(UP)
    horizontal = np.linalg.norm(lean - np.outer(height, np.array(UP)), axis=-1)
    return horizontal < math.tan(math.radians(UPRIGHT_LEAN_DEGREES)) * np.maximum(height, 1e-6)


def measure_frames(positions):
    """An orthonormal frame per bone per frame, built only from measured directions."""
    hip_line = normalise(positions[:, L_HIP] - positions[:, R_HIP])
    pelvis_up = normalise(positions[:, LOWER_BACK] - positions[:, HIP])

    body = orthonormal_frames(pelvis_up, hip_line) @ root_rest_frame(positions).T

    direct = {
        HIP_LINE: hip_line,
        SHOULDER_LINE: normalise(positions[:, L_SHOULDER] - positions[:, R_SHOULDER]),
        # The hip joints and the clavicles run sideways, so the lines across them are too nearly
        # parallel to serve as their own roll reference; the trunk is what they are measured against.
        PELVIS_UP: pelvis_up,
        CHEST_UP: normalise(positions[:, HEAD] - positions[:, NECK]),
        # A foot's roll is pinned by its own shin, not by the knee's bend plane. The knee plane is
        # undefined exactly when the leg is straight, and nothing keeps it off the foot's own axis
        # when it is noisy: measured that way a foot swung through 163 degrees in one frame while
        # its perpendicular component collapsed to 0.19. The ankle hinges perpendicular to the
        # shin, which is always well defined and never parallel to the foot.
        L_SHIN: normalise(positions[:, L_HEEL] - positions[:, L_KNEE]),
        R_SHIN: normalise(positions[:, R_HEEL] - positions[:, R_KNEE]),
    }
    planes = {kind: bend_plane(positions, *joints) for kind, joints in BEND_PLANE_JOINTS.items()}

    frames = np.empty((positions.shape[0], len(SKELETON), 3, 3))
    for i, bone in enumerate(SKELETON):
        if bone.aim is None:
            frames[:, i] = frames[:, bone.parent]
            continue
        direction = normalise(positions[:, bone.aim] - positions[:, bone.source])

        if bone.reference == PARENT_ROLL:
            reference = carried_parent_roll(frames[:, bone.parent], direction)
        elif bone.reference in planes:
            normal, trust = planes[bone.reference]
            rest = zero_twist_reference(direction, bone, body)
            # Interpolated along the arc between the two, never by averaging them: a linear blend
            # of two opposed directions cancels to no length at all, and a reference of no length
            # points anywhere. Slerp also keeps a fully-trusted measurement exactly as measured,
            # which matters -- an earlier version down-weighted the plane wherever it disagreed
            # with the rest guess, and that alone flipped a well-bent elbow through 123 degrees in
            # a single frame.
            reference = smooth_directions(slerp_directions(rest, normal, trust))
        else:
            reference = direct[bone.reference]

        frames[:, i] = orthonormal_frames(direction, reference)
    return frames


def rest_frames(positions, frames):
    """Per bone, the frame its measured directions occupy once the character is in the T-pose.

    A bone's world rotation on any frame is then `measured_frame @ rest_frame.T`, which is the
    identity at rest -- so the rest pose the BVH declares really is the T-pose, and the retarget
    downstream gets the calibration pose it needs rather than a BVH's usual degenerate one.

    Bones run parents-first, because a segment with no canonical rest direction takes the mean one
    the actor holds it at, and that has to be read in the parent's rest frame.
    """
    rest = np.empty((len(SKELETON), 3, 3))
    for i, bone in enumerate(SKELETON):
        if bone.parent < 0:
            rest[i] = root_rest_frame(positions)
        elif bone.aim is None:
            rest[i] = rest[bone.parent]
        elif bone.rest_dir is not None:
            rest[i] = orthonormal_frames(np.array(bone.rest_dir), np.array(bone.rest_ref))
        else:
            mean = np.mean(np.einsum("fba,fb->fa", frames[:, bone.parent],
                                     positions[:, bone.aim] - positions[:, bone.source]), axis=0)
            direction = normalise(rest[bone.parent] @ mean)
            reference = (carried_parent_roll(rest[bone.parent], direction)
                         if bone.reference == PARENT_ROLL else np.array(bone.rest_ref))
            rest[i] = orthonormal_frames(direction, reference)
    return rest


Skeleton = collections.namedtuple("Skeleton", "rest offsets end_sites spread")


def fit_skeleton(positions, frames):
    """The one rest pose every converted clip shares: a T-pose with the actor's own proportions.

    Offsets are the mean over the whole dataset and never per clip, because
    batch_retarget.check_skeleton_matches rejects any BVH whose bone lengths differ from the setup
    blend's by more than a millimetre. The spread reported alongside is how well one rigid
    skeleton represents the data at all.
    """
    rest = rest_frames(positions, frames)
    offsets, end_sites, spread = {}, {}, {}

    def local_offset(frame_index, from_joint, to_joint):
        local = np.einsum("fba,fb->fa", frames[:, frame_index],
                          positions[:, to_joint] - positions[:, from_joint])
        mean = np.mean(local, axis=0)
        # How far the offset wanders from its mean *as a vector*, not merely how much its length
        # changes: a constant-length offset that swings within the frame is the whole of the
        # forward-kinematics error, and a length-only measure reads zero on it.
        wander = float(np.mean(np.linalg.norm(local - mean, axis=-1)))
        return rest[frame_index] @ mean, wander

    for bone in SKELETON:
        if bone.parent < 0:
            continue
        parent = SKELETON[bone.parent]
        offsets[bone.name], spread[bone.name] = local_offset(bone.parent, parent.source,
                                                             bone.source)

    for i, bone in enumerate(SKELETON):
        if bone.end_site is not None:
            end_sites[bone.name], spread[bone.name + " End Site"] = local_offset(
                i, bone.source, bone.end_site)
        elif bone.aim is None:
            # A toe tip and the crown of the head were never captured, so continue the parent bone.
            end_sites[bone.name] = offsets[bone.name] * END_SITE_FRACTION

    return Skeleton(rest, offsets, end_sites, spread)


def rest_pose(skeleton):
    """Where every joint sits with all rotations at identity -- the pose the BVH declares."""
    positions = np.zeros((len(SKELETON), 3))
    for i, bone in enumerate(SKELETON):
        if bone.parent >= 0:
            positions[i] = positions[bone.parent] + skeleton.offsets[bone.name]
    return positions


def validate_rest_pose(skeleton):
    """Checks the declared rest really is a T-pose, which catches a mislabelled axis.

    Positions cannot see roll, so this is only half the check; validate_joint_bends is the other.
    """
    pose = rest_pose(skeleton)
    hips, head = pose[BONE_INDEX["Hips"]], pose[BONE_INDEX["Head"]]
    left, right = pose[BONE_INDEX["LeftHand"]], pose[BONE_INDEX["RightHand"]]
    toe = pose[BONE_INDEX["LeftToe"]]

    problems = []
    if head[1] <= hips[1]:
        problems.append("the head is not above the hips")
    if toe[1] >= hips[1]:
        problems.append("the toes are not below the hips")
    if np.dot(left, LEFT) <= 0 or np.dot(right, RIGHT) <= 0:
        problems.append("the hands are not on their own sides of the body")
    if problems:
        raise ConversionError("The rest pose is not a T-pose: " + "; ".join(problems)
                              + ". An axis is mislabelled.")

    mirror = max(abs(left[0] + right[0]), abs(left[1] - right[1]), abs(left[2] - right[2]))
    log("rest pose: {:.1f} cm tall, arm span {:.1f} cm, left/right asymmetry {:.2f} cm"
        .format(float(head[1] - toe[1]), float(np.linalg.norm(left - right)), float(mirror)))


def validate_joint_bends(positions, frames, skeleton):
    """Checks that knees bend backwards and elbows forwards.

    The roll about a bone is the one thing the point cloud does not determine, so each limb's rest
    roll is anatomy written into SKELETON.rest_ref. Get one of those signs wrong and nothing here
    moves by a millimetre -- the limb simply comes out of the retarget inside out. These two facts
    are what the guess is worth checking against.
    """
    world = frames @ skeleton.rest.transpose(0, 2, 1)

    def bend(parent_bone, joint, tip, expected_sign, description):
        # Undoing the parent's rotation puts the limb back where it would sit with the character in
        # its rest pose, so which way it leans can be read straight off the forward axis.
        parent = BONE_INDEX[parent_bone]
        direction = normalise(positions[:, tip] - positions[:, joint])
        deposed = np.einsum("fba,fb->fa", world[:, parent], direction)
        forward = float(np.mean(deposed @ np.array(FORWARD)))
        if forward * expected_sign <= 0:
            raise ConversionError(
                "{} by {:+.3f} along its own forward axis, the wrong way. The rest roll in "
                "SKELETON.rest_ref for {} has the wrong sign.".format(description, forward,
                                                                      parent_bone))
        return forward

    log("joint bends: knees {:+.3f}/{:+.3f} back, elbows {:+.3f}/{:+.3f} forward".format(
        bend("LeftUpLeg", L_KNEE, L_HEEL, -1, "The left shin leans"),
        bend("RightUpLeg", R_KNEE, R_HEEL, -1, "The right shin leans"),
        bend("LeftArm", L_ELBOW, L_HAND, +1, "The left forearm leans"),
        bend("RightArm", R_ELBOW, R_HAND, +1, "The right forearm leans")))


# --- per-clip solve and BVH output -----------------------------------------------------


def solve_channels(positions, frames, skeleton):
    """Root world positions plus a ZYX euler per bone per frame."""
    world = frames @ skeleton.rest.transpose(0, 2, 1)

    local = np.empty_like(world)
    for i, bone in enumerate(SKELETON):
        if bone.parent < 0:
            local[:, i] = world[:, i]
        else:
            local[:, i] = np.einsum("fba,fbc->fac", world[:, bone.parent], world[:, i])

    angles = matrices_to_zyx_degrees(local)
    residual = float(np.max(np.abs(zyx_degrees_to_matrices(angles) - local)))
    if residual > 1e-4:
        raise ConversionError(
            "A ZYX euler does not reproduce its own rotation matrix (residual {:.2e}), so a bone "
            "is sitting on that order's gimbal singularity. Write the file with the ZXY channel "
            "order instead.".format(residual))
    return positions[:, HIP], angles


def write_bvh(path, skeleton, root_positions, angles):
    children = collections.defaultdict(list)
    for i, bone in enumerate(SKELETON):
        if bone.parent >= 0:
            children[bone.parent].append(i)

    def offset(vector):
        return "OFFSET {:.6f} {:.6f} {:.6f}".format(*vector)

    lines = ["HIERARCHY"]

    def declare(index, depth):
        bone = SKELETON[index]
        pad = "\t" * depth
        if bone.parent < 0:
            lines.append(pad + "ROOT " + bone.name)
            lines.append(pad + "{")
            lines.append(pad + "\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(pad + "\tCHANNELS 6 Xposition Yposition Zposition "
                               "Zrotation Yrotation Xrotation")
        else:
            lines.append(pad + "JOINT " + bone.name)
            lines.append(pad + "{")
            lines.append(pad + "\t" + offset(skeleton.offsets[bone.name]))
            lines.append(pad + "\tCHANNELS 3 Zrotation Yrotation Xrotation")

        for child in children[index]:
            declare(child, depth + 1)
        if bone.name in skeleton.end_sites:
            lines.append(pad + "\tEnd Site")
            lines.append(pad + "\t{")
            lines.append(pad + "\t\t" + offset(skeleton.end_sites[bone.name]))
            lines.append(pad + "\t}")
        lines.append(pad + "}")

    declare(0, 0)
    lines.append("MOTION")
    lines.append("Frames: {}".format(len(root_positions)))
    lines.append("Frame Time: {:.7f}".format(1.0 / OUTPUT_FPS))

    # Channels run position (root only) then Z, Y, X rotation; the solver returns eulers as
    # X, Y, Z, so each joint's triple is emitted reversed.
    rows = np.concatenate((root_positions, angles[:, :, ::-1].reshape(len(angles), -1)), axis=1)
    lines.extend(" ".join("{:.6f}".format(value) for value in row) for row in rows)

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


# --- reporting -------------------------------------------------------------------------


def forward_kinematics(skeleton, root_positions, angles):
    """Re-derives joint positions from what was written, so a file can be checked against the
    point cloud it came from rather than against the solver's own intermediates."""
    local = zyx_degrees_to_matrices(angles)
    rotation = np.empty_like(local)
    position = np.empty((len(root_positions), len(SKELETON), 3))

    for i, bone in enumerate(SKELETON):
        if bone.parent < 0:
            rotation[:, i], position[:, i] = local[:, i], root_positions
            continue
        rotation[:, i] = rotation[:, bone.parent] @ local[:, i]
        position[:, i] = (position[:, bone.parent]
                          + rotation[:, bone.parent] @ skeleton.offsets[bone.name])
    return position


def worst_bone_step(skeleton, frames):
    """The largest frame-to-frame rotation on any bone, as (degrees, twist of it, bone, frame).

    Joint positions can be perfect while a limb spins about itself, because roll is the half of the
    solve the point cloud does not determine -- so this is the only automatic warning of a roll
    reference that moved when the character did not. Measure the whole rotation and not just its
    twist: an earlier version reported twist alone and read a clean 13.6 degrees while a hand was
    visibly snapping through 125, because most of that was swing.
    """
    world = frames @ skeleton.rest.transpose(0, 2, 1)
    worst = (0.0, 0.0, "none", 0)
    for i, bone in enumerate(SKELETON):
        delta = np.einsum("fba,fbc->fac", world[:-1, i], world[1:, i])
        cosine = np.clip((np.trace(delta, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))

        step = int(np.argmax(angle))
        if angle[step] > worst[0]:
            twist = abs(float(np.degrees(twist_about(delta[step:step + 1],
                                                     skeleton.rest[i][:, 0])[0])))
            worst = (float(angle[step]), twist, bone.name, step + 2)
    return worst


def twist_about(rotations, axis):
    """The twist half of each rotation's swing-twist decomposition about `axis`, in radians.

    Rolling a seed vector and measuring how far it turned would be simpler and wrong: it reads a
    bone's swing as twist whenever the swing is large, which for a thigh at a full stride is most
    of the time.
    """
    trace = np.trace(rotations, axis1=-2, axis2=-1)
    w = np.sqrt(np.maximum(0.0, 1.0 + trace)) / 2.0
    # Frame-to-frame rotations are far from 180 degrees, where this form would lose precision.
    scale = 1.0 / np.maximum(4.0 * w, 1e-9)
    vector = np.stack((rotations[..., 2, 1] - rotations[..., 1, 2],
                       rotations[..., 0, 2] - rotations[..., 2, 0],
                       rotations[..., 1, 0] - rotations[..., 0, 1]), axis=-1) * scale[..., None]
    along = vector @ axis
    return 2.0 * np.arctan2(along, w)


def report_accuracy(name, skeleton, positions, root_positions, angles):
    reconstructed = forward_kinematics(skeleton, root_positions, angles)
    source = np.stack([positions[:, bone.source] for bone in SKELETON], axis=1)
    error = np.linalg.norm(reconstructed - source, axis=-1)
    worst = int(np.argmax(np.max(error, axis=0)))

    # A wrong velocity accumulation makes the character travel at the wrong speed while its legs
    # keep their stride, which no position comparison against the same reconstruction would see.
    toe = reconstructed[:, BONE_INDEX["LeftToe"]]
    planted = toe[:-1, 1] < np.percentile(toe[:, 1], 25.0)
    skate = np.linalg.norm(np.diff(toe[:, ::2], axis=0), axis=-1) * OUTPUT_FPS

    step, twist, step_bone, step_frame = worst_bone_step(skeleton, measure_frames(positions))
    log("{}: joint error mean {:.4f} cm, max {:.4f} cm at {}; planted toe slides {:.1f} cm/s; "
        "worst bone step {:.1f} deg ({:.1f} twist) at {} frame {}"
        .format(name, float(np.mean(error)), float(np.max(error)), SKELETON[worst].name,
                float(np.median(skate[planted])) if np.any(planted) else float("nan"),
                step, twist, step_bone, step_frame))
    return float(np.mean(error)), float(np.max(error)), step, step_bone, step_frame


def report_scale(positions, scale):
    """Reads the scaled actor back in anthropometric terms, so a wrong factor is obvious.

    Hip height and leg length are the two rigid anchors worth trusting; the reduced 21-joint
    marker set puts its head joint at the base of the skull, so overall stature reads short and
    says nothing useful about the scale.
    """
    hip_height = float(np.median(positions[:, HIP, 1]))
    thigh = float(np.median(np.linalg.norm(positions[:, L_KNEE] - positions[:, L_HIP], axis=-1)))
    shin = float(np.median(np.linalg.norm(positions[:, L_HEEL] - positions[:, L_KNEE], axis=-1)))
    log("scale {:g} cm per unit: hips stand {:.1f} cm, thigh {:.1f} cm, shin {:.1f} cm "
        "(LAFAN's actor: 91.9, 43.5, 42.4)".format(scale, hip_height, thigh, shin))


def report_skeleton(skeleton):
    log("rest skeleton, {} bones:".format(len(SKELETON)))
    for bone in SKELETON:
        if bone.parent < 0:
            continue
        offset = skeleton.offsets[bone.name]
        log("  {:<13} {:>6.2f} cm   offset {:>7.2f} {:>7.2f} {:>7.2f}   wander {:.3f} cm"
            .format(bone.name, float(np.linalg.norm(offset)), *offset,
                    skeleton.spread[bone.name]))
    worst, value = max(skeleton.spread.items(), key=lambda item: item[1])
    log("widest offset wander across the dataset: {} at {:.3f} cm -- the cost of representing this "
        "actor with one rigid skeleton, and it lands in the joint error below".format(worst, value))


def report_padding(clips):
    """The release pads short sequences by repeating their first and last frame, and a frozen pose
    is exactly what a motion matching database will happily match against."""
    joints = clips[:, :, :63]
    leading = np.sum(np.all(np.isclose(joints[:, 1:], joints[:, :1], atol=1e-6), axis=2), axis=1)
    trailing = np.sum(np.all(np.isclose(joints[:, :-1], joints[:, -1:], atol=1e-6), axis=2), axis=1)
    padded = int(np.count_nonzero((leading > 0) | (trailing > 0)))
    log("padding: {} of {} clips repeat a frame at an end; worst run {} leading, {} trailing"
        .format(padded, len(clips), int(np.max(leading)), int(np.max(trailing))))


def report_overlap(clips):
    """Holden's preprocessing cut long takes into overlapping windows; where it did, consecutive
    clips share frames and the database fills with near-duplicate poses."""
    strides = collections.Counter()
    for i in range(min(len(clips) - 1, 200)):
        distances = np.linalg.norm(clips[i, :, :63] - clips[i + 1, 0, :63], axis=-1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < 1e-4:
            strides[nearest] += 1
    if not strides:
        log("overlap: no consecutive clip pair shares a frame; the windows look independent")
        return
    stride, count = strides.most_common(1)[0]
    log("overlap: {} of the pairs sampled start at frame {} of {} of their predecessor, so "
        "roughly {:.0f}% of every clip is a repeat of the one before it"
        .format(count, stride, clips.shape[1], 100.0 * (1.0 - stride / clips.shape[1])))


# --- driver ----------------------------------------------------------------------------


def trim_padding(positions):
    """Drops the repeated frames the release padded a sequence's ends with."""
    same = np.all(np.isclose(positions, positions[:1], atol=1e-4), axis=(1, 2))
    start = int(np.argmin(same)) if same[0] else 0
    same = np.all(np.isclose(positions, positions[-1:], atol=1e-4), axis=(1, 2))
    end = len(positions) - int(np.argmin(same[::-1])) if same[-1] else len(positions)
    return positions[start:end]


def prepare(clips, orientation, scale, trim):
    """Every clip in world-space centimetres at the output frame rate."""
    decimation = SOURCE_FPS // OUTPUT_FPS

    prepared = []
    for clip in clips:
        # The root channels are per-frame deltas, so the whole 60 Hz sequence has to be integrated
        # before anything is thrown away; decimating the deltas instead would halve the character's
        # travel speed while its legs kept their stride.
        world = (reconstruct_world(clip) @ orientation.T)[::decimation] * scale
        prepared.append(trim_padding(world) if trim else world)
    return prepared


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every file written has to declare the same skeleton, because
    # batch_retarget.check_skeleton_matches measures each clip against the setup blend's own
    # import and rejects a millimetre of disagreement. Converting the train and test splits in one
    # run is what keeps them to one skeleton, and so to one setup blend.
    parser.add_argument("--npz", required=True, nargs="+",
                        help="Edinburgh .npz files to convert, all sharing one fitted skeleton.")
    parser.add_argument("--out-dir", help="Folder to write the BVH files into.")
    parser.add_argument("--name", nargs="+", default=["edin_locomotion"],
                        help="File stem per --npz; a zero-padded clip index is appended.")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE_CM,
                        help="Centimetres per source unit.")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N clips.")
    parser.add_argument("--survey", action="store_true",
                        help="Report what is in the file and write nothing.")
    parser.add_argument("--verify", action="store_true",
                        help="Re-derive joint positions from the solve and report the error.")
    parser.add_argument("--trim-padding", action="store_true",
                        help="Drop repeated frames the release padded a sequence's ends with.")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if len(args.name) != len(args.npz):
        raise ConversionError("Give one --name per --npz; got {} names for {} files."
                              .format(len(args.name), len(args.npz)))

    loaded = []
    for npz, name in zip(args.npz, args.name):
        npz_path = resolve(npz)
        if not os.path.isfile(npz_path):
            raise ConversionError("No .npz at " + npz_path)
        clips = load_clips(npz_path)
        log("{}: {} clips x {} frames x {} channels"
            .format(os.path.basename(npz_path), *clips.shape))
        if args.limit:
            clips = clips[:args.limit]
        if args.survey:
            report_padding(clips)
            report_overlap(clips)
        loaded.append((name, clips))

    # Which way round the character lies is a property of the release, not of any one clip, and
    # the splits have to agree on it or the shared skeleton is fitted across two datasets lying at
    # right angles to each other.
    sample = np.concatenate([clips[:, ::10, :63].reshape(-1, 21, 3) for _, clips in loaded])
    orientation = detect_orientation(sample)
    splits = [(name, prepare(clips, orientation, args.scale, args.trim_padding))
              for name, clips in loaded]

    prepared = [clip for _, clips in splits for clip in clips]
    positions = np.concatenate(prepared)
    frames = measure_frames(positions)
    skeleton = fit_skeleton(positions, frames)

    report_scale(positions, args.scale)
    report_skeleton(skeleton)
    validate_rest_pose(skeleton)
    validate_joint_bends(positions, frames, skeleton)

    if args.survey:
        log("survey only; nothing written")
        return
    if not args.out_dir:
        raise ConversionError("--out-dir is required unless --survey is passed.")

    out_dir = resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    errors = []
    written = 0
    for name, clips in splits:
        width = max(2, len(str(len(clips) - 1)))
        for index, clip in enumerate(clips):
            root_positions, angles = solve_channels(clip, measure_frames(clip), skeleton)
            path = os.path.join(out_dir, "{}_{:0{}d}.bvh".format(name, index, width))
            write_bvh(path, skeleton, root_positions, angles)
            written += 1
            if args.verify:
                errors.append(report_accuracy(os.path.basename(path), skeleton, clip,
                                              root_positions, angles))
            elif index % 200 == 0:
                log("{}: wrote {} of {}".format(name, index + 1, len(clips)))

    log("wrote {} BVH files to {}".format(written, out_dir))
    if errors:
        log("overall joint error: mean {:.4f} cm, worst {:.4f} cm"
            .format(float(np.mean([e[0] for e in errors])), float(np.max([e[1] for e in errors]))))
        worst = max(errors, key=lambda e: e[2])
        log("worst bone step anywhere: {:.1f} deg at {} frame {}; {} clips step more than 40 deg"
            .format(worst[2], worst[3], worst[4],
                    sum(1 for e in errors if e[2] > 40.0)))
    log("done")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except ConversionError as error:
        log("ERROR: " + str(error))
        sys.exit(1)
