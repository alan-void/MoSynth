"""
Bridge between the exported pose database and the motion field's state arrays.

:func:`load_animations` reads the database Unity exported and repacks it into the
(x, v, y) triples the motion field is defined over. :func:`get_pose_arrays` goes the
other way, flattening a synthesized pose into lists PythonNET can marshal into C#.
"""

import os

import numpy as np

from core.pose import Pose, PoseDelta
from core.pose_set import PoseSet
from core.simulation_frame import derive_pose_set_frames
from core.skeleton import Joint, Skeleton
from debugging.python_net import connect_debugger
from formats.pose_set_importer import deserialize_pose_set

# Attaching costs a socket timeout per import and injects a PyCharm egg into
# sys.path; set MOSYNTH_PYCHARM_DEBUG=0 to skip it for batch work such as training.
if os.environ.get('MOSYNTH_PYCHARM_DEBUG', '1') != '0':
    connect_debugger()


def build_state_indices(clips, n_poses):
    """
    Frame indices that can serve as motion field states.

    A state at frame i needs the next frame's velocity too (it becomes pose_y),
    so frame i + 1 must be in the same clip and the last frame of every clip is
    dropped.
    """
    if not clips:
        return np.arange(max(0, n_poses - 1), dtype=np.int64)

    ranges = [np.arange(c['start'], min(c['end'], n_poses) - 1, dtype=np.int64)
              for c in clips]
    ranges = [r for r in ranges if r.size > 0]
    if not ranges:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(ranges)


VIRTUAL_ROOT_NAME = 'SimulationFrame'


def with_virtual_root(skeleton: Skeleton) -> Skeleton:
    """
    Copy a skeleton under an extra parentless joint standing for the character frame.

    The packed layout reserves slot 0 for that frame, which no rig bone corresponds to.
    A matching joint keeps the tree aligned with the array, so bone weights, feature rows
    and FK indices all agree, and `fk_root_space` pins slot 0 to the origin.

    Copied rather than reparented in place, so the pose set's own skeleton goes on
    describing the pose set's own frame-free arrays.
    """
    joints = list(skeleton)
    index_of = {joint: i for i, joint in enumerate(joints)}

    copies = [Joint(joint.name, joint.default_local_position, joint.default_local_rotation)
              for joint in joints]
    for joint, copy in zip(joints, copies):
        parent = joint.parent()
        if parent is not None:
            copies[index_of[parent]].add_child(copy)

    virtual_root = Joint(VIRTUAL_ROOT_NAME)
    virtual_root.add_child(copies[0])

    return Skeleton(skeleton.name, virtual_root)


def load_animations(data_dir='../Assets/StreamingAssets/MMDatabases/MotionMatchingData',
                    db_name='MotionMatchingData'):
    """
    Load a pose database and repack it into motion field states. Each usable frame i
    becomes a triple: the pose (``pose_x``), how it is moving now (``pose_v``), and how
    it is moving one frame later (``pose_y``, what the field learns to predict).

    ``pose_x`` is made translation- and yaw-invariant, so "walking north at the origin"
    and "walking east across the map" are the same state. That motion is not lost -- it
    lives in ``pose_v`` as a rate in the character frame. The database stores poses in
    clip space, so both the frame and its rate are derived here; see
    :func:`core.simulation_frame.derive_pose_set_frames`.

    The returned skeleton carries the extra frame joint the packed layout expects and so
    is one bone longer than ``pose_set.skeleton``, which keeps describing the file's own
    arrays. See :func:`with_virtual_root`.

    ``pose_contacts`` holds the contact flags of frame i itself and ``next_contacts``
    those of frame i + 1, the pose one step from state i lands on; both are
    (n_states, n_contacts), one column per contact bone of the database.

    :returns: ``(skeleton, pose_x, pose_v, pose_y, pose_contacts, next_contacts,
        frame_time, pose_set)``
    """
    pose_set: PoseSet = deserialize_pose_set(data_dir, db_name)

    skeleton: Skeleton = with_virtual_root(pose_set.skeleton)
    bone_count = len(list(skeleton))

    quats = pose_set.local_rotations  # (n_poses, n_joints, 4)
    lav = pose_set.local_angular_velocities  # (n_poses, n_joints, 3) — rotation vectors, rad/s, xyz

    # Indexed by global pose index, and clip-safe: no rate differences across a boundary.
    frames, rates = derive_pose_set_frames(pose_set)

    idx = build_state_indices(pose_set.clips, quats.shape[0])
    nxt = idx + 1
    n_target_poses = idx.shape[0]

    # The rates below are per-SECOND, so integrating them needs an explicit time step:
    #   pose_x[i].add(PoseDelta.from_array(pose_v[i]).scaled(frame_time)) == pose_x[i+1]
    # Angular rates stay as rotation vectors; see PoseDelta.

    # pose_x: current pose at frame i, expressed in the character frame
    #   slot 0 — frame position: zeroed (translation-invariant)
    #   slot 1 — root bone position in the frame
    #   slot 2 — frame rotation: identity (yaw-invariant)
    #   slot 3 — root bone rotation in the frame
    #   slots 4+ — the remaining joints' parent-local rotations
    pose_x = np.zeros((n_target_poses, bone_count + 2, 4), dtype=np.float32)
    pose_x[:, 1, :3] = frames.root_position[idx]
    pose_x[:, 2, :] = [0., 0., 0., 1.]  # identity quaternion [x,y,z,w]
    pose_x[:, 3, :] = frames.root_rotation[idx]
    pose_x[:, 4:, :] = quats[idx, 1:, :]

    # pose_v: velocity at frame i  =  (Pose(i+1) − Pose(i)) / frame_time
    #   slot 0 — frame translational rate, in the frame's own space
    #   slot 1 — root bone translational rate within the frame
    #   slot 2 — frame rotational rate: yaw only, so a rotation vector along +y
    #   slots 3+ — joint angular rates as rotation vectors (trailing component unused)
    pose_v = np.zeros((n_target_poses, bone_count + 2, 4), dtype=np.float32)
    pose_v[:, 0, :3] = rates.linear[idx]
    pose_v[:, 1, :3] = rates.root_linear[idx]
    pose_v[:, 2, 1] = rates.yaw[idx]
    pose_v[:, 3, :3] = rates.root_angular[idx]
    pose_v[:, 4:, :3] = lav[idx, 1:, :]

    # pose_y: future velocity at frame i+1  =  (Pose(i+2) − Pose(i+1)) / frame_time
    #   Same layout as pose_v, shifted one frame forward. Frame i+2 can be one past the
    #   end of a clip; the derivation reconstructs that frame.
    pose_y = np.zeros((n_target_poses, bone_count + 2, 4), dtype=np.float32)
    pose_y[:, 0, :3] = rates.linear[nxt]
    pose_y[:, 1, :3] = rates.root_linear[nxt]
    pose_y[:, 2, 1] = rates.yaw[nxt]
    pose_y[:, 3, :3] = rates.root_angular[nxt]
    pose_y[:, 4:, :3] = lav[nxt, 1:, :]

    pose_contacts = pose_set.foot_contacts[idx, :].copy()
    next_contacts = pose_set.foot_contacts[nxt, :].copy()

    return (skeleton, pose_x, pose_v, pose_y, pose_contacts, next_contacts,
            pose_set.frameTime, pose_set)


def get_pose_arrays(skeleton: Skeleton,
                    current_x: np.ndarray,
                    current_v: np.ndarray,
                    pose_contacts: np.ndarray):
    """
    Flatten one synthesized pose into the arrays the C# side reads back. Plain lists
    rather than ndarrays, because PythonNET marshals those directly.

    Indexed by rig bone, so one shorter than the packed layout: C# stores no character
    frame, it derives one from bone 0. The pose is already in its own frame, so bone 0
    carries the frame-local root state and the frame's own rates are folded into bone
    0's velocities, where C# reads them back out.

    :param pose_contacts: (n_contacts,) the pose's contact flags, in the database's contact
        slot order.
    :returns: ``(positions, quaternions, linear_velocities, angular_velocities, contacts)``,
        ``contacts`` being a list of bools, one per contact slot
    """
    p_x = Pose.from_array(current_x)
    p_v = PoseDelta.from_array(current_v)

    pos = skeleton.get_local_positions(p_x)[0][1:]
    quats = p_x.quats[0][1:]

    rotvecs = p_v.rotvecs[0]
    frame_rotvec = rotvecs[0]  # yaw rate, a rotation vector along +y

    # Bone 0's channels under an identity frame: the frame's own motion plus the root's
    # within it. The cross term is the root swinging about the frame origin.
    lv = np.zeros_like(pos)
    lv[0] = p_v.rootVel[0] + np.cross(frame_rotvec, p_x.hipPos[0]) + p_v.hipVel[0]

    lav = rotvecs[1:].copy()
    lav[0] = lav[0] + frame_rotvec

    pos = np.ascontiguousarray(pos, dtype=np.float32).flatten().tolist()
    quats = np.ascontiguousarray(quats, dtype=np.float32).flatten().tolist()
    lv = np.ascontiguousarray(lv, dtype=np.float32).flatten().tolist()
    lav = np.ascontiguousarray(lav, dtype=np.float32).flatten().tolist()

    contacts = [bool(flag) for flag in np.asarray(pose_contacts).ravel()]

    return pos, quats, lv, lav, contacts
