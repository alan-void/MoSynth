"""
Bridge between the exported pose database and the motion field's state arrays.

:func:`load_animations` reads the database Unity exported and repacks it into the
(x, v, y) triples the motion field is defined over. :func:`get_pose_arrays` goes the
other way, flattening a synthesized pose into lists PythonNET can marshal into C#.
"""

import os

import numpy as np

from Pose import Pose, PoseDelta
from Skeleton import Skeleton

from Animation import PoseSet
from pose_set_importer import deserialize_pose_set

from debugging.python_net import connect_debugger

# Attaching costs a socket timeout per import and injects a PyCharm egg into
# sys.path, which is pure overhead for batch work such as training. Opt in.
if os.environ.get('MOSYNTH_PYCHARM_DEBUG', '1') != '0':
    connect_debugger()


def build_state_indices(clips, n_poses):
    """
    Frame indices that can serve as motion field states.

    A state at frame i needs its own velocity `lv[i]` *and* the next frame's
    velocity `lv[i + 1]` (which becomes pose_y). Frame i + 1 must therefore
    belong to the same clip, so the last frame of every clip is dropped --
    otherwise a state would predict its successor from the first frame of an
    unrelated animation.
    """
    if not clips:
        return np.arange(max(0, n_poses - 1), dtype=np.int64)

    ranges = [np.arange(c['start'], min(c['end'], n_poses) - 1, dtype=np.int64)
              for c in clips]
    ranges = [r for r in ranges if r.size > 0]
    if not ranges:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(ranges)


def load_animations(data_dir='../Assets/StreamingAssets/MMDatabases/MotionMatchingData',
                    db_name='MotionMatchingData'):
    """
    Load a pose database and repack it into motion field states. Each usable frame i
    becomes a triple: the pose (``pose_x``), how it is moving now (``pose_v``), and how
    it is moving one frame later (``pose_y``, what the field learns to predict).

    ``pose_x`` is made translation- and yaw-invariant, so "walking north at the origin"
    and "walking east across the map" are the same state. That motion is not lost -- it
    lives in ``pose_v`` as a rate in the root's own frame. See the slot notes below.

    :returns: ``(skeleton, pose_x, pose_v, pose_y, pose_contacts, frame_time, pose_set)``
    """
    pose_set: PoseSet = deserialize_pose_set(data_dir, db_name)

    skeleton: Skeleton = pose_set.skeleton
    bone_count = len(list(skeleton))

    pos = pose_set.local_positions  # (n_poses, n_joints, 3)
    quats = pose_set.local_rotations  # (n_poses, n_joints, 4)
    lv = pose_set.local_velocities  # (n_poses, n_joints, 3) — m/s
    lav = pose_set.local_angular_velocities  # (n_poses, n_joints, 3) — rotation vectors, rad/s, xyz

    idx = build_state_indices(pose_set.clips, pos.shape[0])
    nxt = idx + 1
    n_target_poses = idx.shape[0]

    # NOTE: lv/lav are per-SECOND rates (PoseExtractor divides by FrameTime), and
    # they are stored here as-is. Integrating them requires an explicit time step:
    #   pose_x[i].add(PoseDelta.from_array(pose_v[i]).scaled(frame_time)) == pose_x[i+1]
    # Angular rates stay as rotation vectors; see PoseDelta for why converting
    # them to quaternions here would alias every joint faster than pi rad/s.

    # pose_x: current pose at frame i, expressed in root-relative frame
    #   slot 0 — root position: zeroed (translation-invariant)
    #   slot 1 — hip position: kept as-is
    #   slot 2 — root bone quat: identity (yaw-invariant)
    #   slots 3+ — remaining joint quats
    pose_x = np.zeros((n_target_poses, bone_count + 2, 4), dtype=np.float32)
    pose_x[:, 1, :3] = pos[idx, 1, :]  # hip
    pose_x[:, 2, :] = [0., 0., 0., 1.]  # identity quaternion [x,y,z,w]
    pose_x[:, 3:, :] = quats[idx, 1:, :]

    # pose_v: velocity at frame i  =  (Pose(i+1) − Pose(i)) / frame_time
    #   slot 0 — root translational rate (root-local space)
    #   slot 1 — hip translational rate
    #   slots 2+ — joint angular rates as rotation vectors (trailing component unused)
    pose_v = np.zeros((n_target_poses, bone_count + 2, 4), dtype=np.float32)
    pose_v[:, 0, :3] = lv[idx, 0, :]
    pose_v[:, 1, :3] = lv[idx, 1, :]
    pose_v[:, 2:, :3] = lav[idx, :, :]

    # pose_y: future velocity at frame i+1  =  (Pose(i+2) − Pose(i+1)) / frame_time
    #   Same layout as pose_v, shifted one frame forward.
    pose_y = np.zeros((n_target_poses, bone_count + 2, 4), dtype=np.float32)
    pose_y[:, 0, :3] = lv[nxt, 0, :]
    pose_y[:, 1, :3] = lv[nxt, 1, :]
    pose_y[:, 2:, :3] = lav[nxt, :, :]

    # pose_contacts: foot contact flags aligned to frame i
    pose_contacts = pose_set.foot_contacts[idx, :].copy()

    return skeleton, pose_x, pose_v, pose_y, pose_contacts, pose_set.frameTime, pose_set

def get_pose_arrays(skeleton: Skeleton,
                    current_x: np.ndarray,
                    current_v: np.ndarray,
                    pose_contacts: np.ndarray):
    """
    Flatten one synthesized pose into the arrays the C# side reads back. Plain lists
    rather than ndarrays, because PythonNET marshals those directly.

    :returns: ``(positions, quaternions, linear_velocities, angular_velocities,
        left_foot_contact, right_foot_contact)``
    """
    p_x = Pose.from_array(current_x)
    p_v = PoseDelta.from_array(current_v)

    pos = skeleton.get_local_positions(p_x)[0]
    lv = np.zeros_like(pos)
    lv[0] = p_v.rootVel
    quats = p_x.quats[0]
    lav = p_v.rotvecs[0]

    # Ensure flat float arrays for PythonNet auto-marshalling
    pos = np.ascontiguousarray(pos, dtype=np.float32).flatten().tolist()
    quats = np.ascontiguousarray(quats, dtype=np.float32).flatten().tolist()
    lv = np.ascontiguousarray(lv, dtype=np.float32).flatten().tolist()
    lav = np.ascontiguousarray(lav, dtype=np.float32).flatten().tolist()

    return pos, quats, lv, lav, bool(pose_contacts[0]), bool(pose_contacts[1])