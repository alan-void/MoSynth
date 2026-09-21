import os
import struct
import numpy as np
from scipy.spatial.transform import Rotation

from core.pose_set import PoseSet
from core.skeleton import Joint, Skeleton
from formats.binary_reading import read_csharp_string

# Unity is y-up and left-handed, so a character faces +z.
_CHARACTER_FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def read_skeleton(f) -> tuple[Skeleton, int, np.ndarray]:
    """
    Reads the skeleton block at the head of an open .mmpose file, leaving the stream
    positioned at the clip table.

    Unity writes this block because the Python side has no ScriptableObject to read the bone
    tree from: names key the per-bone weight table, parent indices drive FK, and the rest
    offsets give root-space positions. It lives in the same file as the poses so that the two
    cannot drift apart -- which is what the separate .mmskeleton file used to allow.

    The character frame the poses are matched in is structural rather than stored: it is read
    off joint 0, along the bone-local axis that points forward in the rest pose. Both fall out
    of the bone entries above, so nothing about the frame is written to the file.

    :return: ``(skeleton, simulation_frame_bone_index, simulation_frame_forward)``
    """
    skeleton_data = []

    n_joints = struct.unpack('<I', f.read(4))[0]

    for _ in range(n_joints):
        name = read_csharp_string(f)
        parent_index = struct.unpack('<i', f.read(4))[0]  # the root's -1, written as 0xFFFFFFFF
        local_offset = struct.unpack('<3f', f.read(12))  # x, y, z
        rest_rotation = struct.unpack('<4f', f.read(16))  # x, y, z, w

        skeleton_data.append({
            'name': name,
            'parent_index': parent_index,
            'local_offset': local_offset,
            'rest_rotation': rest_rotation
        })

    joints = [
        Joint(datum['name'],
              np.asarray(datum['local_offset'], dtype=np.float32),
              Rotation.from_quat(datum['rest_rotation']))
        for datum in skeleton_data
    ]
    for joint_idx, joint in enumerate(joints):
        datum = skeleton_data[joint_idx]
        parent_idx = int(datum['parent_index'])
        if parent_idx == -1: continue
        parent_joint = joints[parent_idx]
        parent_joint.add_child(joint)

    assert joints[0].parent() is None, "root bone should be the first bone"

    # The C# side's Skeleton.RestLocalAxis(0, forward): joint 0 has no parent, so its rest
    # character rotation is its own rest local rotation.
    sim_frame_bone_index = 0
    sim_frame_forward = np.asarray(
        joints[sim_frame_bone_index].default_local_rotation.inv().apply(_CHARACTER_FORWARD),
        dtype=np.float32)

    return Skeleton("char", joints[0]), sim_frame_bone_index, sim_frame_forward


def deserialize_pose_set(path: str, file_name: str) -> PoseSet:
    """
    Reads the .mmpose file from the given path and constructs a PoseSet object.

    The simulation frame definition rides along on the returned object as
    ``sim_frame_bone_index`` and ``sim_frame_forward``, which is everything
    :mod:`core.simulation_frame` needs to derive the character frame of every pose. Both are
    read off the skeleton rather than the file -- see :func:`read_skeleton`.
    """
    pose_path = os.path.join(path, f"{file_name}.mmpose")

    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Could not find {file_name}.mmpose at {path}")

    with open(pose_path, 'rb') as f:
        # 1. Deserialize Skeleton
        skeleton, sim_frame_bone_index, sim_frame_forward = read_skeleton(f)

        # 2. Deserialize Pose Data
        # Read Clips
        n_clips = struct.unpack('<I', f.read(4))[0]
        clips = []
        frame_time = 0.0
        for _ in range(n_clips):
            start = struct.unpack('<I', f.read(4))[0]
            end = struct.unpack('<I', f.read(4))[0]
            f_time = struct.unpack('<f', f.read(4))[0]
            clips.append({'start': start, 'end': end})
            frame_time = f_time  # Overwritten per clip, usually constant across a PoseSet

        # Read Header Counts
        n_poses, n_joints, n_tags = struct.unpack('<I I I', f.read(12))

        # Pre-allocate numpy arrays for fast assignment
        pos = np.zeros((n_poses, n_joints, 3), dtype=np.float32)
        quats = np.zeros((n_poses, n_joints, 4), dtype=np.float32)
        vel = np.zeros((n_poses, n_joints, 3), dtype=np.float32)
        ang_vel = np.zeros((n_poses, n_joints, 3), dtype=np.float32)

        foot_contacts = np.zeros((n_poses, 2), dtype=bool)

        # Read Poses
        for i in range(n_poses):
            # JointLocalPositions (float3 = 12 bytes per joint)
            pos[i] = np.frombuffer(f.read(n_joints * 12), dtype=np.float32).reshape(n_joints, 3)

            # JointLocalRotations (quaternion = 16 bytes per joint)
            quats[i] = np.frombuffer(f.read(n_joints * 16), dtype=np.float32).reshape(n_joints, 4)

            # JointLocalVelocities
            vel[i] = np.frombuffer(f.read(n_joints * 12), dtype=np.float32).reshape(n_joints, 3)

            # JointLocalAngularVelocities
            ang_vel[i] = np.frombuffer(f.read(n_joints * 12), dtype=np.float32).reshape(n_joints, 3)

            # Foot Contacts (uint = 4 bytes each)
            left_contact = struct.unpack('<I', f.read(4))[0] == 1
            right_contact = struct.unpack('<I', f.read(4))[0] == 1
            foot_contacts[i] = [left_contact, right_contact]

        # Gait phase and its rate, one pair per pose. Unity evaluates these from each clip's
        # authored footfalls and writes them here so that nothing on this side reconstructs
        # phase from the contacts -- which is how the two definitions used to drift apart.
        phase_block = f.read(n_poses * 8)
        if len(phase_block) < n_poses * 8:
            raise ValueError(
                f"{file_name}.mmpose ends before its gait phase block; it is truncated, or "
                f"predates the block. Regenerate the databases.")

        phase_pairs = np.frombuffer(phase_block, dtype=np.float32).reshape(n_poses, 2)
        phase, phase_rate = phase_pairs[:, 0].copy(), phase_pairs[:, 1].copy()

        # Read Tags
        tags = []
        for _ in range(n_tags):
            tag_name = read_csharp_string(f)
            n_ranges = struct.unpack('<I', f.read(4))[0]

            ranges = []
            for _ in range(n_ranges):
                start_range = struct.unpack('<I', f.read(4))[0]
                end_range = struct.unpack('<I', f.read(4))[0]
                ranges.append((start_range, end_range))

            tags.append({
                'name': tag_name,
                'ranges': ranges
            })

    # 3. Instantiate the Python PoseSet
    pose_set = PoseSet(
        skeleton=skeleton,
        frame_time=frame_time,
        local_pos=pos,
        local_quats=quats,
        foot_contacts=foot_contacts,
        local_vel=vel,
        local_angular_vel=ang_vel,
        clips=clips,
        phase=phase,
        phase_rate=phase_rate,
    )

    # Optionally attach data not directly in the __init__ definition to the python object
    pose_set.tags = tags
    pose_set.sim_frame_bone_index = sim_frame_bone_index
    pose_set.sim_frame_forward = sim_frame_forward

    return pose_set

# Example Usage:
# my_pose_set = deserialize_pose_set("path/to/directory", "RunAnimation")
