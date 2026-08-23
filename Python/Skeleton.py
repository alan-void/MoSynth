from __future__ import annotations

from typing import Union, Iterator

import numpy as np
from scipy.spatial.transform import Rotation

from Pose import Pose


class Skeleton:
    def __init__(self, name: str, root_joint: Joint):
        self.name = name
        self.root_joint = root_joint

    def __iter__(self) -> Iterator[Joint]:
        """Iterates joints in depth-first order starting from the root_joint."""
        return iter(self.root_joint)

    def fk_root_space(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs forward kinematics while ignoring the root.

        Joint 0 is the character frame, which is pinned to the origin with identity
        rotation: that is what makes the result independent of where the character
        stands and which way it faces. Joint 1 is the rig's own root bone, and its
        translation is the one the pose carries, in that frame.

        Returns a tuple of:
            - Root space positions: np.ndarray of shape (..., num_joints, 3)
            - World space rotations: scipy.spatial.transform.Rotation of shape (..., num_joints)
        """

        joints = list(self)
        num_joints = len(joints)
        joint_to_idx = {j: i for i, j in enumerate(joints)}

        batch_shape = pose.quats.shape[:-2]

        global_positions = (
            np.zeros(batch_shape + (num_joints, 3), dtype=np.float32))
        global_rotations = (
            np.ones(batch_shape + (num_joints, 4), dtype=np.float32))

        for i, joint in enumerate(joints):

            parent = joint.parent()
            if parent is None:
                # the character frame
                global_rotations[..., i, :] = Rotation.identity().as_quat()
                continue

            local_rot = Rotation.from_quat(pose.quats[..., i, :])
            local_pos = joint.default_local_position
            if i == 1:  # the rig's root bone, offset from the character frame
                local_pos = pose.hipPos

            parent_idx = joint_to_idx[parent]
            parent_rot = Rotation.from_quat(
                global_rotations[..., parent_idx, :])
            parent_pos = global_positions[..., parent_idx, :]

            global_rotations[..., i, :] = (parent_rot * local_rot).as_quat()
            global_positions[..., i, :] = parent_pos + parent_rot.apply(local_pos)

        return global_positions, global_rotations

    def get_local_positions(self, pose: Pose) -> np.ndarray:
        """
        Returns the local positions of all joints in the skeleton based on the given pose.
        The local position of each joint is relative to its parent joint.
        """
        joints = list(self)
        num_joints = len(joints)

        batch_shape = pose.quats.shape[:-2]
        local_positions = np.zeros(batch_shape + (num_joints, 3), dtype=np.float32)

        for i, joint in enumerate(joints):
            parent = joint.parent()
            if parent is None:
                # the character frame
                local_positions[..., i, :] = pose.rootPos
                continue
            if i == 1:  # the rig's root bone, offset from the character frame
                local_positions[..., i, :] = pose.hipPos
            else:
                local_positions[..., i, :] = joint.default_local_position

        return local_positions



class Joint:
    def __init__(self,
                 name,
                 local_pos=np.zeros(3, dtype=np.float32),
                 local_rot=Rotation.identity()):
        self.name = name
        self._parent = None
        self._children: list[Joint] = []
        self.default_local_position = local_pos
        self.default_local_rotation = local_rot

    def add_child(self, child: Joint):
        assert child._parent is None, "Child joint already has a parent."
        self._children.append(child)
        child._parent = self

    def parent(self) -> Union[Joint, None]:
        """Returns the parent joint, or None if this is the root."""
        return self._parent

    def children(self) -> list[Joint]:
        """Returns a list of child joints."""
        return self._children.copy()

    def dfs(self) -> Iterator[Joint]:
        """Generator that yields this joint and all descendants in depth-first order."""
        yield self
        for child in self._children:
            yield from child.dfs()

    def __iter__(self) -> Iterator[Joint]:
        """Allow `iter(joint)` to iterate DFS from this joint."""
        return self.dfs()
