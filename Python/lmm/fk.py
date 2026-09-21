"""
Forward kinematics over a bone selection, in the two forms the learned motion matching code needs.

The loss of Holden et al. 2020 is written in *two spaces*: a pose is predicted joint-locally and
then scored again after forward kinematics, "which ensure the output pose changes smoothly in
time". So FK has to be differentiable and run inside the training step, which is what
:func:`forward_kinematics` is for. :func:`forward_kinematics_reference` is the same map written in
numpy and scipy, kept because a silent axis, order or handedness error in the torch version would
not throw -- it would just train against a slightly wrong character space -- and two independent
spellings agreeing is the only cheap way to catch that. ``test_lmm_fk`` holds them to each other
and both to the database.

Conventions are inherited unchanged from :mod:`training.training_data`: quaternions xyzw, y-up left-handed
with the character facing +z, every rate per second.

**The root's translation is not predicted.** Its ``x`` and ``z`` are what the character frame
transform removed, so they are zero by construction; only its height above the ground is free, and
that is the one float :func:`forward_kinematics` takes beside the rotations. Every other bone sits
at its rest offset from its parent, which is why a reconstructed pose cannot violate a bone length.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from training.training_data import rotations_to_6d


def parents_within(parents: np.ndarray, bones: np.ndarray) -> np.ndarray:
    """
    Parent index of each selected bone, expressed within the selection; -1 for its root.

    :param parents: (n_bones,) parent index per database bone, -1 for the root.
    :param bones: the selected database bone indices, ascending.
    """
    position = {int(bone): i for i, bone in enumerate(bones)}
    return np.array([position.get(int(parents[bone]), -1) for bone in bones], dtype=np.int64)


def rest_offsets(positions: np.ndarray, rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """
    Each bone's offset from its parent, in the parent's own space.

    Recovered from one character-space frame rather than read from the skeleton, because a
    :class:`training.training_data.TrainingSet` carries no rest transforms. It is constant across frames --
    that is what makes it recoverable at all -- and ``test_lmm_fk`` checks it by taking a different
    frame and getting the same answer.

    :param positions: (n_bones, 3) joint positions of one frame, in the character frame.
    :param rotations: (n_bones, 4) joint rotations of that frame, xyzw, in the character frame.
    :param parents: (n_bones,) parent index within this selection, -1 for the root.
    :return: (n_bones, 3) float32, zero for the root.
    """
    positions = np.asarray(positions, dtype=np.float64)
    frames = Rotation.from_quat(np.asarray(rotations, dtype=np.float64))

    offsets = np.zeros((len(parents), 3))
    for i, parent in enumerate(parents):
        if parent < 0:
            continue
        offsets[i] = frames[int(parent)].inv().apply(positions[i] - positions[int(parent)])
    return offsets.astype(np.float32)


def local_rotations(rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """
    Character-space joint rotations expressed against their parent instead.

    The decompressor regresses these rather than character-space rotations, because that is what a
    rig is posed with and what makes the character-space error of a deep chain a *consequence* of
    the prediction rather than a separate output that can disagree with it.

    :param rotations: (n, n_bones, 4) character-space rotations, xyzw.
    :param parents: (n_bones,) parent index within the selection, -1 for the root.
    :return: (n, n_bones, 4) xyzw. The root keeps its character-space rotation, since the character
        frame is its parent.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    n, n_bones = rotations.shape[0], rotations.shape[1]
    local = np.empty_like(rotations)

    for j in range(n_bones):
        parent = int(parents[j])
        if parent < 0:
            local[:, j] = rotations[:, j]
            continue
        local[:, j] = (Rotation.from_quat(rotations[:, parent]).inv()
                       * Rotation.from_quat(rotations[:, j])).as_quat()

    return local.astype(np.float32)


def local_rotations_6d(rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """:func:`local_rotations` in the two-axis form -- see :func:`training.training_data.rotations_to_6d`."""
    return rotations_to_6d(local_rotations(rotations, parents))


# Torch -------------------------------------------------------------------------------------------

def six_d_to_matrix(six: torch.Tensor) -> torch.Tensor:
    """
    The two-axis rotation representation as a rotation matrix, by Gram-Schmidt.

    Differentiable, and the torch counterpart of :func:`training.training_data.rotations_from_6d` -- which
    goes on to a quaternion, a step nothing here needs.

    :param six: (..., 6) two concatenated columns.
    :return: (..., 3, 3) with the columns orthonormalised.
    """
    column0 = torch.nn.functional.normalize(six[..., :3], dim=-1)
    second = six[..., 3:]
    second = second - (column0 * second).sum(dim=-1, keepdim=True) * column0
    column1 = torch.nn.functional.normalize(second, dim=-1)
    column2 = torch.cross(column0, column1, dim=-1)
    return torch.stack([column0, column1, column2], dim=-1)


def matrix_to_six_d(matrix: torch.Tensor) -> torch.Tensor:
    """A rotation matrix's first two columns, concatenated -- the inverse of :func:`six_d_to_matrix`."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


class Hierarchy:
    """
    A bone selection grouped by depth, with the index tensors forward kinematics gathers through.

    Everything at one depth can be posed in a single batched product, because none of them is an
    ancestor of another -- which turns FK from one small matrix multiply per bone into one per
    *level*, about nine for a locomotion rig against twenty-seven bones. The indices are built once
    and held on the device for the same reason: rebuilding them per call means a host-to-device
    copy per level, which on a step this small costs more than the arithmetic it feeds.

    :param parents: (n_bones,) parent index within the selection, -1 for the root. Must be in
        depth-first order, so a bone's parent precedes it.
    :param device: where the index tensors live; the default is the CPU.
    """

    def __init__(self, parents, device=None):
        self.parents = np.asarray(parents, dtype=np.int64)

        depth = np.zeros(self.parents.size, dtype=np.int64)
        for j, parent in enumerate(self.parents):
            depth[j] = 0 if int(parent) < 0 else depth[int(parent)] + 1

        self.n_bones = int(self.parents.size)
        self.root = int(np.flatnonzero(depth == 0)[0])
        self.levels = []
        for d in range(1, int(depth.max()) + 1):
            bones = np.flatnonzero(depth == d)
            self.levels.append((
                torch.as_tensor(bones, dtype=torch.long, device=device),
                torch.as_tensor(self.parents[bones], dtype=torch.long, device=device)))


def forward_kinematics(local_six: torch.Tensor, root_height: torch.Tensor,
                       offsets: torch.Tensor, hierarchy: Hierarchy) -> tuple:
    """
    Character-space joint positions and rotations, from joint-local rotations and the root height.

    :param local_six: (m, n_bones, 6) joint-local rotations in the two-axis form.
    :param root_height: (m,) the root bone's height above the ground plane.
    :param offsets: (n_bones, 3) rest offsets, as :func:`rest_offsets` returns them.
    :param hierarchy: the bone tree, on the same device.
    :return: ``(positions (m, n_bones, 3), rotations (m, n_bones, 3, 3))``, both character-space.
    """
    matrices = six_d_to_matrix(local_six)
    m = local_six.shape[0]
    offsets = offsets.to(matrices.dtype)

    positions = torch.zeros(m, hierarchy.n_bones, 3,
                            dtype=matrices.dtype, device=matrices.device)
    rotations = torch.zeros(m, hierarchy.n_bones, 3, 3,
                            dtype=matrices.dtype, device=matrices.device)

    positions[:, hierarchy.root, 1] = root_height
    rotations[:, hierarchy.root] = matrices[:, hierarchy.root]

    for bones, of_parent in hierarchy.levels:
        parent_rotation = rotations[:, of_parent]
        rotations[:, bones] = parent_rotation @ matrices[:, bones]
        positions[:, bones] = positions[:, of_parent] + (
            parent_rotation @ offsets[bones].unsqueeze(-1).unsqueeze(0).expand(m, -1, -1, -1)
        ).squeeze(-1)

    return positions, rotations


def character_vector_from_fk(local_six: torch.Tensor, root_height: torch.Tensor,
                             offsets: torch.Tensor, hierarchy: Hierarchy) -> torch.Tensor:
    """
    :func:`forward_kinematics` packed as ``Q`` -- positions then character-space 6D rotations.

    The same layout :func:`lmm.dataset.character_vector_layout` declares, so a predicted ``Q`` and
    the database's own are directly comparable float for float.
    """
    positions, rotations = forward_kinematics(local_six, root_height, offsets, hierarchy)
    m = positions.shape[0]
    return torch.cat([positions.reshape(m, -1), matrix_to_six_d(rotations).reshape(m, -1)], dim=1)


# Numpy, for the cross-check ----------------------------------------------------------------------

def forward_kinematics_reference(local_six: np.ndarray, root_height: np.ndarray,
                                 offsets: np.ndarray, parents) -> tuple:
    """
    :func:`forward_kinematics` in numpy and scipy, for ``test_lmm_fk`` to hold the torch one to.

    :return: ``(positions (m, n_bones, 3), rotations (m, n_bones, 4))``, the rotations xyzw.
    """
    from training.training_data import rotations_from_6d

    local = rotations_from_6d(np.asarray(local_six, dtype=np.float64))
    m, n_bones = local.shape[0], local.shape[1]

    positions = np.zeros((m, n_bones, 3))
    positions[:, 0, 1] = np.asarray(root_height, dtype=np.float64)

    rotations = [Rotation.from_quat(local[:, 0].astype(np.float64))]
    for j in range(1, n_bones):
        parent = int(parents[j])
        rotations.append(rotations[parent] * Rotation.from_quat(local[:, j].astype(np.float64)))
        # np.tile rather than np.broadcast_to: scipy needs a writable buffer to read through.
        positions[:, j] = positions[:, parent] + \
            rotations[parent].apply(np.tile(np.asarray(offsets[j], dtype=np.float64), (m, 1)))

    quaternions = np.stack([rotation.as_quat() for rotation in rotations], axis=1)
    return positions, quaternions
