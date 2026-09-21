from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from typing import List

from core.quaternions import blend_quaternions


class Pose:
    """
    Represents skeletal poses with root translation, hip translation,
    and joint rotation quaternions.
    """

    rootPos: np.ndarray
    hipPos: np.ndarray
    quats: np.ndarray

    def __init__(self, root: np.ndarray, hips: np.ndarray, quats: np.ndarray):
        """
        :param root: Vector3 root position (..., 3)
        :param hips: Vector3 hips position (..., 3)
        :param quats: Joint rotations quaternions (..., num_bones, 4)
        """
        assert root.shape[-1] == 3, "Root position must be a 3D vector."
        assert hips.shape[-1] == 3, "Hips position must be a 3D vector."
        assert root.shape[:-1] == hips.shape[:-1] and root.shape[:-1] == quats.shape[:-2], "Batch sizes must match."
        assert len(root.shape) > 1, "Root position must have at least 2 dimensions (batch, 3)."

        self.rootPos = np.asarray(root, dtype=np.float32)
        self.hipPos = np.asarray(hips, dtype=np.float32)
        self.quats = quats

    @staticmethod
    def from_array(pose: np.ndarray) -> Pose:
        """Unpacks a flat pose tensor into a PoseData instance."""
        root = pose[..., 0, :3].copy()
        hips = pose[..., 1, :3].copy()
        quats = pose[..., 2:, :].copy()
        if len(root.shape) == 1:
            root = np.expand_dims(root, axis=0)
            hips = np.expand_dims(hips, axis=0)
            quats = np.expand_dims(quats, axis=0)

        return Pose(root, hips, quats)

    def pack(self) -> np.ndarray:
        """Packs the PoseData back into a single flat array representation."""
        # Get raw quaternion array to determine shape
        num_bones = self.quats.shape[-2]
        batch_shape = self.quats.shape[:-2]

        result = np.zeros((*batch_shape, num_bones + 2, 4), dtype=np.float32)
        result[..., 0, :3] = self.rootPos
        result[..., 1, :3] = self.hipPos
        result[..., 2:, :] = self.quats
        return result

    def add(self, delta: PoseDelta) -> Pose:
        """
        Integrates a delta onto this pose.

        The delta must already be scaled to the desired time step (use
        PoseDelta.scaled(dt)); this method applies it verbatim. In particular
        `pose_x[i].add(pose_v[i].scaled(frame_time)) == pose_x[i + 1]`.

        The root translation is expressed in root-local space (matching
        PoseExtractor.ExtractPoseVelocities), so it is rotated by the root
        bone rotation before being applied.
        """
        r_root: Rotation = Rotation.from_quat(self.quats[..., 0, :])

        new_root = self.rootPos + r_root.apply(delta.rootVel)

        # Hips vector addition
        new_hips = self.hipPos + delta.hipVel

        # Joint rotations composition. The delta is left-multiplied, matching
        # MathExtensions.AngularVelocity: next = delta * current.
        new_quats = np.asarray(Rotation.as_quat(
            Rotation.from_rotvec(delta.rotvecs) *
            Rotation.from_quat(self.quats)
        ))

        return Pose(new_root, new_hips, new_quats)

    def __sub__(self, other: Pose) -> Pose:
        """
        Subtracts a base pose 'other' from 'self' to get the delta pose.
        """
        # Inverse root rotation of 'other' (b)

        r_b0_inv: Rotation = Rotation.from_quat(other.quats[..., 0, :]).inv()

        # Relative root position & rotation
        diff_root = np.array(
            r_b0_inv.apply(self.rootPos - other.rootPos)
        )

        # Hips vector difference
        diff_hips = self.hipPos - other.hipPos

        # Relative joint rotations: inverse(b) * a
        diff_quats = np.array(Rotation.as_quat(
            Rotation.from_quat(other.quats).inv() *
            Rotation.from_quat(self.quats)
        ))
        # Because we use [x, y, z, w], the scalar part 'w' is now at index -1
        flip = diff_quats[..., -1] < 0
        diff_quats[flip] = -diff_quats[flip]

        return Pose(diff_root, diff_hips, diff_quats)

    @staticmethod
    def lerp(a: Pose, b: Pose, t: float) -> Pose:
        """Linear interpolation for translations and SLERP for rotations."""
        root = (1.0 - t) * a.rootPos + t * b.rootPos
        hips = (1.0 - t) * a.hipPos + t * b.hipPos

        # SciPy Slerp interpolation across time points [0, 1]
        # Note: Slerp expects a Rotation object representing a sequence of rotations.
        # Since 'a.quats' and 'b.quats' might be batches, we handle this by creating
        # a sequence of the two states.

        # We need to construct a Rotation object that represents the start and end
        # stack arrays to fit R.from_quat
        q_stacked = np.stack([a.quats, b.quats])
        r_stacked = Rotation.from_quat(q_stacked)

        slerper = Slerp([0.0, 1.0], r_stacked)
        interpolated_rotations = slerper([t])

        # Slerp returns an array of rotations. We take the one corresponding to 't'
        # Using indexing [0] to extract the Rotation object for time 't'
        quats = np.array(interpolated_rotations[:1].as_quat())

        return Pose(root, hips, quats)

    @staticmethod
    def blend(poses: Pose, weights: np.ndarray) -> Pose:
        """
        Weighted blend across axis 0.

        `poses` is a single Pose whose leading axis is the blend axis, so pair
        this with `Pose.concatenate`. Any remaining batch axes are preserved:
        blending (N_blend, B, ...) leaves a batch of B. `weights` is 1-D of
        length N_blend and is reshaped for broadcast the same way
        `blend_quaternions` does it -- indexing it with `[..., np.newaxis]`
        would align it against the *trailing* axes and fail on batched input.
        """
        weights = np.asarray(weights, dtype=np.float32)

        # poses.rootPos (N, ..., 3), poses.hipPos (N, ..., 3), poses.quats (N, ..., num_bones, 4)
        w = weights.reshape((-1,) + (1,) * (poses.rootPos.ndim - 1))
        roots = np.sum(poses.rootPos * w, axis=0)
        hips = np.sum(poses.hipPos * w, axis=0)
        quats = blend_quaternions(poses.quats, weights, axis=0)

        # Collapsing the blend axis of an unbatched input leaves (3,) / (B, 4);
        # Pose requires an explicit batch axis. Batched input already has one.
        if roots.ndim == 1:
            roots = np.expand_dims(roots, axis=0)
            hips = np.expand_dims(hips, axis=0)
            quats = np.expand_dims(quats, axis=0)

        return Pose(roots, hips, quats)

    @staticmethod
    def concatenate(poses: List[Pose]) -> Pose:
        """Concatenates a list of Pose instances along the batch dimension."""
        roots = np.concatenate([p.rootPos for p in poses], axis=0)
        hips = np.concatenate([p.hipPos for p in poses], axis=0)
        quats = np.concatenate([p.quats for p in poses], axis=0)
        return Pose(roots, hips, quats)

    @staticmethod
    def stack(poses: List[Pose]) -> Pose:
        """
        Stacks poses on a NEW leading axis, preserving each one's batch shape.

        This is the input `blend` wants when the operands are themselves
        batched: `concatenate` would fold the batch into the blend axis, so
        blending two batches of A poses would produce one pose rather than A.
        """
        roots = np.stack([p.rootPos for p in poses], axis=0)
        hips = np.stack([p.hipPos for p in poses], axis=0)
        quats = np.stack([p.quats for p in poses], axis=0)
        return Pose(roots, hips, quats)


class PoseDelta:
    """
    A pose velocity / delta: root and hip translational rates plus per-joint
    angular rates.

    Angular rates are stored as rotation *vectors* (angle * axis, rad/s), never
    as quaternions. A unit quaternion can only carry a rotation in [0, 2*pi) and
    scipy canonicalises it back to [0, pi], so round-tripping a rad/s rate
    through a quaternion silently aliases every joint turning faster than
    pi rad/s (~2.6% of this dataset, of which the majority come back pointing
    the opposite way). Keeping rotation vectors makes `scaled` exact.

    The packed layout matches Pose so the (..., num_bones + 2, 4) arrays coming
    from / going to C# are unchanged:
        slot 0  [:3] -> root translational rate (root-local space)
        slot 1  [:3] -> hip translational rate
        slots 2+[:3] -> per-joint angular rate as a rotation vector
    The trailing component of each slot is unused and kept at 0.
    """

    rootVel: np.ndarray
    hipVel: np.ndarray
    rotvecs: np.ndarray

    def __init__(self, root_vel: np.ndarray, hip_vel: np.ndarray, rotvecs: np.ndarray):
        """
        :param root_vel: Vector3 root translational rate (..., 3)
        :param hip_vel: Vector3 hips translational rate (..., 3)
        :param rotvecs: Joint angular rates as rotation vectors (..., num_bones, 3)
        """
        assert root_vel.shape[-1] == 3, "Root velocity must be a 3D vector."
        assert hip_vel.shape[-1] == 3, "Hip velocity must be a 3D vector."
        assert rotvecs.shape[-1] == 3, "Angular rates must be rotation vectors."
        assert root_vel.shape[:-1] == hip_vel.shape[:-1] and \
               root_vel.shape[:-1] == rotvecs.shape[:-2], "Batch sizes must match."
        assert len(root_vel.shape) > 1, "Root velocity must have at least 2 dimensions (batch, 3)."

        self.rootVel = np.asarray(root_vel, dtype=np.float32)
        self.hipVel = np.asarray(hip_vel, dtype=np.float32)
        self.rotvecs = np.asarray(rotvecs, dtype=np.float32)

    @staticmethod
    def from_array(delta: np.ndarray) -> PoseDelta:
        """Unpacks a flat delta tensor into a PoseDelta instance."""
        root_vel = delta[..., 0, :3].copy()
        hip_vel = delta[..., 1, :3].copy()
        rotvecs = delta[..., 2:, :3].copy()
        if len(root_vel.shape) == 1:
            root_vel = np.expand_dims(root_vel, axis=0)
            hip_vel = np.expand_dims(hip_vel, axis=0)
            rotvecs = np.expand_dims(rotvecs, axis=0)

        return PoseDelta(root_vel, hip_vel, rotvecs)

    def pack(self) -> np.ndarray:
        """Packs the PoseDelta back into a single flat array representation."""
        num_bones = self.rotvecs.shape[-2]
        batch_shape = self.rotvecs.shape[:-2]

        result = np.zeros((*batch_shape, num_bones + 2, 4), dtype=np.float32)
        result[..., 0, :3] = self.rootVel
        result[..., 1, :3] = self.hipVel
        result[..., 2:, :3] = self.rotvecs
        return result

    def scaled(self, scale_factor: float) -> PoseDelta:
        """Scales the delta by a given factor. Exact: rates are linear."""
        return PoseDelta(self.rootVel * scale_factor,
                         self.hipVel * scale_factor,
                         self.rotvecs * scale_factor)

    @staticmethod
    def blend(deltas: PoseDelta, weights: np.ndarray) -> PoseDelta:
        """
        Weighted blend across axis 0. Mirrors Pose.blend: `deltas` is a single
        PoseDelta whose leading axis is the blend axis, and any remaining batch
        axes are preserved.
        """
        weights = np.asarray(weights, dtype=np.float32)

        w_vel = weights.reshape((-1,) + (1,) * (deltas.rootVel.ndim - 1))
        root_vel = np.sum(deltas.rootVel * w_vel, axis=0)
        hip_vel = np.sum(deltas.hipVel * w_vel, axis=0)
        # Rotation vectors blend linearly - that is the correct operation for
        # angular rates, and needs no nlerp/renormalisation.
        w = weights.reshape((-1,) + (1,) * (deltas.rotvecs.ndim - 1))
        rotvecs = np.sum(deltas.rotvecs * w, axis=0)

        if root_vel.ndim == 1:
            root_vel = np.expand_dims(root_vel, axis=0)
            hip_vel = np.expand_dims(hip_vel, axis=0)
            rotvecs = np.expand_dims(rotvecs, axis=0)

        return PoseDelta(root_vel, hip_vel, rotvecs)

    @staticmethod
    def concatenate(deltas: List[PoseDelta]) -> PoseDelta:
        """Concatenates a list of PoseDelta instances along the batch dimension."""
        root_vels = np.concatenate([d.rootVel for d in deltas], axis=0)
        hip_vels = np.concatenate([d.hipVel for d in deltas], axis=0)
        rotvecs = np.concatenate([d.rotvecs for d in deltas], axis=0)
        return PoseDelta(root_vels, hip_vels, rotvecs)

    @staticmethod
    def stack(deltas: List[PoseDelta]) -> PoseDelta:
        """Stacks deltas on a new leading axis. See Pose.stack."""
        root_vels = np.stack([d.rootVel for d in deltas], axis=0)
        hip_vels = np.stack([d.hipVel for d in deltas], axis=0)
        rotvecs = np.stack([d.rotvecs for d in deltas], axis=0)
        return PoseDelta(root_vels, hip_vels, rotvecs)

