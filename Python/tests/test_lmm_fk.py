"""
Forward kinematics, which sits on both sides of the learned motion matching loss.

Nothing here would throw if it were wrong. A transposed rotation, a parent composed on the wrong
side, or a level visited out of order all produce a perfectly finite character space -- just not
the one the database is measured in -- and the only symptom would be a model that trains to a
plausible loss and poses badly. So this is checked three ways, each of which would catch a
different class of mistake:

* against the **database**, whose character-space positions and rotations are derived by a wholly
  separate route in :mod:`training_data`;
* against a **second spelling** of the same map in numpy and scipy, which shares no code with the
  torch one;
* as a **gradient**, since a version that is correct but not differentiable is useless to the
  trainer and the failure appears only under ``backward``.

Torch is optional, as it is in ``test_lmm_model``: the suite's stated property is that it needs
neither Unity nor venv extras.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmm_dataset  # noqa: E402
import neural_packing  # noqa: E402
from test_lmm_dataset import straight_walk_set  # noqa: E402

try:
    import torch

    import lmm_fk
    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only on a bare interpreter
    HAS_TORCH = False


def packed(training_set, spec):
    """``(local rotations (n, B, 6), root height (n,), true positions, true rotations)``."""
    y = lmm_dataset.pose_vector(training_set, spec)
    q = lmm_dataset.character_vector(training_set, spec)
    pose_layout, character_layout = spec.pose_layout(), spec.character_layout()

    return (np.ascontiguousarray(
                neural_packing.block(y, pose_layout, 'rotations_6d')
                .reshape(-1, spec.n_bones, 6)),
            np.ascontiguousarray(
                neural_packing.block(y, pose_layout, 'root_height').reshape(-1)),
            neural_packing.block(q, character_layout, 'positions')
            .reshape(-1, spec.n_bones, 3),
            neural_packing.block(q, character_layout, 'rotations_6d')
            .reshape(-1, spec.n_bones, 6))


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class AgainstTheDatabaseTests(unittest.TestCase):
    """The strongest check: FK of what the model predicts must reproduce what it is scored against."""

    def setUp(self):
        self.training_set = straight_walk_set()
        self.spec = lmm_dataset.build_spec(self.training_set)
        self.local, self.height, self.positions, self.rotations = packed(
            self.training_set, self.spec)
        self.hierarchy = lmm_fk.Hierarchy(self.spec.parents)

    def run_fk(self):
        return lmm_fk.forward_kinematics(
            torch.from_numpy(self.local), torch.from_numpy(self.height),
            torch.from_numpy(self.spec.rest_offsets), self.hierarchy)

    def test_it_reproduces_every_joint_position(self):
        positions, _ = self.run_fk()

        np.testing.assert_allclose(positions.numpy(), self.positions, rtol=0, atol=1e-5)

    def test_it_reproduces_every_joint_rotation(self):
        _, rotations = self.run_fk()

        np.testing.assert_allclose(lmm_fk.matrix_to_six_d(rotations).numpy(), self.rotations,
                                   rtol=0, atol=1e-5)

    def test_the_packed_form_is_the_character_vector_the_loss_compares_against(self):
        character = lmm_fk.character_vector_from_fk(
            torch.from_numpy(self.local), torch.from_numpy(self.height),
            torch.from_numpy(self.spec.rest_offsets), self.hierarchy)

        np.testing.assert_allclose(
            character.numpy(), lmm_dataset.character_vector(self.training_set, self.spec),
            rtol=0, atol=1e-5)

    def test_the_rest_offsets_are_the_same_from_any_frame(self):
        # If they were not, recovering them from frame 0 would silently encode that frame's pose
        # into every reconstruction.
        late = lmm_fk.rest_offsets(
            self.training_set.positions[-1][self.spec.bone_indices],
            self.training_set.rotations[-1][self.spec.bone_indices], self.spec.parents)

        np.testing.assert_allclose(late, self.spec.rest_offsets, rtol=0, atol=1e-5)


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class AgainstTheReferenceTests(unittest.TestCase):
    """Two independent spellings of the same map, sharing no code."""

    def setUp(self):
        self.spec = lmm_dataset.build_spec(straight_walk_set())
        rng = np.random.default_rng(11)
        self.local = rng.standard_normal((16, self.spec.n_bones, 6)).astype(np.float32)
        self.height = rng.standard_normal(16).astype(np.float32)

    def test_arbitrary_rotations_give_the_same_positions(self):
        # Random 6D columns are not orthonormal, so this also checks that both sides project them
        # the same way rather than only that they compose the same way.
        positions, _ = lmm_fk.forward_kinematics(
            torch.from_numpy(self.local), torch.from_numpy(self.height),
            torch.from_numpy(self.spec.rest_offsets), lmm_fk.Hierarchy(self.spec.parents))
        reference, _ = lmm_fk.forward_kinematics_reference(
            self.local, self.height, self.spec.rest_offsets, self.spec.parents)

        np.testing.assert_allclose(positions.numpy(), reference, rtol=0, atol=1e-5)


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class DifferentiabilityTests(unittest.TestCase):
    def test_a_gradient_reaches_both_the_rotations_and_the_root_height(self):
        spec = lmm_dataset.build_spec(straight_walk_set())
        local = torch.randn(4, spec.n_bones, 6, requires_grad=True)
        height = torch.randn(4, requires_grad=True)

        positions, _ = lmm_fk.forward_kinematics(
            local, height, torch.from_numpy(spec.rest_offsets), lmm_fk.Hierarchy(spec.parents))
        positions.abs().sum().backward()

        self.assertTrue(torch.isfinite(local.grad).all())
        self.assertTrue(torch.isfinite(height.grad).all())
        self.assertGreater(float(local.grad.abs().sum()), 0.0)
        self.assertGreater(float(height.grad.abs().sum()), 0.0)


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class HierarchyTests(unittest.TestCase):
    def test_every_bone_appears_once_below_its_parents_level(self):
        parents = np.array([-1, 0, 1, 0, 3, 4], dtype=np.int64)
        hierarchy = lmm_fk.Hierarchy(parents)

        seen = [hierarchy.root] + [int(b) for bones, _ in hierarchy.levels for b in bones]
        self.assertEqual(sorted(seen), list(range(parents.size)))

        depth_of = {hierarchy.root: 0}
        for level, (bones, of_parent) in enumerate(hierarchy.levels, start=1):
            for bone, parent in zip(bones.tolist(), of_parent.tolist()):
                self.assertEqual(depth_of[parent], level - 1)
                depth_of[bone] = level


class RotationConversionTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, 'torch is not installed')
    def test_local_rotations_compose_back_to_the_character_space_ones(self):
        training_set = straight_walk_set()
        spec = lmm_dataset.build_spec(training_set)
        local = lmm_fk.local_rotations(training_set.rotations[:, spec.bone_indices], spec.parents)

        from scipy.spatial.transform import Rotation
        for bone in range(1, spec.n_bones):
            parent = int(spec.parents[bone])
            composed = (Rotation.from_quat(training_set.rotations[:, spec.bone_indices[parent]])
                        * Rotation.from_quat(local[:, bone])).as_matrix()
            expected = Rotation.from_quat(
                training_set.rotations[:, spec.bone_indices[bone]]).as_matrix()

            np.testing.assert_allclose(composed, expected, rtol=0, atol=1e-5)

    @unittest.skipUnless(HAS_TORCH, 'torch is not installed')
    def test_the_six_d_round_trip_is_the_identity_on_a_real_rotation(self):
        from scipy.spatial.transform import Rotation
        from training_data import rotations_to_6d

        rng = np.random.default_rng(5)
        rotations = Rotation.random(20, random_state=5).as_quat().astype(np.float32)
        six = rotations_to_6d(rotations)

        matrices = lmm_fk.six_d_to_matrix(torch.from_numpy(six)).numpy()
        np.testing.assert_allclose(matrices, Rotation.from_quat(rotations).as_matrix(),
                                   rtol=0, atol=1e-5)
        np.testing.assert_allclose(
            lmm_fk.matrix_to_six_d(torch.from_numpy(matrices)).numpy(), six, rtol=0, atol=1e-5)
        del rng


if __name__ == '__main__':
    unittest.main()
