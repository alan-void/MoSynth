"""
Labels, pairing and the closed-form terms of the style dataset.

The fixture is a seven-bone biped whose styles differ from neutral in known ways -- a constant
spine pitch, a lower root, a slower walk, a hop that only ever lands on one foot -- so pairing and
the posture fit can be checked against the answer they should recover exactly.
"""

import math
import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_set import PoseSet  # noqa: E402
from core.skeleton import Joint, Skeleton  # noqa: E402
from lmm.fk import local_rotations  # noqa: E402
from style import dataset  # noqa: E402
from training import training_data  # noqa: E402

FRAME_TIME = 1.0 / 30.0
ROOT_HEIGHT = 0.9
STEP_FRAMES = 15
LEG_SWING = 0.4
SAD_PITCH = 0.3
SAD_DROP = 0.05
NEUTRAL_SPEED = 1.2
SAD_SPEED = 0.9
CONTACT_BONES = ['lfoot', 'rfoot']
SPINE = 1


def build_skeleton() -> Skeleton:
    """root -> spine -> head, root -> lleg -> lfoot, root -> rleg -> rfoot."""
    def joint(name, offset):
        return Joint(name, np.asarray(offset, dtype=np.float32), Rotation.identity())

    root = joint('root', (0.0, 0.0, 0.0))
    spine, head = joint('spine', (0.0, 0.3, 0.0)), joint('head', (0.0, 0.5, 0.0))
    lleg, lfoot = joint('lleg', (-0.1, -0.1, 0.0)), joint('lfoot', (0.0, -0.8, 0.0))
    rleg, rfoot = joint('rleg', (0.1, -0.1, 0.0)), joint('rfoot', (0.0, -0.8, 0.0))
    root.add_child(spine)
    spine.add_child(head)
    root.add_child(lleg)
    lleg.add_child(lfoot)
    root.add_child(rleg)
    rleg.add_child(rfoot)
    return Skeleton('biped', root)


def clip(n_frames, speed=NEUTRAL_SPEED, start_phase=0.0, spine_pitch=0.0, drop=0.0,
         idle=False, hop=False, start_z=0.0):
    """
    One clip as ``(positions, rotations, contacts, phase, phase_rate)``.

    A walk swings the legs with the phase, planting the right foot over the first half-cycle and
    the left over the second -- so the cycle is zeroed on a right footfall, as GaitPhase zeroes
    it. An idle stands on both feet and sweeps at the standing rate. A hop lands only the right.
    """
    skeleton = list(build_skeleton())
    n_bones = len(skeleton)
    slope = math.pi / STEP_FRAMES
    if idle:
        slope = 2.0 * math.pi * FRAME_TIME
    unwrapped = start_phase + slope * np.arange(n_frames)
    phase = np.mod(unwrapped, 2.0 * math.pi)

    positions = np.zeros((n_frames, n_bones, 3), dtype=np.float32)
    for bone, joint in enumerate(skeleton):
        positions[:, bone] = joint.default_local_position
    velocity = 0.0 if idle else speed
    positions[:, 0] = np.stack([np.zeros(n_frames), np.full(n_frames, ROOT_HEIGHT - drop),
                                start_z + velocity * FRAME_TIME * np.arange(n_frames)], axis=1)

    swing = 0.0 if idle else LEG_SWING
    rotations = np.zeros((n_frames, n_bones, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    rotations[:, 1] = Rotation.from_rotvec(
        np.tile([spine_pitch, 0.0, 0.0], (n_frames, 1))).as_quat()
    rotations[:, 3] = Rotation.from_rotvec(
        np.stack([swing * np.sin(phase), np.zeros(n_frames), np.zeros(n_frames)], 1)).as_quat()
    rotations[:, 5] = Rotation.from_rotvec(
        np.stack([-swing * np.sin(phase), np.zeros(n_frames), np.zeros(n_frames)], 1)).as_quat()

    contacts = np.ones((n_frames, 2), dtype=bool)
    if not idle:
        contacts[:, 1] = phase < math.pi
        contacts[:, 0] = phase >= math.pi
    if hop:
        # The right lands once per cycle and the left never does.
        contacts[:, 0] = False
        contacts[:, 1] = phase < 0.5 * math.pi

    return positions, rotations, contacts, phase.astype(np.float32), \
        np.full(n_frames, slope / FRAME_TIME, dtype=np.float32)


def build_pose_set(clips) -> PoseSet:
    """Concatenate clips into a database, with stored rates the extractor would have written."""
    parts = [clip(**spec) for spec in clips]
    positions = np.concatenate([part[0] for part in parts])
    rotations = np.concatenate([part[1] for part in parts])
    contacts = np.concatenate([part[2] for part in parts])
    phase = np.concatenate([part[3] for part in parts])
    phase_rate = np.concatenate([part[4] for part in parts])

    ranges, start = [], 0
    for part in parts:
        ranges.append((start, start + part[0].shape[0]))
        start += part[0].shape[0]

    n_bones = positions.shape[1]
    velocities = np.zeros_like(positions)
    angular = np.zeros_like(positions)
    for start, end in ranges:
        velocities[start:end - 1] = (positions[start + 1:end] - positions[start:end - 1]) \
            / FRAME_TIME
        velocities[end - 1] = velocities[end - 2]
        delta = (Rotation.from_quat(rotations[start + 1:end].reshape(-1, 4))
                 * Rotation.from_quat(rotations[start:end - 1].reshape(-1, 4)).inv())
        angular[start:end - 1] = delta.as_rotvec().reshape(-1, n_bones, 3) / FRAME_TIME
        angular[end - 1] = angular[end - 2]

    pose_set = PoseSet(skeleton=build_skeleton(), frame_time=FRAME_TIME, local_pos=positions,
                       local_quats=rotations, foot_contacts=contacts, local_vel=velocities,
                       local_angular_vel=angular,
                       clips=[{'start': s, 'end': e} for s, e in ranges],
                       contact_bone_names=CONTACT_BONES, phase=phase, phase_rate=phase_rate)
    pose_set.sim_frame_bone_index = 0
    pose_set.sim_frame_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return pose_set


# The standard fixture: three styles, two gaits, and a transition take.
STANDARD_CLIPS = [
    ({'n_frames': 120}, 'style.Neutral', 'gait.FW'),
    ({'n_frames': 60, 'idle': True}, 'style.Neutral', 'gait.ID'),
    # Shifted by whole frames of phase, so every styled frame has an exact neutral match.
    ({'n_frames': 120, 'speed': SAD_SPEED, 'start_phase': 3 * math.pi / STEP_FRAMES,
      'spine_pitch': SAD_PITCH, 'drop': SAD_DROP}, 'style.Sad', 'gait.FW'),
    ({'n_frames': 60, 'idle': True, 'spine_pitch': SAD_PITCH, 'drop': SAD_DROP},
     'style.Sad', 'gait.ID'),
    ({'n_frames': 90, 'hop': True}, 'style.Hop', 'gait.FW'),
    ({'n_frames': 40, 'speed': 0.5}, 'style.Sad', 'gait.TR1'),
]


def standard_set():
    """``(training_set, labels)`` for :data:`STANDARD_CLIPS`."""
    pose_set = build_pose_set([spec for spec, _, _ in STANDARD_CLIPS])
    document = {'clips': [{'style': style, 'gait': gait} for _, style, gait in STANDARD_CLIPS]}
    labels = dataset.parse_labels(document, len(pose_set.clips))
    return training_data.build_training_set(pose_set), labels


class LabelTests(unittest.TestCase):
    def test_tags_are_read_by_leaf_and_names_kept_verbatim(self):
        labels = dataset.parse_labels(
            {'clips': [{'style': 'style.Old', 'gait': 'gait.FW'},
                       {'style': 'Old', 'gait': 'tr2'}]}, 2)

        self.assertEqual(labels[0], dataset.ClipLabel('style.Old', 'FW'))
        self.assertTrue(labels[1].is_transition)
        self.assertEqual(dataset.style_key(labels[0].style), dataset.style_key(labels[1].style))

    def test_a_count_that_differs_from_the_database_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'database holds 3'):
            dataset.parse_labels({'clips': [{'style': 'Neutral', 'gait': 'FW'}]}, 3)

    def test_an_unknown_gait_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'gait'):
            dataset.parse_labels({'clips': [{'style': 'Neutral', 'gait': 'XX'}]}, 1)

    def test_labels_of_empty_clips_are_dropped_with_them(self):
        labels = [dataset.ClipLabel('A', 'FW'), dataset.ClipLabel('B', 'FW')]
        clips = [{'start': 0, 'end': 0}, {'start': 0, 'end': 10}]

        self.assertEqual(dataset.labels_for_stored_clips(labels, clips, 10), [labels[1]])


class DescriptorTests(unittest.TestCase):
    def test_forward_is_the_frames_z_and_lateral_its_x(self):
        training_set, _ = standard_set()
        spec = dataset.build_spec(training_set)
        descriptors = dataset.descriptor(training_set.positions, training_set.root_velocity,
                                         training_set.root_yaw_rate, spec.contact_bones)

        walk = slice(0, 119)
        np.testing.assert_allclose(descriptors[walk, 0], NEUTRAL_SPEED, atol=1e-3)
        np.testing.assert_allclose(descriptors[walk, 1], 0.0, atol=1e-3)
        self.assertEqual(descriptors.shape[1], spec.descriptor_size)

    def test_styled_local_rotations_match_the_lmm_definition(self):
        training_set, _ = standard_set()
        spec = dataset.build_spec(training_set, excluded_bones=['head'])
        rotations = Rotation.random(20 * training_set.n_bones, random_state=3).as_quat()
        rotations = rotations.reshape(20, training_set.n_bones, 4)

        ours = dataset.styled_local_rotations(rotations, spec)
        reference = local_rotations(rotations[:, spec.styled_bones], spec.styled_parents)
        gap = Rotation.from_quat(ours.reshape(-1, 4)) \
            * Rotation.from_quat(reference.reshape(-1, 4)).inv()
        np.testing.assert_allclose(gap.magnitude(), 0.0, atol=1e-5)
        self.assertEqual(spec.styled_bone_names, ['root', 'spine', 'lleg', 'lfoot', 'rleg',
                                                  'rfoot'])

    def test_history_is_clamped_to_the_clip_start(self):
        indices = dataset.history_indices(np.array([[0, 10], [10, 20]]), 20, np.array([3, 6]))

        np.testing.assert_array_equal(indices[12], [10, 10])
        np.testing.assert_array_equal(indices[19], [16, 13])

    def test_the_input_vector_has_the_size_the_spec_declares(self):
        training_set, _ = standard_set()
        spec = dataset.build_spec(training_set)
        inputs = dataset.frame_inputs(training_set, spec)

        self.assertEqual(inputs.x.shape, (training_set.n_frames, spec.input_size))
        self.assertTrue(np.isfinite(inputs.x).all())


class FilterTests(unittest.TestCase):
    def test_a_style_that_lands_on_one_foot_is_dropped_as_cadence_defined(self):
        training_set, labels = standard_set()
        kept, dropped = dataset.filter_styles(training_set, labels, 'Neutral')

        self.assertEqual(kept, ['style.Neutral', 'style.Sad'])
        self.assertIn('style.Hop', dropped)
        self.assertIn('cadence', dropped['style.Hop'])

    def test_an_excluded_style_is_reported_as_such(self):
        training_set, labels = standard_set()
        _, dropped = dataset.filter_styles(training_set, labels, 'Neutral',
                                           excluded_styles=['Sad'])

        self.assertEqual(dropped['style.Sad'], 'excluded by the config')


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.training_set, self.labels = standard_set()
        self.styles = ['style.Neutral', 'style.Sad']
        self.gaits = dataset.neutral_gaits(self.labels, 'Neutral')
        self.pairs = dataset.pair_frames(self.training_set, self.labels, self.styles, self.gaits)

    def test_gaits_come_from_the_neutral_style(self):
        self.assertEqual(self.gaits, ['FW', 'ID'])

    def test_styled_frames_pair_with_the_neutral_frame_of_the_same_phase_and_gait(self):
        sad = self.pairs.style == 1
        phase = self.training_set.phase
        gap = np.angle(np.exp(1j * (phase[self.pairs.input[sad]]
                                    - phase[self.pairs.target[sad]])))

        np.testing.assert_allclose(gap, 0.0, atol=1e-4)
        neutral_clips = {0: 'FW', 1: 'ID'}
        for pair in np.flatnonzero(sad):
            source_clip = 0 if self.pairs.input[pair] < 120 else 1
            self.assertEqual(neutral_clips[source_clip], self.gaits[self.pairs.gait[pair]])

    def test_neutral_frames_pair_with_themselves(self):
        neutral = self.pairs.style == 0

        np.testing.assert_array_equal(self.pairs.input[neutral], self.pairs.target[neutral])
        self.assertEqual(int(neutral.sum()), 180)

    def test_transition_and_dropped_styles_are_not_paired(self):
        self.assertFalse(np.isin(self.pairs.target, np.arange(450, 490)).any())  # TR1
        self.assertFalse(np.isin(self.pairs.target, np.arange(360, 450)).any())  # Hop

    def test_successors_stay_within_a_clip_and_a_split(self):
        has_next = self.pairs.next >= 0
        np.testing.assert_array_equal(self.pairs.target[self.pairs.next[has_next]],
                                      self.pairs.target[has_next] + 1)
        np.testing.assert_array_equal(self.pairs.validation[self.pairs.next[has_next]],
                                      self.pairs.validation[has_next])
        self.assertEqual(int(self.pairs.validation[self.pairs.target < 120].sum()), 12)


class ClosedFormTests(unittest.TestCase):
    def setUp(self):
        self.training_set, labels = standard_set()
        self.styles = ['style.Neutral', 'style.Sad']
        self.gaits = dataset.neutral_gaits(labels, 'Neutral')
        self.spec = dataset.build_spec(self.training_set)
        self.inputs = dataset.frame_inputs(self.training_set, self.spec)
        self.pairs = dataset.pair_frames(self.training_set, labels, self.styles, self.gaits)
        centroids = dataset.gait_centroids(self.training_set, labels, 'Neutral', self.gaits,
                                           self.inputs.gait_features)
        self.closed = dataset.fit_closed_form(self.training_set, self.pairs, self.inputs, 2,
                                              self.gaits, centroids, 0.35)

    def test_the_posture_term_recovers_a_constant_offset(self):
        expected = Rotation.from_rotvec([SAD_PITCH, 0.0, 0.0])
        for g in range(len(self.gaits)):
            posture = Rotation.from_quat(self.closed.posture_rotations[1, g])
            spine = SPINE  # styled bones are every bone here, so the index is the skeleton's
            self.assertLess((posture[spine] * expected.inv()).magnitude(), 1e-3)
            others = [b for b in range(self.spec.n_styled) if b != spine]
            np.testing.assert_allclose(posture[others].magnitude(), 0.0, atol=1e-3)
            self.assertAlmostEqual(float(self.closed.posture_heights[1, g]), -SAD_DROP, places=4)

    def test_neutral_is_the_identity_style(self):
        np.testing.assert_allclose(self.closed.posture_rotations[0, ..., 3], 1.0)
        np.testing.assert_allclose(self.closed.posture_heights[0], 0.0)
        np.testing.assert_allclose(self.closed.speed_ratios[0], 1.0)

    def test_the_speed_ratio_is_styled_over_neutral_and_one_when_idle(self):
        self.assertAlmostEqual(float(self.closed.speed_ratios[1, 0]),
                               SAD_SPEED / NEUTRAL_SPEED, places=3)
        self.assertEqual(float(self.closed.speed_ratios[1, 1]), 1.0)

    def test_gait_weights_pick_the_nearest_centroid_and_sum_to_one(self):
        weights = dataset.gait_weights(self.inputs.gait_features, self.closed.gait_centroids,
                                       0.35)

        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        self.assertGreater(weights[60, 0], 0.99)   # mid neutral walk
        self.assertGreater(weights[150, 1], 0.99)  # neutral idle

    def test_the_root_residual_is_what_the_speed_ratio_leaves(self):
        targets = dataset.pair_targets(self.training_set, self.pairs, self.inputs, self.closed)
        walking = (self.pairs.style == 1) & (self.pairs.gait == 0)

        np.testing.assert_allclose(targets.root_residual[walking], 0.0, atol=2e-2)
        np.testing.assert_allclose(targets.speed_ratio[walking], SAD_SPEED / NEUTRAL_SPEED,
                                   atol=1e-2)


if __name__ == '__main__':
    unittest.main()
