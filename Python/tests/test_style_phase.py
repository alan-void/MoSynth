"""
The causal gait phase, rule by rule, and the golden vectors the C# port is held to.

The golden file is asserted here as well as in Unity: if the Python rule changes, this suite fails
until the file is regenerated, and then the C# suite fails until the port follows.
"""

import json
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from style import phase as online  # noqa: E402

DT = 1.0 / 30.0
TAU = 2.0 * math.pi
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                           'online_gait_phase_golden.json')


def wrapped_gap(a, b):
    return abs((a - b + math.pi) % TAU - math.pi)


def run(ticks, speed=1.0, tracker=None):
    """Feed ``(left, right)`` pairs; return the reported ``(phase, rate)`` per tick."""
    tracker = tracker or online.OnlineGaitPhase()
    return [tracker.update(DT, left, right, speed) for left, right in ticks]


class RuleTests(unittest.TestCase):
    def test_the_first_tick_initialises_and_a_foot_already_down_is_no_footfall(self):
        tracker = online.OnlineGaitPhase()
        phase, rate = tracker.update(DT, True, True, 1.0)

        self.assertEqual(phase, 0.0)
        self.assertAlmostEqual(rate, TAU / tracker.standing_period)
        # Still down on the next tick: no rising edge, so the phase only extrapolates.
        phase, _ = tracker.update(DT, True, True, 1.0)
        self.assertAlmostEqual(phase, TAU * DT)

    def test_a_right_footfall_converges_on_zero_and_a_left_on_pi(self):
        tracker = online.OnlineGaitPhase(correction_half_life=0.0)
        run([(False, False)] * 10 + [(False, True)], tracker=tracker)
        self.assertAlmostEqual(tracker.phase, 0.0)

        run([(False, True)] * 10 + [(True, True)], tracker=tracker)
        self.assertAlmostEqual(tracker.phase, math.pi)

    def test_the_correction_is_continuous_and_decays(self):
        tracker = online.OnlineGaitPhase()
        before = run([(False, False)] * 10, tracker=tracker)[-1][0]
        after = tracker.update(DT, False, True, 1.0)[0]

        # One tick of extrapolation plus one tick of the offset's decay, not a jump onto the target.
        decayed = 1.0 - math.exp(-math.log(2.0) * DT / tracker.correction_half_life)
        self.assertLess(wrapped_gap(after, before), TAU * DT + decayed * math.pi + 1e-6)
        self.assertGreater(wrapped_gap(after, 0.0), 0.1)
        for _ in range(30):
            tracker.update(DT, False, True, 1.0)
        self.assertLess(wrapped_gap(tracker.phase, 30 * DT * tracker.rate), 0.05)

    def test_alternating_feet_set_the_rate_to_half_a_cycle_per_interval(self):
        ticks = [(False, False)] * 2 + [(True, False)] + [(False, False)] * 14 + [(False, True)]
        _, rate = run(ticks)[-1]

        self.assertAlmostEqual(rate, 0.5 * TAU / (15 * DT), places=5)

    def test_the_same_foot_twice_is_a_whole_cycle(self):
        ticks = [(False, False)] * 2 + [(True, False)] + [(False, False)] * 19 + [(True, False)]
        _, rate = run(ticks)[-1]

        self.assertAlmostEqual(rate, TAU / (20 * DT), places=5)

    def test_a_recontact_inside_the_min_step_interval_is_ignored(self):
        tracker = online.OnlineGaitPhase()
        run([(False, False)] * 2 + [(True, False)] + [(False, False)] * 14 + [(False, True)],
            tracker=tracker)
        rate = tracker.rate
        run([(False, False)] * 2 + [(False, True)], tracker=tracker)  # 3 ticks: too fast

        self.assertEqual(tracker.rate, rate)

    def test_both_feet_on_one_tick_leave_the_rate_alone_for_the_second(self):
        tracker = online.OnlineGaitPhase(correction_half_life=0.0)
        run([(False, False)] * 2 + [(False, True)] + [(False, False)] * 20 + [(True, True)],
            tracker=tracker)

        # Left first (half a cycle over 21 ticks), then right on the same tick: interval 0.
        self.assertAlmostEqual(tracker.rate, 0.5 * TAU / (21 * DT), places=5)
        self.assertAlmostEqual(tracker.phase, 0.0)

    def test_the_rate_is_clamped(self):
        tracker = online.OnlineGaitPhase()
        run([(False, False)] * 2 + [(True, False)] + [(False, False)] * 5 + [(False, True)],
            tracker=tracker)
        self.assertAlmostEqual(tracker.rate, tracker.max_rate)

    def test_standing_returns_to_the_standing_rate_after_the_timeout(self):
        tracker = online.OnlineGaitPhase()
        ticks = [(False, False)] * 2 + [(True, False)] + [(False, False)] * 14 + [(False, True)]
        run(ticks, tracker=tracker)
        walking = tracker.rate

        run([(False, True)] * 20, speed=0.0, tracker=tracker)
        self.assertEqual(tracker.rate, walking)
        run([(False, True)] * 15, speed=0.0, tracker=tracker)
        self.assertAlmostEqual(tracker.rate, TAU / tracker.standing_period)

    def test_causal_phase_runs_the_rule_over_a_clip(self):
        contacts = np.zeros((40, 2), dtype=bool)
        contacts[10:20, 1] = True
        contacts[25:35, 0] = True
        phase, rate = online.causal_phase(contacts, np.ones(40), DT)

        expected = run([tuple(row) for row in contacts])
        np.testing.assert_allclose(phase, [p for p, _ in expected], atol=1e-6)
        np.testing.assert_allclose(rate, [r for _, r in expected], atol=1e-5)


class GoldenTests(unittest.TestCase):
    def setUp(self):
        with open(GOLDEN_PATH, 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def test_the_rule_reproduces_the_golden_vectors(self):
        tracker = online.OnlineGaitPhase.from_params(self.document['params'])
        dt = self.document['dt']
        for i, ((left, right, speed), (phase, rate)) in enumerate(
                zip(self.document['inputs'], self.document['expected'])):
            got_phase, got_rate = tracker.update(dt, bool(left), bool(right), speed)
            self.assertLess(wrapped_gap(got_phase, phase), 1e-3, f'tick {i}')
            self.assertAlmostEqual(got_rate, rate, delta=1e-3, msg=f'tick {i}')

    def test_the_file_is_what_the_generator_writes(self):
        regenerated = json.loads(json.dumps(online.golden()))

        self.assertEqual(regenerated, self.document,
                         'regenerate with: python -m style.phase --golden '
                         'tests/fixtures/online_gait_phase_golden.json')

    def test_the_sequence_covers_every_case(self):
        inputs = np.asarray(self.document['inputs'])
        contact = inputs[:, :2] > 0.5
        rises = contact[1:] & ~contact[:-1]

        self.assertTrue((rises[:, 0] & rises[:, 1]).any(), 'a double contact')
        self.assertTrue((inputs[:40, 2] < 0.1).all(), 'a stand')
        self.assertTrue((inputs[-10:, 2] < 0.1).all(), 'a stop')
        rates = np.asarray(self.document['expected'])[:, 1]
        self.assertGreater(len(np.unique(np.round(rates, 4))), 5, 'jittered strides')


if __name__ == '__main__':
    unittest.main()
