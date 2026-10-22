"""
A gait phase read causally off contact flags, one tick at a time.

The baked phase (``GaitPhase`` in Unity, read by :mod:`training.training_data`) interpolates
between footfalls, so it needs the *next* footfall to place a frame -- look-ahead the style stage
does not have. :class:`OnlineGaitPhase` is the causal rule instead: a rising contact edge is a
footfall, the phase extrapolates at the last stride's rate in between, and the error left at a
footfall is closed by a decaying offset rather than a snap, so the reported phase stays continuous.
Its conventions are ``GaitPhase``'s: half a cycle per footfall, a whole one when the same foot falls
twice, the cycle zeroed on a right footfall, and a fixed standing sweep below a ground speed.

This is a port of the C# ``OnlineGaitPhase``. The two must agree tick for tick, because the style
network trains on the phase this module computes and runs on the one C# computes; the shared golden
vectors in ``tests/fixtures/online_gait_phase_golden.json`` hold both to that. Regenerate
them with::

    python -m style.phase --golden tests/fixtures/online_gait_phase_golden.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

TAU = 2.0 * math.pi

LEFT, RIGHT = 0, 1

DEFAULT_PARAMS = {
    'standingSpeed': 0.1,
    'standingPeriod': 1.0,
    'minStepInterval': 0.15,
    'correctionHalfLife': 0.1,
    'minRate': TAU / 2.0,
    'maxRate': TAU / 0.4,
    'standTimeout': 1.0,
}


def wrap(angle: float) -> float:
    """Radians into ``[0, Tau)``."""
    wrapped = angle - math.floor(angle / TAU) * TAU
    # A value a hair below a multiple of Tau can floor onto it, which would return exactly Tau.
    return 0.0 if wrapped >= TAU else wrapped


def wrap_to_pi(angle: float) -> float:
    """Radians into ``(-pi, pi]``."""
    wrapped = wrap(angle)
    return wrapped - TAU if wrapped > math.pi else wrapped


class OnlineGaitPhase:
    """
    The causal gait phase of one character. Call :meth:`update` once per tick.

    Parameter names and defaults match the C# class; see this module's docstring.
    """

    def __init__(self, standing_speed: float = DEFAULT_PARAMS['standingSpeed'],
                 standing_period: float = DEFAULT_PARAMS['standingPeriod'],
                 min_step_interval: float = DEFAULT_PARAMS['minStepInterval'],
                 correction_half_life: float = DEFAULT_PARAMS['correctionHalfLife'],
                 min_rate: float = DEFAULT_PARAMS['minRate'],
                 max_rate: float = DEFAULT_PARAMS['maxRate'],
                 stand_timeout: float = DEFAULT_PARAMS['standTimeout']):
        self.standing_speed = standing_speed
        self.standing_period = standing_period
        self.min_step_interval = min_step_interval
        self.correction_half_life = correction_half_life
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.stand_timeout = stand_timeout
        self.reset()

    @classmethod
    def from_params(cls, params: dict) -> 'OnlineGaitPhase':
        """Build from the camelCase dict the golden file stores."""
        return cls(params['standingSpeed'], params['standingPeriod'], params['minStepInterval'],
                   params['correctionHalfLife'], params['minRate'], params['maxRate'],
                   params['standTimeout'])

    def params(self) -> dict:
        return {
            'standingSpeed': self.standing_speed,
            'standingPeriod': self.standing_period,
            'minStepInterval': self.min_step_interval,
            'correctionHalfLife': self.correction_half_life,
            'minRate': self.min_rate,
            'maxRate': self.max_rate,
            'standTimeout': self.stand_timeout,
        }

    def reset(self) -> None:
        """Forget everything, so the next :meth:`update` initialises again."""
        self._phase = 0.0
        self.rate = 0.0
        self._offset = 0.0
        self._time = 0.0
        self._last_footfall_time = -math.inf
        self._last_foot = None
        self._last_foot_time = [-math.inf, -math.inf]
        self._previous_contact = [False, False]
        self._initialized = False

    @property
    def phase(self) -> float:
        """The reported phase, in ``[0, Tau)``."""
        return wrap(self._phase + self._offset)

    def update(self, dt: float, left: bool, right: bool, ground_speed: float) \
            -> tuple[float, float]:
        """
        Advance one tick.

        :param dt: seconds since the previous tick. Ignored on the first call.
        :param left: whether the left contact bone is planted this tick.
        :param right: whether the right contact bone is planted this tick.
        :param ground_speed: the character frame's planar speed, m/s.
        :return: ``(phase in [0, Tau), rate in rad/s)``.
        """
        contacts = (bool(left), bool(right))

        if not self._initialized:
            # A foot already down when tracking starts is not a footfall.
            self._previous_contact = list(contacts)
            self._phase = 0.0
            self.rate = TAU / self.standing_period
            self._offset = 0.0
            self._time = 0.0
            self._initialized = True
            return self.phase, self.rate

        self._time += dt
        self._phase = wrap(self._phase + self.rate * dt)

        for foot in (LEFT, RIGHT):
            if not contacts[foot] or self._previous_contact[foot]:
                continue
            if self._time - self._last_foot_time[foot] < self.min_step_interval:
                continue
            self._footfall(foot)

        self._previous_contact = list(contacts)

        if ground_speed < self.standing_speed \
                and self._time - self._last_footfall_time >= self.stand_timeout:
            self.rate = TAU / self.standing_period

        if self.correction_half_life > 0.0:
            self._offset *= math.exp(-math.log(2.0) * dt / self.correction_half_life)
        else:
            self._offset = 0.0

        return self.phase, self.rate

    def _footfall(self, foot: int) -> None:
        target = 0.0 if foot == RIGHT else math.pi

        if self._last_footfall_time > -math.inf:
            interval = self._time - self._last_footfall_time
            cycles = 1.0 if foot == self._last_foot else 0.5
            if interval > 0.0:
                self.rate = min(max(cycles * TAU / interval, self.min_rate), self.max_rate)

        # The phase jumps onto the target and the offset absorbs the jump, so what is reported
        # moves continuously and only converges on the target as the offset decays.
        error = wrap_to_pi(target - self._phase)
        self._phase = wrap(self._phase + error)
        self._offset -= error

        self._last_footfall_time = self._time
        self._last_foot = foot
        self._last_foot_time[foot] = self._time


def causal_phase(contacts: np.ndarray, ground_speed: np.ndarray, frame_time: float,
                 tracker: OnlineGaitPhase | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Run :class:`OnlineGaitPhase` over one clip, from its first frame.

    :param contacts: (n, 2) contact flags, left then right -- ``GaitPhase.Foot``'s order, which is
        the contact slot order.
    :param ground_speed: (n,) the character frame's planar speed, m/s.
    :param frame_time: seconds per frame.
    :param tracker: a configured tracker to run; it is reset first. The defaults when None.
    :return: ``(phase, rate)``, both (n,) float32.
    """
    contacts = np.asarray(contacts).astype(bool)
    ground_speed = np.asarray(ground_speed, dtype=np.float64)
    if contacts.ndim != 2 or contacts.shape[1] != 2:
        raise ValueError(f'causal phase reads a left and a right contact, got {contacts.shape}')

    tracker = tracker or OnlineGaitPhase()
    tracker.reset()

    n = contacts.shape[0]
    phase = np.zeros(n, dtype=np.float32)
    rate = np.zeros(n, dtype=np.float32)
    for i in range(n):
        phase[i], rate[i] = tracker.update(frame_time, contacts[i, 0], contacts[i, 1],
                                           ground_speed[i])
    return phase, rate


# Golden vectors --------------------------------------------------------------------------------

GOLDEN_DT = 1.0 / 30.0
GOLDEN_TICKS = 320
GOLDEN_SEED = 7

# Comparisons closer than this to a threshold could resolve differently in C#'s float arithmetic
# than in Python's doubles, so the generator refuses a sequence that lands on one.
_BOUNDARY_MARGIN = 1e-4


def _golden_inputs() -> list[list[float]]:
    """
    A scripted sequence: a stand, a walk start, jittered alternating footfalls, a double contact,
    a repeated foot, a re-contact inside ``minStepInterval``, and a stop back to standing.
    """
    rng = np.random.default_rng(GOLDEN_SEED)
    contact = np.zeros((GOLDEN_TICKS, 2), dtype=bool)

    def plant(foot, land, lift):
        contact[land:min(lift, GOLDEN_TICKS), foot] = True

    # Standing on both feet, then the left lifts first to start walking.
    plant(LEFT, 0, 45)
    plant(RIGHT, 0, 57)

    # Alternating footfalls with a jittered step and a stance a little longer than a step.
    tick, foot = 50, LEFT
    while tick < 125:
        plant(foot, tick, tick + 20)
        tick += 15 + int(rng.integers(-2, 3))
        foot = 1 - foot

    # Both feet off, then both land on the same tick.
    contact[125:140] = False
    plant(LEFT, 140, 150)
    plant(RIGHT, 140, 190)

    # The left falls twice running while the right stays planted -- a missed right contact.
    plant(LEFT, 158, 165)
    plant(LEFT, 172, 182)

    # A right footfall, then a lift and a re-contact four ticks later: too fast to be a step.
    contact[190:195, RIGHT] = False
    plant(RIGHT, 195, 198)
    plant(RIGHT, 199, 215)

    # Back to alternating, then the stop: both feet stay planted from the last footfall on.
    tick, foot = 208, LEFT
    while tick < 262:
        plant(foot, tick, tick + 20)
        tick += 14 + int(rng.integers(-2, 3))
        foot = 1 - foot
    last_footfall = tick
    plant(foot, last_footfall, GOLDEN_TICKS)
    plant(1 - foot, last_footfall - 12, GOLDEN_TICKS)

    # Speed: stands, ramps to walking, jitters, and decays to rest over 40 ticks after the stop.
    speed = np.zeros(GOLDEN_TICKS)
    speed[:42] = 0.02
    speed[42:60] = np.linspace(0.02, 1.2, 18)
    speed[60:last_footfall] = 1.2 + rng.uniform(-0.05, 0.05, last_footfall - 60)
    decay = np.linspace(1.2, 0.0, 40)
    speed[last_footfall:last_footfall + 40] = decay[:GOLDEN_TICKS - last_footfall]
    speed[last_footfall + 40:] = 0.01

    return [[float(contact[i, 0]), float(contact[i, 1]), round(float(speed[i]), 6)]
            for i in range(GOLDEN_TICKS)]


def _check_no_boundary_ticks(inputs, params, dt) -> None:
    """Refuse a sequence whose threshold tests sit within float noise of their threshold."""
    time, last_footfall, last_foot_time = 0.0, -math.inf, [-math.inf, -math.inf]
    previous = [bool(inputs[0][0]), bool(inputs[0][1])]

    for i, (left, right, speed) in enumerate(inputs[1:], start=1):
        time += dt
        for foot, now in ((LEFT, bool(left)), (RIGHT, bool(right))):
            if now and not previous[foot]:
                since = time - last_foot_time[foot]
                if abs(since - params['minStepInterval']) < _BOUNDARY_MARGIN:
                    raise ValueError(f'tick {i}: a rising edge sits on minStepInterval')
                if since >= params['minStepInterval']:
                    last_footfall = time
                    last_foot_time[foot] = time
        previous = [bool(left), bool(right)]

        if abs(speed - params['standingSpeed']) < _BOUNDARY_MARGIN:
            raise ValueError(f'tick {i}: the speed sits on standingSpeed')
        if speed < params['standingSpeed'] \
                and abs(time - last_footfall - params['standTimeout']) < _BOUNDARY_MARGIN:
            raise ValueError(f'tick {i}: the standing test sits on standTimeout')


def golden() -> dict:
    """
    ``{"params", "dt", "inputs": [[left, right, speed], ...], "expected": [[phase, rate], ...]}``,
    the document both the C# and the Python suites assert against.
    """
    params = dict(DEFAULT_PARAMS)
    inputs = _golden_inputs()
    _check_no_boundary_ticks(inputs, params, GOLDEN_DT)

    tracker = OnlineGaitPhase.from_params(params)
    expected = [list(tracker.update(GOLDEN_DT, bool(left), bool(right), speed))
                for left, right, speed in inputs]
    return {'params': params, 'dt': GOLDEN_DT, 'inputs': inputs, 'expected': expected}


def _format_golden(document: dict) -> str:
    """The document as JSON with one tick per line, so a regenerated file diffs readably."""
    def rows(values):
        return ',\n'.join(f'  {json.dumps(row)}' for row in values)

    return (f'{{\n "params": {json.dumps(document["params"])},\n'
            f' "dt": {json.dumps(document["dt"])},\n'
            f' "inputs": [\n{rows(document["inputs"])}\n ],\n'
            f' "expected": [\n{rows(document["expected"])}\n ]\n}}\n')


def _main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--golden', required=True, help='where to write the golden vectors')
    args = parser.parse_args(argv)

    document = golden()
    os.makedirs(os.path.dirname(os.path.abspath(args.golden)), exist_ok=True)
    with open(args.golden, 'w', encoding='utf-8', newline='\n') as f:
        f.write(_format_golden(document))
    print(f'{len(document["inputs"])} ticks -> {args.golden}')


if __name__ == '__main__':
    _main()
