"""
Compares the C# and Python definitions of a character-frame pose on the same frames.

Both sides are unit-tested against the same properties -- that a rigid translation leaves no motion
inside the frame, that the root sits over the frame origin -- but until this existed, nothing had
compared the numbers the two actually produce. That is the gap worth closing, because a
disagreement here does not throw. A model trained on one definition and run on the other simply
produces bad motion, and the cause is invisible from the symptom.

Driven from Unity by ``MoSynth/Pfnn/Check Training Agreement``, which does the C# half and hands the
arrays over. Positions and rotations should agree to float precision. Rates are deliberately not
compared: the two sides differ there by construction -- Python differences consecutive frame-local
poses of a stored database, C# composes the instantaneous rate implied by a pose's own velocity
channels -- so a difference would say nothing that is not already documented.
"""

from __future__ import annotations

import numpy as np

from training_data import load_database


def compare(data_dir: str, db_name: str, frames, csharp_positions, csharp_rotations,
            bone_count: int) -> str:
    """
    :param data_dir: the database folder under StreamingAssets.
    :param db_name: base file name, i.e. the Unity asset's name.
    :param frames: database frame indices the C# arrays were sampled at.
    :param csharp_positions: flat (len(frames), bone_count, 3), from ``CharacterSpacePose.Extract``.
    :param csharp_rotations: flat (len(frames), bone_count, 4) xyzw, from the same call.
    :param bone_count: bones per frame, to reshape by.
    :return: a one-line report, which Unity logs.
    """
    frames = np.asarray(frames, dtype=np.int64)
    theirs_positions = np.asarray(csharp_positions, dtype=np.float64) \
        .reshape(frames.size, bone_count, 3)
    theirs_rotations = np.asarray(csharp_rotations, dtype=np.float64) \
        .reshape(frames.size, bone_count, 4)

    training_set = load_database(data_dir, db_name, with_features=False)
    if training_set.n_bones != bone_count:
        return (f'MISMATCH: the database has {training_set.n_bones} bones but Unity sent '
                f'{bone_count}. The two are not reading the same rig.')

    ours_positions = training_set.positions[frames].astype(np.float64)
    ours_rotations = training_set.rotations[frames].astype(np.float64)

    position_error = np.abs(ours_positions - theirs_positions)

    # A quaternion double-covers SO(3), so compare the rotation rather than the four floats: a
    # negated quaternion is the same orientation and would otherwise read as a total disagreement.
    dot = np.abs(np.sum(ours_rotations * theirs_rotations, axis=-1))
    angle_error = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))

    worst_position = int(np.unravel_index(np.argmax(position_error), position_error.shape)[1])
    worst_rotation = int(np.unravel_index(np.argmax(angle_error), angle_error.shape)[1])

    # The residual grows down the chain because forward kinematics composes in float32 on the C#
    # side and float64 here, so reporting the depth of the worst bone is what tells accumulated
    # rounding apart from a genuine disagreement, which would not care how deep a joint sits.
    depths = _depths(training_set.parents)

    return (f'agreement over {frames.size} frames x {bone_count} bones: '
            f'position max {position_error.max():.2e} m (mean {position_error.mean():.2e}, '
            f'worst {training_set.bone_names[worst_position]} at depth {depths[worst_position]}), '
            f'rotation max {np.degrees(angle_error.max()):.2e} deg '
            f'(mean {np.degrees(angle_error.mean()):.2e}, '
            f'worst {training_set.bone_names[worst_rotation]} at depth {depths[worst_rotation]}, '
            f'deepest bone is at {depths.max()}). '
            f'Rates are not compared -- the two sides differ there by construction.')


def _depths(parents: np.ndarray) -> np.ndarray:
    """How many joints sit between each bone and the root."""
    depths = np.zeros(parents.size, dtype=np.int64)
    for i in range(1, parents.size):
        depths[i] = depths[parents[i]] + 1
    return depths
