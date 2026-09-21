"""
The vector-packing primitives every neural model here shares: named blocks, per-block loss
weights, input normalisation, and bone selection.

:mod:`training.training_data` builds the per-frame arrays; each model then decides how to lay those
arrays out as one flat vector. That decision is the model's, but the *machinery* for making it
is not: a block layout, a way to slice a block back out, a check that a checkpoint was written
against the layout being read, and weights that make a three-float block count for as much as a
hundred-float one. All four were written for the PFNN first and are none of them specific to it.

**There are two copies of this code right now.** :mod:`pfnn.dataset` still carries its own, and
this module is a deliberate copy rather than a refactor of it -- the PFNN work was in flight when
learned motion matching needed the same primitives, and editing that module would have collided.
Fixing one copy and not the other is the failure to watch for; the duplication goes away when
:mod:`pfnn.dataset` re-exports from here.

Conventions are inherited unchanged from :mod:`training.training_data`: quaternions xyzw, y-up left-handed
with the character facing +z, every rate per second, and the ground plane written as ``(x, z)``.
"""

from __future__ import annotations

import numpy as np


def layout(blocks) -> list:
    """
    ``(name, offset, float_count)`` per block, from ``(name, float_count)`` pairs in vector order.

    :param blocks: the blocks of one vector, in the order they are concatenated.
    """
    result, offset = [], 0
    for name, count in blocks:
        result.append((name, offset, count))
        offset += count
    return result


def vector_size(block_layout) -> int:
    """Total floats described by a layout."""
    if not block_layout:
        return 0
    _, offset, count = block_layout[-1]
    return offset + count


def block(vectors: np.ndarray, block_layout, name: str) -> np.ndarray:
    """The named block of a packed vector, or of a batch of them, as a view."""
    for entry_name, offset, count in block_layout:
        if entry_name == name:
            return vectors[..., offset:offset + count]
    raise KeyError(f'no block named {name!r}; have {[n for n, _, _ in block_layout]}')


def check_blocks(stored_names, block_layout, what: str = 'output') -> None:
    """
    Refuse a checkpoint packed with a different set of blocks than this code reads.

    The failure this catches is silent rather than loud. Blocks are appended, so every block an
    older checkpoint does carry still slices out correctly, and the new one comes back as a
    truncated view instead of raising -- a character that moves badly for no visible reason. The
    checkpoint stores the names it was written with for the same reason ``.mmpose`` carries a
    skeleton block instead of a version number: the content is the check.

    :param stored_names: the block names the checkpoint was written with, in order.
    :param block_layout: ``(name, offset, count)`` per block, as this code reads them.
    :param what: what the blocks describe, for the message.
    :raises ValueError: naming the first block that differs.
    """
    expected = [name for name, _, _ in block_layout]
    stored = [str(name) for name in stored_names]
    if stored == expected:
        return

    for i, name in enumerate(expected):
        if i >= len(stored):
            raise ValueError(f'checkpoint has no {name!r} {what} block; it was trained against an '
                             f'older layout and has to be retrained')
        if stored[i] != name:
            raise ValueError(f'checkpoint {what} block {i} is {stored[i]!r} where this code reads '
                             f'{name!r}; it has to be retrained')

    raise ValueError(f'checkpoint carries {what} blocks this code does not read: '
                     f'{stored[len(expected):]}; it has to be retrained')


def block_weights(block_layout, importance) -> np.ndarray:
    """
    Per-float weights that make each block count for what it is worth, not for how wide it is.

    Normalising the targets equalises the *floats*, which is not the same thing and is the trap this
    exists to avoid: the root velocity is three floats beside a hundred and sixty of joint rotation,
    so an unweighted loss spends about 2% of its gradient on the only block that decides whether the
    character travels, stands still, or turns.

    Weights are scaled so they average one, which keeps the loss the same order of magnitude as the
    unweighted mean and lets an existing learning rate carry over.

    :param block_layout: ``(name, offset, count)`` per block.
    :param importance: what each block is worth, by name.
    :raises KeyError: if a block has no importance, rather than silently weighting it zero.
    """
    total = vector_size(block_layout)
    weights = np.zeros(total, dtype=np.float32)

    for name, offset, count in block_layout:
        if name not in importance:
            raise KeyError(f'no loss importance given for the {name!r} block; '
                           f'have {sorted(importance)}')
        if count:
            weights[offset:offset + count] = importance[name] / count

    return (weights * (total / weights.sum())).astype(np.float32)


def normalization(vectors: np.ndarray):
    """
    ``(mean, std)`` over a set of packed vectors, with constant columns left alone.

    A column that never varies -- a contact flag on a database with no flight phase, say -- would
    otherwise be divided by zero and arrive at the network as an infinity.
    """
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def select_bones(bone_names, parents: np.ndarray, excluded_names) -> np.ndarray:
    """
    The bones a model predicts: every bone the caller did not exclude, in skeleton order.

    The exclusion list is authored on the Unity config and travels here by name, so a joint that
    moves in the hierarchy keeps its setting. Names it does not recognise are an error rather than
    a silent no-op: a misspelt exclusion would quietly train a different model than intended, and
    the symptom would not appear until the checkpoint failed to load.

    :param bone_names: every bone of the database, in depth-first order.
    :param parents: (n_bones,) parent index per bone, -1 for the root.
    :param excluded_names: bone names not to predict.
    :raises ValueError: if a name is unknown, if the root is excluded, or if the remaining set is
        not closed under parent -- a model predicts rotations, and a rotation needs its parent's
        frame to be resolved against.
    """
    bone_names = list(bone_names)
    excluded = set(excluded_names)

    unknown = sorted(excluded.difference(bone_names))
    if unknown:
        raise ValueError(f'excluded bones not in this skeleton: {", ".join(unknown)}')

    keep = np.array([name not in excluded for name in bone_names], dtype=bool)
    if not keep[0]:
        raise ValueError(f'the root bone {bone_names[0]!r} cannot be excluded: it is the frame '
                         'every other rotation is resolved against')

    orphans = [bone_names[i] for i in range(1, len(bone_names))
               if keep[i] and not keep[parents[i]]]
    if orphans:
        raise ValueError(
            'these bones are kept but their parent is excluded, so there is no frame to apply '
            f'their rotation in: {", ".join(orphans)}. Exclude a bone together with its subtree.')

    return np.flatnonzero(keep).astype(np.int64)
