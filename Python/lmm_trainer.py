"""
Fits the compressor/decompressor pair of Learned Motion Matching, and bakes the latents.

This is the first of the three training stages -- the one phase A needs. It learns a latent ``Z``
that, together with the matching feature vector ``X`` a classic matcher already searches, is enough
to reconstruct the whole pose. The stepper and the projector are fitted against these latents in
later phases, which is why the latents are baked and stored rather than recomputed.

**The loss is Algorithm 1 of the paper**, and its shape is the method:

* Every pose term is **L1 on denormalised quantities**. L1 rather than MSE because the mean of two
  plausible poses is generally not a pose while their median is, and denormalised because the
  weights below trade a metre of position error against a radian of angle -- a trade that only
  means anything in real units.
* Every pose is scored **twice, in two spaces**: joint-locally as the network predicted it, and
  again in character space after forward kinematics. The paper is explicit that this is the point
  -- a naive per-joint loss gives "jittery, low quality motion", because it cannot see that a small
  error at the hip is a large one at the hand.
* Both spaces are scored **again as velocities**, differenced across the frame pair. That is what
  makes the output change smoothly in time rather than only sit in the right place per frame.
* Three regularisers act on the latent: magnitude, energy, and rate of change. The last is what
  leaves ``Z`` steppable at all in phase B, and it applies to a **per-second** derivative -- a
  factor of sixty against the per-frame delta it would be easy to write instead.

The weights are in :mod:`lmm_dataset`; they are the author's released training code, since the
paper states none.

Validation holds out a **contiguous tail** rather than a random subset, for the reason
``pfnn_trainer`` does: neighbouring frames of an animation are nearly the same pose, so a random
split puts near-duplicates of the validation set into training and reports a number that measures
nothing. The parameters that scored best on it are what gets written, not the last ones.

Runs from the Unity Editor through PythonNET, or standalone::

    python lmm_trainer.py ../Assets/StreamingAssets/MMDatabases/MM_LafanCorrected_Edinburg \\
        MM_LafanCorrected_Edinburg --out model.lmm.npz --iterations 150000
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch

import lmm_dataset
import lmm_fk
import lmm_io
from lmm_model import Compressor, Decompressor, resolve_device
from training_data import load_database

LOSS_COLUMNS = ('local', 'character', 'local_velocity', 'character_velocity',
                'latent', 'total', 'validation')

# Frames baked per forward pass. Large enough that the per-call overhead vanishes, small enough
# that a 200k-frame database does not need the whole latent table resident on the GPU at once.
BAKE_BATCH = 4096

# Pairs the held-out score is taken over. The tail of a large database is tens of thousands of
# frames and scoring all of them every thousand iterations would cost more than the training step.
VALIDATION_SAMPLE = 8192


def _noop_progress(stage: str, fraction: float) -> None:
    pass


def train(data_dir: str,
          db_name: str,
          out_path: str,
          excluded_bones=(),
          feature_weights=(),
          latent_size: int = lmm_dataset.DEFAULT_LATENT_SIZE,
          latent_velocity_weight: float = lmm_dataset.LATENT_VELOCITY_WEIGHT,
          compressor_hidden: int = 0,
          decompressor_hidden: int = 0,
          iterations: int = 150000,
          batch_size: int = 256,
          learning_rate: float = 1e-3,
          weight_decay: float = 1e-3,
          learning_rate_decay: float = 0.99,
          decay_interval: int = 1000,
          validation_fraction: float = 0.1,
          validation_interval: int = 1000,
          max_seconds: float = 0.0,
          seed: int = 42,
          device: str = 'auto',
          progress=None) -> dict:
    """
    Build the training set, fit the autoencoder, bake the latents, and write the checkpoint.

    :param data_dir: the database folder under StreamingAssets.
    :param db_name: base file name, i.e. the Unity asset's name.
    :param out_path: where to write the ``.lmm.npz``.
    :param excluded_bones: bone names the decompressor does not predict.
    :param feature_weights: one authored search weight per feature definition, which travels into
        the checkpoint so the phase C projector approximates the search that is actually run.
    :param latent_velocity_weight: ``w_vreg``, the penalty on how fast the latent moves. The one
        loss weight worth tuning -- see :mod:`lmm_dataset`. Too low and the phase B stepper has
        noise to advance; too high and the latent cannot carry enough to reconstruct from.
    :param compressor_hidden: hidden width; 0 takes the reference implementation's.
    :param decompressor_hidden: likewise.
    :param iterations: optimiser steps. Counted in steps rather than epochs because the paper's
        schedule is, and because an epoch means something different on every database size.
    :param learning_rate_decay: multiplied into the learning rate every ``decay_interval`` steps.
    :param validation_interval: steps between held-out scores. The best-scoring parameters are
        what is written.
    :param max_seconds: stop after this long regardless, 0 for no limit. A training run that has to
        fit a budget should be cut by the clock rather than by guessing an iteration count.
    :param progress: ``(stage, fraction)`` callback, which the Editor drives a progress bar from.
    :return: a summary dict, logged by the caller.
    """
    progress = progress or _noop_progress
    started = time.time()
    torch.manual_seed(seed)

    progress('Reading database', 0.0)
    training_set = load_database(data_dir, db_name, with_features=True)
    spec = lmm_dataset.build_spec(training_set, excluded_bones, latent_size)

    progress('Packing vectors', 0.05)
    x, y, q, latent_exists = lmm_dataset.build_vectors(training_set, spec)
    pairs = lmm_dataset.training_pairs(training_set)
    if pairs.size < 2:
        raise ValueError(
            f'{pairs.size} usable frame pairs in {db_name}: nothing to train on. A latent spans a '
            'frame and its successor, so a database of one-frame clips, or one whose feature '
            'vectors are all invalid, contributes none.')

    # By index, and the pairs are still in database order, so this is the tail of the animation
    # rather than a scattering of frames from all over it.
    split = max(1, int(round(pairs.size * (1.0 - validation_fraction))))
    train_pairs, validation_pairs = pairs[:split], pairs[split:]

    pose_layout = spec.pose_layout()
    character_layout = spec.character_layout()
    y_mean, y_std = lmm_dataset.group_normalization(y[latent_exists], pose_layout)
    q_mean, q_std = lmm_dataset.group_normalization(q[latent_exists], character_layout)

    torch_device = resolve_device(device)
    x_t = torch.from_numpy(x).to(torch_device)
    y_t = torch.from_numpy(y).to(torch_device)
    q_t = torch.from_numpy(q).to(torch_device)

    y_mean_t = torch.from_numpy(y_mean).to(torch_device)
    y_std_t = torch.from_numpy(y_std).to(torch_device)
    q_mean_t = torch.from_numpy(q_mean).to(torch_device)
    q_std_t = torch.from_numpy(q_std).to(torch_device)
    offsets_t = torch.from_numpy(spec.rest_offsets).to(torch_device)

    rotation_slice = _slice(pose_layout, 'rotations_6d')
    height_slice = _slice(pose_layout, 'root_height')
    hierarchy = lmm_fk.Hierarchy(spec.parents, torch_device)
    frame_time = spec.frame_time

    def weights_of(block_layout, table):
        return torch.from_numpy(_float_weights(block_layout, table)).to(torch_device)

    local_weights = weights_of(pose_layout, lmm_dataset.LOCAL_WEIGHTS)
    character_weights = weights_of(character_layout, lmm_dataset.CHARACTER_WEIGHTS)
    local_rate_weights = weights_of(pose_layout, lmm_dataset.LOCAL_RATE_WEIGHTS)
    character_rate_weights = weights_of(character_layout, lmm_dataset.CHARACTER_RATE_WEIGHTS)

    compressor = Compressor(
        spec.pose_size, spec.character_size, spec.latent_size,
        **({'hidden_units': compressor_hidden} if compressor_hidden else {})).to(torch_device)
    decompressor = Decompressor(
        spec.feature_size, spec.latent_size, spec.pose_size,
        **({'hidden_units': decompressor_hidden} if decompressor_hidden else {})).to(torch_device)

    # amsgrad and this weight decay are the released training code's; the paper says RAdam, which
    # differs mainly in the warmup that a decaying schedule already provides.
    optimizer = torch.optim.AdamW(
        list(compressor.parameters()) + list(decompressor.parameters()),
        lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

    def decode(frames: torch.Tensor):
        """``(Y_hat, Q_hat, Z)`` for a batch of frames, all in the database's own units."""
        latent = compressor.encode((y_t[frames] - y_mean_t) / y_std_t,
                                   (q_t[frames] - q_mean_t) / q_std_t)
        predicted = decompressor.decode(x_t[frames], latent) * y_std_t + y_mean_t

        local_six = predicted[:, rotation_slice].reshape(frames.shape[0], spec.n_bones, 6)
        character = lmm_fk.character_vector_from_fk(
            local_six, predicted[:, height_slice.start], offsets_t, hierarchy)
        return predicted, character, latent

    def evaluate(frames: torch.Tensor):
        """Algorithm 1 over a batch of pair start frames; the columns of ``LOSS_COLUMNS``."""
        # Both frames of every pair go through as one batch. The step is dominated by the number of
        # operations rather than their size, so two passes of n cost nearly twice one pass of 2n.
        both = torch.cat([frames, frames + 1])
        predicted, character, latent = decode(both)
        n = frames.shape[0]

        local = _weighted(predicted, y_t[both], local_weights)
        chr_space = _weighted(character, q_t[both], character_weights)

        local_rate = _weighted((predicted[:n] - predicted[n:]) / frame_time,
                               (y_t[frames] - y_t[frames + 1]) / frame_time, local_rate_weights)
        character_rate = _weighted((character[:n] - character[n:]) / frame_time,
                                   (q_t[frames] - q_t[frames + 1]) / frame_time,
                                   character_rate_weights)

        regularisation = (
            lmm_dataset.LATENT_SPARSITY_WEIGHT * latent.abs().mean()
            + lmm_dataset.LATENT_MAGNITUDE_WEIGHT * latent.pow(2).mean()
            + latent_velocity_weight
            * ((latent[:n] - latent[n:]) / frame_time).abs().mean())

        return local, chr_space, local_rate, character_rate, regularisation

    train_t = torch.from_numpy(train_pairs).to(torch_device)
    validation_t = torch.from_numpy(
        validation_pairs[np.linspace(0, validation_pairs.size - 1,
                                     min(VALIDATION_SAMPLE, validation_pairs.size)).astype(np.int64)]
        if validation_pairs.size else validation_pairs).to(torch_device)

    # Accumulated on the device. Reading a loss back per step would synchronise the GPU six times
    # an iteration, which on a step this small costs more than the arithmetic it reports on.
    losses = []
    running = torch.zeros(len(LOSS_COLUMNS) - 1, device=torch_device)
    running_count = 0
    best_validation, best_parameters, best_iteration = float('inf'), None, 0
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ran = 0

    for step in range(iterations):
        compressor.train()
        decompressor.train()
        frames = train_t[torch.randint(train_t.numel(), (batch_size,), generator=generator)
                         .to(torch_device)]

        optimizer.zero_grad(set_to_none=True)
        terms = evaluate(frames)
        loss = sum(terms)
        loss.backward()
        optimizer.step()

        running += torch.stack([term.detach() for term in terms] + [loss.detach()])
        running_count += 1
        ran = step + 1

        if (step + 1) % decay_interval == 0:
            schedule.step()

        if (step + 1) % validation_interval == 0 or step + 1 == iterations:
            validation = _validate(evaluate, compressor, decompressor, validation_t)
            losses.append(tuple((running / max(1, running_count)).tolist()) + (validation,))
            running = torch.zeros(len(LOSS_COLUMNS) - 1, device=torch_device)
            running_count = 0

            if validation < best_validation:
                best_validation, best_iteration = validation, step + 1
                best_parameters = (copy.deepcopy(compressor.state_dict()),
                                   copy.deepcopy(decompressor.state_dict()))

            progress('Training', 0.05 + 0.85 * (step + 1) / iterations)

            if max_seconds and time.time() - started > max_seconds:
                break

    # The held-out curve of a small database turns back up well before the last step, so writing
    # the final parameters would ship a model measurably worse than one this run already had.
    if best_parameters is not None:
        compressor.load_state_dict(best_parameters[0])
        decompressor.load_state_dict(best_parameters[1])

    progress('Baking latents', 0.92)
    latents = _bake_latents(compressor, y_t, q_t, y_mean_t, y_std_t, q_mean_t, q_std_t,
                            latent_exists, spec.latent_size, torch_device)
    diagnostics = _latent_diagnostics(decompressor, x_t, y_t, y_mean, y_std, latents,
                                      latent_exists, validation_pairs, pose_layout, torch_device)

    z_mean = latents[latent_exists].mean(axis=0).astype(np.float32)
    z_std = latents[latent_exists].std(axis=0).astype(np.float32)
    z_std[z_std < 1e-6] = 1.0

    progress('Writing checkpoint', 0.98)
    lmm_io.save_checkpoint(
        out_path,
        compressor_weights=compressor.weights(), compressor_biases=compressor.biases(),
        decompressor_weights=decompressor.weights(), decompressor_biases=decompressor.biases(),
        # Unity's FeatureSet already standardised X the way the paper's section 3 does, so a second
        # layer here would divide by a spread that has already been divided out.
        x_mean=np.zeros(spec.feature_size, dtype=np.float32),
        x_std=np.ones(spec.feature_size, dtype=np.float32),
        y_mean=y_mean, y_std=y_std, q_mean=q_mean, q_std=q_std, z_mean=z_mean, z_std=z_std,
        latents=latents, latent_valid=latent_exists,
        feature_weights=lmm_dataset.feature_weights(training_set, feature_weights),
        bone_names=spec.bone_names,
        output_blocks=[name for name, _, _ in pose_layout],
        character_blocks=[name for name, _, _ in character_layout],
        parents=spec.parents, rest_offsets=spec.rest_offsets,
        feature_names=training_set.feature_names,
        feature_widths=training_set.feature_widths,
        feature_counts=training_set.feature_counts,
        n_trajectory_features=training_set.n_trajectory_features,
        pose_offset=int(_pose_offset(training_set)),
        latent_size=spec.latent_size, frame_time=spec.frame_time,
        n_frames=training_set.n_frames,
        losses=losses, loss_columns=LOSS_COLUMNS)

    final = losses[-1] if losses else (float('nan'),) * len(LOSS_COLUMNS)
    summary = {
        'out_path': out_path,
        'frames': int(training_set.n_frames),
        'pairs': int(pairs.size),
        'train_pairs': int(train_pairs.size),
        'bones': spec.n_bones,
        'feature_size': spec.feature_size,
        'latent_size': spec.latent_size,
        'pose_size': spec.pose_size,
        'character_size': spec.character_size,
        'parameters': compressor.parameter_count() + decompressor.parameter_count(),
        'iterations_run': ran,
        'best_iteration': best_iteration,
        'local': final[0],
        'character': final[1],
        'local_velocity': final[2],
        'character_velocity': final[3],
        'latent_regularisation': final[4],
        'train_loss': final[5],
        'val_loss': best_validation,
        'device': str(torch_device),
        'seconds': time.time() - started,
    }
    summary.update(diagnostics)

    print(f"[LMM] trained {summary['parameters']} parameters over {summary['bones']} bones on "
          f"{summary['train_pairs']}/{summary['pairs']} pairs "
          f"(X {summary['feature_size']} + Z {summary['latent_size']} -> Y {summary['pose_size']}, "
          f"Q {summary['character_size']}) in {summary['seconds']:.1f}s on {summary['device']}: "
          f"{summary['iterations_run']} iterations, best held-out {summary['val_loss']:.4f} at "
          f"{summary['best_iteration']} -> {out_path}")
    print(f"[LMM] loss terms at the end: local {summary['local']:.4f}, character "
          f"{summary['character']:.4f}, local velocity {summary['local_velocity']:.4f}, "
          f"character velocity {summary['character_velocity']:.4f}, latent "
          f"{summary['latent_regularisation']:.4f}")
    print(f"[LMM] latent: a linear model predicts its step from (X, Z) with held-out R^2 "
          f"{summary['latent_step_predictability']:+.3f} -- the lower bound on what a phase B "
          f"stepper could learn. It moves {summary['latent_step_share'] * 100:.1f}% of its own "
          f"spread per frame, and linear extrapolation scores "
          f"{summary['latent_extrapolation_share'] * 100:.0f}% of holding still")
    print(f"[LMM] latent, secondary: lag-1 autocorrelation "
          f"{summary['latent_autocorrelation']:.3f} median, "
          f"{summary['latent_min_autocorrelation']:.3f} worst dimension; "
          f"{summary['latent_dimensions_used']} of {summary['latent_size']} dimensions carry 90% "
          f"of the variance; D(X||0) is {summary['latent_ablation_ratio']:.2f}x the error of "
          f"D(X||Z)")
    return summary


def _slice(block_layout, name: str) -> slice:
    for entry, offset, count in block_layout:
        if entry == name:
            return slice(offset, offset + count)
    raise KeyError(f'no block named {name!r}')


def _float_weights(block_layout, weights) -> np.ndarray:
    """
    Per-float weights whose *sum* is ``sum_b w_b * mean|.|_b`` over the named blocks.

    Spelling the per-block means as one weighted sum is exactly equivalent -- a block's mean is the
    sum over its floats divided by its width -- and it turns a dozen small reductions into one.
    That matters more than it looks: the training step is bound by how many kernels it launches,
    not by their size.

    Blocks the term does not name get zero, which is what excludes them.
    """
    total = sum(count for _, _, count in block_layout)
    per_float = np.zeros(total, dtype=np.float32)

    for name, offset, count in block_layout:
        weight = weights.get(name)
        if weight and count:
            per_float[offset:offset + count] = weight / count

    if not per_float.any():
        raise KeyError(f'none of {sorted(weights)} is a block of this layout')
    return per_float


def _weighted(predicted: torch.Tensor, target: torch.Tensor, per_float: torch.Tensor):
    """``sum_b w_b * mean|predicted_b - target_b|``, as one reduction. See :func:`_float_weights`."""
    return ((predicted - target).abs() * per_float).sum(dim=1).mean()


def _validate(evaluate, compressor, decompressor, frames: torch.Tensor) -> float:
    if not frames.numel():
        return float('nan')

    compressor.eval()
    decompressor.eval()
    with torch.no_grad():
        return float(sum(evaluate(frames)))


def _pose_offset(training_set) -> int:
    """Float offset where the pose half of a matching feature vector starts."""
    widths = np.asarray(training_set.feature_widths, dtype=np.int64)
    counts = np.asarray(training_set.feature_counts, dtype=np.int64)
    return int((widths[:training_set.n_trajectory_features]
                * counts[:training_set.n_trajectory_features]).sum())


def _bake_latents(compressor, y_t, q_t, y_mean_t, y_std_t, q_mean_t, q_std_t,
                  latent_exists: np.ndarray, latent_size: int, device) -> np.ndarray:
    """
    Every compressible frame's latent, indexed by database frame.

    Frames with no successor in their clip keep a row of zeros and are marked invalid, rather than
    being dropped: the runtime indexes this table by the frame the matcher picked, so a compacted
    table would need a second index nothing else in the pipeline carries.
    """
    latents = np.zeros((y_t.shape[0], latent_size), dtype=np.float32)
    frames = np.flatnonzero(latent_exists).astype(np.int64)

    compressor.eval()
    with torch.no_grad():
        for start in range(0, frames.size, BAKE_BATCH):
            batch = torch.from_numpy(frames[start:start + BAKE_BATCH]).to(device)
            encoded = compressor.encode((y_t[batch] - y_mean_t) / y_std_t,
                                        (q_t[batch] - q_mean_t) / q_std_t)
            latents[frames[start:start + BAKE_BATCH]] = encoded.cpu().numpy()

    return latents


def _latent_diagnostics(decompressor, x_t, y_t, y_mean, y_std, latents: np.ndarray,
                        latent_exists: np.ndarray, validation_pairs: np.ndarray,
                        pose_layout, device) -> dict:
    """
    Whether the latent is carrying anything, and whether phase B will be able to advance it.

    The gate is not the ablation ratio. A decompressor handed ``X`` alone already places most of the
    body, because ``X`` holds both feet, their velocities and the hips -- so that ratio mostly
    measures how informative the query is, and a modest one is not evidence that ``Z`` is empty.

    What phase B needs is a latent a small network can *advance*, which is a question about how
    **predictable** the latent's step is, not about how slow it is. ``latent_step_predictability``
    asks it directly: the held-out R-squared of a ridge regression from ``(X, Z)`` to ``Z' - Z``,
    which is the linear lower bound on what a stepper could learn. A negative or near-zero value
    means there is no function there to fit and phase B cannot work; a high one means it can.

    Two cheaper numbers sit beside it and neither may be read on its own, because both reward a
    latent that has simply gone quiet:

    * ``latent_extrapolation_share`` -- a linear extrapolation of the last two latents against
      holding the current one. Useful as a sanity check, but raising the velocity regulariser
      improves it while leaving predictability flat, which is exactly the trap.
    * the lag-1 autocorrelation, which an undertrained latent scores *higher* on, because it
      carries less and therefore moves less.
    """
    triples = np.flatnonzero(
        latent_exists[:-2] & latent_exists[1:-1] & latent_exists[2:]).astype(np.int64)
    if triples.size:
        first, second, third = latents[triples], latents[triples + 1], latents[triples + 2]
        hold = float(np.abs(third - second).mean())
        extrapolated = float(np.abs(third - (2 * second - first)).mean())
        step_share = hold / max(1e-9, float(latents[latent_exists].std()))
    else:
        hold, extrapolated, step_share = float('nan'), float('nan'), float('nan')

    predictability = _step_predictability(x_t, latents, latent_exists)

    pairs = np.flatnonzero(latent_exists[:-1] & latent_exists[1:]).astype(np.int64)
    if pairs.size:
        current, following = latents[pairs], latents[pairs + 1]
        centred, next_centred = current - current.mean(axis=0), following - following.mean(axis=0)
        spread = current.std(axis=0) * following.std(axis=0)
        correlation = np.where(spread > 1e-9, (centred * next_centred).mean(axis=0) / np.maximum(spread, 1e-9), 0.0)
    else:
        correlation = np.zeros(latents.shape[1])

    variance = np.sort(latents[latent_exists].var(axis=0))[::-1] if latent_exists.any() \
        else np.zeros(latents.shape[1])
    cumulative = np.cumsum(variance) / max(1e-12, variance.sum())

    frames = validation_pairs if validation_pairs.size else np.arange(min(2048, len(latents)))
    frames = frames[:2048]
    if frames.size == 0:
        return {'latent_ablation_ratio': float('nan')}

    index = torch.from_numpy(np.asarray(frames, dtype=np.int64)).to(device)
    latent = torch.from_numpy(latents[frames]).to(device)
    weights = torch.from_numpy(
        _float_weights(pose_layout, lmm_dataset.LOCAL_WEIGHTS)).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)
    y_mean_t = torch.from_numpy(y_mean).to(device)

    decompressor.eval()
    with torch.no_grad():
        def error(code):
            predicted = decompressor.decode(x_t[index], code) * y_std_t + y_mean_t
            return float(((predicted - y_t[index]).abs() * weights).sum(dim=1).mean())

        with_latent = error(latent)
        without_latent = error(torch.zeros_like(latent))

    return {
        'latent_reconstruction': with_latent,
        'latent_ablation_ratio': without_latent / with_latent if with_latent else float('nan'),
        'latent_autocorrelation': float(np.median(correlation)),
        'latent_min_autocorrelation': float(correlation.min()),
        'latent_dimensions_used': int(np.searchsorted(cumulative, 0.9) + 1),
        'latent_step_share': step_share,
        'latent_extrapolation_share': extrapolated / hold if hold else float('nan'),
        'latent_step_predictability': predictability,
    }


def _step_predictability(x_t, latents: np.ndarray, latent_exists: np.ndarray) -> float:
    """
    Held-out R-squared of a ridge regression from ``(X, Z)`` to the latent's next step.

    The linear lower bound on what the phase B stepper could learn, and the only diagnostic here
    that a latent cannot improve by going quiet -- R-squared is scored against that latent's own
    variance, so shrinking the steps shrinks the target too.

    The split is the same contiguous tail the trainer validates on, because adjacent frames are
    near-duplicates and a random split would fit the test set through its neighbours.
    """
    pairs = np.flatnonzero(latent_exists[:-1] & latent_exists[1:]).astype(np.int64)
    if pairs.size < 64:
        return float('nan')

    features = np.concatenate([
        x_t.cpu().numpy().astype(np.float64)[pairs],
        latents.astype(np.float64)[pairs],
        np.ones((pairs.size, 1)),
    ], axis=1)
    target = latents.astype(np.float64)[pairs + 1] - latents.astype(np.float64)[pairs]

    cut = max(1, int(pairs.size * 0.9))
    fit, held = features[:cut], features[cut:]
    fit_target, held_target = target[:cut], target[cut:]
    if held.shape[0] < 2:
        return float('nan')

    coefficients = np.linalg.solve(
        fit.T @ fit + 1e-3 * np.eye(fit.shape[1]), fit.T @ fit_target)
    residual = ((held_target - held @ coefficients) ** 2).sum()
    variance = ((held_target - held_target.mean(axis=0)) ** 2).sum()
    return float(1.0 - residual / variance) if variance else float('nan')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('database', help='the generated database folder under StreamingAssets')
    parser.add_argument('name', help='base file name, i.e. the Unity asset name')
    parser.add_argument('--out', default=None, help='where to write the .lmm.npz')
    parser.add_argument('--exclude', nargs='*', default=[], help='bone names not to predict')
    parser.add_argument('--latent-size', type=int, default=lmm_dataset.DEFAULT_LATENT_SIZE)
    parser.add_argument('--latent-velocity-weight', type=float,
                        default=lmm_dataset.LATENT_VELOCITY_WEIGHT,
                        help='w_vreg: how hard the latent is held to moving smoothly')
    parser.add_argument('--iterations', type=int, default=150000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--validation-interval', type=int, default=1000)
    parser.add_argument('--max-seconds', type=float, default=0.0,
                        help='stop after this long regardless; 0 for no limit')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    return parser.parse_args(argv)


def _main(argv=None) -> None:
    args = _parse_args(argv)
    out = args.out or f'{args.name}.lmm.npz'
    train(args.database, args.name, out,
          excluded_bones=args.exclude,
          latent_size=args.latent_size,
          latent_velocity_weight=args.latent_velocity_weight,
          iterations=args.iterations, batch_size=args.batch_size,
          learning_rate=args.learning_rate, weight_decay=args.weight_decay,
          validation_fraction=args.validation_fraction,
          validation_interval=args.validation_interval,
          max_seconds=args.max_seconds, seed=args.seed, device=args.device)


if __name__ == '__main__':
    _main()
