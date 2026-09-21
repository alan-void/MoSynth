"""
Fits the networks of Learned Motion Matching, in the order they depend on each other.

First the compressor/decompressor pair, which learns a latent ``Z`` that, together with the
matching feature vector ``X`` a classic matcher already searches, is enough to reconstruct the
whole pose. Then, against latents that pair has baked and that nothing may move again, the
**stepper** -- see :func:`fit_stepper` -- which advances ``[X Z]`` between searches so the database
need not be played, and finally the **projector** -- see :func:`fit_projector` -- which answers a
query with a state the database holds, and so replaces the search itself.

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
  leaves ``Z`` steppable at all by the stepper, and it applies to a **per-second** derivative -- a
  factor of sixty against the per-frame delta it would be easy to write instead.

The weights are in :mod:`lmm.dataset`; they are the author's released training code, since the
paper states none.

Validation holds out a **contiguous tail** rather than a random subset, for the reason
``pfnn.trainer`` does: neighbouring frames of an animation are nearly the same pose, so a random
split puts near-duplicates of the validation set into training and reports a number that measures
nothing. The parameters that scored best on it are what gets written, not the last ones.

Runs from the Unity Editor through PythonNET, or standalone::

    python -m lmm.trainer ../Assets/StreamingAssets/MMDatabases/MM_LafanCorrected_Edinburg \\
        MM_LafanCorrected_Edinburg --out model.lmm.npz --iterations 150000

Either of the later networks alone, against a checkpoint whose autoencoder is already fitted --
which is what to run when tuning them, since the autoencoder is the half-hour half::

    python -m lmm.trainer <database> <name> --out model.lmm.npz --stepper-only
    python -m lmm.trainer <database> <name> --out model.lmm.npz --projector-only
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch

from lmm import dataset, fk
from lmm import io as lmm_io
from formats.feature_set_importer import read_feature_set
from lmm.model import Compressor, Decompressor, Projector, Stepper, resolve_device
from training.training_data import load_database

LOSS_COLUMNS = ('local', 'character', 'local_velocity', 'character_velocity',
                'latent', 'total', 'validation')

# The stepper's own curve, fitted in a second loop whose terms measure something else
# entirely, so they are a separate table in the checkpoint rather than more columns on this one.
STEPPER_LOSS_COLUMNS = ('features', 'latent', 'feature_rate', 'latent_rate', 'total', 'validation')

# The projector's, likewise. `distance` is how far the projector thinks it moved against how far
# the true nearest neighbour is -- see fit_projector.
PROJECTOR_LOSS_COLUMNS = ('features', 'latent', 'distance', 'total', 'validation')

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
          latent_size: int = dataset.DEFAULT_LATENT_SIZE,
          latent_velocity_weight: float = dataset.LATENT_VELOCITY_WEIGHT,
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
          stepper: bool = True,
          stepper_hidden: int = 0,
          stepper_window: int = dataset.DEFAULT_STEPPER_WINDOW,
          stepper_iterations: int = 30000,
          stepper_patience: int = 5,
          stepper_max_seconds: float = 0.0,
          projector: bool = True,
          projector_hidden: int = 0,
          projector_iterations: int = 30000,
          projector_patience: int = 5,
          projector_sigma: float = dataset.PROJECTOR_SIGMA,
          projector_max_seconds: float = 0.0,
          seed: int = 42,
          device: str = 'auto',
          progress=None) -> dict:
    """
    Build the training set, fit the autoencoder, bake the latents, fit the stepper, and write the
    checkpoint.

    :param data_dir: the database folder under StreamingAssets.
    :param db_name: base file name, i.e. the Unity asset's name.
    :param out_path: where to write the ``.lmm.npz``.
    :param excluded_bones: bone names the decompressor does not predict.
    :param feature_weights: one authored search weight per feature definition, which travels into
        the checkpoint so the projector approximates the search that is actually run.
    :param latent_velocity_weight: ``w_vreg``, the penalty on how fast the latent moves. The one
        loss weight worth tuning -- see :mod:`lmm.dataset`. Too low and the stepper has
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
    :param stepper: also fit the stepper, against the latents this run just baked. Off
        writes an autoencoder-only checkpoint, which is what the ``DecompressorOnly`` mode needs
        and all it needs.
    :param stepper_window: frames the stepper is unrolled over; see :func:`fit_stepper`.
    :param projector: also fit the projector, which replaces the search itself. Needs the
        stepper, because the ``Full`` mode runs both -- the projector answers a search and the
        stepper carries that answer to the next one.
    :param projector_sigma: how far the projector's training queries are displaced; see
        :func:`fit_projector`.
    :param progress: ``(stage, fraction)`` callback, which the Editor drives a progress bar from.
    :return: a summary dict, logged by the caller.
    """
    if projector and not stepper:
        raise ValueError('a projector cannot be fitted without a stepper: the Full mode runs both, '
                         'and the checkpoint refuses to hold one without the other.')

    progress = progress or _noop_progress
    started = time.time()
    torch.manual_seed(seed)

    progress('Reading database', 0.0)
    training_set = load_database(data_dir, db_name, with_features=True)
    spec = dataset.build_spec(training_set, excluded_bones, latent_size)

    progress('Packing vectors', 0.05)
    x, y, q, latent_exists = dataset.build_vectors(training_set, spec)
    pairs = dataset.training_pairs(training_set)
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
    y_mean, y_std = dataset.group_normalization(y[latent_exists], pose_layout)
    q_mean, q_std = dataset.group_normalization(q[latent_exists], character_layout)

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
    hierarchy = fk.Hierarchy(spec.parents, torch_device)
    frame_time = spec.frame_time

    def weights_of(block_layout, table):
        return torch.from_numpy(_float_weights(block_layout, table)).to(torch_device)

    local_weights = weights_of(pose_layout, dataset.LOCAL_WEIGHTS)
    character_weights = weights_of(character_layout, dataset.CHARACTER_WEIGHTS)
    local_rate_weights = weights_of(pose_layout, dataset.LOCAL_RATE_WEIGHTS)
    character_rate_weights = weights_of(character_layout, dataset.CHARACTER_RATE_WEIGHTS)

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
        character = fk.character_vector_from_fk(
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
            dataset.LATENT_SPARSITY_WEIGHT * latent.abs().mean()
            + dataset.LATENT_MAGNITUDE_WEIGHT * latent.pow(2).mean()
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
    # Shared out between the fits that will actually run, so the bar does not sit at 55% for half
    # an hour when the autoencoder is the only thing being trained.
    autoencoder_span = 0.85 if not stepper else (0.45 if projector else 0.55)
    stepper_span = 0.22 if projector else 0.30

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

            progress('Training', 0.05 + autoencoder_span * (step + 1) / iterations)

            if max_seconds and time.time() - started > max_seconds:
                break

    # The held-out curve of a small database turns back up well before the last step, so writing
    # the final parameters would ship a model measurably worse than one this run already had.
    if best_parameters is not None:
        compressor.load_state_dict(best_parameters[0])
        decompressor.load_state_dict(best_parameters[1])

    progress('Baking latents', 0.05 + autoencoder_span + 0.02)
    latents = _bake_latents(compressor, y_t, q_t, y_mean_t, y_std_t, q_mean_t, q_std_t,
                            latent_exists, spec.latent_size, torch_device)
    diagnostics = _latent_diagnostics(decompressor, x_t, y_t, y_mean, y_std, latents,
                                      latent_exists, validation_pairs, pose_layout, torch_device)

    z_mean = latents[latent_exists].mean(axis=0).astype(np.float32)
    z_std = latents[latent_exists].std(axis=0).astype(np.float32)
    z_std[z_std < 1e-6] = 1.0

    fitted_stepper = fitted_projector = None
    stepper_base = 0.05 + autoencoder_span + 0.02
    if stepper:
        # Against the table that was just baked, not against a compressor that is still moving --
        # see fit_stepper. The latents are an input to it from here on.
        fitted_stepper, rate_mean, rate_std, stepper_losses, stepper_summary = fit_stepper(
            x, latents, latent_exists, spec.frame_time,
            hidden_units=stepper_hidden, window=stepper_window,
            iterations=stepper_iterations, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay,
            learning_rate_decay=learning_rate_decay, decay_interval=decay_interval,
            validation_fraction=validation_fraction, validation_interval=validation_interval,
            patience=stepper_patience, max_seconds=stepper_max_seconds, seed=seed, device=device,
            progress=lambda stage, fraction: progress(stage, stepper_base + stepper_span * fraction))
        diagnostics.update(stepper_summary)

    if projector:
        # Against the same fixed latents, and against the authored weights the stage will search
        # with -- the projector is approximating that metric, not a uniform one.
        projector_base = stepper_base + stepper_span
        fitted_projector, projector_losses, projector_summary = fit_projector(
            x, latents, latent_exists,
            dataset.feature_weights(training_set, feature_weights),
            np.concatenate([np.zeros(spec.feature_size, dtype=np.float32), z_mean]),
            np.concatenate([np.ones(spec.feature_size, dtype=np.float32), z_std]),
            hidden_units=projector_hidden, iterations=projector_iterations,
            batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
            learning_rate_decay=learning_rate_decay, decay_interval=decay_interval,
            validation_fraction=validation_fraction, validation_interval=validation_interval,
            patience=projector_patience, sigma=projector_sigma,
            max_seconds=projector_max_seconds, seed=seed, device=device,
            progress=lambda stage, fraction: progress(stage, projector_base + 0.23 * fraction))
        diagnostics.update(projector_summary)

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
        feature_weights=dataset.feature_weights(training_set, feature_weights),
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
        losses=losses, loss_columns=LOSS_COLUMNS,
        stepper_weights=fitted_stepper.weights() if fitted_stepper else None,
        stepper_biases=fitted_stepper.biases() if fitted_stepper else None,
        xz_rate_mean=rate_mean if fitted_stepper else None,
        xz_rate_std=rate_std if fitted_stepper else None,
        stepper_losses=stepper_losses if fitted_stepper else None,
        stepper_loss_columns=STEPPER_LOSS_COLUMNS if fitted_stepper else (),
        projector_weights=fitted_projector.weights() if fitted_projector else None,
        projector_biases=fitted_projector.biases() if fitted_projector else None,
        projector_losses=projector_losses if fitted_projector else None,
        projector_loss_columns=PROJECTOR_LOSS_COLUMNS if fitted_projector else ())

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
          f"{summary['latent_step_predictability']:+.3f} -- the lower bound on what a "
          f"stepper could learn. It moves {summary['latent_step_share'] * 100:.1f}% of its own "
          f"spread per frame, and linear extrapolation scores "
          f"{summary['latent_extrapolation_share'] * 100:.0f}% of holding still")
    print(f"[LMM] latent, secondary: lag-1 autocorrelation "
          f"{summary['latent_autocorrelation']:.3f} median, "
          f"{summary['latent_min_autocorrelation']:.3f} worst dimension; "
          f"{summary['latent_dimensions_used']} of {summary['latent_size']} dimensions carry 90% "
          f"of the variance; D(X||0) is {summary['latent_ablation_ratio']:.2f}x the error of "
          f"D(X||Z)")
    if fitted_stepper:
        _log_stepper(summary)
    if fitted_projector:
        _log_projector(summary)
    return summary


def _log_stepper(summary: dict) -> None:
    """The stepper's lines of a training log: what was fitted, and how far it wanders."""
    stopped = 'stopped early' if summary['stepper_stopped_early'] else 'ran to the limit'
    print(f"[LMM] stepper: {summary['stepper_parameters']} parameters over "
          f"{summary['stepper_train_windows']}/{summary['stepper_windows']} windows of "
          f"{summary['stepper_window']} frames in {summary['stepper_seconds']:.1f}s, "
          f"{summary['stepper_iterations_run']} iterations ({stopped}), best held-out "
          f"{summary['stepper_val_loss']:.4f} at {summary['stepper_best_iteration']}")
    drift = ', '.join(
        f"{n}f {summary[f'stepper_feature_drift_{n}']:.3f}/"
        f"{summary[f'stepper_latent_drift_{n}']:.3f}"
        for n in sorted(dataset.DRIFT_HORIZONS)
        if f'stepper_feature_drift_{n}' in summary)
    print(f"[LMM] stepper free-run drift (X/Z, in units of each half's own spread): {drift}. "
          "Ten frames is the search cadence, so that is the one that decides whether the state "
          "survives between searches")


def refit_stepper(checkpoint_path: str, data_dir: str, db_name: str,
                  out_path: str = None, progress=None, **options) -> dict:
    """
    Fit only the stepper, against a checkpoint whose autoencoder is already trained.

    The autoencoder is the expensive half and the stepper is fitted against latents it has already
    produced, so re-running it to try a different window or a longer stepper schedule would be
    paying half an hour for nothing.

    It reads the ``.mmfeatures`` alone and not the ``.mmpose`` beside it. Everything else it needs
    is in the checkpoint: ``latent_valid`` is exactly the frame mask the autoencoder derived from
    the clip ranges, so there is no second definition of which frames carry a latent for the two to
    disagree about -- and it saves reading three hundred megabytes of poses that nothing here looks at.

    :param out_path: where to write; the checkpoint is overwritten in place when this is None.
    :param options: passed to :func:`fit_stepper`.
    :return: the fit's summary, logged.
    """
    checkpoint = lmm_io.load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(
            f'no usable LMM checkpoint at {checkpoint_path}. The stepper is fitted against latents '
            'an autoencoder has already baked, so there has to be one to fit against.')

    features = read_feature_set(data_dir, db_name).features
    x = np.ascontiguousarray(features, dtype=np.float32)

    if x.shape[0] != checkpoint.n_frames:
        raise ValueError(
            f'{db_name} now holds {x.shape[0]} frames but the checkpoint was baked over '
            f'{checkpoint.n_frames}. The latents are indexed by frame, so regenerating the '
            'database invalidates them -- retrain the autoencoder.')
    if x.shape[1] != checkpoint.feature_size:
        raise ValueError(f'{db_name} now produces {x.shape[1]}-float queries but the checkpoint '
                         f'was trained on {checkpoint.feature_size}. Retrain the autoencoder.')

    progress = progress or _noop_progress
    stepper, rate_mean, rate_std, losses, summary = fit_stepper(
        x, checkpoint.latents, checkpoint.latent_valid, checkpoint.frame_time,
        progress=progress, **options)

    progress('Writing checkpoint', 0.98)
    arguments = lmm_io.checkpoint_arguments(checkpoint)
    arguments.update(
        stepper_weights=stepper.weights(), stepper_biases=stepper.biases(),
        xz_rate_mean=rate_mean, xz_rate_std=rate_std,
        stepper_losses=losses, stepper_loss_columns=STEPPER_LOSS_COLUMNS)
    lmm_io.save_checkpoint(out_path or checkpoint_path, **arguments)

    summary['out_path'] = out_path or checkpoint_path
    _log_stepper(summary)
    return summary


def fit_stepper(x: np.ndarray, latents: np.ndarray, latent_exists: np.ndarray,
                frame_time: float, *,
                hidden_units: int = 0,
                window: int = dataset.DEFAULT_STEPPER_WINDOW,
                iterations: int = 30000,
                batch_size: int = 256,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-3,
                learning_rate_decay: float = 0.99,
                decay_interval: int = 1000,
                validation_fraction: float = 0.1,
                validation_interval: int = 1000,
                patience: int = 5,
                max_seconds: float = 0.0,
                seed: int = 42,
                device: str = 'auto',
                progress=None):
    """
    Fit the stepper against latents that are already fixed, and measure how far it drifts.

    **The latents are an input, never a parameter.** The autoencoder's compressor is not merely
    frozen here, it does not run at all: the stepper is fitted against the table baked into the
    checkpoint. Letting the two train together would give the pair a much cheaper way to make the
    latent steppable than learning to step it -- make it constant -- which is latent collapse
    arriving through the back door.

    Training is **unrolled**, never on single-frame pairs: the state the stepper is asked to
    advance at frame *n* is the state it produced at frame *n-1*, errors and all. A stepper fitted
    on true states only is a stepper that has never seen its own mistakes, and it compounds them.
    No window crosses a clip boundary -- see :func:`dataset.stepper_windows`.

    The loss is scored on both halves of the state and on both halves of the rate, each divided by
    **one scalar spread for the whole half**, so the weights in
    :data:`dataset.STEPPER_WEIGHTS` trade a feature error against a latent error in comparable
    terms. It is then divided by the window length, which keeps its magnitude -- and so the
    effective learning rate -- independent of how far the unrolling goes.

    :param x: (n, feature_size) every database frame's matching feature vector.
    :param latents: (n, latent_size) the baked latents, indexed by the same frames.
    :param latent_exists: (n,) bool, which of those frames carry a latent.
    :param frame_time: seconds per database frame; what turns a step into a rate.
    :param hidden_units: 0 takes the reference implementation's 512.
    :param window: frames to unroll over. See :data:`dataset.DEFAULT_STEPPER_WINDOW`.
    :param patience: stop after this many held-out scores without an improvement; 0 never stops
        early. **This is the stopping rule, not ``iterations``**, which is a ceiling. Measured on
        Edinburgh the held-out score bottoms out around iteration 2,000 of 30,000 and rises
        monotonically after it, so a fixed count is nine parts waste and one part fit -- and the
        right count is a property of the database, not something to guess per run. The autoencoder
        has no equivalent because its held-out curve is still falling at its iteration limit.
    :return: ``(stepper, rate_mean, rate_std, losses, summary)``.
    """
    progress = progress or _noop_progress
    started = time.time()
    torch.manual_seed(seed)

    windows = dataset.latent_runs(latent_exists, window)
    if windows.size < 2:
        raise ValueError(
            f'no run of {window + 1} consecutive frames in this database carries a latent, so the '
            'stepper has nothing to unroll over. Either the clips are shorter than the window or '
            'their feature vectors are invalid.')

    feature_size, latent_size = x.shape[1], latents.shape[1]
    torch_device = resolve_device(device)

    state = torch.from_numpy(
        np.concatenate([x, latents], axis=1).astype(np.float32)).to(torch_device)
    # The rate that carries each frame to the next. Rows spanning a clip boundary are meaningless
    # and are never indexed: every window lies inside one clip, by construction.
    rates = (state[1:] - state[:-1]) / frame_time

    usable = torch.from_numpy(np.ascontiguousarray(latent_exists, dtype=bool)).to(torch_device)
    rate_mean = rates[usable[:-1]].mean(dim=0)
    rate_std = rates[usable[:-1]].std(dim=0)
    rate_std = torch.where(rate_std < 1e-6, torch.ones_like(rate_std), rate_std)

    x_scale, z_scale = dataset.state_scales(x, latents, latent_exists)

    stepper = Stepper(feature_size, latent_size,
                      **({'hidden_units': hidden_units} if hidden_units else {})).to(torch_device)
    optimizer = torch.optim.AdamW(stepper.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay, amsgrad=True)
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

    weights = dataset.STEPPER_WEIGHTS

    def evaluate(starts: torch.Tensor):
        """The unrolled loss over a batch of window starts; the columns of STEPPER_LOSS_COLUMNS."""
        features = state[starts, :feature_size]
        latent = state[starts, feature_size:]
        feature_error = latent_error = feature_rate_error = latent_rate_error = 0.0

        for n in range(window):
            predicted = stepper.rate(features, latent) * rate_std + rate_mean
            truth = rates[starts + n]

            feature_rate_error = feature_rate_error + (
                predicted[:, :feature_size] - truth[:, :feature_size]).abs().mean() / x_scale
            latent_rate_error = latent_rate_error + (
                predicted[:, feature_size:] - truth[:, feature_size:]).abs().mean() / z_scale

            features = features + predicted[:, :feature_size] * frame_time
            latent = latent + predicted[:, feature_size:] * frame_time

            target = state[starts + n + 1]
            feature_error = feature_error + (
                features - target[:, :feature_size]).abs().mean() / x_scale
            latent_error = latent_error + (
                latent - target[:, feature_size:]).abs().mean() / z_scale

        return (weights['features'] * feature_error / window,
                weights['latent'] * latent_error / window,
                weights['feature_rate'] * feature_rate_error / window,
                weights['latent_rate'] * latent_rate_error / window)

    split = max(1, int(round(windows.size * (1.0 - validation_fraction))))
    train_t = torch.from_numpy(windows[:split]).to(torch_device)
    held_out = windows[split:]
    validation_t = torch.from_numpy(
        held_out[np.linspace(0, held_out.size - 1,
                             min(VALIDATION_SAMPLE, held_out.size)).astype(np.int64)]
        if held_out.size else held_out).to(torch_device)

    losses = []
    running = torch.zeros(len(STEPPER_LOSS_COLUMNS) - 1, device=torch_device)
    running_count = 0
    best_validation, best_parameters, best_iteration = float('inf'), None, 0
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ran, stalled, stopped_early = 0, 0, False

    for step in range(iterations):
        stepper.train()
        starts = train_t[torch.randint(train_t.numel(), (batch_size,), generator=generator)
                         .to(torch_device)]

        optimizer.zero_grad(set_to_none=True)
        terms = evaluate(starts)
        loss = sum(terms)
        loss.backward()
        optimizer.step()

        running += torch.stack([term.detach() for term in terms] + [loss.detach()])
        running_count += 1
        ran = step + 1

        if (step + 1) % decay_interval == 0:
            schedule.step()

        if (step + 1) % validation_interval == 0 or step + 1 == iterations:
            if validation_t.numel():
                stepper.eval()
                with torch.no_grad():
                    validation = float(sum(evaluate(validation_t)))
            else:
                validation = float('nan')

            losses.append(tuple((running / max(1, running_count)).tolist()) + (validation,))
            running = torch.zeros(len(STEPPER_LOSS_COLUMNS) - 1, device=torch_device)
            running_count = 0

            if validation < best_validation:
                best_validation, best_iteration = validation, step + 1
                best_parameters = copy.deepcopy(stepper.state_dict())
                stalled = 0
            else:
                stalled += 1

            progress('Fitting stepper', (step + 1) / iterations)

            if patience and stalled >= patience:
                stopped_early = True
                break
            if max_seconds and time.time() - started > max_seconds:
                break

    if best_parameters is not None:
        stepper.load_state_dict(best_parameters)

    drift = _stepper_drift(stepper, state, rate_mean, rate_std, latent_exists,
                           held_out[0] if held_out.size else 0,
                           feature_size, frame_time, x_scale, z_scale, torch_device)

    final = losses[-1] if losses else (float('nan'),) * len(STEPPER_LOSS_COLUMNS)
    summary = {
        'stepper_windows': int(windows.size),
        'stepper_train_windows': int(split),
        'stepper_window': int(window),
        'stepper_parameters': stepper.parameter_count(),
        'stepper_iterations_run': ran,
        'stepper_best_iteration': best_iteration,
        'stepper_stopped_early': stopped_early,
        'stepper_features': final[0],
        'stepper_latent': final[1],
        'stepper_feature_rate': final[2],
        'stepper_latent_rate': final[3],
        'stepper_train_loss': final[4],
        'stepper_val_loss': best_validation,
        'stepper_seconds': time.time() - started,
    }
    summary.update(drift)

    return stepper, rate_mean.cpu().numpy(), rate_std.cpu().numpy(), losses, summary


def _stepper_drift(stepper, state, rate_mean, rate_std, latent_exists: np.ndarray,
                   first_held_out: int, feature_size: int, frame_time: float,
                   x_scale: float, z_scale: float, device) -> dict:
    """
    How far a free-running stepper has wandered after each of :data:`DRIFT_HORIZONS` frames.

    Reported in units of each half's own spread, so ``0.25`` means the state is a quarter of a
    standard deviation from where the database says it should be. Free-running is the only honest
    measurement: a stepper scored one frame at a time from true states never has to live with its
    own error, and compounding error is the failure mode the stepper exists to bound.

    Measured on runs starting at or after ``first_held_out``, which is the same contiguous tail the
    fit validated on -- a run the stepper trained over would be reporting memorisation.
    """
    horizons = tuple(sorted(dataset.DRIFT_HORIZONS))
    runs = dataset.latent_runs(latent_exists, horizons[-1])
    runs = runs[runs >= first_held_out]
    if runs.size == 0:
        return {f'stepper_feature_drift_{n}': float('nan') for n in horizons}

    starts = torch.from_numpy(
        runs[np.linspace(0, runs.size - 1,
                         min(VALIDATION_SAMPLE, runs.size)).astype(np.int64)]).to(device)

    features = state[starts, :feature_size]
    latent = state[starts, feature_size:]
    drift = {}

    stepper.eval()
    with torch.no_grad():
        for n in range(1, horizons[-1] + 1):
            predicted = stepper.rate(features, latent) * rate_std + rate_mean
            features = features + predicted[:, :feature_size] * frame_time
            latent = latent + predicted[:, feature_size:] * frame_time

            if n not in horizons:
                continue

            target = state[starts + n]
            drift[f'stepper_feature_drift_{n}'] = float(
                (features - target[:, :feature_size]).abs().mean() / x_scale)
            drift[f'stepper_latent_drift_{n}'] = float(
                (latent - target[:, feature_size:]).abs().mean() / z_scale)

    return drift


def _log_projector(summary: dict) -> None:
    """The projector's lines of a training log: what was fitted, and how near it gets."""
    stopped = 'stopped early' if summary['projector_stopped_early'] else 'ran to the limit'
    print(f"[LMM] projector: {summary['projector_parameters']} parameters over "
          f"{summary['projector_train_queries']}/{summary['projector_queries']} query frames in "
          f"{summary['projector_seconds']:.1f}s, {summary['projector_iterations_run']} iterations "
          f"({stopped}), best held-out {summary['projector_val_loss']:.4f} at "
          f"{summary['projector_best_iteration']}")
    print(f"[LMM] projector recall on held-out queries: it answers from "
          f"{summary['projector_distance_ratio']:.3f}x as far as the true nearest neighbour, and "
          f"lands inside the database's nearest 1% "
          f"{summary['projector_top_percent_recall'] * 100:.0f}% of the time. Its latent is "
          f"{summary['projector_latent_error']:.3f} of the latent spread from the right one")


def refit_projector(checkpoint_path: str, data_dir: str, db_name: str,
                    out_path: str = None, progress=None, **options) -> dict:
    """
    Fit only the projector, against a checkpoint whose autoencoder and stepper are already trained.

    The same bargain :func:`refit_stepper` offers, and a better one: the projector is the network
    with the most left to tune -- how far the training noise reaches, how long the fit runs -- and
    none of that touches the latent space it is aiming at.

    Reads the ``.mmfeatures`` alone, for the reason :func:`refit_stepper` does: everything else it
    needs is in the checkpoint, including which frames carry a latent.

    :param out_path: where to write; the checkpoint is overwritten in place when this is None.
    :param options: passed to :func:`fit_projector`.
    :return: the fit's summary, logged.
    """
    checkpoint = lmm_io.load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(
            f'no usable LMM checkpoint at {checkpoint_path}. The projector is fitted against '
            'latents an autoencoder has already baked, so there has to be one to fit against.')
    if not checkpoint.has_stage(lmm_io.STAGE_STEPPER):
        raise ValueError(
            f'{checkpoint_path} carries no stepper. The Full mode needs both networks -- the '
            'projector answers a search and the stepper carries the answer to the next one -- so a '
            'checkpoint cannot hold one without the other. Fit the stepper first.')

    features = read_feature_set(data_dir, db_name).features
    x = np.ascontiguousarray(features, dtype=np.float32)

    if x.shape[0] != checkpoint.n_frames:
        raise ValueError(
            f'{db_name} now holds {x.shape[0]} frames but the checkpoint was baked over '
            f'{checkpoint.n_frames}. The latents are indexed by frame, so regenerating the '
            'database invalidates them -- retrain the autoencoder.')
    if x.shape[1] != checkpoint.feature_size:
        raise ValueError(f'{db_name} now produces {x.shape[1]}-float queries but the checkpoint '
                         f'was trained on {checkpoint.feature_size}. Retrain the autoencoder.')

    progress = progress or _noop_progress
    projector, losses, summary = fit_projector(
        x, checkpoint.latents, checkpoint.latent_valid, checkpoint.feature_weights,
        np.concatenate([checkpoint.x_mean, checkpoint.z_mean]),
        np.concatenate([checkpoint.x_std, checkpoint.z_std]),
        progress=progress, **options)

    progress('Writing checkpoint', 0.98)
    arguments = lmm_io.checkpoint_arguments(checkpoint)
    arguments.update(
        projector_weights=projector.weights(), projector_biases=projector.biases(),
        projector_losses=losses, projector_loss_columns=PROJECTOR_LOSS_COLUMNS)
    lmm_io.save_checkpoint(out_path or checkpoint_path, **arguments)

    summary['out_path'] = out_path or checkpoint_path
    _log_projector(summary)
    return summary


def fit_projector(x: np.ndarray, latents: np.ndarray, latent_exists: np.ndarray,
                  feature_weights: np.ndarray, state_mean: np.ndarray, state_std: np.ndarray, *,
                  hidden_units: int = 0,
                  iterations: int = 30000,
                  batch_size: int = 256,
                  learning_rate: float = 1e-3,
                  weight_decay: float = 1e-3,
                  learning_rate_decay: float = 0.99,
                  decay_interval: int = 1000,
                  validation_fraction: float = 0.1,
                  validation_interval: int = 1000,
                  patience: int = 5,
                  sigma: float = dataset.PROJECTOR_SIGMA,
                  max_seconds: float = 0.0,
                  seed: int = 42,
                  device: str = 'auto',
                  progress=None):
    """
    Fit the network that replaces the search, and measure how well it recalls what it replaced.

    Every iteration displaces a database frame's query vector by noise, finds **the true weighted
    nearest neighbour of the displaced vector**, and asks the projector to produce that neighbour's
    state. The emphasis is the whole method: targeting the frame the noise was added to would fit a
    denoiser, a network that undoes a perturbation. The classic matcher does not undo anything --
    asked for a query no frame answers, it returns whichever frame answers it best, and that is
    usually a different frame entirely. See :func:`dataset.nearest_neighbours`.

    The metric is the **authored** search weights, carried in the checkpoint. A projector fitted
    under a uniform metric approximates a search nobody runs, and the comparison against the
    classic matcher is then quietly measuring two different things.

    Three terms, weighted by :data:`dataset.PROJECTOR_WEIGHTS`. Two are the obvious ones -- the
    state it answers with against the state it should have answered with, each divided by its own
    half's spread so the weights trade comparable things. The third is the **distance**: how far
    the projector's answer sits from the query, against how far the true nearest neighbour sits.
    That scalar is what the stage's accept rule compares, so a projector that is close in ``X`` but
    systematically wrong about how close would make every accept decision on the tick path wrong in
    the same direction.

    The latents are an input and never a parameter, exactly as in :func:`fit_stepper`.

    :param x: (n, feature_size) every database frame's matching feature vector.
    :param latents: (n, latent_size) the baked latents.
    :param latent_exists: (n,) bool; both the queries drawn from and the candidates searched.
    :param feature_weights: (feature_size,) the authored search weights, one per float.
    :param state_mean: (feature_size + latent_size,) what the answer is denormalised against --
        the checkpoint's ``x_mean`` and ``z_mean`` concatenated. The projector regresses the state
        itself, so there is nothing new to measure here and no second set of numbers to drift.
    :param state_std: likewise, ``x_std`` and ``z_std``.
    :param sigma: the upper end of the per-sample noise; see :data:`dataset.PROJECTOR_SIGMA`.
    :param patience: as :func:`fit_stepper`'s, and the stopping rule for the same reason.
    :return: ``(projector, losses, summary)``.
    """
    progress = progress or _noop_progress
    started = time.time()
    torch.manual_seed(seed)

    frames = np.flatnonzero(latent_exists).astype(np.int64)
    if frames.size < 2:
        raise ValueError('no frame of this database carries a latent, so the projector has '
                         'nothing to project onto.')

    feature_size, latent_size = x.shape[1], latents.shape[1]
    torch_device = resolve_device(device)

    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch_device)
    z_t = torch.from_numpy(np.ascontiguousarray(latents, dtype=np.float32)).to(torch_device)

    # The frames a search may return, gathered once. The projector is asked to approximate a lookup
    # over exactly this set, which is what the stage masks its own search down to.
    candidate_frames = torch.from_numpy(frames).to(torch_device)
    candidates = x_t[candidate_frames].contiguous()
    candidate_latents = z_t[candidate_frames].contiguous()

    weights = torch.from_numpy(
        np.ascontiguousarray(feature_weights, dtype=np.float32)).to(torch_device)
    norms = dataset.candidate_norms(candidates, weights)
    noise_scale = torch.from_numpy(
        dataset.projector_noise_scale(x, latent_exists)).to(torch_device)

    x_scale, z_scale = dataset.state_scales(x, latents, latent_exists)
    out_mean = torch.from_numpy(
        np.ascontiguousarray(state_mean, dtype=np.float32)).to(torch_device)
    out_std = torch.from_numpy(np.ascontiguousarray(state_std, dtype=np.float32)).to(torch_device)

    projector = Projector(feature_size, latent_size,
                          **({'hidden_units': hidden_units} if hidden_units else {})).to(torch_device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay, amsgrad=True)
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

    term_weights = dataset.PROJECTOR_WEIGHTS

    def answer(queries: torch.Tensor):
        """``(X_hat, Z_hat)`` in the database's own units."""
        features, latent = projector.project(queries)
        return (features * out_std[:feature_size] + out_mean[:feature_size],
                latent * out_std[feature_size:] + out_mean[feature_size:])

    def evaluate(queries: torch.Tensor):
        """The three terms of the loss over a batch of already-displaced queries."""
        with torch.no_grad():
            nearest = dataset.nearest_neighbours(queries, candidates, weights, norms)
            target_x = candidates[nearest]
            target_z = candidate_latents[nearest]
            target_distance = dataset.weighted_distance(queries, target_x, weights)

        features, latent = answer(queries)
        distance = dataset.weighted_distance(queries, features, weights)

        return (term_weights['features'] * (features - target_x).abs().mean() / x_scale,
                term_weights['latent'] * (latent - target_z).abs().mean() / z_scale,
                term_weights['distance'] * (distance - target_distance).abs().mean() / x_scale)

    split = max(1, int(round(frames.size * (1.0 - validation_fraction))))
    train_t = torch.from_numpy(frames[:split]).to(torch_device)
    held_out = frames[split:]
    validation_frames = torch.from_numpy(
        held_out[np.linspace(0, held_out.size - 1,
                             min(VALIDATION_SAMPLE, held_out.size)).astype(np.int64)]
        if held_out.size else held_out).to(torch_device)

    # Drawn once and reused at every score. Re-rolling it would make the held-out curve wander for
    # reasons that have nothing to do with the model, and patience would stop on the noise.
    validation_generator = torch.Generator(device='cpu').manual_seed(seed + 1)
    validation_queries = _displace(
        x_t[validation_frames], noise_scale, sigma, validation_generator, torch_device)

    losses = []
    running = torch.zeros(len(PROJECTOR_LOSS_COLUMNS) - 1, device=torch_device)
    running_count = 0
    best_validation, best_parameters, best_iteration = float('inf'), None, 0
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ran, stalled, stopped_early = 0, 0, False

    for step in range(iterations):
        projector.train()
        chosen = train_t[torch.randint(train_t.numel(), (batch_size,), generator=generator)
                         .to(torch_device)]
        queries = _displace(x_t[chosen], noise_scale, sigma, generator, torch_device)

        optimizer.zero_grad(set_to_none=True)
        terms = evaluate(queries)
        loss = sum(terms)
        loss.backward()
        optimizer.step()

        running += torch.stack([term.detach() for term in terms] + [loss.detach()])
        running_count += 1
        ran = step + 1

        if (step + 1) % decay_interval == 0:
            schedule.step()

        if (step + 1) % validation_interval == 0 or step + 1 == iterations:
            if validation_queries.numel():
                projector.eval()
                with torch.no_grad():
                    validation = float(sum(evaluate(validation_queries)))
            else:
                validation = float('nan')

            losses.append(tuple((running / max(1, running_count)).tolist()) + (validation,))
            running = torch.zeros(len(PROJECTOR_LOSS_COLUMNS) - 1, device=torch_device)
            running_count = 0

            if validation < best_validation:
                best_validation, best_iteration = validation, step + 1
                best_parameters = copy.deepcopy(projector.state_dict())
                stalled = 0
            else:
                stalled += 1

            progress('Fitting projector', (step + 1) / iterations)

            if patience and stalled >= patience:
                stopped_early = True
                break
            if max_seconds and time.time() - started > max_seconds:
                break

    if best_parameters is not None:
        projector.load_state_dict(best_parameters)

    recall = _projector_recall(projector, answer, validation_queries, candidates,
                               candidate_latents, weights, norms, z_scale)

    final = losses[-1] if losses else (float('nan'),) * len(PROJECTOR_LOSS_COLUMNS)
    summary = {
        'projector_queries': int(frames.size),
        'projector_train_queries': int(split),
        'projector_parameters': projector.parameter_count(),
        'projector_iterations_run': ran,
        'projector_best_iteration': best_iteration,
        'projector_stopped_early': stopped_early,
        'projector_sigma': float(sigma),
        'projector_features': final[0],
        'projector_latent': final[1],
        'projector_distance': final[2],
        'projector_train_loss': final[3],
        'projector_val_loss': best_validation,
        'projector_seconds': time.time() - started,
    }
    summary.update(recall)

    return projector, losses, summary


def _displace(frames: torch.Tensor, noise_scale: torch.Tensor, sigma: float,
              generator: torch.Generator, device) -> torch.Tensor:
    """
    Push a batch of query vectors off the database, by a different amount each.

    The scale is drawn per sample over ``[0, sigma]`` rather than fixed, so one batch spans a query
    a database frame answers almost exactly and one no frame answers well. A projector trained at a
    single displacement is good at that displacement and guesses everywhere else, and the runtime
    supplies every displacement: the controller can ask for anything at all.
    """
    if frames.numel() == 0:
        return frames

    scale = torch.rand((frames.shape[0], 1), generator=generator).to(device) * sigma
    noise = torch.randn(frames.shape, generator=generator).to(device)
    return frames + scale * noise_scale * noise


def _projector_recall(projector, answer, queries: torch.Tensor, candidates: torch.Tensor,
                      candidate_latents: torch.Tensor, weights: torch.Tensor,
                      norms: torch.Tensor, z_scale: float) -> dict:
    """
    :func:`dataset.recall_against_search` over the held-out queries this fit validated on.

    Reported straight out of the fit as well as by ``lmm.runtime`` afterwards, because it is the
    number that says whether the projector is usable and a training loss cannot: a loss falling
    steadily says nothing about whether the answers are the ones the search would have given.
    """
    if queries.numel() == 0:
        return {'projector_distance_ratio': float('nan'),
                'projector_top_percent_recall': float('nan'),
                'projector_latent_error': float('nan')}

    projector.eval()
    with torch.no_grad():
        features, latent = answer(queries)

    recall = dataset.recall_against_search(
        queries, features, latent, candidates, candidate_latents, weights, norms, z_scale)
    return {f'projector_{key}': value for key, value in recall.items()}


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
    Whether the latent is carrying anything, and whether a stepper will be able to advance it.

    The gate is not the ablation ratio. A decompressor handed ``X`` alone already places most of the
    body, because ``X`` holds both feet, their velocities and the hips -- so that ratio mostly
    measures how informative the query is, and a modest one is not evidence that ``Z`` is empty.

    What the stepper needs is a latent a small network can *advance*, which is a question about how
    **predictable** the latent's step is, not about how slow it is. ``latent_step_predictability``
    asks it directly: the held-out R-squared of a ridge regression from ``(X, Z)`` to ``Z' - Z``,
    which is the linear lower bound on what a stepper could learn. A negative or near-zero value
    means there is no function there to fit and the stepper cannot work; a high one means it can.

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
        _float_weights(pose_layout, dataset.LOCAL_WEIGHTS)).to(device)
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

    The linear lower bound on what the stepper could learn, and the only diagnostic here
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
    parser.add_argument('--latent-size', type=int, default=dataset.DEFAULT_LATENT_SIZE)
    parser.add_argument('--latent-velocity-weight', type=float,
                        default=dataset.LATENT_VELOCITY_WEIGHT,
                        help='w_vreg: how hard the latent is held to moving smoothly')
    parser.add_argument('--iterations', type=int, default=150000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--validation-interval', type=int, default=1000)
    parser.add_argument('--max-seconds', type=float, default=0.0,
                        help='stop after this long regardless; 0 for no limit')
    parser.add_argument('--no-stepper', action='store_true',
                        help='write an autoencoder-only checkpoint, for the DecompressorOnly mode')
    parser.add_argument('--stepper-only', action='store_true',
                        help='fit only the stepper, against the checkpoint already at --out')
    parser.add_argument('--no-projector', action='store_true',
                        help='stop after the stepper, for the Stepper mode')
    parser.add_argument('--projector-only', action='store_true',
                        help='fit only the projector, against the checkpoint already at --out')
    parser.add_argument('--stepper-window', type=int,
                        default=dataset.DEFAULT_STEPPER_WINDOW,
                        help='frames the stepper is unrolled over while training')
    parser.add_argument('--stepper-iterations', type=int, default=30000,
                        help='ceiling on stepper steps; --stepper-patience is the stopping rule')
    parser.add_argument('--stepper-patience', type=int, default=5,
                        help='held-out scores without an improvement before stopping; 0 never does')
    parser.add_argument('--stepper-hidden', type=int, default=0,
                        help='stepper hidden width; 0 takes the reference implementation of 512')
    parser.add_argument('--stepper-max-seconds', type=float, default=0.0)
    parser.add_argument('--projector-iterations', type=int, default=30000,
                        help='ceiling on projector steps; --projector-patience stops the fit')
    parser.add_argument('--projector-patience', type=int, default=5,
                        help='held-out scores without an improvement before stopping; 0 never does')
    parser.add_argument('--projector-hidden', type=int, default=0,
                        help='projector hidden width; 0 takes the reference implementation of 512')
    parser.add_argument('--projector-sigma', type=float, default=dataset.PROJECTOR_SIGMA,
                        help='how far training queries are displaced, in noise scales')
    parser.add_argument('--projector-max-seconds', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    return parser.parse_args(argv)


def _main(argv=None) -> None:
    args = _parse_args(argv)
    out = args.out or f'{args.name}.lmm.npz'

    stepper_options = dict(
        window=args.stepper_window, iterations=args.stepper_iterations,
        patience=args.stepper_patience, hidden_units=args.stepper_hidden,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, validation_fraction=args.validation_fraction,
        validation_interval=args.validation_interval, max_seconds=args.stepper_max_seconds,
        seed=args.seed, device=args.device)

    projector_options = dict(
        iterations=args.projector_iterations, patience=args.projector_patience,
        hidden_units=args.projector_hidden, sigma=args.projector_sigma,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, validation_fraction=args.validation_fraction,
        validation_interval=args.validation_interval, max_seconds=args.projector_max_seconds,
        seed=args.seed, device=args.device)

    if args.stepper_only:
        refit_stepper(out, args.database, args.name, **stepper_options)
        return

    if args.projector_only:
        refit_projector(out, args.database, args.name, **projector_options)
        return

    train(args.database, args.name, out,
          excluded_bones=args.exclude,
          latent_size=args.latent_size,
          latent_velocity_weight=args.latent_velocity_weight,
          iterations=args.iterations, batch_size=args.batch_size,
          learning_rate=args.learning_rate, weight_decay=args.weight_decay,
          validation_fraction=args.validation_fraction,
          validation_interval=args.validation_interval,
          max_seconds=args.max_seconds,
          stepper=not args.no_stepper,
          stepper_hidden=args.stepper_hidden,
          stepper_window=args.stepper_window,
          stepper_iterations=args.stepper_iterations,
          stepper_patience=args.stepper_patience,
          stepper_max_seconds=args.stepper_max_seconds,
          projector=not args.no_projector and not args.no_stepper,
          projector_hidden=args.projector_hidden,
          projector_iterations=args.projector_iterations,
          projector_patience=args.projector_patience,
          projector_sigma=args.projector_sigma,
          projector_max_seconds=args.projector_max_seconds,
          seed=args.seed, device=args.device)


if __name__ == '__main__':
    _main()
