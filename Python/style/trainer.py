"""
Trains the style network on a labelled style database.

:mod:`style.dataset` decides what a pair, an input and a target are; this module fits the dynamics
network on what the closed-form posture term leaves over, and writes the ``.style.npz``.

Losses, per sample ``p`` and the pair of its styled successor ``q``:

* **rotation**: the dynamics delta's 6D against the target remainder ``P^-1 R_in^-1 R_target``
* **height** and **root residual**: mean squared error on the normalised outputs
* **FK position**: the composed styled bones, through :func:`lmm.fk.forward_kinematics`, against
  the styled frame's own
* **foot slide**: contact-bone planar velocity while the *styled* frame's contact holds,
  root motion included
* **smoothness**: the dynamics delta's change from ``p`` to ``q``
* **magnitude**: a small pull of the raw output toward zero, i.e. toward the posture term alone

Validation holds out a contiguous tail of every clip, not a random subset: neighbouring frames are
near-duplicates, and a random split measures nothing.

Runs from the Unity Editor through PythonNET, or standalone::

    python -m style.trainer ../Assets/StreamingAssets/StyleDatabases/Style100 Style100 \\
        --out Style100.style.npz --epochs 60
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from formats.pose_set_importer import deserialize_pose_set
from lmm.fk import (Hierarchy, forward_kinematics, local_rotations_6d, matrix_to_six_d,
                    parents_within, rest_offsets, six_d_to_matrix)
from pfnn.dataset import normalization
from pfnn.model import resolve_device
from style import dataset
from style import io as style_io
from style.model import StyleNetwork
from training.training_data import TrainingSet, build_training_set

# Loss weights. Starting points for tuning, not derived values.
ROTATION_WEIGHT = 1.0
HEIGHT_WEIGHT = 0.5
ROOT_WEIGHT = 0.5
POSITION_WEIGHT = 10.0
SLIDE_WEIGHT = 0.1
SMOOTH_WEIGHT = 1.0
MAGNITUDE_WEIGHT = 1e-3

# The gate's reference set: at most this many training inputs, k neighbours per novelty score,
# and the percentile of the reference set's own leave-one-out score that sets the threshold.
GATE_REFERENCE_SIZE = 20000
GATE_K = 8
GATE_PERCENTILE = 95.0

_EVAL_BATCH = 8192


def _noop_progress(stage: str, fraction: float) -> None:
    pass


def train(data_dir: str,
          db_name: str,
          out_path: str,
          neutral_style: str = 'Neutral',
          excluded_styles=(),
          excluded_bones=(),
          hidden_units: int = 512,
          layers: int = 3,
          embed_dim: int = 32,
          history_offsets=(0.1, 0.2),
          gait_sigma: float = 0.35,
          epochs: int = 60,
          batch_size: int = 1024,
          learning_rate: float = 1e-3,
          weight_decay: float = 1e-4,
          validation_fraction: float = 0.1,
          seed: int = 42,
          device: str = 'auto',
          progress=None) -> dict:
    """
    Read the database and its style labels, fit the network, and write the checkpoint.

    :param data_dir: the database folder under StreamingAssets.
    :param db_name: base file name, i.e. the Unity asset's name. ``<db_name>.style_labels.json``
        must sit beside the ``.mmpose``.
    :param out_path: where to write the ``.style.npz``.
    :param neutral_style: the identity style every other style is paired against.
    :param excluded_styles: style names not to train, on top of the cadence filter.
    :param excluded_bones: bone names not to style, authored on the Unity config.
    :param gait_sigma: width of the soft gait weights, in m/s of ``style.dataset.gait_features``.
    :param progress: ``(stage, fraction)`` callback, which the Editor drives a progress bar from.
    :return: a summary dict, logged by the caller.
    """
    progress = progress or _noop_progress
    started = time.time()

    progress('Reading database', 0.0)
    pose_set = deserialize_pose_set(data_dir, db_name)
    labels = dataset.load_labels(data_dir, db_name, len(pose_set.clips))
    labels = dataset.labels_for_stored_clips(labels, pose_set.clips,
                                             len(pose_set.local_positions))
    training_set = build_training_set(pose_set)

    return train_training_set(
        training_set, labels, out_path, neutral_style=neutral_style,
        excluded_styles=excluded_styles, excluded_bones=excluded_bones,
        hidden_units=hidden_units, layers=layers, embed_dim=embed_dim,
        history_offsets=history_offsets, gait_sigma=gait_sigma, epochs=epochs,
        batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
        validation_fraction=validation_fraction, seed=seed, device=device,
        progress=progress, started=started)


def train_training_set(training_set: TrainingSet, labels, out_path: str,
                       neutral_style: str = 'Neutral', excluded_styles=(), excluded_bones=(),
                       hidden_units: int = 512, layers: int = 3, embed_dim: int = 32,
                       history_offsets=(0.1, 0.2), gait_sigma: float = 0.35, epochs: int = 60,
                       batch_size: int = 1024, learning_rate: float = 1e-3,
                       weight_decay: float = 1e-4, validation_fraction: float = 0.1,
                       seed: int = 42, device: str = 'auto', progress=None,
                       started: float | None = None) -> dict:
    """
    :func:`train` on a training set already in memory, with one label per entry of its
    ``clip_ranges``.
    """
    progress = progress or _noop_progress
    started = time.time() if started is None else started
    torch.manual_seed(seed)

    if len(labels) != len(training_set.clip_ranges):
        raise ValueError(f'{len(labels)} style labels for {len(training_set.clip_ranges)} clips')

    progress('Pairing frames', 0.05)
    spec = dataset.build_spec(training_set, excluded_bones, history_offsets)
    styles, dropped = dataset.filter_styles(training_set, labels, neutral_style, excluded_styles)
    gait_names = dataset.neutral_gaits(labels, neutral_style)
    inputs = dataset.frame_inputs(training_set, spec)

    pairs = dataset.pair_frames(training_set, labels, styles, gait_names, validation_fraction)
    paired = set(np.unique(pairs.style).tolist())
    if 0 not in paired:
        raise ValueError(f'no frame of the neutral style {neutral_style!r} is usable: every one '
                         'has a baked phase rate of zero')
    if len(paired) < len(styles):
        for s, name in enumerate(styles):
            if s not in paired:
                dropped[name] = 'no usable frame to pair (baked phase rate zero throughout)'
        styles = [name for s, name in enumerate(styles) if s in paired]
        pairs = dataset.pair_frames(training_set, labels, styles, gait_names,
                                    validation_fraction)

    centroids = dataset.gait_centroids(training_set, labels, neutral_style, gait_names,
                                       inputs.gait_features)
    closed_form = dataset.fit_closed_form(training_set, pairs, inputs, len(styles), gait_names,
                                          centroids, gait_sigma)
    targets = dataset.pair_targets(training_set, pairs, inputs, closed_form)

    has_next = pairs.next >= 0
    train_samples = np.flatnonzero(has_next & ~pairs.validation)
    val_samples = np.flatnonzero(has_next & pairs.validation)
    if train_samples.size < 2:
        raise ValueError(f'{train_samples.size} training pairs: nothing to train on')

    progress('Packing tensors', 0.1)
    train_inputs = np.unique(pairs.input[train_samples])
    x_mean, x_std = normalization(inputs.x[train_inputs])

    dynamics_height = (inputs.root_height[pairs.target] - inputs.root_height[pairs.input]
                       - targets.posture_height).astype(np.float32)
    _, height_std = normalization(dynamics_height[train_samples, np.newaxis])
    _, root_std = normalization(targets.root_residual[train_samples])
    y_std = np.concatenate([np.ones(6 * spec.n_styled, dtype=np.float32), height_std,
                            root_std]).astype(np.float32)
    y_mean = np.zeros_like(y_std)

    torch_device = resolve_device(device)
    batch = _Batcher(training_set, spec, inputs, pairs, targets, dynamics_height,
                     x_mean, x_std, y_std, torch_device)

    network = StyleNetwork(spec.input_size, spec.output_size, len(styles),
                           hidden_units, layers, embed_dim).to(torch_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    generator = torch.Generator(device='cpu').manual_seed(seed)

    losses = []
    per_style = np.full(len(styles), np.nan)
    for epoch in range(epochs):
        network.train()
        order = train_samples[torch.randperm(train_samples.size, generator=generator).numpy()]
        total, batches = 0.0, 0
        for start in range(0, order.size, batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch.losses(network, order[start:start + batch_size])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1

        validation, per_style = _validate(network, batch, val_samples, pairs.style, len(styles))
        losses.append((total / max(1, batches), validation))
        progress('Training', 0.1 + 0.85 * (epoch + 1) / max(1, epochs))

    if epochs == 0:
        validation, per_style = _validate(network, batch, val_samples, pairs.style, len(styles))

    progress('Building gate', 0.97)
    gate = _gate_reference(inputs, train_inputs, seed)

    checkpoint = style_io.StyleCheckpoint(
        parameters={name: value.detach().cpu().numpy()
                    for name, value in network.state_dict().items()},
        hidden_units=hidden_units, layers=layers, embed_dim=embed_dim,
        style_names=list(styles), gait_names=list(gait_names),
        gait_centroids=closed_form.gait_centroids, gait_sigma=closed_form.gait_sigma,
        posture_rotations=closed_form.posture_rotations,
        posture_heights=closed_form.posture_heights, speed_ratios=closed_form.speed_ratios,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        bone_names=list(spec.bone_names), parents=spec.parents,
        styled_bone_names=spec.styled_bone_names,
        contact_bone_names=list(spec.contact_bone_names), frame_time=spec.frame_time,
        history_offsets=np.asarray(spec.history_offsets, dtype=np.float64),
        losses=np.asarray(losses, dtype=np.float32).reshape(-1, 2), **gate)
    style_io.save_checkpoint(out_path, checkpoint)
    progress('Done', 1.0)

    summary = {
        'out_path': out_path,
        'samples': int(train_samples.size + val_samples.size),
        'train_samples': int(train_samples.size),
        'styles_kept': list(styles),
        'styles_dropped': dict(dropped),
        'gaits': list(gait_names),
        'held_out_transition_clips': sum(1 for label in labels if label.is_transition),
        'styled_bones': spec.n_styled,
        'input_size': spec.input_size,
        'output_size': spec.output_size,
        'parameters': network.parameter_count(),
        'epochs_run': len(losses),
        'train_loss': losses[-1][0] if losses else float('nan'),
        'val_loss': losses[-1][1] if losses else float('nan'),
        'val_rotation_loss_per_style': {name: float(per_style[s])
                                        for s, name in enumerate(styles)},
        'gate_threshold': gate['gate_threshold'],
        'device': str(torch_device),
        'seconds': time.time() - started,
    }

    print(f"[Style] trained {summary['parameters']} parameters, {len(styles)} styles "
          f"({len(dropped)} dropped) over {spec.n_styled} bones on "
          f"{summary['train_samples']}/{summary['samples']} pairs in "
          f"{summary['seconds']:.1f}s on {summary['device']}: train {summary['train_loss']:.5f}, "
          f"val {summary['val_loss']:.5f} -> {out_path}")
    return summary


class _Batcher:
    """
    The per-frame and per-pair tensors, kept on the CPU and moved to the device a batch at a time,
    and the loss over a batch of pairs.
    """

    def __init__(self, training_set: TrainingSet, spec: dataset.StyleSpec,
                 inputs: dataset.FrameInputs, pairs: dataset.Pairs,
                 targets: dataset.PairTargets, dynamics_height: np.ndarray,
                 x_mean, x_std, y_std, device):
        self.device = device
        self.n_styled = spec.n_styled
        self.frame_time = float(spec.frame_time)

        # FK runs over the styled bones plus every contact bone and its ancestors, so the foot
        # slide is measured on a contact bone even when it is not styled itself.
        fk_bones = set(spec.styled_bones.tolist())
        for bone in spec.contact_bones:
            bone = int(bone)
            while bone >= 0:
                fk_bones.add(bone)
                bone = int(spec.parents[bone])
        fk_bones = np.array(sorted(fk_bones), dtype=np.int64)
        fk_parents = parents_within(spec.parents, fk_bones)
        self.styled_slots = torch.as_tensor(np.searchsorted(fk_bones, spec.styled_bones),
                                            device=device)
        self.contact_slots = torch.as_tensor(np.searchsorted(fk_bones, spec.contact_bones),
                                             device=device)
        self.hierarchy = Hierarchy(fk_parents, device=device)
        self.offsets = torch.from_numpy(rest_offsets(
            training_set.positions[0, fk_bones], training_set.rotations[0, fk_bones],
            fk_parents)).to(device)

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype)

        self.x = tensor((inputs.x - x_mean) / x_std)
        self.fk_local = tensor(local_rotations_6d(training_set.rotations[:, fk_bones], fk_parents))
        self.target_local = tensor(inputs.local_6d)
        self.target_positions = tensor(training_set.positions[:, spec.styled_bones])
        self.root_height = tensor(inputs.root_height)
        self.root_velocity = tensor(training_set.root_velocity)
        self.contacts = tensor(training_set.contacts)

        self.pair_input = torch.as_tensor(pairs.input)
        self.pair_target = torch.as_tensor(pairs.target)
        self.pair_style = torch.as_tensor(pairs.style)
        self.pair_next = torch.as_tensor(pairs.next)
        self.posture = tensor(targets.posture_6d)
        self.posture_height = tensor(targets.posture_height)
        self.speed_ratio = tensor(targets.speed_ratio)
        self.root_residual = tensor(targets.root_residual)
        self.dynamics_height = tensor(dynamics_height)

        self.height_std = float(y_std[6 * spec.n_styled])
        self.root_std = torch.as_tensor(y_std[6 * spec.n_styled + 1:], device=device)
        self.identity_6d = torch.as_tensor(dataset.IDENTITY_6D, device=device)

    def _forward(self, network, pair):
        frame = self.pair_input[pair]
        to = self.device
        out = network(self.x[frame].to(to), self.pair_style[pair].to(to))
        m = out.shape[0]

        raw_rotation = out[:, :6 * self.n_styled].reshape(m, self.n_styled, 6)
        dynamics = six_d_to_matrix(self.identity_6d + raw_rotation)
        posture = six_d_to_matrix(self.posture[pair].to(to))

        local = six_d_to_matrix(self.fk_local[frame].to(to))
        styled_in = local[:, self.styled_slots]
        composed = local.index_copy(1, self.styled_slots, styled_in @ posture @ dynamics)

        height_norm = out[:, 6 * self.n_styled]
        root_norm = out[:, 6 * self.n_styled + 1:]
        height = self.root_height[frame].to(to) + self.posture_height[pair].to(to) \
            + height_norm * self.height_std
        positions, _ = forward_kinematics(matrix_to_six_d(composed), height, self.offsets,
                                          self.hierarchy)

        root_velocity = self.speed_ratio[pair].to(to).unsqueeze(1) \
            * self.root_velocity[frame].to(to) + root_norm[:, :3] * self.root_std[:3]

        return {'raw_rotation': raw_rotation, 'dynamics': dynamics, 'posture': posture,
                'styled_in': styled_in, 'height_norm': height_norm, 'root_norm': root_norm,
                'positions': positions, 'root_velocity': root_velocity}

    def losses(self, network, pair):
        """``(weighted total, per-sample rotation loss)`` over a batch of pair indices."""
        pair = torch.as_tensor(pair, dtype=torch.long)
        to = self.device
        p = self._forward(network, pair)
        q = self._forward(network, self.pair_next[pair])
        target = self.pair_target[pair]

        target_local = six_d_to_matrix(self.target_local[target].to(to))
        remainder = p['posture'].transpose(-1, -2) @ p['styled_in'].transpose(-1, -2) \
            @ target_local
        rotation = (matrix_to_six_d(p['dynamics']) - matrix_to_six_d(remainder)).square() \
            .mean(dim=(1, 2))

        height = (p['height_norm'] - self.dynamics_height[pair].to(to) / self.height_std) \
            .square().mean()
        root = (p['root_norm'] - self.root_residual[pair].to(to) / self.root_std).square().mean()
        position = (p['positions'][:, self.styled_slots]
                    - self.target_positions[target].to(to)).square().mean()

        # World foot velocity, approximated in p's frame: the frame-local displacement plus the
        # root's travel. The omega x r term of the frame's own turn is left out.
        feet_p = p['positions'][:, self.contact_slots]
        feet_q = q['positions'][:, self.contact_slots]
        foot_velocity = (feet_q - feet_p) / self.frame_time + p['root_velocity'].unsqueeze(1)
        planted = self.contacts[target].to(to)
        slide = (planted * foot_velocity[..., [0, 2]].square().sum(-1)).sum() \
            / planted.sum().clamp_min(1.0)

        smooth = (matrix_to_six_d(q['dynamics']) - matrix_to_six_d(p['dynamics'])).square() \
            .mean() + (q['height_norm'] - p['height_norm']).square().mean()
        magnitude = p['raw_rotation'].square().mean() + p['height_norm'].square().mean()

        total = (ROTATION_WEIGHT * rotation.mean() + HEIGHT_WEIGHT * height + ROOT_WEIGHT * root
                 + POSITION_WEIGHT * position + SLIDE_WEIGHT * slide + SMOOTH_WEIGHT * smooth
                 + MAGNITUDE_WEIGHT * magnitude)
        return total, rotation.detach()


def _validate(network, batch: _Batcher, samples: np.ndarray, styles: np.ndarray,
              n_styles: int) -> tuple[float, np.ndarray]:
    """``(mean weighted loss, per-style mean rotation loss)`` over the held-out pairs."""
    if samples.size == 0:
        return float('nan'), np.full(n_styles, np.nan)

    network.eval()
    total, rotation = 0.0, np.zeros(samples.size)
    with torch.no_grad():
        for start in range(0, samples.size, _EVAL_BATCH):
            chunk = samples[start:start + _EVAL_BATCH]
            loss, per_sample = batch.losses(network, chunk)
            total += float(loss) * chunk.size
            rotation[start:start + chunk.size] = per_sample.cpu().numpy()

    counts = np.bincount(styles[samples], minlength=n_styles)
    sums = np.bincount(styles[samples], weights=rotation, minlength=n_styles)
    per_style = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return total / samples.size, per_style


def _gate_reference(inputs: dataset.FrameInputs, frames: np.ndarray, seed: int) -> dict:
    """
    The gate's reference set and threshold: normalised gate vectors of the training inputs, and
    the given percentile of their own leave-one-out mean k-NN distance.
    """
    vectors = dataset.gate_vectors(inputs.descriptors[frames], inputs.phase_rate[frames])
    mean, std = normalization(vectors)
    rng = np.random.default_rng(seed)
    if vectors.shape[0] > GATE_REFERENCE_SIZE:
        vectors = vectors[rng.choice(vectors.shape[0], GATE_REFERENCE_SIZE, replace=False)]
    reference = ((vectors - mean) / std).astype(np.float32)

    k = min(GATE_K, reference.shape[0] - 1)
    if k < 1:
        threshold = 1.0
    else:
        distance, _ = cKDTree(reference).query(reference, k=k + 1)
        threshold = float(np.percentile(distance[:, 1:].mean(axis=1), GATE_PERCENTILE))
        threshold = max(threshold, 1e-6)
    return {'gate_reference': reference, 'gate_mean': mean, 'gate_std': std,
            'gate_threshold': threshold, 'gate_k': max(1, k)}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('database', help='the generated database folder under StreamingAssets')
    parser.add_argument('name', help='base file name, i.e. the Unity asset name')
    parser.add_argument('--out', default=None, help='where to write the .style.npz')
    parser.add_argument('--neutral', default='Neutral', help='the identity style')
    parser.add_argument('--exclude-styles', nargs='*', default=[])
    parser.add_argument('--exclude-bones', nargs='*', default=[], help='bone names not to style')
    parser.add_argument('--hidden-units', type=int, default=512)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--history-offsets', type=float, nargs='*', default=[0.1, 0.2])
    parser.add_argument('--gait-sigma', type=float, default=0.35)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    return parser.parse_args(argv)


def _main(argv=None) -> None:
    args = _parse_args(argv)
    out = args.out or f'{args.name}.style.npz'

    def report(stage, fraction):
        print(f'[Style] {stage} {100.0 * fraction:.0f}%', flush=True)

    summary = train(args.database, args.name, out, neutral_style=args.neutral,
                    excluded_styles=args.exclude_styles, excluded_bones=args.exclude_bones,
                    hidden_units=args.hidden_units, layers=args.layers,
                    embed_dim=args.embed_dim, history_offsets=args.history_offsets,
                    gait_sigma=args.gait_sigma, epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                    validation_fraction=args.validation_fraction, seed=args.seed,
                    device=args.device, progress=report)
    for name, reason in summary['styles_dropped'].items():
        print(f'[Style] dropped {name}: {reason}')


if __name__ == '__main__':
    _main()
