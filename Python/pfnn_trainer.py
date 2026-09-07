"""
Trains a phase-functioned network on a generated pose database.

The loop is ordinary supervised regression -- the interesting parts are all upstream, in what
:mod:`pfnn_dataset` decides an input and a target are. Loss is mean squared error on the
**normalised** output, so every block of the target counts equally regardless of its units; a raw
loss would let the joint velocities, which are numerically the largest block, drown out the phase
increment, which is one float and controls whether the character walks at all.

Validation holds out a **contiguous tail** rather than a random subset. Neighbouring frames of an
animation are nearly the same pose, so a random split puts near-duplicates of the validation set
into training and reports a validation loss that measures nothing.

Runs from the Unity Editor through PythonNET, or standalone::

    python pfnn_trainer.py Assets/StreamingAssets/MMDatabases/MotionMatchingData MotionMatchingData \\
        --out MotionMatchingData.pfnn.npz --epochs 150
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import pfnn_dataset
import pfnn_io
from pfnn_model import PhaseFunctionedNetwork, resolve_device
from training_data import load_database


def _noop_progress(stage: str, fraction: float) -> None:
    pass


def train(data_dir: str,
          db_name: str,
          out_path: str,
          excluded_bones=(),
          window_radius: int = pfnn_dataset.DEFAULT_WINDOW_RADIUS,
          window_stride: int = pfnn_dataset.DEFAULT_WINDOW_STRIDE,
          hidden_units: int = 256,
          dropout: float = 0.3,
          epochs: int = 150,
          batch_size: int = 32,
          learning_rate: float = 1e-4,
          weight_decay: float = 2.5e-3,
          validation_fraction: float = 0.1,
          seed: int = 42,
          device: str = 'auto',
          with_features: bool = True,
          progress=None) -> dict:
    """
    Build the training set, fit the network, and write the checkpoint.

    :param data_dir: the database folder under StreamingAssets.
    :param db_name: base file name, i.e. the Unity asset's name.
    :param out_path: where to write the ``.pfnn.npz``.
    :param excluded_bones: bone names the model does not predict, authored on the Unity config.
    :param with_features: read the ``.mmfeatures`` beside the poses. A database generated from a
        config that authors no matching features has none.
    :param progress: ``(stage, fraction)`` callback, which the Editor drives a progress bar from.
    :return: a summary dict, logged by the caller.
    """
    progress = progress or _noop_progress
    started = time.time()
    torch.manual_seed(seed)

    progress('Reading database', 0.0)
    training_set = load_database(data_dir, db_name, with_features=with_features)
    spec = pfnn_dataset.build_spec(training_set, excluded_bones, window_radius, window_stride)

    progress('Packing vectors', 0.05)
    x, y, phase, _ = pfnn_dataset.build_vectors(training_set, spec)
    if x.shape[0] < 2:
        raise ValueError(f'{x.shape[0]} usable frames in {db_name}: nothing to train on. A clip '
                         'with no measurable gait cycle contributes none -- see GaitPhase in Unity.')

    # The split is by index, and the packed samples are still in database order, so this is the tail
    # of the animation rather than a scattering of frames from all over it.
    split = max(1, int(round(x.shape[0] * (1.0 - validation_fraction))))
    x_mean, x_std = pfnn_dataset.normalization(x[:split])
    y_mean, y_std = pfnn_dataset.normalization(y[:split])

    torch_device = resolve_device(device)
    x_t = torch.from_numpy((x - x_mean) / x_std).to(torch_device)
    y_t = torch.from_numpy((y - y_mean) / y_std).to(torch_device)
    phase_t = torch.from_numpy(phase).to(torch_device)

    network = PhaseFunctionedNetwork(spec.input_size, spec.output_size,
                                     hidden_units, dropout).to(torch_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    losses = []
    generator = torch.Generator(device='cpu').manual_seed(seed)

    for epoch in range(epochs):
        network.train()
        order = torch.randperm(split, generator=generator).to(torch_device)
        total, batches = 0.0, 0

        for start in range(0, split, batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                network(x_t[batch], phase_t[batch]), y_t[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1

        network.eval()
        with torch.no_grad():
            validation = float(torch.nn.functional.mse_loss(
                network(x_t[split:], phase_t[split:]), y_t[split:])) if split < x.shape[0] \
                else float('nan')

        losses.append((total / max(1, batches), validation))
        progress('Training', (epoch + 1) / epochs)

    weights = [layer.weights.detach().cpu().numpy() for layer in network.layers]
    biases = [layer.biases.detach().cpu().numpy() for layer in network.layers]

    pfnn_io.save_checkpoint(out_path, weights, biases, x_mean, x_std, y_mean, y_std,
                            spec.bone_names, spec.window_offsets, spec.frame_time,
                            hidden_units, dropout, losses)

    summary = {
        'out_path': out_path,
        'samples': int(x.shape[0]),
        'train_samples': int(split),
        'bones': spec.n_bones,
        'input_size': spec.input_size,
        'output_size': spec.output_size,
        'parameters': network.parameter_count(),
        'epochs_run': len(losses),
        'train_loss': losses[-1][0] if losses else float('nan'),
        'val_loss': losses[-1][1] if losses else float('nan'),
        'device': str(torch_device),
        'seconds': time.time() - started,
    }

    print(f"[PFNN] trained {summary['parameters']} parameters over {summary['bones']} bones on "
          f"{summary['train_samples']}/{summary['samples']} frames "
          f"({summary['input_size']} -> {summary['output_size']}) in "
          f"{summary['seconds']:.1f}s on {summary['device']}: "
          f"train {summary['train_loss']:.5f}, val {summary['val_loss']:.5f} -> {out_path}")
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('database', help='the generated database folder under StreamingAssets')
    parser.add_argument('name', help='base file name, i.e. the Unity asset name')
    parser.add_argument('--out', default=None, help='where to write the .pfnn.npz')
    parser.add_argument('--exclude', nargs='*', default=[], help='bone names not to predict')
    parser.add_argument('--window-radius', type=int, default=pfnn_dataset.DEFAULT_WINDOW_RADIUS)
    parser.add_argument('--window-stride', type=int, default=pfnn_dataset.DEFAULT_WINDOW_STRIDE)
    parser.add_argument('--hidden-units', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=2.5e-3)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-features', action='store_true',
                        help='skip the .mmfeatures, for a database that has none')
    return parser.parse_args(argv)


def _main(argv=None) -> None:
    args = _parse_args(argv)
    out = args.out or f'{args.name}.pfnn.npz'
    train(args.database, args.name, out,
          excluded_bones=args.exclude,
          window_radius=args.window_radius, window_stride=args.window_stride,
          hidden_units=args.hidden_units, dropout=args.dropout,
          epochs=args.epochs, batch_size=args.batch_size,
          learning_rate=args.learning_rate, weight_decay=args.weight_decay,
          validation_fraction=args.validation_fraction, seed=args.seed,
          device=args.device, with_features=not args.no_features)


if __name__ == '__main__':
    _main()
