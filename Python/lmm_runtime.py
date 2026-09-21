"""
Runs a trained Learned Motion Matching model one frame at a time, for the Unity stage and for
offline inspection.

The policy here is **stateless**, as ``pfnn_runtime``'s is: it takes a whole query and returns a
whole pose, while the character's state -- which feature vector it is holding, which latent -- lives
on the C# side, where the profiler and the inspector can see it and a benchmark run can reset it
without reaching across the boundary.

**The latent table stays in Python.** The ``DecompressorOnly`` runtime mode reads a latent per tick
by database frame, and that lookup is free on this side of the boundary while marshalling a
200 000-row table across it at startup is not. So the stage passes a frame index and gets a pose
back, in the one call it was already making. Once the stepper is running the latent is no longer
one of the baked rows, so the stage carries it and :meth:`LmmPolicy.tick` takes it back --
advancing the state and decompressing it in the one call, because they always happen together.

Forward kinematics is the caller's job, as it is for the PFNN: the network predicts **joint-local**
rotations, and the authority on the rest offsets that turn those into a posed character is the rig.
The stage does it with ``SkeletonData``; :func:`reconstruction_report` below does it with
:mod:`lmm_fk`, which is the same definition the loss was written in.

Everything crossing the PythonNET boundary is a flat Python list of floats, never an ndarray --
pythonnet marshals a list straight into a C# ``float[]``.

Standalone, to judge a checkpoint before Unity is involved::

    python lmm_runtime.py <checkpoint>.lmm.npz \\
        --database ../Assets/StreamingAssets/MMDatabases/MM_LafanCorrected \\
        --name MM_LafanCorrected --report --rollout --project --full-rollout
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import lmm_dataset
import lmm_fk
import lmm_io
import neural_packing
from lmm_model import Decompressor, Projector, Stepper, resolve_device

# Frames scored per forward pass by the offline report. Nothing runtime-critical depends on it.
REPORT_BATCH = 4096


class LmmPolicy:
    """
    A loaded checkpoint, ready to decompress.

    :param checkpoint_path: the ``.lmm.npz`` to load.
    :param log: where to report a load failure. Unity passes its own callable.
    :param device: ``auto``, ``cpu`` or ``cuda``. A single-sample forward pass is small enough that
        CPU usually beats the per-call transfer to a GPU.
    """

    def __init__(self, checkpoint_path: str, log=print, device: str = 'cpu'):
        checkpoint = lmm_io.load_checkpoint(checkpoint_path, log=log)
        if checkpoint is None:
            raise FileNotFoundError(
                f'no usable LMM checkpoint at {checkpoint_path}. Press Train on the config.')

        self.checkpoint = checkpoint
        self.device = resolve_device(device)

        self._pose_layout = lmm_dataset.pose_vector_layout(checkpoint.n_bones)
        # Before anything is fed through it, not at the first odd-looking frame.
        neural_packing.check_blocks(checkpoint.output_blocks, self._pose_layout, what='pose')

        expected = neural_packing.vector_size(self._pose_layout)
        if checkpoint.pose_size != expected:
            raise ValueError(f'checkpoint predicts {checkpoint.pose_size} pose floats where this '
                             f'layout reads {expected}; it has to be retrained')

        self.decompressor = Decompressor(
            checkpoint.feature_size, checkpoint.latent_size, checkpoint.pose_size,
            hidden_units=checkpoint.decompressor_weights[0].shape[0],
            hidden_layers=len(checkpoint.decompressor_weights) - 1).to(self.device)
        self.decompressor.load_parameters(
            checkpoint.decompressor_weights, checkpoint.decompressor_biases)
        self.decompressor.eval()

        self._x_mean = torch.from_numpy(checkpoint.x_mean).to(self.device)
        self._x_std = torch.from_numpy(checkpoint.x_std).to(self.device)
        self._y_mean = checkpoint.y_mean
        self._y_std = checkpoint.y_std
        self._latents = checkpoint.latents

        # The two halves of the state, concatenated once: what the projector's answer is
        # denormalised against. It regresses the state itself, so these are the statistics the
        # checkpoint already carries rather than a second set to keep in step with the first.
        self._state_mean = torch.from_numpy(
            np.concatenate([checkpoint.x_mean, checkpoint.z_mean])).to(self.device)
        self._state_std = torch.from_numpy(
            np.concatenate([checkpoint.x_std, checkpoint.z_std])).to(self.device)

        self.stepper = None
        if checkpoint.has_stage(lmm_io.STAGE_STEPPER):
            self._load_stepper(checkpoint)

        self.projector = None
        if checkpoint.has_stage(lmm_io.STAGE_PROJECTOR):
            self._load_projector(checkpoint)

    def _load_stepper(self, checkpoint) -> None:
        """
        The stepper network, and the statistics that turn its answer back into real rates.

        Refused rather than loaded half-built if the rate statistics are missing: a stepper whose
        output is left normalised would advance the state by roughly the right shape at entirely
        the wrong speed, which looks like a tuning problem rather than a broken file.
        """
        if checkpoint.xz_rate_mean is None or checkpoint.xz_rate_std is None:
            raise ValueError('this checkpoint carries a stepper but not the rate statistics it was '
                             'normalised against; it has to be retrained')

        self.stepper = Stepper(
            checkpoint.feature_size, checkpoint.latent_size,
            hidden_units=checkpoint.stepper_weights[0].shape[0],
            hidden_layers=len(checkpoint.stepper_weights) - 1).to(self.device)
        self.stepper.load_parameters(checkpoint.stepper_weights, checkpoint.stepper_biases)
        self.stepper.eval()

        self._rate_mean = torch.from_numpy(checkpoint.xz_rate_mean).to(self.device)
        self._rate_std = torch.from_numpy(checkpoint.xz_rate_std).to(self.device)

    def _load_projector(self, checkpoint) -> None:
        """The projector network, which answers a query with a state instead of finding one."""
        self.projector = Projector(
            checkpoint.feature_size, checkpoint.latent_size,
            hidden_units=checkpoint.projector_weights[0].shape[0],
            hidden_layers=len(checkpoint.projector_weights) - 1).to(self.device)
        self.projector.load_parameters(
            checkpoint.projector_weights, checkpoint.projector_biases)
        self.projector.eval()

    # Description of the packing, for the stage to validate against its own database --------------

    def bone_names(self) -> list:
        """The bones this model predicts, in the order the pose blocks are packed."""
        return list(self.checkpoint.bone_names)

    def pose_blocks(self) -> list:
        """The names of the pose vector's blocks, in order."""
        return [name for name, _, _ in self._pose_layout]

    def pose_block_offsets(self) -> list:
        """``[offset, count]`` per pose block, flattened -- the slices the stage reads."""
        return [value for _, offset, count in self._pose_layout for value in (offset, count)]

    def feature_names(self) -> list:
        """The matching features the query vector was packed from, in vector order."""
        return list(self.checkpoint.feature_names)

    def feature_widths(self) -> list:
        return [int(width) for width in self.checkpoint.feature_widths]

    def feature_counts(self) -> list:
        return [int(count) for count in self.checkpoint.feature_counts]

    def feature_weights(self) -> list:
        """The authored search weights the model was fitted against, one per float of the query."""
        return [float(weight) for weight in self.checkpoint.feature_weights]

    def feature_size(self) -> int:
        return int(self.checkpoint.feature_size)

    def latent_size(self) -> int:
        return int(self.checkpoint.latent_size)

    def n_frames(self) -> int:
        """Database frames the latent table is indexed by; the staleness check at load."""
        return int(self.checkpoint.n_frames)

    def frame_time(self) -> float:
        return float(self.checkpoint.frame_time)

    def stages_trained(self) -> list:
        """Which of ``autoencoder``, ``stepper``, ``projector`` this checkpoint carries."""
        return list(self.checkpoint.stages_trained)

    def invalid_latent_frames(self) -> list:
        """
        Database frames with no latent -- the last frame of every clip, and any frame whose
        matching feature vector Unity marked invalid.

        Returned as the exceptions rather than as a per-frame mask because there are a few thousand
        of them against a few hundred thousand frames, and marshalling the mask would cost more than
        the information in it.
        """
        return [int(frame) for frame in np.flatnonzero(~self.checkpoint.latent_valid)]

    def describe(self) -> str:
        checkpoint = self.checkpoint
        return (f'{checkpoint.n_bones} bones, X {checkpoint.feature_size} + Z '
                f'{checkpoint.latent_size} -> Y {checkpoint.pose_size}, '
                f'{checkpoint.n_frames} frames, stages {"+".join(checkpoint.stages_trained)}, '
                f'final val loss {checkpoint.final_loss:.5f}')

    # The tick --------------------------------------------------------------------------------

    def decompress_frame(self, features, frame: int) -> list:
        """
        Reconstruct a pose from a query vector and the latent baked for one database frame.

        The ``DecompressorOnly`` path: the stage decides which frame it is holding and this looks
        the latent up, so nothing but the query and the pose crosses the boundary per tick.

        :param features: the matching feature vector, normalised exactly as the database is -- i.e.
            what ``MotionMatchingStage.FillQueryVector`` produces.
        :param frame: a database frame index that has a latent; see :meth:`invalid_latent_frames`.
        """
        return self.decompress(features, self._latents[frame])

    def decompress(self, features, latent) -> list:
        """
        Reconstruct a pose from a query vector and an arbitrary latent.

        The general entry point, used by the ablation diagnostics now and by the stepper later,
        when the latent the stage carries is no longer one of the baked rows.

        :return: ``pose_size`` floats, in the order :meth:`pose_blocks` declares.
        """
        packed = np.asarray(features, dtype=np.float32).ravel()
        if packed.size != self.checkpoint.feature_size:
            raise ValueError(f'got a {packed.size}-float query but the checkpoint expects '
                             f'{self.checkpoint.feature_size}')

        code = np.asarray(latent, dtype=np.float32).ravel()
        if code.size != self.checkpoint.latent_size:
            raise ValueError(f'got a {code.size}-float latent but the checkpoint expects '
                             f'{self.checkpoint.latent_size}')

        with torch.no_grad():
            x = (torch.from_numpy(packed).to(self.device) - self._x_mean) / self._x_std
            y = self.decompressor.decode(x.unsqueeze(0),
                                         torch.from_numpy(code).to(self.device).unsqueeze(0))
            normalised = y[0].cpu().numpy()

        return (normalised * self._y_std + self._y_mean).astype(np.float32).tolist()

    def tick(self, features, latent, delta_time: float):
        """
        Advance the state by one synthesis tick and reconstruct the pose it now describes.

        The stepper's path, and one boundary crossing rather than two: its answer is only
        ever wanted as the decompressor's input, so splitting them would marshal a sixty-five-float
        state across for no reason.

        Advancing *before* decompressing is what the ``DecompressorOnly`` mode does with its
        playhead -- it moves it by ``delta_time`` and then reads the frame it landed on -- so the
        two modes differ in where the state comes from and in nothing else.

        :param features: the query vector the character is holding, in the database's own units.
        :param latent: the latent beside it.
        :param delta_time: seconds since the last tick. The stepper answers in rates per second, so
            synthesis need not run at the rate the database was sampled at.
        :return: ``(pose, features, latent)`` -- the pose to write, and the state to carry into the
            next tick.
        """
        if self.stepper is None:
            raise ValueError('this checkpoint has no stepper, so there is nothing to advance the '
                             'state with. Train one, or run the DecompressorOnly mode.')

        with torch.no_grad():
            x, z = self._advance(self._as_state(features, latent), float(delta_time))
            y = self.decompressor.decode(x, z)[0].cpu().numpy()
            state = (x[0].cpu().numpy(), z[0].cpu().numpy())

        return ((y * self._y_std + self._y_mean).astype(np.float32).tolist(),
                state[0].astype(np.float32).tolist(),
                state[1].astype(np.float32).tolist())

    def step(self, features, latent, delta_time: float):
        """
        Advance the state without reconstructing a pose, for diagnostics and offline rollouts.

        :return: ``(features, latent)`` after ``delta_time`` seconds.
        """
        if self.stepper is None:
            raise ValueError('this checkpoint has no stepper')

        with torch.no_grad():
            x, z = self._advance(self._as_state(features, latent), float(delta_time))

        return (x[0].cpu().numpy().astype(np.float32).tolist(),
                z[0].cpu().numpy().astype(np.float32).tolist())

    def has_stepper(self) -> bool:
        """Whether this checkpoint can advance a state; the stepper-based modes refuse one that cannot."""
        return self.stepper is not None

    def has_projector(self) -> bool:
        """Whether this checkpoint can answer a query without searching. The ``Full`` mode's test."""
        return self.projector is not None

    def project(self, features):
        """
        Answer a query with a state the database could have held -- the projector's search tick.

        Kept separate from :meth:`tick` rather than folded into it, because it runs on search ticks
        only and because **the accept decision stays in C#**, where the authored feature weights
        live and where the classic matcher makes the same decision about the same numbers. Handing
        back a verdict instead of a candidate would move that comparison to the side of the
        boundary that cannot be held to the matcher's own implementation of it.

        :param features: the query, as ``MotionMatchingStage.FillQueryVector`` produces it.
        :return: ``(features, latent)`` -- the state to take if the caller decides it is better
            than the one it is holding.
        """
        if self.projector is None:
            raise ValueError('this checkpoint has no projector, so there is nothing to answer a '
                             'query with. Train one, or run a mode that searches.')

        packed = np.asarray(features, dtype=np.float32).ravel()
        if packed.size != self.checkpoint.feature_size:
            raise ValueError(f'got a {packed.size}-float query but the checkpoint expects '
                             f'{self.checkpoint.feature_size}')

        with torch.no_grad():
            query = torch.from_numpy(packed).to(self.device).unsqueeze(0)
            x, z = self._answer(query)

        return (x[0].cpu().numpy().astype(np.float32).tolist(),
                z[0].cpu().numpy().astype(np.float32).tolist())

    def project_batch(self, features: np.ndarray):
        """Answer many queries at once. For the offline reports; :meth:`project` is the tick path."""
        if self.projector is None:
            raise ValueError('this checkpoint has no projector')

        with torch.no_grad():
            query = torch.from_numpy(
                np.ascontiguousarray(features, dtype=np.float32)).to(self.device)
            x, z = self._answer(query)

        return x.cpu().numpy(), z.cpu().numpy()

    def _answer(self, query):
        """The projector's reply, denormalised. Caller holds ``torch.no_grad()``."""
        size = self.checkpoint.feature_size
        x, z = self.projector.project(query)
        return (x * self._state_std[:size] + self._state_mean[:size],
                z * self._state_std[size:] + self._state_mean[size:])

    def latent(self, frame: int) -> list:
        """One baked latent, for diagnostics and for seeding the stepper."""
        return self._latents[frame].astype(np.float32).tolist()

    def _as_state(self, features, latent):
        """``(1, feature_size)`` and ``(1, latent_size)`` on the device, sizes checked."""
        packed = np.asarray(features, dtype=np.float32).ravel()
        if packed.size != self.checkpoint.feature_size:
            raise ValueError(f'got a {packed.size}-float query but the checkpoint expects '
                             f'{self.checkpoint.feature_size}')

        code = np.asarray(latent, dtype=np.float32).ravel()
        if code.size != self.checkpoint.latent_size:
            raise ValueError(f'got a {code.size}-float latent but the checkpoint expects '
                             f'{self.checkpoint.latent_size}')

        return (torch.from_numpy(packed).to(self.device).unsqueeze(0),
                torch.from_numpy(code).to(self.device).unsqueeze(0))

    def _advance(self, state, delta_time: float):
        """
        One integration of the stepper, batched. Caller holds ``torch.no_grad()``.

        The network answers with a **rate per second**, denormalised here, and the state is
        integrated as every other predicted motion channel in this project is. That is what makes
        the tick rate and the database's frame rate independent of each other.
        """
        features, latent = state
        size = self.checkpoint.feature_size

        rate = self.stepper.rate(features, latent) * self._rate_std + self._rate_mean
        return (features + rate[:, :size] * delta_time,
                latent + rate[:, size:] * delta_time)

    # Batched, for the offline report ------------------------------------------------------------

    def decompress_batch(self, features: np.ndarray, latents: np.ndarray) -> np.ndarray:
        """Many poses at once, in the database's own units. Not for the tick path."""
        with torch.no_grad():
            x = (torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(self.device)
                 - self._x_mean) / self._x_std
            code = torch.from_numpy(np.ascontiguousarray(latents, dtype=np.float32)).to(self.device)
            normalised = self.decompressor.decode(x, code).cpu().numpy()
        return normalised * self._y_std + self._y_mean

    def step_batch(self, features: np.ndarray, latents: np.ndarray, delta_time: float):
        """Advance many states at once. Not for the tick path; :meth:`tick` is."""
        if self.stepper is None:
            raise ValueError('this checkpoint has no stepper')

        with torch.no_grad():
            state = (
                torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(self.device),
                torch.from_numpy(np.ascontiguousarray(latents, dtype=np.float32)).to(self.device))
            x, z = self._advance(state, float(delta_time))

        return x.cpu().numpy(), z.cpu().numpy()


def reconstruction_report(policy: LmmPolicy, training_set, frames: int = 0,
                          holdout: float = 0.0, log=print) -> dict:
    """
    Score a checkpoint against the database it was trained on, in metres rather than loss units.

    This is the honest test of the decompressor. The training loss is computed on weighted blocks
    in mixed units, which makes it comparable between runs and comparable to nothing else -- it cannot say
    whether a foot is a centimetre or a hand's breadth out of place.

    Joint positions come from **forward kinematics of the predicted rotations**, which is the only
    place they can come from: the decompressor predicts joint-local rotations and the root's height,
    and nothing else. That is also what the stage poses the rig with, so the number measured here is
    the number the character exhibits.

    :param frames: how many latent-carrying frames to score, evenly spread; 0 scores all of them.
    :param holdout: score only the last fraction of the training pairs -- the same contiguous tail
        the trainer validated on. 0 scores the whole database, which reports how well the model
        memorised it rather than how well it generalises.
    :return: a summary dict, also logged.
    """
    spec = lmm_dataset.build_spec(training_set, _excluded_for(policy, training_set),
                                  policy.latent_size())
    x, y, q, latent_exists = lmm_dataset.build_vectors(training_set, spec)

    if holdout:
        pairs = lmm_dataset.training_pairs(training_set)
        scored = pairs[max(1, int(round(pairs.size * (1.0 - holdout)))):]
    else:
        scored = np.flatnonzero(latent_exists).astype(np.int64)

    if scored.size == 0:
        raise ValueError('no frame of this database has a latent, so there is nothing to score')
    if frames and frames < scored.size:
        scored = scored[np.linspace(0, scored.size - 1, frames).astype(np.int64)]

    pose_layout = spec.pose_layout()
    character_layout = spec.character_layout()
    offsets = torch.from_numpy(spec.rest_offsets)
    hierarchy = lmm_fk.Hierarchy(spec.parents)

    fk_errors, height_errors, ablation = [], [], []
    speeds, true_speeds, contact_hits = [], [], []

    for start in range(0, scored.size, REPORT_BATCH):
        batch = scored[start:start + REPORT_BATCH]
        predicted = policy.decompress_batch(x[batch], policy.checkpoint.latents[batch])
        without = policy.decompress_batch(x[batch], np.zeros_like(policy.checkpoint.latents[batch]))
        truth = y[batch]

        positions = joint_positions(predicted, pose_layout, spec.n_bones, offsets, hierarchy)
        true_positions = neural_packing.block(q[batch], character_layout, 'positions') \
            .reshape(-1, spec.n_bones, 3)
        fk_errors.append(np.linalg.norm(positions - true_positions, axis=2))
        height_errors.append(np.abs(
            neural_packing.block(predicted, pose_layout, 'root_height')
            - neural_packing.block(truth, pose_layout, 'root_height')))
        ablation.append(np.abs(without - truth).mean() / max(1e-9, np.abs(predicted - truth).mean()))

        speeds.append(np.linalg.norm(
            neural_packing.block(predicted, pose_layout, 'root_velocity')[:, [0, 2]], axis=1))
        true_speeds.append(np.linalg.norm(
            neural_packing.block(truth, pose_layout, 'root_velocity')[:, [0, 2]], axis=1))
        contact_hits.append((neural_packing.block(predicted, pose_layout, 'contacts') > 0.5) ==
                            (neural_packing.block(truth, pose_layout, 'contacts') > 0.5))

    fk_error = np.concatenate(fk_errors)
    mean_speed = float(np.concatenate(speeds).mean())
    true_mean_speed = float(np.concatenate(true_speeds).mean())

    summary = {
        'frames_scored': int(scored.size),
        'held_out': bool(holdout),
        'mean_joint_error': float(fk_error.mean()),
        'joint_error_std': float(fk_error.std()),
        'worst_bone_error': float(fk_error.mean(axis=0).max()),
        'worst_bone': spec.bone_names[int(fk_error.mean(axis=0).argmax())],
        'root_height_error': float(np.concatenate(height_errors).mean()),
        'mean_root_speed': mean_speed,
        'database_root_speed': true_mean_speed,
        'root_speed_ratio': float(mean_speed / true_mean_speed) if true_mean_speed else float('nan'),
        'contact_agreement': float(np.concatenate(contact_hits).mean()),
        'latent_ablation_ratio': float(np.mean(ablation)),
    }

    where = 'held-out' if holdout else 'all'
    log(f"[LMM] reconstruction over {summary['frames_scored']} {where} frames: mean joint error "
        f"{summary['mean_joint_error'] * 100:.2f} cm, std {summary['joint_error_std'] * 100:.2f} cm, "
        f"worst bone {summary['worst_bone']} at {summary['worst_bone_error'] * 100:.2f} cm, "
        f"root height {summary['root_height_error'] * 100:.2f} cm")
    log(f"[LMM] root speed {summary['mean_root_speed']:.3f} m/s against the database's "
        f"{summary['database_root_speed']:.3f} ({summary['root_speed_ratio'] * 100:.1f}%), "
        f"contacts agree {summary['contact_agreement'] * 100:.1f}% of the time")
    log(f"[LMM] latent ablation: D(X||0) is {summary['latent_ablation_ratio']:.2f}x the error of "
        f"D(X||Z) -- a diagnostic of how informative X is, not a pass mark")
    return summary


def joint_positions(predicted: np.ndarray, pose_layout, n_bones: int, offsets, hierarchy):
    """
    (n, n_bones, 3) where a predicted pose actually puts every joint, in the character frame.

    The one place joint positions can come from: the decompressor predicts joint-*local* rotations
    and the root's height, and nothing else. Every report here measures against these rather than
    against anything read out of the database, because this is what the stage poses the rig with.
    """
    local_six = torch.from_numpy(np.ascontiguousarray(
        neural_packing.block(predicted, pose_layout, 'rotations_6d').reshape(-1, n_bones, 6)))
    height = torch.from_numpy(np.ascontiguousarray(
        neural_packing.block(predicted, pose_layout, 'root_height').reshape(-1)))

    with torch.no_grad():
        positions, _ = lmm_fk.forward_kinematics(local_six, height, offsets, hierarchy)
    return positions.numpy()


def rollout_report(policy: LmmPolicy, training_set, seeds: int = 512,
                   holdout: float = 0.1, horizons=(), log=print) -> dict:
    """
    Free-run the stepper from held-out database states and measure how far it has wandered.

    This is the honest test of the stepper: it is asked to advance a state it produced itself,
    over and over, with nothing correcting it. Scored one frame at a time from true states it would
    look far better and mean far less, because the failure mode is compounding error and a
    single-step score cannot see compounding.

    Two kinds of number per horizon. The **drift** is the state's distance from where the database
    says it should be, in units of each half's own spread -- comparable between runs, meaningless
    on its own. The **joint error** is what that drift does to the character, in metres, after
    decompressing the drifted state and running forward kinematics on it.

    Ten frames is the horizon that decides whether the stepper works, because the stage searches
    every ``searchInterval`` seconds -- 10/60 by default, which is ten database frames at 60 Hz.
    Beyond that the numbers say whether the model degrades or explodes, which is a different
    question and one the ``Full`` mode's projector will ask again with a longer cadence.

    :param seeds: how many states to run from, evenly spread over the scored region.
    :param holdout: run only from the last fraction of the eligible starts, which is the tail the
        fit validated on. 0 runs from the whole database and reports memorisation.
    :param horizons: frame counts to report at; empty takes :data:`lmm_dataset.DRIFT_HORIZONS`.
    """
    if not policy.has_stepper():
        raise ValueError('this checkpoint has no stepper to roll out. Train one with '
                         'lmm_trainer.py --stepper-only.')

    horizons = tuple(sorted(horizons or lmm_dataset.DRIFT_HORIZONS))
    spec = lmm_dataset.build_spec(training_set, _excluded_for(policy, training_set),
                                  policy.latent_size())
    x, _, q, latent_exists = lmm_dataset.build_vectors(training_set, spec)
    latents = policy.checkpoint.latents
    x_scale, z_scale = lmm_dataset.state_scales(x, latents, latent_exists)

    runs = lmm_dataset.latent_runs(latent_exists, horizons[-1])
    if holdout:
        runs = runs[int(round(runs.size * (1.0 - holdout))):]
    if runs.size == 0:
        raise ValueError(f'no run of {horizons[-1] + 1} consecutive frames carries a latent, so '
                         'there is nothing long enough to roll out over')

    starts = runs[np.linspace(0, runs.size - 1, min(seeds, runs.size)).astype(np.int64)]
    features, latent = x[starts], latents[starts]

    pose_layout, character_layout = spec.pose_layout(), spec.character_layout()
    offsets = torch.from_numpy(spec.rest_offsets)
    hierarchy = lmm_fk.Hierarchy(spec.parents)

    summary = {'rollout_seeds': int(starts.size), 'held_out': bool(holdout)}
    for step in range(1, horizons[-1] + 1):
        features, latent = policy.step_batch(features, latent, spec.frame_time)
        if step not in horizons:
            continue

        frames = starts + step
        positions = joint_positions(policy.decompress_batch(features, latent),
                                    pose_layout, spec.n_bones, offsets, hierarchy)
        truth = neural_packing.block(q[frames], character_layout, 'positions') \
            .reshape(-1, spec.n_bones, 3)

        summary[f'feature_drift_{step}'] = float(np.abs(features - x[frames]).mean() / x_scale)
        summary[f'latent_drift_{step}'] = float(np.abs(latent - latents[frames]).mean() / z_scale)
        summary[f'joint_error_{step}'] = float(np.linalg.norm(positions - truth, axis=2).mean())

    summary['diverged'] = not all(np.isfinite(value) for key, value in summary.items()
                                  if key.startswith(('feature_', 'latent_', 'joint_')))

    where = 'held-out' if holdout else 'all'
    log(f"[LMM] stepper free run from {summary['rollout_seeds']} {where} states:")
    for step in horizons:
        log(f"[LMM]   {step:>3} frames: X drifts {summary[f'feature_drift_{step}']:.3f} and Z "
            f"{summary[f'latent_drift_{step}']:.3f} of their own spread, and the character stands "
            f"{summary[f'joint_error_{step}'] * 100:.2f} cm from the database")
    log('[LMM] ten frames is the default search cadence, so that row is the one that decides '
        'whether the state survives between searches')
    return summary


def projector_report(policy: LmmPolicy, training_set, queries: int = 2048,
                     holdout: float = 0.1, sigmas=(0.25, 0.5, 1.0), log=print) -> dict:
    """
    Score the projector against the search it replaces, on queries no frame answers exactly.

    This is the honest test of the projector, and the reason it has to be its own report: the
    training loss falls steadily whether or not the answers are the ones the search would have given. What
    matters is the comparison against the lookup itself -- see
    :func:`lmm_dataset.recall_against_search` for what each number means.

    Measured at several displacements rather than one averaged draw, because the two ends behave
    differently and an average over them hides it. At a small displacement the true neighbour is
    nearly the frame the query came from and a near miss is cheap; at a large one the query is
    somewhere the database barely reaches, which is exactly where a controller asking for the
    impossible puts it.

    :param queries: how many held-out frames to displace and answer.
    :param holdout: draw only from the last fraction of the latent-carrying frames -- the tail the
        fit validated on. 0 draws from the whole database and reports memorisation.
    :param sigmas: the displacements to report at, in units of
        :func:`lmm_dataset.projector_noise_scale`.
    """
    if not policy.has_projector():
        raise ValueError('this checkpoint has no projector to score. Train one with '
                         'lmm_trainer.py --projector-only.')

    spec = lmm_dataset.build_spec(training_set, _excluded_for(policy, training_set),
                                  policy.latent_size())
    x, _, _, latent_exists = lmm_dataset.build_vectors(training_set, spec)
    latents = policy.checkpoint.latents
    _, z_scale = lmm_dataset.state_scales(x, latents, latent_exists)

    frames = np.flatnonzero(latent_exists).astype(np.int64)
    if frames.size == 0:
        raise ValueError('no frame of this database has a latent, so there is nothing to project')

    device = policy.device
    candidates = torch.from_numpy(x[frames]).to(device)
    candidate_latents = torch.from_numpy(latents[frames]).to(device)
    weights = torch.from_numpy(policy.checkpoint.feature_weights).to(device)
    norms = lmm_dataset.candidate_norms(candidates, weights)
    noise_scale = torch.from_numpy(
        lmm_dataset.projector_noise_scale(x, latent_exists)).to(device)

    drawn = frames[int(round(frames.size * (1.0 - holdout))):] if holdout else frames
    drawn = drawn[np.linspace(0, drawn.size - 1, min(queries, drawn.size)).astype(np.int64)]
    seeds = torch.from_numpy(x[drawn]).to(device)

    generator = torch.Generator(device='cpu').manual_seed(11)
    summary = {'projector_queries_scored': int(drawn.size), 'held_out': bool(holdout)}

    for sigma in sigmas:
        noise = torch.randn(seeds.shape, generator=generator).to(device)
        displaced = seeds + sigma * noise_scale * noise

        answered_x, answered_z = policy.project_batch(displaced.cpu().numpy())
        recall = lmm_dataset.recall_against_search(
            displaced, torch.from_numpy(answered_x).to(device),
            torch.from_numpy(answered_z).to(device),
            candidates, candidate_latents, weights, norms, z_scale)

        for key, value in recall.items():
            summary[f'{key}_{sigma:g}'] = value

    where = 'held-out' if holdout else 'all'
    log(f"[LMM] projector against the search it replaces, over "
        f"{summary['projector_queries_scored']} {where} queries:")
    for sigma in sigmas:
        log(f"[LMM]   displaced {sigma:>4g}: answers from "
            f"{summary[f'distance_ratio_{sigma:g}']:.3f}x the true nearest neighbour's distance, "
            f"inside the nearest 1% {summary[f'top_percent_recall_{sigma:g}'] * 100:.0f}% of the "
            f"time, latent {summary[f'latent_error_{sigma:g}']:.3f} of its spread out")
    return summary


def full_rollout_report(policy: LmmPolicy, training_set, seeds: int = 256, frames: int = 900,
                        holdout: float = 0.1, search_interval: int = 10,
                        acceptance_ratio: float = 0.95, log=print) -> dict:
    """
    Run the whole method free of the database for thirty seconds and see whether it survives.

    Projector, stepper and decompressor together, in the loop ``LmmStage`` runs in its ``Full``
    mode: every ``search_interval`` frames the query is answered by the projector and taken only if
    it is nearer than the state already held, and in between the stepper carries that state
    forward. Nothing reads the database. **This is the only honest test of the projector**, because
    every per-frame score the three networks produce stays plausible long after the loop as a whole has
    stopped producing motion.

    The controller is held still: each seed goes on asking for the trajectory its own frame asked
    for. That is a control signal a character can follow indefinitely, which a replayed one cannot
    be -- the clips here average two seconds and this runs for thirty -- and it is the harder ask,
    because nothing in it ever pulls a drifting state back towards the data.

    Three things are measured against the seeds' own frames: the speed the character travels at,
    how far its joints reach from its root, and whether any of it stopped being finite. A model
    that has quietly collapsed stands still with its arms at its sides, and reports an excellent
    per-frame loss while doing it.

    :param seeds: states to run from, evenly spread over the scored region.
    :param frames: database frames to run for; 900 is thirty seconds at 30 Hz.
    :param search_interval: frames between projections. Ten is the stage's default cadence.
    :param acceptance_ratio: a projection is taken when it is this much nearer than the held state,
        matching ``LmmStage.acceptanceRatio``.
    """
    if not policy.has_projector():
        raise ValueError('this checkpoint has no projector, so there is no full loop to run')
    if not policy.has_stepper():
        raise ValueError('this checkpoint has no stepper, so a projected state cannot be carried')

    spec = lmm_dataset.build_spec(training_set, _excluded_for(policy, training_set),
                                  policy.latent_size())
    x, _, q, latent_exists = lmm_dataset.build_vectors(training_set, spec)
    latents = policy.checkpoint.latents
    pose_offset = policy.checkpoint.pose_offset
    weights = policy.checkpoint.feature_weights

    eligible = np.flatnonzero(latent_exists).astype(np.int64)
    if holdout:
        eligible = eligible[int(round(eligible.size * (1.0 - holdout))):]
    if eligible.size == 0:
        raise ValueError('no held-out frame carries a latent to start from')

    starts = eligible[np.linspace(0, eligible.size - 1, min(seeds, eligible.size)).astype(np.int64)]
    state_x, state_z = x[starts].copy(), latents[starts].copy()

    # Held for the whole run: the controller asking for what it asked for at the seed frame.
    request = x[starts][:, :pose_offset].copy()

    pose_layout, character_layout = spec.pose_layout(), spec.character_layout()
    offsets = torch.from_numpy(spec.rest_offsets)
    hierarchy = lmm_fk.Hierarchy(spec.parents)

    speeds, extents = [], []
    searches = accepted = 0

    for step in range(frames):
        if step and step % search_interval == 0:
            query = np.concatenate([request, state_x[:, pose_offset:]], axis=1)
            candidate_x, candidate_z = policy.project_batch(query)

            # The same comparison the stage makes, in the same squared-distance metric: take the
            # answer only when it is enough nearer than what is already held.
            held = (((query - state_x) ** 2) * weights).sum(axis=1)
            offered = (((query - candidate_x) ** 2) * weights).sum(axis=1)
            take = offered < held * acceptance_ratio

            state_x = np.where(take[:, None], candidate_x, state_x)
            state_z = np.where(take[:, None], candidate_z, state_z)
            searches += take.size
            accepted += int(take.sum())

        state_x, state_z = policy.step_batch(state_x, state_z, spec.frame_time)

        predicted = policy.decompress_batch(state_x, state_z)
        speeds.append(np.linalg.norm(
            neural_packing.block(predicted, pose_layout, 'root_velocity')[:, [0, 2]], axis=1))
        positions = joint_positions(predicted, pose_layout, spec.n_bones, offsets, hierarchy)
        extents.append(np.linalg.norm(positions - positions[:, :1], axis=2).max(axis=1))

    true_positions = neural_packing.block(q[starts], character_layout, 'positions') \
        .reshape(-1, spec.n_bones, 3)
    true_speed = float(np.linalg.norm(
        training_set.root_velocity[starts][:, [0, 2]], axis=1).mean())
    true_extent = float(np.linalg.norm(
        true_positions - true_positions[:, :1], axis=2).max(axis=1).mean())

    speed = float(np.mean(speeds))
    extent = float(np.mean(extents))
    summary = {
        'full_seeds': int(starts.size),
        'full_frames': int(frames),
        'full_seconds': float(frames * spec.frame_time),
        'full_search_interval': int(search_interval),
        'full_accept_rate': float(accepted / searches) if searches else float('nan'),
        'full_mean_speed': speed,
        'full_database_speed': true_speed,
        'full_speed_ratio': float(speed / true_speed) if true_speed else float('nan'),
        'full_mean_extent': extent,
        'full_database_extent': true_extent,
        'full_extent_ratio': float(extent / true_extent) if true_extent else float('nan'),
        'full_diverged': not (np.isfinite(speed) and np.isfinite(extent)),
    }

    log(f"[LMM] full loop free-run: {summary['full_seeds']} seeds for "
        f"{summary['full_frames']} frames ({summary['full_seconds']:.0f} s), projecting every "
        f"{summary['full_search_interval']} and taking the answer "
        f"{summary['full_accept_rate'] * 100:.0f}% of the time")
    log(f"[LMM]   travels at {summary['full_mean_speed']:.3f} m/s against the seeds' "
        f"{summary['full_database_speed']:.3f} ({summary['full_speed_ratio']:.2f}x), reaching "
        f"{summary['full_mean_extent']:.3f} m from the root against "
        f"{summary['full_database_extent']:.3f} ({summary['full_extent_ratio']:.2f}x)")
    if summary['full_diverged']:
        log('[LMM]   the run produced values that are not finite: the loop diverged')
    return summary


def _excluded_for(policy: LmmPolicy, training_set) -> list:
    """The bones the checkpoint does not predict, as names, against this database's skeleton."""
    predicted = set(policy.bone_names())
    missing = predicted.difference(training_set.bone_names)
    if missing:
        raise ValueError('this checkpoint predicts bones the database does not have: '
                         + ', '.join(sorted(missing)))
    return [name for name in training_set.bone_names if name not in predicted]


def _main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('checkpoint', help='the .lmm.npz to load')
    parser.add_argument('--database', default=None, help='database folder, for --report')
    parser.add_argument('--name', default=None, help='database base file name, for --report')
    parser.add_argument('--report', action='store_true',
                        help='score the checkpoint against its database, in metres')
    parser.add_argument('--rollout', action='store_true',
                        help='free-run the stepper from held-out states and report its drift')
    parser.add_argument('--project', action='store_true',
                        help='score the projector against the search it replaces')
    parser.add_argument('--full-rollout', action='store_true',
                        help='run projector, stepper and decompressor together, free of the '
                             'database, and report whether the loop survives')
    parser.add_argument('--full-frames', type=int, default=900,
                        help='database frames the full loop is run for; 900 is thirty seconds')
    parser.add_argument('--holdout', type=float, nargs='?', const=0.1, default=0.0,
                        help='score only the last fraction of the pairs, as the trainer validated')
    parser.add_argument('--frames', type=int, default=0,
                        help='frames to score, evenly spread; 0 scores all of them')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)

    policy = LmmPolicy(args.checkpoint, device=args.device)
    print(f'[LMM] {args.checkpoint}: {policy.describe()}')

    if not any((args.report, args.rollout, args.project, args.full_rollout)):
        return
    if not args.database or not args.name:
        parser.error('every report needs --database and --name')

    from training_data import load_database
    training_set = load_database(args.database, args.name)

    if args.report:
        reconstruction_report(policy, training_set, args.frames, holdout=args.holdout)
    if args.rollout:
        rollout_report(policy, training_set, holdout=args.holdout or 0.1)
    if args.project:
        projector_report(policy, training_set, holdout=args.holdout or 0.1)
    if args.full_rollout:
        full_rollout_report(policy, training_set, frames=args.full_frames,
                            holdout=args.holdout or 0.1)


if __name__ == '__main__':
    _main()
