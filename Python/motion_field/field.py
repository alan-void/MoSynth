from __future__ import annotations

import json

import numpy as np
import torch

from core.pose import Pose, PoseDelta
from core.skeleton import Skeleton
from motion_field.io import ValueFunctionData, load_value_function

# Packed pose layout: row 0 the character frame's position, row 1 the rig root
# bone's position within that frame, row 2 the frame's rotation, row 3 the root
# bone's rotation within it, rows 4+ the remaining joint quaternions.

def wrap_angle(angle):
    """Wrap radians into [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def root_yaw(packed_x: np.ndarray) -> np.ndarray:
    """
    Yaw carried by the character frame's quaternion of a packed pose, in radians.

    Exact rather than approximate: `simulation_frame.derive_frames` builds every
    frame rotation as [0, sin(psi/2), 0, cos(psi/2)], so it is a pure yaw by
    construction and reduces to 2*atan2(q.y, q.w). Note the xyzw component order
    -- the wxyz formula the reference implementation uses would read x and z here
    and return nonsense.
    """
    root_quat_row = 2
    q = np.asarray(packed_x)[..., root_quat_row, :]
    return 2.0 * np.arctan2(q[..., 1], q[..., 3])


def state_locomotion_scores(poses_v: np.ndarray, speed_threshold: float) -> np.ndarray:
    """
    Per-state "is this state moving" score in [0, 1], from root speed.

    The reference implementation scored states by locomotion-set membership
    (walk vs jog); with a single set that generalises to moving vs idle, which
    is the distinction that matters: it is what keeps an idle pose from being a
    zero-cost place for the reward's heading term to hide in. Ramped rather
    than thresholded so slow turn-around frames keep partial credit instead of
    flickering across a cliff.
    """
    speeds = np.linalg.norm(np.asarray(poses_v)[:, 0, :3], axis=-1)
    return np.clip(speeds / max(float(speed_threshold), 1e-6), 0.0, 1.0).astype(np.float32)


def resolve_device(preference=None) -> str:
    if preference in (None, '', 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if preference == 'cuda' and not torch.cuda.is_available():
        print('[MotionField] CUDA requested but unavailable; falling back to CPU.')
        return 'cpu'
    return preference


def _weight_pair(value) -> tuple[float, float]:
    """One bone's (position, velocity) weight, from a scalar or a 2-sequence."""
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value), float(value)

    pair = list(value)
    if len(pair) != 2:
        raise ValueError(f'a bone weight must be a scalar or (pos, vel), got {value!r}')
    return float(pair[0]), float(pair[1])


def pack_bone_weights(skeleton: Skeleton, bone_weights, log=print) -> np.ndarray | None:
    """
    Normalize a per-bone weight table onto the feature row order.

    Weights are keyed by joint name rather than by index. The feature rows follow
    `list(skeleton)`, a depth-first walk, which happens to match the order joints
    are serialized in but is not guaranteed to -- resolving by name means a
    reordered skeleton moves the weights with it instead of silently applying
    them to the wrong bones.

    :param bone_weights: `None`, a `{joint name: weight}` mapping (weight being a
        scalar or a `(position, velocity)` pair), or an array already shaped
        `(joint_count, 2)` in feature-row order.
    :return: `(joint_count, 2)` float32, or `None` when every weight is 1 --
        callers treat `None` as "uniform" and skip the multiply entirely.
    """
    if bone_weights is None:
        return None

    names = [joint.name for joint in skeleton]
    count = len(names)

    if hasattr(bone_weights, 'keys'):
        table = np.ones((count, 2), dtype=np.float32)
        slot_of = {name: i for i, name in enumerate(names)}
        unknown = []
        for key in bone_weights.keys():
            slot = slot_of.get(str(key))
            if slot is None:
                unknown.append(str(key))
                continue
            table[slot] = _weight_pair(bone_weights[key])
        if unknown:
            log(f'[MotionField] bone weights name joints this skeleton does not '
                f'have, ignored: {sorted(unknown)}')
    else:
        table = np.asarray(bone_weights, dtype=np.float32)
        if table.shape != (count, 2):
            raise ValueError(f'bone_weights must be ({count}, 2) for this skeleton, '
                             f'got {table.shape}')
        table = np.ascontiguousarray(table, dtype=np.float32)

    # Row 0 is the character frame, which `fk_root_space` pins to the origin with
    # identity rotation -- its position and velocity are structurally zero, so its
    # weight cannot change the feature. Normalising it to 1 lets a table that
    # differs only there collapse to `None` below instead of reading as a metric
    # change. Row 1 is the rig's root bone, whose translation `build_motion_states`
    # replaces with the rest-pose offset -- constant across states, so equally
    # unable to move a distance.
    table[0] = 1.0
    table[1] = 1.0

    if np.all(np.abs(table - 1.0) <= 1e-6):
        return None
    return table


def load_bone_weights_file(path: str | None):
    """
    Read a `{"JointName": [pos, vel], ...}` JSON file for the CLIs.

    Unity passes the same mapping straight across PythonNET into
    `pack_bone_weights`, so this exists only to give the headless entry points
    parity with the Editor buttons.
    """
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


class MotionField:
    def __init__(self,
                 poses_x: np.ndarray,
                 poses_v: np.ndarray, poses_y: np.ndarray,
                 skeleton: Skeleton,
                 frame_time: float,
                 device=None,
                 pos_weight: float = 1.0,
                 vel_weight: float = 0.06,
                 k_neighbors: int = 15,
                 tug_ratio: float = 0.1,
                 knn_chunk: int = 64,
                 bone_weights=None,
                 locomotion_factor: float = 1.0,
                 locomotion_speed_threshold: float = 0.5,
                 value_function_path: str = None):
        """
        Initialize the MotionField.

        :param poses_x: Pose in array format (root, hips, quats) for each state. Shape: (state_count, num_bones + 2, 4)
        :param poses_v: Pose velocity in array format (root vel, hips vel, rotvecs) for each state, per second. Shape: (state_count, num_bones + 2, 4)
        :param poses_y: Next pose velocity in array format (root vel, hips vel, rotvecs) for each state, per second. Shape: (state_count, num_bones + 2, 4)
        :param skeleton: The Skeleton object used for forward kinematics of the poses.
        :param frame_time: Seconds per frame of the source database. Velocities are per-second rates, so this is needed to turn them into a one-frame step.
        :param device: 'cuda', 'cpu' or None/'auto' to pick whatever is present.
        :param pos_weight: Weight on the position half of the similarity feature.
        :param vel_weight: Weight on the velocity half of the similarity feature.
        :param k_neighbors: Default neighborhood size for k-NN and for the action set.
        :param tug_ratio: Drift correction strength toward the database, per step.
        :param knn_chunk: Queries per batched k-NN chunk. Bounds VRAM at roughly
            chunk * state_count * feature_count * 3 * 4 bytes.
        :param bone_weights: Optional per-joint emphasis on top of
            `pos_weight`/`vel_weight`. See `pack_bone_weights`.
        :param locomotion_factor: Reward bonus for landing in moving states --
            the reference implementation's `state_score * factor` term. Without
            it every reward is <= 0 and a database state that stands still while
            facing the goal scores a perfect 0 forever, so the policy freezes
            into any idle pose the data contains. 0 restores that behavior.
        :param locomotion_speed_threshold: Root speed, m/s, at which a state
            counts as fully moving. The score ramps linearly from 0 below it.
        :param value_function_path: Optional `.mffield.npz` enabling `optimal_action`.
        """
        self.current_frame = None
        assert (poses_x.shape[:-2] ==
                poses_v.shape[:-2] ==
                poses_y.shape[:-2]), "Inconsistent batch shapes"

        self.frame_time = frame_time
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.k_neighbors = k_neighbors
        self.tug_ratio = tug_ratio
        self.knn_chunk = max(1, int(knn_chunk))
        self.device = resolve_device(device)

        self.poses_x = poses_x
        self.poses_v = poses_v
        self.poses_y = poses_y
        self.skeleton = skeleton
        self.states_count = poses_x.shape[0]
        self.bone_count = poses_x.shape[-2] - 2
        self.bone_weights = pack_bone_weights(skeleton, bone_weights)

        # How much each database state "moves", for the locomotion reward term.
        self.locomotion_factor = float(locomotion_factor)
        self.locomotion_speed_threshold = float(locomotion_speed_threshold)
        self.state_scores = state_locomotion_scores(poses_v, self.locomotion_speed_threshold)

        self.state_features = self.build_motion_states(poses_x, poses_v)
        self.feature_shape = self.state_features.shape[-2:]
        self.states = torch.from_numpy(self.state_features).to(self.device)

        self.value_function: ValueFunctionData | None = None
        if value_function_path:
            self.load_value_function(value_function_path)

        # Last decision, for the Unity debug visualizer. Written by
        # optimal_action / greedy_action, read via get_debug_arrays().
        self.last_indices = None
        self.last_weights = None
        self.last_action = None

    def get_batched_knn(self, queries: np.ndarray, k: int = None, chunk: int = None):
        """
        Brute-force k-NN for many queries at once.

        :param queries: (query_count, feature_count, 3)
        :return: (indices (query_count, k) int64, weights (query_count, k) float32)
            where the weights are the normalised inverse-square distances the
            paper calls similarity weights.
        """
        k = self.k_neighbors if k is None else k
        chunk = self.knn_chunk if chunk is None else max(1, int(chunk))

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 2:
            queries = queries[np.newaxis, ...]

        query_tensor = torch.from_numpy(queries).to(self.device)
        states = self.states.unsqueeze(0)  # (1, state_count, feature_count, 3)

        indices = np.empty((queries.shape[0], k), dtype=np.int64)
        distances = np.empty((queries.shape[0], k), dtype=np.float32)

        # Chunked because the difference tensor is
        # chunk * state_count * feature_count * 3 floats -- a few hundred MB at
        # chunk=64 on this database, and quadratic in the chunk size.
        for start in range(0, query_tensor.shape[0], chunk):
            q = query_tensor[start:start + chunk].unsqueeze(1)  # (c, 1, F, 3)
            diff = states - q  # (c, S, F, 3)
            per_joint = torch.norm(diff, dim=3)  # (c, S, F)
            summed = torch.sum(per_joint, dim=2)  # (c, S)
            top_k = torch.topk(summed, k=k, largest=False)
            indices[start:start + chunk] = top_k.indices.cpu().numpy()
            distances[start:start + chunk] = top_k.values.cpu().numpy()

        return indices, MotionField.calculate_similarity_weights(distances)

    def get_knn(self, motion_state, k):
        """Single-query k-NN. Returns (indices (k,), distances (k,))."""
        query = np.ascontiguousarray(motion_state, dtype=np.float32)
        if query.ndim == 3:
            query = query[0]

        query_tensor = torch.from_numpy(query).to(self.device)

        diff = self.states - query_tensor  # (state_count, feature_count, 3)
        point_distances = torch.norm(diff, dim=2)
        sum_distances = torch.sum(point_distances, dim=1)

        top_k = torch.topk(sum_distances, k=k, largest=False)
        return top_k.indices.cpu().numpy(), top_k.values.cpu().numpy()

    @staticmethod
    def calculate_similarity_weights(distances: np.ndarray) -> np.ndarray:
        """Normalized inverse-square distance weights, over the last axis."""
        weights = 1.0 / (np.asarray(distances, dtype=np.float32) ** 2 + 1e-8)
        return (weights / np.sum(weights, axis=-1, keepdims=True)).astype(np.float32)

    @staticmethod
    def build_action_weights(weights: np.ndarray) -> np.ndarray:
        """
        The candidate action set for one state.

        One action per neighbour: take the similarity weights and pin that
        neighbour's weight to 1 before renormalising, so each action leans on a
        different neighbour while still respecting the similarity distribution.
        Action i is row i.

        :param weights: (k,) similarity weights
        :return: (k, k) float32, each row a convex combination
        """
        k = weights.shape[-1]
        actions = np.tile(np.asarray(weights, dtype=np.float32), (k, 1))
        actions[np.arange(k), np.arange(k)] = 1.0
        return actions / np.sum(actions, axis=1, keepdims=True)

    # ------------------------------------------------------------------ #
    # Integration
    # ------------------------------------------------------------------ #

    def compute_new_states(self,
                           current_pose_x: np.ndarray,
                           delta_time: float,
                           indices: np.ndarray,
                           action_weights: np.ndarray,
                           tug_indices: np.ndarray,
                           tug_ratio: float = None):
        """
        Integrate several candidate actions from a pose, in one batch.

        :param current_pose_x: (num_bones+2, 4), (1, num_bones+2, 4) or
            (action_count, num_bones+2, 4) if each action starts somewhere else
        :param indices: (k,) neighbour state ids shared by every action, or
            (action_count, k) when each action has its own neighbourhood --
            which is what the trainer needs to evaluate every database state at
            once.
        :param action_weights: (action_count, k) convex weights per action
        :param tug_indices: (action_count,) database state each action is tugged toward
        :return: (x, v), both (action_count, num_bones+2, 4)
        """
        tug_ratio = self.tug_ratio if tug_ratio is None else tug_ratio
        # The tug is a contraction applied per CALL, so its strength has to
        # follow the step size: at delta_time < frame_time a fixed ratio pulls
        # onto a database pose faster than the dt-scaled advance can move away
        # -- a literal freeze. Compounding it per database frame makes n small
        # steps tug exactly as much as one full frame. Identity when
        # delta_time == frame_time, which is what training uses.
        tug_ratio = 1.0 - (1.0 - tug_ratio) ** (delta_time / self.frame_time)
        action_weights = np.ascontiguousarray(action_weights, dtype=np.float32)
        action_count = action_weights.shape[0]
        tug_indices = np.asarray(tug_indices).reshape(-1)

        x0 = np.asarray(current_pose_x, dtype=np.float32)
        if x0.ndim == 2:
            x0 = x0[np.newaxis, ...]
        if x0.shape[0] == 1 and action_count > 1:
            x0 = np.broadcast_to(x0, (action_count,) + x0.shape[1:])
        assert x0.shape[0] == action_count, \
            f'expected {action_count} start poses, got {x0.shape[0]}'

        current = Pose.from_array(x0)  # from_array copies, so this is writable

        # The frame slot is a PER-FRAME INCREMENT, not a world pose. Database
        # states are stored in their own character frame (framePos == 0, frame
        # quat == identity, see motion_field.action_predictor.load_animations) and the tug
        # below blends against one of them. So 'tug_ratio' would drag an
        # accumulated world position toward the origin every frame -- a 0.9^n
        # collapse that reaches the origin in about 30 frames. Unity owns the
        # world placement anyway and integrates it from the returned velocities
        # (MotionSynthesisComponent.ApplyPoseToSkeletonTransforms), so zeroing
        # here also makes training and runtime structurally identical: the frame
        # of the returned pose is exactly the increment this step produced.
        current.rootPos[...] = 0.0
        current.quats[..., 0, :] = (0.0, 0.0, 0.0, 1.0)

        # Blend the neighbour velocities under each action's weights.
        indices = np.asarray(indices)
        gather = 'kbc,ak->abc' if indices.ndim == 1 else 'akbc,ak->abc'
        blended_v = np.einsum(gather, self.poses_v[indices, ...], action_weights)
        blended_y = np.einsum(gather, self.poses_y[indices, ...], action_weights)

        next_pose_x = current.add(PoseDelta.from_array(blended_v).scaled(delta_time))
        next_pose_v = PoseDelta.from_array(blended_y)

        # Drift correction: pull a little toward an actual database state so the
        # field cannot wander into regions it has no data for.
        nearest_pose_x = Pose.from_array(self.poses_x[tug_indices]).add(
            PoseDelta.from_array(self.poses_v[tug_indices]).scaled(delta_time))
        nearest_pose_v = PoseDelta.from_array(self.poses_y[tug_indices])

        tug_weights = np.array([1.0 - tug_ratio, tug_ratio], dtype=np.float32)
        tugged_pose_x = Pose.blend(Pose.stack([next_pose_x, nearest_pose_x]), tug_weights)
        tugged_pose_v = PoseDelta.blend(
            PoseDelta.stack([next_pose_v, nearest_pose_v]), tug_weights)

        return tugged_pose_x.pack(), tugged_pose_v.pack()

    def load_value_function(self, path: str, log=None) -> bool:
        """
        Load a trained value function. Returns False, leaving the field in greedy
        mode, when the file is missing or unreadable rather than raising -- Unity
        calls this inside Py.GIL() where an exception surfaces as an opaque
        managed error.

        Whether the file matches the config that will run it is decided in Unity
        (`MotionFieldConfig.hasTrained`) before this is called, so nothing is
        re-checked here. The caller does have to get it right, though:
        every index in the value function is a row of the pose database, so a
        file trained against a re-extracted `.mmpose` addresses the wrong poses
        without any symptom this module could detect.

        :param path:
        :param log: optional `callable(str)` for the outcome. Defaults to
            `print`, which under PythonNET goes to a stdout Unity does not
            display -- so Unity passes its own.
        """
        log = log or print
        self.value_function = load_value_function(path, log=log)
        if self.value_function is not None:
            log(f'[MotionField] loaded value function {path} '
                f'({self.value_function.values.shape[0]} states x '
                f'{self.value_function.theta_count} headings, '
                f'final residual {self.value_function.final_residual:.6f})')
        return self.value_function is not None

    def interpolate_value(self, indices: np.ndarray, weights: np.ndarray,
                          theta: np.ndarray) -> np.ndarray:
        """
        V(s', theta') by bilinear interpolation.

        Linear across the two bracketing headings on the task grid, then
        weighted by the motion-space similarity weights. Sampling the value
        function at only the discrete database states is what makes it behave as
        if it were continuous over the whole field.

        :param indices: (n, k) neighbour state ids
        :param weights: (n, k) similarity weights
        :param theta: (n,) goal headings
        :return: (n,)
        """
        vf = self.value_function
        coord = (np.asarray(theta, dtype=np.float32) + np.pi) / vf.theta_spacing
        lower = np.floor(coord)
        alpha = (coord - lower)[:, np.newaxis]
        i0 = (lower.astype(np.int64) % vf.theta_count)[:, np.newaxis]
        i1 = (i0 + 1) % vf.theta_count

        v = (1.0 - alpha) * vf.values[indices, i0] + alpha * vf.values[indices, i1]
        return np.sum(v * weights, axis=1)

    def value_reward(self, theta: float, next_x: np.ndarray, next_v: np.ndarray) -> float:
        """
        Reward for landing in a state, using the trained value function.

        This is the Bellman objective the training optimised:
        `Q = R(s, a) + gamma * V(s')`. The immediate term keeps the runtime
        consistent with the value iteration -- scoring on the discounted future
        value alone would optimise something the value function was never fitted
        to.

        :param theta: the character's current heading in the goal frame, radians
        :param next_x: packed pose after the action, (num_bones+2, 4) or (1, ...)
        :param next_v: matching packed velocity
        :return: scalar Q
        """
        return float(self.value_rewards(theta, next_x, next_v)[0])

    def value_rewards(self, theta: float, next_x: np.ndarray, next_v: np.ndarray) -> np.ndarray:
        """Batched `value_reward` over several candidate outcomes. Returns (n,)."""
        assert self.value_function is not None, 'no value function loaded'

        next_x = np.asarray(next_x, dtype=np.float32)
        next_v = np.asarray(next_v, dtype=np.float32)
        if next_x.ndim == 2:
            next_x = next_x[np.newaxis, ...]
            next_v = next_v[np.newaxis, ...]

        # Each action rotates the character, so the goal moves in its frame.
        theta_prime = wrap_angle(theta + root_yaw(next_x)).astype(np.float32)

        indices, weights = self.get_batched_knn(self.build_motion_states(next_x, next_v),
                                                self.value_function.k_neighbors)
        future = self.interpolate_value(indices, weights, theta_prime)
        # The locomotion bonus is paid on the arrival state, interpolated
        # through the same neighbourhood as V -- the immediate reward must match
        # what the value iteration optimised or the argmax drifts off-policy.
        arrival = np.sum(self.state_scores[indices] * weights, axis=1)
        return (-np.abs(theta_prime) + self.locomotion_factor * arrival
                + self.value_function.gamma * future)

    # ------------------------------------------------------------------ #
    # Policies
    # ------------------------------------------------------------------ #

    def optimal_action(self, theta: float,
                       current_x: np.ndarray, current_v: np.ndarray,
                       delta_time: float):
        """
        Step the field along the action with the highest long-term value.

        Unlike `greedy_action`, which only asks which action points the
        character closest to the goal on the very next frame, this scores each
        candidate with `R + gamma * V(s')` and so will accept a worse immediate
        heading when it sets up a better turn. That anticipation is the whole
        reason for training a value function.

        :param theta: The character's current heading expressed in the goal
            frame, radians. Unity computes it as
            `-SignedAngle(root.forward, desiredWorldDir, up)`.
        :return: (new_x, new_v), both (1, num_bones+2, 4)
        """
        if self.value_function is None:
            return self.greedy_action(theta, current_x, current_v, delta_time)

        k = self.value_function.k_neighbors
        indices, distances = self.get_knn(self.build_motion_states(current_x, current_v), k=k)
        weights = MotionField.calculate_similarity_weights(distances)

        actions = MotionField.build_action_weights(weights)
        xs, vs = self.compute_new_states(current_x, delta_time, indices, actions,
                                         tug_indices=indices)

        best = int(np.argmax(self.value_rewards(theta, xs, vs)))
        self._record_decision(indices, weights, best)
        return xs[best:best + 1], vs[best:best + 1]

    def greedy_action(self, theta: float,
                      current_x: np.ndarray, current_v: np.ndarray,
                      delta_time: float):
        """
        One-step-greedy fallback: pick whichever action best aligns the heading
        with the goal on the next frame. Used when no value function is loaded.
        """
        indices, distances = self.get_knn(
            self.build_motion_states(current_x, current_v), k=self.k_neighbors)
        weights = MotionField.calculate_similarity_weights(distances)

        actions = MotionField.build_action_weights(weights)
        xs, vs = self.compute_new_states(current_x, delta_time, indices, actions,
                                         tug_indices=indices)

        # Heading, plus the locomotion bonus over each action's convex blend of
        # the neighbourhood. Without the bonus an idle-targeting action wins
        # every tie: standing produces exactly zero yaw, which beats the yaw
        # wiggle of any actual walk cycle once the character is aligned.
        arrival = actions @ self.state_scores[indices]
        rewards = (-np.abs(wrap_angle(theta + root_yaw(xs)))
                   + self.locomotion_factor * arrival)
        best = int(np.argmax(rewards))
        self._record_decision(indices, weights, best)
        return xs[best:best + 1], vs[best:best + 1]

    def get_next_pose(self, current_x: np.ndarray, current_v: np.ndarray, delta_time):
        """Debug playback: walk the database frame by frame from the nearest match."""
        if self.current_frame is None:
            indices, _ = self.get_knn(self.build_motion_states(current_x, current_v), k=self.k_neighbors)
            self.current_frame = int(indices[0])
        else:
            self.current_frame = (self.current_frame + 1) % self.states_count

        return (self.poses_x[self.current_frame][np.newaxis, ...],
                self.poses_v[self.current_frame][np.newaxis, ...])

    def get_next_pose_from_field(self, current_x: np.ndarray, current_v: np.ndarray, delta_time):
        """Debug: snap to the successor of the nearest database state."""
        indices, _ = self.get_knn(self.build_motion_states(current_x, current_v), k=self.k_neighbors)
        nxt = min(int(indices[0]) + 1, self.states_count - 1)
        return self.poses_x[nxt][np.newaxis, ...], self.poses_v[nxt][np.newaxis, ...]

    def build_motion_states(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Build motion states from packed pose and velocity. Used to construct the motion field.

        Frame-invariant by construction: `fk_root_space` forces joint 0, the
        character frame, to the origin with identity rotation, so where the
        character is in the world never influences which states it matches.

        Root-translation-invariant too: both FK passes use the root bone's
        rest-pose offset instead of the pose's own, following the reference
        metric. A live root offset is a rigid translation of every joint, so
        it would enter all rows at once and drown the per-joint shape signal
        this metric exists to compare. Changing the metric here means retraining
        every field by hand: Unity's staleness check watches config fields, and
        an edit in this method moves none of them.

        :param x: Packed poses, shape (..., bone_count + 2, 4)
        :param v: Packed per-second velocities, shape (..., bone_count + 2, 4)
        :return: (..., bone_count * 2, 3)
        """

        current_pose = Pose.from_array(x)  # from_array copies, so this is writable

        # Joint 1 is the rig's root bone; joint 0 is the character frame, which
        # carries no rest offset of its own. The exact offset is irrelevant --
        # any constant cancels when two states are subtracted -- it only has to
        # be the SAME for every state.
        rest_hips = list(self.skeleton)[1].default_local_position
        current_pose.hipPos[...] = rest_hips

        p_a, _ = self.skeleton.fk_root_space(current_pose)

        next_pose = current_pose.add(PoseDelta.from_array(v).scaled(self.frame_time))
        next_pose.hipPos[...] = rest_hips

        p_b, _ = self.skeleton.fk_root_space(next_pose)

        velocity = (p_b - p_a) / self.frame_time  # m/s

        position_scale = self.pos_weight
        velocity_scale = self.vel_weight
        if self.bone_weights is not None:
            # (bone_count, 1) broadcasts across xyz and across the batch axes.
            position_scale = self.pos_weight * self.bone_weights[:, 0:1]
            velocity_scale = self.vel_weight * self.bone_weights[:, 1:2]

        return np.concatenate(
            [position_scale * p_a, velocity_scale * velocity],
            axis=-2).astype(np.float32)  # (batch, bone_count*2, 3)

    # ------------------------------------------------------------------ #
    # Debug introspection (consumed by MotionFieldVisualizer in Unity)
    # ------------------------------------------------------------------ #

    def _record_decision(self, indices, weights, best: int) -> None:
        """Remember the neighbourhood and the action the policy just chose."""
        self.last_indices = np.ravel(indices)
        self.last_weights = np.ravel(weights)
        self.last_action = int(best)

    def get_debug_arrays(self):
        """
        The last policy decision, as flat lists for PythonNET marshalling.

        `chosen_slot` indexes into the returned neighbour list, so the pose the
        tug pulled toward is `indices[chosen_slot]` -- that is exactly the value
        handed to `compute_new_states(tug_indices=...)`, not a re-derivation of
        it, so the Unity highlight cannot drift out of sync with the policy.

        :return: (neighbour_indices, similarity_weights, chosen_slot).
            ([], [], -1) before the first `optimal_action`/`greedy_action` call
            -- the playback and nearest modes never populate it.
        """
        if self.last_indices is None or self.last_action is None:
            return [], [], -1

        return (np.asarray(self.last_indices, dtype=np.int32).tolist(),
                np.asarray(self.last_weights, dtype=np.float32).tolist(),
                self.last_action)
