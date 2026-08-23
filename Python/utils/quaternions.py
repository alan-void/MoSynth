"""
Quaternion helpers for the motion field's blending.

Everything here uses xyzw component order, matching the packed pose layout the Unity
side writes -- not the wxyz order much of the quaternion literature assumes.
"""

import numpy as np


def blend_quaternions(quaternions, weights, axis=0):
    """
    Blends multiple quaternions using weighted NLERP along a specified axis.

    :param quaternions: A (..., 4) numpy array of quaternions.
    :param weights: A numpy array of weights. Can be 1D or match the shape of quaternions (excluding the last dimension).
    :param axis: The axis along which to blend the quaternions when weights are 1D. Ignored otherwise.
    :return: A blended and normalized numpy array of quaternions with the blended axis collapsed.

    NLERP rather than SLERP because SLERP only handles two rotations, while blending a
    KNN neighbourhood means combining k at once. The cost is a non-constant angular
    rate, which does not matter when the neighbours are already close together.

    Signs are aligned against the first quaternion first: q and -q are the same
    rotation, so without that two near-equal neighbours can cancel instead of reinforce.
    """

    quats = np.asarray(quaternions, dtype=np.float64)

    # Convert negative axes to positive to easily compare with ndim
    if axis < 0:
        axis += quats.ndim

    # Ensure the last dimension is always the 4 quaternion components
    if quats.shape[-1] != 4:
        raise ValueError(f"The last dimension must be 4, but got {quats.shape[-1]}")
    if axis == quats.ndim - 1:
        raise ValueError("Cannot blend along the last axis (which represents quaternion components).")

    w = np.asarray(weights, dtype=np.float64)

    # Apply standard numpy-like weighting conventions (similar to np.average)
    if w.ndim == 1:
        # Standard 1: 1D weights matching the length of the blend axis
        if w.shape[0] != quats.shape[axis]:
            raise ValueError(
                f"1D weights (length {w.shape[0]}) must match the size of quaternions along axis {axis} (size {quats.shape[axis]}).")

        # Reshape to broadcast: e.g., if quats is (2, 3, 4) and axis=1, w becomes (1, 3, 1)
        w_shape = [1] * quats.ndim
        w_shape[axis] = w.shape[0]
        w = w.reshape(w_shape)

    elif w.shape == quats.shape[:-1]:
        # Standard 2: N-D weights matching the batch shape of the quaternions
        # (e.g., individual weight for every single quaternion in the ND grid)
        w = np.expand_dims(w, axis=-1)

    else:
        raise ValueError(f"Weights shape {w.shape} is incompatible with quaternions shape {quats.shape}.")

    # Normalize weights along the blend axis so they sum to 1
    w = w / np.sum(w, axis=axis, keepdims=True)

    # Extract the first quaternion along the blend axis to use as the baseline.
    # Passing [0] instead of 0 preserves the dimension as size 1 for broadcasting.
    base_q = np.take(quats, [0], axis=axis)

    # Compute dot products along the quaternion component axis (-1)
    # keepdims=True keeps the shape as (..., 1) to easily multiply back with (..., 4)
    dot_products = np.sum(quats * base_q, axis=-1, keepdims=True)

    # Create the sign mask (-1 for negative dot products, 1 otherwise)
    signs = np.where(dot_products < 0, -1.0, 1.0)

    # Apply weights and signs, then sum along the requested blend axis
    blended_q = np.sum(quats * signs * w, axis=axis)

    # Calculate the L2 norm along the new last axis
    norms = np.linalg.norm(blended_q, axis=-1, keepdims=True)

    # Normalize, avoiding division by zero
    return blended_q / np.maximum(norms, 1e-12)
