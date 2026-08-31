"""
The phase-functioned network itself (Holden et al. 2017).

A PFNN is an ordinary three-layer MLP whose weights are not constants but a function of the gait
phase. That function is a cubic Catmull-Rom spline through four control points, wrapped so phase 0
and phase 2*pi give the same network -- which is what lets one small model hold a whole gait cycle
without the layers having to encode "where in the step am I" themselves.

The one implementation choice worth stating: **the outputs are blended, not the weights.** Because
the spline is a linear combination of its control points, and a layer is linear in its weights,

    (sum_k w_k * A_k) @ x + sum_k w_k * b_k  ==  sum_k w_k * (A_k @ x + b_k)

The right-hand side is four dense matmuls and a weighted sum. The left-hand side has to build a
distinct weight matrix for every item in the batch first, which for a 512-unit layer is millions of
floats per sample and is what makes naive PFNN implementations slow. ``test_pfnn_model`` asserts the
two agree.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# The spline's control points. Four is the paper's choice and is not a free parameter here: the
# cyclic Catmull-Rom below indexes exactly four, and the checkpoint format stores that shape.
PHASE_CONTROL_POINTS = 4

TAU = 2.0 * math.pi


def catmull_rom_weights(phase: torch.Tensor) -> torch.Tensor:
    """
    How much each of the four control points contributes, at a given phase.

    Returned as weights rather than as an interpolated value so the same numbers can be applied to
    every layer at once, and so the blend can be moved to the far side of the matmul.

    :param phase: (n,) gait phase in radians. Wrapped, so any real value is accepted.
    :return: (n, 4) float weights, one row per sample, summing to 1.
    """
    scaled = (phase % TAU) / TAU * PHASE_CONTROL_POINTS
    index = torch.floor(scaled)
    w = (scaled - index).unsqueeze(1)

    # The cubic in Catmull-Rom form, kept as coefficients on the four surrounding control points.
    coefficients = torch.cat([
        -0.5 * w ** 3 + w ** 2 - 0.5 * w,
        1.5 * w ** 3 - 2.5 * w ** 2 + 1.0,
        -1.5 * w ** 3 + 2.0 * w ** 2 + 0.5 * w,
        0.5 * w ** 3 - 0.5 * w ** 2,
    ], dim=1)

    # Scatter them onto the control points they belong to. The wrap is what makes the phase cyclic:
    # the network at phase 2*pi is the same object as the network at phase 0, not merely close to it.
    base = index.long().unsqueeze(1)
    offsets = torch.tensor([-1, 0, 1, 2], device=phase.device).unsqueeze(0)
    targets = (base + offsets) % PHASE_CONTROL_POINTS

    weights = torch.zeros(phase.shape[0], PHASE_CONTROL_POINTS,
                          dtype=coefficients.dtype, device=phase.device)
    return weights.scatter_add_(1, targets, coefficients)


class PhaseFunctionedLayer(nn.Module):
    """One linear layer whose weights are a phase function. See the module docstring for the blend."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # The paper's initialisation: uniform over +/- sqrt(6 / (in + out)), applied per control
        # point so the network starts out roughly phase-independent in scale but not in value.
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.weights = nn.Parameter(
            torch.empty(PHASE_CONTROL_POINTS, out_features, in_features).uniform_(-bound, bound))
        self.biases = nn.Parameter(torch.zeros(PHASE_CONTROL_POINTS, out_features))

    def forward(self, x: torch.Tensor, phase_weights: torch.Tensor) -> torch.Tensor:
        flat = self.weights.reshape(PHASE_CONTROL_POINTS * self.out_features, self.in_features)
        per_control_point = (x @ flat.t()).view(-1, PHASE_CONTROL_POINTS, self.out_features)
        per_control_point = per_control_point + self.biases
        return torch.einsum('nk,nko->no', phase_weights, per_control_point)

    def blended_weights(self, phase_weights: torch.Tensor):
        """
        The explicit per-sample weight matrix and bias, for one sample's phase.

        Only the test needs this -- it is the slow formulation :meth:`forward` avoids -- but having
        it named makes the equivalence checkable rather than merely asserted in a comment.
        """
        weight = torch.einsum('k,koi->oi', phase_weights, self.weights)
        bias = torch.einsum('k,ko->o', phase_weights, self.biases)
        return weight, bias


class PhaseFunctionedNetwork(nn.Module):
    """
    Three phase-functioned layers with ELU activations and dropout between them.

    :param input_size: floats per input vector, from ``PfnnSpec.input_size``.
    :param output_size: floats per output vector, from ``PfnnSpec.output_size``.
    :param hidden_units: width of the two hidden layers.
    :param dropout: probability of **dropping** a unit, PyTorch's convention. The paper quotes 0.7
        as a retention rate, so its setting is 0.3 here.
    """

    def __init__(self, input_size: int, output_size: int, hidden_units: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout

        self.layer0 = PhaseFunctionedLayer(input_size, hidden_units)
        self.layer1 = PhaseFunctionedLayer(hidden_units, hidden_units)
        self.layer2 = PhaseFunctionedLayer(hidden_units, output_size)
        self.dropout = nn.Dropout(dropout)

    @property
    def layers(self):
        return self.layer0, self.layer1, self.layer2

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        :param x: (n, input_size) normalised input vectors.
        :param phase: (n,) gait phase in radians.
        :return: (n, output_size) normalised output vectors.
        """
        phase_weights = catmull_rom_weights(phase).to(x.dtype)

        h = self.layer0(self.dropout(x), phase_weights)
        h = self.dropout(torch.nn.functional.elu(h))
        h = self.layer1(h, phase_weights)
        h = self.dropout(torch.nn.functional.elu(h))
        return self.layer2(h, phase_weights)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def resolve_device(preference: str = 'auto') -> torch.device:
    """
    The device to run on. ``auto`` takes CUDA when it is there.

    Kept here rather than imported from ``MotionField`` so a PFNN run does not drag in the motion
    field's k-NN machinery and its animation loading.
    """
    if preference in (None, '', 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(preference)
