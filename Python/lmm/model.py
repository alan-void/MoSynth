"""
The networks of Learned Motion Matching (Holden et al. 2020).

All of them are plain MLPs, and that is a constraint: nothing here may use an operation that cannot
be exported to ONNX for a Unity-native inference backend. Every leaf module is a
:class:`torch.nn.Linear` or an activation, which ``test_lmm_model`` asserts.

* **Compressor** ``[Y Q] -> Z``, 516 wide and three hidden layers deep, with ELU.
* **Decompressor** ``[X Z] -> Y``, 512 wide and **one** hidden layer deep, with ReLU.
* **Stepper** ``[X Z] -> d/dt [X Z]``, 512 wide and two hidden layers deep, with ReLU.
* **Projector** ``X -> [X Z]``, 512 wide and **four** hidden layers deep, with ReLU.

Both the depth asymmetry and the mixed activations are the paper's (Table 1): decoding is a smooth
map that needs width, while the projector approximates a piecewise-constant nearest-neighbour lookup
and needs depth most.
"""

from __future__ import annotations

import torch
from torch import nn

# Re-exported so PFNN and LMM agree on what 'auto' means.
from pfnn.model import resolve_device

# Widths read off the reference implementation's ONNX graphs; the paper's Table 1 states the same
# depths and rounds the widths to 512.
COMPRESSOR_HIDDEN_UNITS = 516
COMPRESSOR_HIDDEN_LAYERS = 3
DECOMPRESSOR_HIDDEN_UNITS = 512
DECOMPRESSOR_HIDDEN_LAYERS = 1
STEPPER_HIDDEN_UNITS = 512
STEPPER_HIDDEN_LAYERS = 2
PROJECTOR_HIDDEN_UNITS = 512
PROJECTOR_HIDDEN_LAYERS = 4

__all__ = ['Mlp', 'Compressor', 'Decompressor', 'Stepper', 'Projector', 'resolve_device',
           'COMPRESSOR_HIDDEN_UNITS', 'COMPRESSOR_HIDDEN_LAYERS',
           'DECOMPRESSOR_HIDDEN_UNITS', 'DECOMPRESSOR_HIDDEN_LAYERS',
           'STEPPER_HIDDEN_UNITS', 'STEPPER_HIDDEN_LAYERS',
           'PROJECTOR_HIDDEN_UNITS', 'PROJECTOR_HIDDEN_LAYERS']


class Mlp(nn.Module):
    """
    A stack of linear layers with one activation between them, and nothing else.

    :param input_size: floats per input vector.
    :param output_size: floats per output vector.
    :param hidden_units: width of every hidden layer.
    :param hidden_layers: how many hidden layers; 0 makes this a single linear map.
    :param activation: the module type placed after each hidden layer.
    """

    def __init__(self, input_size: int, output_size: int, hidden_units: int, hidden_layers: int,
                 activation=nn.ELU):
        super().__init__()
        if hidden_layers < 0:
            raise ValueError(f'hidden_layers must be >= 0, got {hidden_layers}')

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers

        modules, width = [], input_size
        for _ in range(hidden_layers):
            modules.append(nn.Linear(width, hidden_units))
            modules.append(activation())
            width = hidden_units
        modules.append(nn.Linear(width, output_size))

        self.layers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def linear_layers(self) -> list:
        """Every :class:`torch.nn.Linear`, in forward order. The parameters a checkpoint stores."""
        return [module for module in self.layers if isinstance(module, nn.Linear)]

    def weights(self) -> list:
        """One ``(out, in)`` array per linear layer, detached onto the CPU."""
        return [layer.weight.detach().cpu().numpy() for layer in self.linear_layers]

    def biases(self) -> list:
        """One ``(out,)`` array per linear layer, detached onto the CPU."""
        return [layer.bias.detach().cpu().numpy() for layer in self.linear_layers]

    def load_parameters(self, weights, biases) -> None:
        """
        Copy stored parameters back in, checking the shapes rather than trusting them.

        A checkpoint whose widths no longer match this code is refused with the layer named.
        """
        layers = self.linear_layers
        if len(weights) != len(layers) or len(biases) != len(layers):
            raise ValueError(f'checkpoint holds {len(weights)} layers where this model has '
                             f'{len(layers)}; it has to be retrained')

        with torch.no_grad():
            for i, (layer, weight, bias) in enumerate(zip(layers, weights, biases)):
                if tuple(weight.shape) != tuple(layer.weight.shape):
                    raise ValueError(f'checkpoint layer {i} is {tuple(weight.shape)} where this '
                                     f'model reads {tuple(layer.weight.shape)}; it has to be '
                                     'retrained')
                layer.weight.copy_(torch.as_tensor(weight, dtype=layer.weight.dtype))
                layer.bias.copy_(torch.as_tensor(bias, dtype=layer.bias.dtype))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Compressor(Mlp):
    """
    Encodes a pose, in both the spaces it is measured in, into a latent.

    Never run at inference: its latents are baked. It is stored so a bake can be reproduced and
    the latent diagnostics re-run.
    """

    def __init__(self, pose_size: int, character_size: int, latent_size: int,
                 hidden_units: int = COMPRESSOR_HIDDEN_UNITS,
                 hidden_layers: int = COMPRESSOR_HIDDEN_LAYERS):
        super().__init__(pose_size + character_size, latent_size, hidden_units, hidden_layers,
                         activation=torch.nn.ELU)
        self.pose_size = pose_size
        self.character_size = character_size
        self.latent_size = latent_size

    def encode(self, pose: torch.Tensor, character: torch.Tensor) -> torch.Tensor:
        """
        :param pose: (n, pose_size) normalised ``Y``.
        :param character: (n, character_size) the same frames' normalised ``Q``.
        """
        return self(torch.cat([pose, character], dim=1))


class Decompressor(Mlp):
    """
    Reconstructs a pose from a matching feature vector and a latent.

    Replaces reading a pose out of the database, and the only network the decompressor-only mode
    needs at runtime.
    """

    def __init__(self, feature_size: int, latent_size: int, pose_size: int,
                 hidden_units: int = DECOMPRESSOR_HIDDEN_UNITS,
                 hidden_layers: int = DECOMPRESSOR_HIDDEN_LAYERS):
        super().__init__(feature_size + latent_size, pose_size, hidden_units, hidden_layers,
                         activation=torch.nn.ReLU)
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.pose_size = pose_size

    def decode(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        :param features: (n, feature_size) matching feature vectors, as Unity normalised them.
        :param latent: (n, latent_size) latents, as the compressor produced them.
        """
        return self(torch.cat([features, latent], dim=1))


class Stepper(Mlp):
    """
    Advances the state ``[X Z]`` a frame at a time, so the database need not be played.

    Replaces walking the ``.mmpose`` between searches. It takes **exactly the vector the
    decompressor takes**, so the stage carries one copy of ``(X, Z)`` for both networks.

    It answers with a **rate per second**, not the next state: the stage integrates
    ``x += rate * dt``, so synthesis can run at a rate the database was not sampled at. The output
    is normalised per element by the rate statistics in the checkpoint.
    """

    def __init__(self, feature_size: int, latent_size: int,
                 hidden_units: int = STEPPER_HIDDEN_UNITS,
                 hidden_layers: int = STEPPER_HIDDEN_LAYERS):
        state_size = feature_size + latent_size
        super().__init__(state_size, state_size, hidden_units, hidden_layers,
                         activation=torch.nn.ReLU)
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.state_size = state_size

    def rate(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        :param features: (n, feature_size) the query vector the character is holding.
        :param latent: (n, latent_size) the latent beside it.
        :return: (n, feature_size + latent_size) the **normalised** per-second rate of change of
            the two concatenated. Denormalising it is the caller's job, because the statistics live
            in the checkpoint rather than in the network.
        """
        return self(torch.cat([features, latent], dim=1))


class Projector(Mlp):
    """
    Answers a query with a state the database could have held: ``X -> [X Z]``.

    Replaces the search itself: it returns the nearest state the database contains -- its feature
    vector and latent -- without any frame being looked up. Deepest of the four, because it
    approximates a piecewise-constant nearest-neighbour lookup.

    Its answer is **normalised per element** against the checkpoint's existing state statistics,
    ``x_mean``/``x_std`` and ``z_mean``/``z_std``.
    """

    def __init__(self, feature_size: int, latent_size: int,
                 hidden_units: int = PROJECTOR_HIDDEN_UNITS,
                 hidden_layers: int = PROJECTOR_HIDDEN_LAYERS):
        super().__init__(feature_size, feature_size + latent_size, hidden_units, hidden_layers,
                         activation=torch.nn.ReLU)
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.state_size = feature_size + latent_size

    def project(self, query: torch.Tensor):
        """
        :param query: (n, feature_size) what the controller is asking for, in the database's units.
        :return: ``(features, latent)``, both **normalised**. Denormalising them is the caller's
            job: the statistics live in the checkpoint rather than in the network.
        """
        state = self(query)
        return state[:, :self.feature_size], state[:, self.feature_size:]
