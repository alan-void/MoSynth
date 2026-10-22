"""
The style network: an MLP whose hidden layers are FiLM-modulated by a learned per-style embedding.

One network serves every style. A style is an embedding ``e_s``, and each hidden layer is scaled
and shifted by ``(gamma, beta)`` generated from it (Perez et al., *FiLM*), so a style changes how
the body moves without each style needing its own weights.

The output is a **residual**: per styled bone a 6D rotation added to the identity's 6D, then the
dynamics height offset and the four root-residual values. The output layer starts at zero, so an
untrained network is exactly the identity, and the regulariser pulls ambiguous inputs back there.
"""

from __future__ import annotations

import torch
from torch import nn


class StyleNetwork(nn.Module):
    """
    :param input_size: floats per input vector, ``StyleSpec.input_size``.
    :param output_size: floats per output vector, ``StyleSpec.output_size``.
    :param n_styles: rows of the style embedding.
    :param hidden_units: width of every hidden layer.
    :param layers: number of hidden layers, each FiLM-modulated.
    :param embed_dim: width of a style embedding.
    """

    def __init__(self, input_size: int, output_size: int, n_styles: int,
                 hidden_units: int = 512, layers: int = 3, embed_dim: int = 32):
        super().__init__()
        if layers < 1:
            raise ValueError(f'the style network needs at least one hidden layer, got {layers}')
        self.input_size = input_size
        self.output_size = output_size
        self.n_styles = n_styles
        self.hidden_units = hidden_units
        self.n_layers = layers
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(n_styles, embed_dim)
        widths = [input_size] + [hidden_units] * layers
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.film = nn.ModuleList(nn.Linear(embed_dim, 2 * hidden_units) for _ in range(layers))
        self.output = nn.Linear(hidden_units, output_size)

        # FiLM starts as the identity modulation, and the output at zero: the identity style.
        for film in self.film:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        :param x: (n, input_size) normalised inputs.
        :param style: (n,) long style indices.
        :return: (n, output_size) normalised outputs.
        """
        embedding = self.embedding(style)
        h = x
        for linear, film in zip(self.hidden, self.film):
            gamma, beta = film(embedding).chunk(2, dim=-1)
            h = torch.nn.functional.elu((1.0 + gamma) * linear(h) + beta)
        return self.output(h)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
