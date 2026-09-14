"""
models.py
----------
Arquitecturas neuronales usadas por los agentes DQN.
"""

import torch
from torch import nn


class DQN(nn.Module):
    """
    Red convolucional que aproxima el valor Q(s, a) de cada acción.

    Entrada esperada
    ----------------
    Tensor con forma:
        (batch_size, 4, 84, 84)

    Salida
    ------
    Tensor con forma:
        (batch_size, n_acciones)
    """

    def __init__(self, n_acciones):
        super().__init__()

        self.extractor_caracteristicas = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )

        self.estimador_q = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_acciones),
        )

    def forward(self, observaciones):
        # Las observaciones llegan como uint8 con valores entre 0 y 255.
        observaciones = observaciones.float() / 255.0

        caracteristicas = self.extractor_caracteristicas(observaciones)
        valores_q = self.estimador_q(caracteristicas)

        return valores_q