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

class DuelingDQN(nn.Module):
    """
    Variante Dueling de la arquitectura DQN.

    En vez de estimar Q(s, a) directamente, separa la red en dos flujos
    después del extractor convolucional:

    - flujo_valor: estima V(s), qué tan bueno es el estado en sí,
      independientemente de la acción que se tome.
    - flujo_ventaja: estima A(s, a), cuánto mejor es cada
      acción respecto al resto de acciones disponibles en ese estado.

    Los dos flujos se combinan como:

        Q(s, a) = V(s) + (A(s, a) - promedio_a'(A(s, a')))

    Se resta el promedio de la ventaja porque, sin esa resta, el problema 
    no es identificable: V y A podrían desplazarse por una constante 
    arbitraria entre sí y el entrenamiento se vuelve inestable. Restar
    el promedio fija ese grado de libertad.

    El flujo de valor puede aprender qué estados son buenos o malos en general 
    sin necesidad de que cada acción individual pase por esa misma señal.

    Entrada/salida: mismas formas que DQN, (batch_size, 4, 84, 84) ->
    (batch_size, n_acciones).
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

            nn.Flatten(),
        )

        self.flujo_valor = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.flujo_ventaja = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_acciones),
        )

    def forward(self, observaciones):
        observaciones = observaciones.float() / 255.0

        caracteristicas = self.extractor_caracteristicas(observaciones)

        valor = self.flujo_valor(caracteristicas)          
        ventaja = self.flujo_ventaja(caracteristicas)       

        valores_q = valor + (
            ventaja - ventaja.mean(dim=1, keepdim=True)
        )

        return valores_q