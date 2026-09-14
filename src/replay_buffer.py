"""
replay_buffer.py
----------------
Buffer de repetición de experiencias para agentes DQN.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacidad,
        forma_observacion=(4, 84, 84),
        seed=42,
    ):
        self.capacidad = capacidad
        self.forma_observacion = forma_observacion

        self.observaciones = np.empty(
            (capacidad, *forma_observacion),
            dtype=np.uint8,
        )

        self.siguientes_observaciones = np.empty(
            (capacidad, *forma_observacion),
            dtype=np.uint8,
        )

        self.acciones = np.empty(capacidad, dtype=np.int64)
        self.recompensas = np.empty(capacidad, dtype=np.float32)
        self.finalizados = np.empty(capacidad, dtype=np.bool_)

        self.posicion = 0
        self.tamano = 0

        #generador independiente y reproducible para el muestreo
        self.rng = np.random.default_rng(seed)

    def agregar(
        self,
        observacion,
        accion,
        recompensa,
        siguiente_observacion,
        finalizado,
    ):
        indice = self.posicion

        self.observaciones[indice] = np.asarray(
            observacion,
            dtype=np.uint8,
        )

        self.siguientes_observaciones[indice] = np.asarray(
            siguiente_observacion,
            dtype=np.uint8,
        )

        self.acciones[indice] = accion
        self.recompensas[indice] = recompensa
        self.finalizados[indice] = finalizado

        #al llenarse, vuelve al inicio y reemplaza experiencias antiguas
        self.posicion = (self.posicion + 1) % self.capacidad
        self.tamano = min(self.tamano + 1, self.capacidad)

    def muestrear(self, batch_size, device):
        if self.tamano < batch_size:
            raise ValueError(
                f"El buffer contiene {self.tamano} experiencias, "
                f"pero se solicitaron {batch_size}."
            )

        indices = self.rng.choice(
            self.tamano,
            size=batch_size,
            replace=False,
        )

        observaciones = torch.from_numpy(
            self.observaciones[indices]
        ).to(device)

        acciones = torch.from_numpy(
            self.acciones[indices]
        ).to(device)

        recompensas = torch.from_numpy(
            self.recompensas[indices]
        ).to(device)

        siguientes_observaciones = torch.from_numpy(
            self.siguientes_observaciones[indices]
        ).to(device)

        finalizados = torch.from_numpy(
            self.finalizados[indices]
        ).to(device)

        return (
            observaciones,
            acciones,
            recompensas,
            siguientes_observaciones,
            finalizados,
        )

    def __len__(self):
        return self.tamano

    @property
    def memoria_aproximada_mb(self):
        memoria_bytes = (
            self.observaciones.nbytes
            + self.siguientes_observaciones.nbytes
            + self.acciones.nbytes
            + self.recompensas.nbytes
            + self.finalizados.nbytes
        )

        return memoria_bytes / (1024 ** 2)