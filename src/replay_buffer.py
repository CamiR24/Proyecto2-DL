"""
replay_buffer.py
----------------
Buffer de repetición de experiencias para agentes DQN.
"""

import numpy as np
import torch

class SumTree:
    """
    Árbol binario que permite almacenar prioridades y realizar
    muestreo proporcional en tiempo O(log N).

    Las hojas contienen las prioridades de las experiencias y los
    nodos internos contienen la suma de las prioridades de sus hijos.
    """

    def __init__(self, capacidad):
        if capacidad <= 0:
            raise ValueError(
                "La capacidad del SumTree debe ser positiva."
            )

        self.capacidad = capacidad

        self.arbol = np.zeros(
            2 * capacidad - 1,
            dtype=np.float32,
        )

    @property
    def prioridad_total(self):
        """Suma de todas las prioridades almacenadas."""
        return float(self.arbol[0])

    def actualizar(self, indice_dato, prioridad):
        """
        Actualiza la prioridad asociada a una posición del buffer.
        """

        if not 0 <= indice_dato < self.capacidad:
            raise IndexError(
                f"Índice fuera del árbol: {indice_dato}"
            )

        indice_arbol = (
            indice_dato + self.capacidad - 1
        )

        cambio = prioridad - self.arbol[indice_arbol]
        self.arbol[indice_arbol] = prioridad

        while indice_arbol != 0:
            indice_arbol = (indice_arbol - 1) // 2
            self.arbol[indice_arbol] += cambio

    def obtener(self, valor):
        """
        Busca la hoja correspondiente a un valor acumulado.

        Parámetros
        ----------
        valor : float
            Valor entre 0 y prioridad_total.

        Retorna
        -------
        indice_dato : int
            Posición correspondiente dentro del replay buffer.
        prioridad : float
            Prioridad almacenada en esa posición.
        """

        total = self.prioridad_total

        if total <= 0:
            raise ValueError(
                "No se puede muestrear un árbol sin prioridades."
            )

        valor = min(
            max(float(valor), 0.0),
            np.nextafter(total, 0.0),
        )

        indice_arbol = 0

        while True:
            hijo_izquierdo = 2 * indice_arbol + 1
            hijo_derecho = hijo_izquierdo + 1

            if hijo_izquierdo >= len(self.arbol):
                break

            if valor <= self.arbol[hijo_izquierdo]:
                indice_arbol = hijo_izquierdo
            else:
                valor -= self.arbol[hijo_izquierdo]
                indice_arbol = hijo_derecho

        indice_dato = (
            indice_arbol - self.capacidad + 1
        )

        prioridad = float(
            self.arbol[indice_arbol]
        )

        return indice_dato, prioridad


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

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer que muestrea experiencias proporcionalmente a su
    prioridad.

    La prioridad de una transición se calcula como:

        p_i = (|error TD_i| + epsilon_per) ** alpha

    alpha = 0 produce muestreo uniforme.
    Valores cercanos a 1 aumentan la priorización.
    """

    def __init__(
        self,
        capacidad,
        forma_observacion=(4, 84, 84),
        seed=42,
        alpha=0.6,
        epsilon_per=1e-6,
    ):
        super().__init__(
            capacidad=capacidad,
            forma_observacion=forma_observacion,
            seed=seed,
        )

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha debe estar entre 0 y 1."
            )

        if epsilon_per <= 0:
            raise ValueError(
                "epsilon_per debe ser positivo."
            )

        self.alpha = alpha
        self.epsilon_per = epsilon_per

        self.arbol_prioridades = SumTree(capacidad)
        self.prioridad_maxima = 1.0

        # Permite que train.py detecte el tipo de buffer 
        self.es_priorizado = True

    def agregar(
        self,
        observacion,
        accion,
        recompensa,
        siguiente_observacion,
        finalizado,
    ):
        indice = self.posicion

        super().agregar(
            observacion=observacion,
            accion=accion,
            recompensa=recompensa,
            siguiente_observacion=siguiente_observacion,
            finalizado=finalizado,
        )

        # Las experiencias nuevas reciben la mayor prioridad vista,
        # para garantizar que sean muestreadas al menos una vez.
        prioridad_ajustada = (
            self.prioridad_maxima ** self.alpha
        )

        self.arbol_prioridades.actualizar(
            indice_dato=indice,
            prioridad=prioridad_ajustada,
        )

    def muestrear(
        self,
        batch_size,
        device,
        beta=0.4,
    ):
        if self.tamano < batch_size:
            raise ValueError(
                f"El buffer contiene {self.tamano} experiencias, "
                f"pero se solicitaron {batch_size}."
            )

        if not 0.0 <= beta <= 1.0:
            raise ValueError(
                "beta debe estar entre 0 y 1."
            )

        prioridad_total = (
            self.arbol_prioridades.prioridad_total
        )

        if prioridad_total <= 0:
            raise ValueError(
                "La prioridad total debe ser positiva."
            )

        indices = np.empty(
            batch_size,
            dtype=np.int64,
        )

        prioridades = np.empty(
            batch_size,
            dtype=np.float32,
        )

        tamano_segmento = (
            prioridad_total / batch_size
        )

        for posicion_batch in range(batch_size):
            limite_inferior = (
                posicion_batch * tamano_segmento
            )

            limite_superior = (
                (posicion_batch + 1) * tamano_segmento
            )

            valor = self.rng.uniform(
                limite_inferior,
                limite_superior,
            )

            indice, prioridad = (
                self.arbol_prioridades.obtener(valor)
            )

            indices[posicion_batch] = indice
            prioridades[posicion_batch] = prioridad

        probabilidades = (
            prioridades / prioridad_total
        )

        # Pesos de importancia:
        # w_i = (N * P(i)) ** (-beta)
        pesos = (
            self.tamano * probabilidades
        ) ** (-beta)

        # Normalizar para que el mayor peso del batch sea 1.
        pesos /= pesos.max()

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

        pesos_tensor = torch.from_numpy(
            pesos.astype(np.float32)
        ).to(device)

        return (
            observaciones,
            acciones,
            recompensas,
            siguientes_observaciones,
            finalizados,
            indices,
            pesos_tensor,
        )

    def actualizar_prioridades(
        self,
        indices,
        errores_td,
    ):
        """
        Actualiza las prioridades después de calcular los errores TD.
        """

        indices = np.asarray(
            indices,
            dtype=np.int64,
        ).reshape(-1)

        errores_td = np.asarray(
            errores_td,
            dtype=np.float32,
        ).reshape(-1)

        if len(indices) != len(errores_td):
            raise ValueError(
                "indices y errores_td deben tener "
                "la misma longitud."
            )

        prioridades_sin_exponente = (
            np.abs(errores_td)
            + self.epsilon_per
        )

        for indice, prioridad in zip(
            indices,
            prioridades_sin_exponente,
        ):
            prioridad = float(prioridad)

            self.arbol_prioridades.actualizar(
                indice_dato=int(indice),
                prioridad=prioridad ** self.alpha,
            )

            self.prioridad_maxima = max(
                self.prioridad_maxima,
                prioridad,
            )

    @property
    def memoria_aproximada_mb(self):
        memoria_base = super().memoria_aproximada_mb

        memoria_arbol = (
            self.arbol_prioridades.arbol.nbytes
            / (1024 ** 2)
        )

        return memoria_base + memoria_arbol