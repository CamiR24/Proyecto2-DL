"""
train.py
----------------
Funciones reutilizables para entrenar agentes de la familia DQN.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class ConfigDQN:
    #entorno
    nombre_entorno: str = "ALE/SpaceInvaders-v5"
    n_acciones: int = 6
    seed: int = 42

    #entrenamiento
    total_pasos: int = 500_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    frecuencia_entrenamiento: int = 4
    inicio_entrenamiento: int = 20_000
    frecuencia_actualizacion_target: int = 10_000
    gradient_clip: float = 10.0

    #replay buffer
    capacidad_buffer: int = 50_000

    #exploración epsilon-greedy
    epsilon_inicial: float = 1.0
    epsilon_final: float = 0.1
    pasos_decay_epsilon: int = 250_000

    #preprocesamiento
    frame_skip: int = 4
    screen_size: int = 84
    stack_size: int = 4
    noop_max: int = 30
    terminal_on_life_loss: bool = True
    clip_reward: bool = True

    def como_diccionario(self):
        return asdict(self)


def calcular_epsilon(paso, config):
    """
    Reduce epsilon linealmente desde epsilon_inicial hasta epsilon_final.
    """

    proporcion = min(
        paso / config.pasos_decay_epsilon,
        1.0,
    )

    epsilon = (
        config.epsilon_inicial
        + proporcion
        * (config.epsilon_final - config.epsilon_inicial)
    )

    return epsilon


def seleccionar_accion(
    modelo,
    observacion,
    epsilon,
    n_acciones,
    device,
    rng,
):
    """
    Selecciona una acción mediante una política epsilon-greedy.
    """

    if rng.random() < epsilon:
        return int(rng.integers(n_acciones))

    observacion_tensor = torch.as_tensor(
        np.asarray(observacion),
        dtype=torch.uint8,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        valores_q = modelo(observacion_tensor)
        accion = valores_q.argmax(dim=1).item()

    return accion

def actualizar_modelo(
    modelo_online,
    modelo_target,
    replay_buffer,
    optimizador,
    config,
    device,
    usar_double_dqn=False,
):
    """
    Realiza una actualización del modelo utilizando un batch del
    replay buffer.

    Si usar_double_dqn=False, utiliza el target del DQN vanilla.
    Si usar_double_dqn=True, separa la selección y evaluación de
    la siguiente acción.
    """

    (
        observaciones,
        acciones,
        recompensas,
        siguientes_observaciones,
        finalizados,
    ) = replay_buffer.muestrear(
        batch_size=config.batch_size,
        device=device,
    )

    #Q(s, a) para las acciones que realmente se ejecutaron
    valores_q = modelo_online(observaciones)

    valores_q_acciones = valores_q.gather(
        dim=1,
        index=acciones.unsqueeze(1),
    ).squeeze(1)

    #el target se calcula sin construir gradientes
    with torch.no_grad():
        if usar_double_dqn:
            #la red online selecciona la mejor acción
            siguientes_acciones = (
                modelo_online(siguientes_observaciones)
                .argmax(dim=1, keepdim=True)
            )

            #la red target evalúa la acción seleccionada
            siguientes_valores_q = (
                modelo_target(siguientes_observaciones)
                .gather(1, siguientes_acciones)
                .squeeze(1)
            )
        else:
            #DQN vanilla: la red target selecciona y evalúa
            siguientes_valores_q = (
                modelo_target(siguientes_observaciones)
                .max(dim=1)
                .values
            )

        no_finalizados = (~finalizados).float()

        targets = (
            recompensas
            + config.gamma
            * no_finalizados
            * siguientes_valores_q
        )

    #huber loss: menos sensible a errores extremos que MSE
    perdida = torch.nn.functional.smooth_l1_loss(
        valores_q_acciones,
        targets,
    )

    optimizador.zero_grad()
    perdida.backward()

    torch.nn.utils.clip_grad_norm_(
        modelo_online.parameters(),
        max_norm=config.gradient_clip,
    )

    optimizador.step()

    return {
        "loss": perdida.item(),
        "q_promedio": valores_q_acciones.detach().mean().item(),
        "target_promedio": targets.mean().item(),
    }