"""
evaluation.py
-------------
Funciones para evaluar agentes entrenados con una política greedy.
"""

import numpy as np
import torch

from ale_utils import crear_entorno


def evaluar_modelo(
    modelo,
    config,
    device,
    n_episodios=5,
    seed_base=1000,
    max_steps=10_000,
):
    """
    Evalúa un modelo utilizando acciones greedy y episodios completos.
    """

    env = crear_entorno(
        nombre_entorno=config.nombre_entorno,
        aplicar_preprocesamiento=True,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        grayscale=True,
        stack_size=config.stack_size,
        terminal_on_life_loss=False,
        clip_reward=False,
        noop_max=config.noop_max,
        full_action_space=False,
    )

    estaba_entrenando = modelo.training
    modelo.eval()

    resultados = []

    try:
        for episodio in range(n_episodios):
            seed = seed_base + episodio
            observacion, info = env.reset(seed=seed)

            recompensa_total = 0.0
            pasos = 0
            terminated = False
            truncated = False

            while (
                not (terminated or truncated)
                and pasos < max_steps
            ):
                observacion_tensor = torch.as_tensor(
                    np.asarray(observacion),
                    dtype=torch.uint8,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    valores_q = modelo(observacion_tensor)
                    accion = valores_q.argmax(dim=1).item()

                (
                    observacion,
                    recompensa,
                    terminated,
                    truncated,
                    info,
                ) = env.step(accion)

                recompensa_total += recompensa
                pasos += 1

            resultados.append(
                {
                    "episodio": episodio,
                    "seed": seed,
                    "recompensa_total": recompensa_total,
                    "pasos": pasos,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
    finally:
        env.close()

        if estaba_entrenando:
            modelo.train()

    recompensas = [
        resultado["recompensa_total"]
        for resultado in resultados
    ]

    resumen = {
        "promedio": float(np.mean(recompensas)),
        "mediana": float(np.median(recompensas)),
        "desviacion": float(np.std(recompensas)),
        "minimo": float(np.min(recompensas)),
        "maximo": float(np.max(recompensas)),
    }

    return resultados, resumen