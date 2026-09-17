"""
train.py
----------------
Funciones reutilizables para entrenar agentes de la familia DQN.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch

import csv
import json
import time
from copy import deepcopy
from pathlib import Path

from ale_utils import crear_entorno
from evaluation import evaluar_modelo
from replay_buffer import ReplayBuffer


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
    inicio_entrenamiento: int = 10_000
    frecuencia_actualizacion_target: int = 10_000
    gradient_clip: float = 10.0

    #replay buffer
    capacidad_buffer: int = 20_000

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

    # Evaluación y guardado
    nombre_experimento: str = "v1_dqn_vanilla"
    frecuencia_evaluacion: int = 50_000
    episodios_evaluacion: int = 5
    seed_evaluacion: int = 1_000
    max_steps_episodio: int = 10_000
    frecuencia_log: int = 1_000

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

def _agregar_fila_csv(ruta, fila):
    ruta = Path(ruta)
    archivo_nuevo = not ruta.exists()

    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=fila.keys(),
        )

        if archivo_nuevo:
            escritor.writeheader()

        escritor.writerow(fila)


def _guardar_checkpoint(
    ruta,
    modelo_online,
    modelo_target,
    optimizador,
    paso,
    config,
):
    torch.save(
        {
            "paso": paso,
            "modelo_online_state_dict": modelo_online.state_dict(),
            "modelo_target_state_dict": modelo_target.state_dict(),
            "optimizador_state_dict": optimizador.state_dict(),
            "config": config.como_diccionario(),
        },
        ruta,
    )


def entrenar_dqn(
    config,
    clase_modelo,
    device,
    directorio_modelos="../models",
    directorio_logs="../logs/entrenamientos",
    usar_double_dqn=False,
    ruta_checkpoint_inicial=None,
):
    """
    Entrena un agente DQN y guarda métricas y checkpoints.

    Si ruta_checkpoint_inicial no es None, en vez de empezar con pesos
    aleatorios se cargan los pesos (online, target y optimizador) de
    ese checkpoint, y el conteo de pasos continúa desde el paso en que
    se guardó (en vez de reiniciar en 0). Útil para extender una
    iteración anterior en vez de re-entrenarla desde cero -- por
    ejemplo, si sospechas que el agente todavía no convergió con el
    presupuesto de pasos original.

    Nota: el replay buffer SIEMPRE arranca vacío al reanudar (las
    experiencias anteriores no se guardan en el checkpoint), así que
    los primeros config.inicio_entrenamiento pasos de la continuación
    se dedican de nuevo a llenarlo antes de retomar las actualizaciones
    de la red.
    """
    carpeta_modelo = (
        Path(directorio_modelos)
        / config.nombre_experimento
    )

    carpeta_logs = (
        Path(directorio_logs)
        / config.nombre_experimento
    )

    carpeta_modelo.mkdir(parents=True, exist_ok=True)
    carpeta_logs.mkdir(parents=True, exist_ok=True)

    ruta_log_episodios = carpeta_logs / "episodios.csv"
    ruta_log_updates = carpeta_logs / "actualizaciones.csv"
    ruta_log_evaluaciones = carpeta_logs / "evaluaciones.csv"

    with (carpeta_modelo / "config.json").open(
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            config.como_diccionario(),
            archivo,
            indent=4,
        )

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    rng = np.random.default_rng(config.seed)

    #entorno de entrenamiento
    env = crear_entorno(
        nombre_entorno=config.nombre_entorno,
        aplicar_preprocesamiento=True,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        grayscale=True,
        stack_size=config.stack_size,
        terminal_on_life_loss=config.terminal_on_life_loss,
        clip_reward=False,
        noop_max=config.noop_max,
        full_action_space=False,
    )

    env.action_space.seed(config.seed)
    observacion, info = env.reset(seed=config.seed)

    #modelos y optimizador
    modelo_online = clase_modelo(
        n_acciones=config.n_acciones
    ).to(device)

    optimizador = torch.optim.Adam(
        modelo_online.parameters(),
        lr=config.learning_rate,
    )

    paso_inicial = 0

    if ruta_checkpoint_inicial is not None:
        checkpoint_inicial = torch.load(
            ruta_checkpoint_inicial,
            map_location=device,
            weights_only=True,
        )

        modelo_online.load_state_dict(
            checkpoint_inicial["modelo_online_state_dict"]
        )

        optimizador.load_state_dict(
            checkpoint_inicial["optimizador_state_dict"]
        )

        paso_inicial = checkpoint_inicial["paso"]

        print(
            f"Reanudando entrenamiento desde el paso "
            f"{paso_inicial:,} (checkpoint: {ruta_checkpoint_inicial})"
        )

        if paso_inicial >= config.total_pasos:
            raise ValueError(
                f"config.total_pasos ({config.total_pasos:,}) debe ser "
                f"mayor al paso del checkpoint cargado ({paso_inicial:,}) "
                "para que la continuación tenga pasos nuevos que entrenar."
            )

    modelo_target = deepcopy(modelo_online).to(device)

    if ruta_checkpoint_inicial is not None:
        modelo_target.load_state_dict(
            checkpoint_inicial["modelo_target_state_dict"]
        )

    modelo_target.eval()

    replay_buffer = ReplayBuffer(
        capacidad=config.capacidad_buffer,
        forma_observacion=(
            config.stack_size,
            config.screen_size,
            config.screen_size,
        ),
        seed=config.seed,
    )

    episodio = 0
    pasos_episodio = 0
    recompensa_real_episodio = 0.0
    mejor_promedio_evaluacion = -np.inf
    ultima_actualizacion = None
    tiempo_inicio = time.time()

    modelo_online.train()

    try:
        for paso in range(paso_inicial + 1, config.total_pasos + 1):
            epsilon = calcular_epsilon(paso, config)

            accion = seleccionar_accion(
                modelo=modelo_online,
                observacion=observacion,
                epsilon=epsilon,
                n_acciones=config.n_acciones,
                device=device,
                rng=rng,
            )

            (
                siguiente_observacion,
                recompensa_real,
                terminated,
                truncated,
                info,
            ) = env.step(accion)

            finalizado = terminated or truncated

            if config.clip_reward:
                recompensa_entrenamiento = float(
                    np.sign(recompensa_real)
                )
            else:
                recompensa_entrenamiento = float(
                    recompensa_real
                )

            replay_buffer.agregar(
                observacion=observacion,
                accion=accion,
                recompensa=recompensa_entrenamiento,
                siguiente_observacion=siguiente_observacion,
                finalizado=finalizado,
            )

            recompensa_real_episodio += recompensa_real
            pasos_episodio += 1

            #actualización de la red online
            if (
                paso >= config.inicio_entrenamiento
                and paso % config.frecuencia_entrenamiento == 0
                and len(replay_buffer) >= config.batch_size
            ):
                ultima_actualizacion = actualizar_modelo(
                    modelo_online=modelo_online,
                    modelo_target=modelo_target,
                    replay_buffer=replay_buffer,
                    optimizador=optimizador,
                    config=config,
                    device=device,
                    usar_double_dqn=usar_double_dqn,
                )

            #actualización periódica de la red target
            if (
                paso >= config.inicio_entrenamiento
                and paso
                % config.frecuencia_actualizacion_target
                == 0
            ):
                modelo_target.load_state_dict(
                    modelo_online.state_dict()
                )

            if finalizado:
                _agregar_fila_csv(
                    ruta_log_episodios,
                    {
                        "episodio": episodio,
                        "paso_global": paso,
                        "recompensa_real": recompensa_real_episodio,
                        "pasos": pasos_episodio,
                        "epsilon": epsilon,
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                )

                episodio += 1
                pasos_episodio = 0
                recompensa_real_episodio = 0.0

                # Sin una semilla nueva para permitir que AtariPreprocessing gestione la pérdida de vida.
                observacion, info = env.reset()
            else:
                observacion = siguiente_observacion

            if (
                ultima_actualizacion is not None
                and paso % config.frecuencia_log == 0
            ):
                tiempo_transcurrido = time.time() - tiempo_inicio

                fila_update = {
                    "paso_global": paso,
                    "loss": ultima_actualizacion["loss"],
                    "q_promedio": ultima_actualizacion["q_promedio"],
                    "target_promedio": ultima_actualizacion[
                        "target_promedio"
                    ],
                    "epsilon": epsilon,
                    "tamano_buffer": len(replay_buffer),
                    "tiempo_segundos": tiempo_transcurrido,
                }

                _agregar_fila_csv(
                    ruta_log_updates,
                    fila_update,
                )

                print(
                    f"Paso {paso:,}/{config.total_pasos:,} | "
                    f"episodio={episodio} | "
                    f"epsilon={epsilon:.3f} | "
                    f"loss={ultima_actualizacion['loss']:.4f} | "
                    f"Q={ultima_actualizacion['q_promedio']:.3f}"
                )

            #evaluación greedy
            if paso % config.frecuencia_evaluacion == 0:
                _, resumen = evaluar_modelo(
                    modelo=modelo_online,
                    config=config,
                    device=device,
                    n_episodios=config.episodios_evaluacion,
                    seed_base=config.seed_evaluacion,
                    max_steps=config.max_steps_episodio,
                )

                _agregar_fila_csv(
                    ruta_log_evaluaciones,
                    {
                        "paso_global": paso,
                        **resumen,
                    },
                )

                _guardar_checkpoint(
                    ruta=carpeta_modelo / "ultimo_checkpoint.pt",
                    modelo_online=modelo_online,
                    modelo_target=modelo_target,
                    optimizador=optimizador,
                    paso=paso,
                    config=config,
                )

                if (
                    resumen["promedio"]
                    > mejor_promedio_evaluacion
                ):
                    mejor_promedio_evaluacion = resumen["promedio"]

                    _guardar_checkpoint(
                        ruta=carpeta_modelo / "mejor_modelo.pt",
                        modelo_online=modelo_online,
                        modelo_target=modelo_target,
                        optimizador=optimizador,
                        paso=paso,
                        config=config,
                    )

                print(
                    "\nEVALUACIÓN | "
                    f"paso={paso:,} | "
                    f"promedio={resumen['promedio']:.2f} | "
                    f"mediana={resumen['mediana']:.2f} | "
                    f"máximo={resumen['maximo']:.2f}\n"
                )

    finally:
        env.close()

    _guardar_checkpoint(
        ruta=carpeta_modelo / "checkpoint_final.pt",
        modelo_online=modelo_online,
        modelo_target=modelo_target,
        optimizador=optimizador,
        paso=config.total_pasos,
        config=config,
    )

    return {
        "modelo_online": modelo_online,
        "modelo_target": modelo_target,
        "mejor_promedio_evaluacion": mejor_promedio_evaluacion,
        "carpeta_modelo": carpeta_modelo,
        "carpeta_logs": carpeta_logs,
    }