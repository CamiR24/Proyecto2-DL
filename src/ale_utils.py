"""
ale_utils.py
------------
Funciones reutilizables para crear entornos de Gymnasium/ALE,
ejecutar agentes y grabar video de las partidas.

Extendido a partir de la versión del Lab 5: crear_entorno ahora puede
aplicar el pipeline de preprocesamiento estándar de DQN (Mnih et al.,
2015) - grayscale, resize a 84x84, frame skip, frame stacking - sin
perder la capacidad de RecordVideo de grabar el video en resolución
completa y a la velocidad real del juego.
"""

import numpy as np
import gymnasium as gym
import ale_py

#registrar los entornos de ALE
gym.register_envs(ale_py)


def crear_entorno(nombre_entorno, video_folder=None, episode_trigger=None,
                   name_prefix="video", render_mode=None,
                   aplicar_preprocesamiento=False, frame_skip=4,
                   screen_size=84, grayscale=True, stack_size=4,
                   terminal_on_life_loss=False, clip_reward=False,
                   noop_max=30, **kwargs):
    """
    Crea y retorna un entorno de Gymnasium, envuelto opcionalmente con
    RecordVideo para grabar episodios y, opcionalmente, con el pipeline
    de preprocesamiento estándar de Atari/DQN.

    Funciona tanto para entornos de Atari como para cualquier otro entorno
    de Gymnasium: si aplicar_preprocesamiento=False (default), el
    comportamiento es idéntico al del Lab 5.

    Parámetros
    ----------
    nombre_entorno : str
        Id del entorno a crear
    video_folder : str, opcional
        Carpeta donde se guardarán los videos. Si es None, no se graba video.
    episode_trigger : callable, opcional
        Función que recibe el índice del episodio y retorna True/False
        indicando si ese episodio debe grabarse.
    name_prefix : str
        Prefijo para los archivos de video generados.
    render_mode : str, opcional
        Modo de renderizado. Si se va a grabar video, se fuerza a
        "rgb_array" (requerido por RecordVideo) en caso de no especificarse.
    aplicar_preprocesamiento : bool
        Si es True, envuelve el entorno con AtariPreprocessing (+ opcional
        TransformReward y FrameStackObservation) DESPUÉS de RecordVideo,
        de forma que el video se grabe en resolución/velocidad completa
        mientras el agente recibe observaciones preprocesadas.
    frame_skip : int
        Cuántos frames físicos se repite cada acción (default 4, igual
        que el paper de DQN). El entorno base se crea con frameskip=1
        para que sea AtariPreprocessing quien controle el skip (evita
        duplicarlo).
    screen_size : int
        Tamaño (screen_size x screen_size) al que se reescala cada frame.
    grayscale : bool
        Si True, convierte la observación a escala de grises.
    stack_size : int
        Número de frames apilados en la observación (default 4). Usa 1
        para desactivar el stacking.
    terminal_on_life_loss : bool
        Si True, cada vida perdida termina el episodio para el agente.
        Recomendado en True solo durante ENTRENAMIENTO; en evaluación y
        en la grabación del video final debe quedar en False, porque el
        puntaje de competencia es el de la partida completa (5 vidas).
    clip_reward : bool
        Si True, recorta la recompensa a {-1, 0, 1} (estabiliza el
        entrenamiento). En evaluación/video déjalo en False para reportar
        el puntaje real del juego.
    noop_max : int
        Máximo de acciones "no-op" aleatorias al inicio de cada episodio
        (evita que el agente memorice la secuencia inicial exacta).
    **kwargs :
        Argumentos adicionales que se pasan directamente a gym.make
        (por ejemplo full_action_space=True/False).

    Retorna
    -------
    env : gymnasium.Env
        El entorno creado
    """
    if video_folder is not None:
        #recordVideo necesita observaciones tipo imagen (rgb_array).
        render_mode = render_mode or "rgb_array"

    if aplicar_preprocesamiento:
        #frameskip=1 en gym.make: el skip real lo hace AtariPreprocessing.
        kwargs["frameskip"] = 1

    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)

    if video_folder is not None:
        if episode_trigger is None:
            episode_trigger = lambda episodio: True

        #RecordVideo se coloca ANTES del preprocesamiento para que el
        #video quede en resolución/velocidad completa, y el agente
        #reciba la observación preprocesada.
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix=name_prefix,
        )

    if aplicar_preprocesamiento:
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=grayscale,
            grayscale_newaxis=False,
            scale_obs=False,  #normaliza dentro de la red
        )

        if clip_reward:
            env = gym.wrappers.TransformReward(env, lambda r: float(np.sign(r)))

        if stack_size > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size)

    return env


def agente_aleatorio(observation, env):
    """
    Agente base que no requiere entrenamiento: muestrea una acción
    aleatoria del espacio de acciones del entorno.

    Parámetros
    ----------
    observation : np.ndarray
        Observación actual del entorno
    env : gymnasium.Env
        El entorno, del cual se toma action_space para muestrear

    Retorna
    -------
    action : int
        Acción muestreada aleatoriamente
    """
    return env.action_space.sample()


class AgenteReglaSimple:
    """
    Agente heurístico simple para Space Invaders: dispara constantemente
    mientras barre de un lado a otro de la pantalla, para cubrir la mayor
    cantidad de columnas posible sin ningún tipo de aprendizaje ni
    percepción del estado del juego.

    Pensado como un SEGUNDO baseline, más informativo que el agente
    aleatorio, para tener dos referencias distintas contra las cuales
    comparar cada iteración de RL.

    Es un objeto porque necesita recordar en qué dirección va y cuántos pasos 
    lleva.
    """

    def __init__(self, pasos_por_tramo=25):
        self.pasos_por_tramo = pasos_por_tramo
        self.contador = 0
        self.direccion = "RIGHT"
        self._acciones = None  #se resuelve la primera vez que se llama

    def _resolver_acciones(self, env):
        #se resuelve dinámicamente, por si el conjunto de acciones cambia según full_action_space
        significados = env.unwrapped.get_action_meanings()
        self._acciones = {
            "RIGHTFIRE": significados.index("RIGHTFIRE") if "RIGHTFIRE" in significados
                         else significados.index("RIGHT"),
            "LEFTFIRE": significados.index("LEFTFIRE") if "LEFTFIRE" in significados
                        else significados.index("LEFT"),
        }

    def __call__(self, observation, env):
        if self._acciones is None:
            self._resolver_acciones(env)

        if self.contador >= self.pasos_por_tramo:
            self.contador = 0
            self.direccion = "LEFT" if self.direccion == "RIGHT" else "RIGHT"

        self.contador += 1
        return self._acciones["RIGHTFIRE"] if self.direccion == "RIGHT" else self._acciones["LEFTFIRE"]

    def reset(self):
        """Reinicia el estado interno. Se llama automáticamente entre
        episodios en correr_episodios/generar_video_agente si el agente
        tiene este método."""
        self.contador = 0
        self.direccion = "RIGHT"


def ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None):
    """
    Ejecuta un episodio completo en el entorno usando la función de agente
    dada, hasta que termine, se trunque o se alcance max_steps.

    Parámetros
    ----------
    env : gymnasium.Env
        Entorno ya creado
    funcion_agente : callable
        Función con firma funcion_agente(observation, env) -> action
    max_steps : int
        Número máximo de pasos a ejecutar

    Retorna
    -------
    resultado : dict
        {
            "pasos": int,          # número de pasos ejecutados
            "recompensa_total": float,  # retorno acumulado del episodio
            "terminated": bool,
            "truncated": bool,
            seed : int, opcional
                Semilla utilizada para reiniciar el entorno y el espacio de acciones.
                Permite reproducir un episodio y comparar agentes bajo las mismas
                condiciones iniciales.
        }
    """
    if seed is not None:
        env.action_space.seed(seed)

    observation, info = env.reset(seed=seed)

    pasos = 0
    recompensa_total = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated) and pasos < max_steps:
        action = funcion_agente(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        recompensa_total += reward
        pasos += 1

    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": terminated,
        "truncated": truncated,
        "seed": seed,
    }


def correr_episodios(
    nombre_entorno,
    funcion_agente,
    n_episodios=10,
    max_steps=10000,
    seed_base=None,
    **kwargs_entorno,
):
    """
    Corre n_episodios SIN grabar video y retorna las métricas
    de cada uno. Pensado para calcular el baseline con una muestra 
    de episodios en vez de uno solo, un único episodio puede no ser
    representativo.

    Retorna
    -------
    metricas : list[dict]
        Una entrada por episodio, con la misma forma que ejecutar_episodio.
    seed_base : int, opcional
        Semilla inicial del experimento. El episodio i utiliza la semilla
        seed_base + i, lo que permite reproducir los resultados y evaluar
        diferentes agentes con las mismas semillas.
    """
    env = crear_entorno(nombre_entorno, video_folder=None, **kwargs_entorno)

    metricas = []
    try:
        for episodio in range(n_episodios):
            if hasattr(funcion_agente, "reset"):
                funcion_agente.reset()

            seed_episodio = (
                seed_base + episodio
                if seed_base is not None
                else None
            )

            resultado_ep = ejecutar_episodio(
                env=env,
                funcion_agente=funcion_agente,
                max_steps=max_steps,
                seed=seed_episodio,
            )

            resultado_ep["episodio"] = episodio
            metricas.append(resultado_ep)
    finally:
        env.close()

    recompensas = [m["recompensa_total"] for m in metricas]
    print(f"{n_episodios} episodios | "
          f"promedio={np.mean(recompensas):.2f} | "
          f"std={np.std(recompensas):.2f} | "
          f"max={np.max(recompensas):.2f} | "
          f"min={np.min(recompensas):.2f}")

    return metricas


def generar_video_agente(nombre_entorno, funcion_agente, video_folder,
                          name_prefix, n_episodios=1, max_steps=10000,
                          **kwargs_entorno):
    """
    Crea el entorno con grabación de video,
    ejecuta n_episodios completos con funcion_agente, cierra el
    entorno y retorna las rutas de los videos generados
    junto con las métricas de cada episodio.

    Parámetros
    ----------
    nombre_entorno : str
        Id del entorno de Gymnasium
    funcion_agente : callable
        Función con firma funcion_agente(observation, env) -> action.
    video_folder : str
        Carpeta donde se guardarán los videos
    name_prefix : str
        Prefijo de los archivos de video
    n_episodios : int
        Número de episodios completos a ejecutar y grabar
    max_steps : int
        Máximo de pasos por episodio
    **kwargs_entorno :
        Argumentos adicionales para crear_entorno (incluye los de
        preprocesamiento: aplicar_preprocesamiento, frame_skip, etc.)

    Retorna
    -------
    resultado : dict
        {
            "videos": [lista de rutas de archivos .mp4 generados],
            "metricas": [lista de dicts con pasos/recompensa por episodio],
        }
    """
    import os

    os.makedirs(video_folder, exist_ok=True)

    #se graban todos los episodios que se van a ejecutar
    env = crear_entorno(
        nombre_entorno,
        video_folder=video_folder,
        episode_trigger=lambda ep: ep < n_episodios,
        name_prefix=name_prefix,
        **kwargs_entorno,
    )

    metricas = []
    try:
        for episodio in range(n_episodios):
            if hasattr(funcion_agente, "reset"):
                funcion_agente.reset()  #reinicia agentes con estado 
            resultado_ep = ejecutar_episodio(env, funcion_agente, max_steps=max_steps)
            resultado_ep["episodio"] = episodio
            metricas.append(resultado_ep)
    finally:
        env.close()

    #recopilar las rutas de los videos generados
    videos = sorted(
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.startswith(name_prefix) and f.endswith(".mp4")
    )

    return {
        "videos": videos,
        "metricas": metricas,
    }


if __name__ == "__main__":
    print("Baseline 1: agente aleatorio")
    correr_episodios(
        nombre_entorno="ALE/SpaceInvaders-v5",
        funcion_agente=agente_aleatorio,
        n_episodios=10,
        aplicar_preprocesamiento=True,
    )

    print("\nBaseline 2: regla simple (dispara y barre la pantalla)")
    correr_episodios(
        nombre_entorno="ALE/SpaceInvaders-v5",
        funcion_agente=AgenteReglaSimple(pasos_por_tramo=25),
        n_episodios=10,
        aplicar_preprocesamiento=True,
    )