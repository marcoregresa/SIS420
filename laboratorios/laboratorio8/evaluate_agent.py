#Laboratorio 8 - SIS420
#Nombre: Escobar Ruiz Marco Antonio - Ing Sistemas
import time  # Se usa para pausar entre pasos y ralentizar la simulación
from stable_baselines3 import PPO  # Importamos el algoritmo entrenado PPO
from flip_cup_env import FlipCupEnv  # Importamos el entorno personalizado que creaste
import pybullet as p


# Cargamos el modelo previamente entrenado desde el archivo .zip
model = PPO.load("ppo_flipcup_policy")

# 🎮 Creamos una instancia del entorno con renderizado activado
# Esto abre la ventana de simulación visual para observar el comportamiento del agente
env = FlipCupEnv(render=True)
# Reiniciamos el entorno para comenzar un nuevo episodio
obs = env.reset()


# Marcador que indica si el episodio terminó
done = False

# Contador de recompensa total obtenida por el agente en este episodio
episode_reward = 0

# El agente toma una acción paso a paso hasta que el entorno indique que terminó
while not done:
    # El modelo predice la mejor acción posible dado el estado actual (obs)
    # deterministic=True evita aleatoriedad: muestra cómo se comporta el agente "en serio"
    action, _ = model.predict(obs, deterministic=True)
    print(f"Action: {action}")
    # Aplicamos esa acción al entorno, que nos devuelve:
    # - la siguiente observación
    # - la recompensa obtenida
    # - si el episodio terminó
    # - un diccionario adicional vacío por ahora
    obs, reward, done, _ = env.step(action)

    # Acumulamos la recompensa total del episodio
    episode_reward += reward

    # Imprimimos la recompensa obtenida en este paso para observar el progreso

    print(f"Reward: {reward:.2f}")

    # Ralentizamos la simulación para que el movimiento se vea claramente (~20 FPS)
    time.sleep(0.05)
env.close()
print(f"\n Recompensa total final del episodio: {episode_reward:.2f}")