from stable_baselines3 import PPO  # Importa el algoritmo PPO (Proximal Policy Optimization)
from stable_baselines3.common.vec_env import DummyVecEnv  # Permite vectorizar el entorno (aunque sea uno solo)
from flip_cup_env import FlipCupEnv  # Tu entorno personalizado basado en PyBullet


# Crear entorno vectorizado
env = DummyVecEnv([lambda: FlipCupEnv(render=False)])

# Definir modelo
# Definimos el modelo PPO
model = PPO(
    "MlpPolicy",                # Red neuronal completamente conectada (MLP)
    env,                        # Entorno vectorizado donde entrenar
    verbose=1,                  # Mostrar información de entrenamiento en consola
    tensorboard_log="./ppo_flipcup_tensorboard/",  # Carpeta donde guardar logs para visualizar en TensorBoard
    learning_rate=3e-4,         # Tasa de aprendizaje (valor típico para PPO)
    n_steps=1024,               # Cantidad de pasos que recolecta antes de actualizar la red
    batch_size=64,             # Tamaño de lote para el entrenamiento
    gamma=0.99,                 # Factor de descuento para recompensas futuras
    gae_lambda=0.95,            # Parámetro de GAE (maneja el trade-off entre sesgo y varianza)
    ent_coef=0.01               # Coeficiente de entropía: promueve exploración
)
model.tb_log_name = "flipcup_run"
#  Entrenamos el modelo durante 10,000 pasos de tiempo (bastante corto, ideal para tests rápidos)
model = PPO.load("ppo_flipcup_policy", env=env)

model.learn(total_timesteps=10_000)

# Guardamos el modelo entrenado en disco
model.save("ppo_flipcup_policy")

