import mo_gymnasium as mo_gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

# Convierte recompensas vectoriales a valores escalares mediante la suma de sus componentes.
def get_scalar_reward(reward):
    return np.sum(reward) if isinstance(reward, (list, np.ndarray)) else reward

# Política ε-greedy: con probabilidad ε elige una acción aleatoria,
# de lo contrario elige la acción con el mayor valor Q.
def epsilon_greedy(q_values, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)

# Función softmax para convertir un vector de valores Q en probabilidades de selección
# según la temperatura dada. Luego selecciona una acción según esa distribución.
def softmax_action(q_values, temperature=1.0):
    exp_values = np.exp(q_values / temperature)
    probs = exp_values / np.sum(exp_values)
    return np.random.choice(len(q_values), p=probs)

# --------------------------------------------------------------
# Q-LEARNING BÁSICO
# --------------------------------------------------------------
# Entrena un agente usando Q-learning estándar. Cada `render_every` episodios
# recrea el entorno en modo gráfico ("human") para visualizar el progreso.
# Parámetros:
#   env_name    – nombre del entorno de mo_gymnasium (ej. "four-room-v0")
#   episodes    – número total de episodios de entrenamiento
#   alpha       – tasa de aprendizaje (learning rate)
#   gamma       – factor de descuento para recompensas futuras
#   epsilon     – probabilidad de exploración en ε-greedy
#   render_every– cada cuántos episodios recrear el entorno con render gráfico
# Devuelve:
#   Una tabla Q (diccionario) con valores Q para cada estado y acción.
def train_generic(env_name="four-room-v0",
                  episodes=1000,
                  alpha=0.1,
                  gamma=0.99,
                  epsilon=0.1,
                  render_every=100):
    q_table = defaultdict(lambda: np.zeros(mo_gym.make(env_name).action_space.n))
    env = mo_gym.make(env_name)
    episode_rewards = []

    for ep in range(episodes):
        # Determina si en este episodio se usará render
        if (ep + 1) % render_every == 0:
            env.close()
            env = mo_gym.make(env_name, render_mode="human")
        else:
            env.close()
            env = mo_gym.make(env_name)

        # Reinicia el entorno y obtiene el estado inicial
        obs, _ = env.reset()
        state = tuple(obs)
        total_reward_ep = 0

        while True:
            # Selecciona acción usando política ε-greedy
            action = epsilon_greedy(q_table[state], epsilon)
            # Ejecuta la acción en el entorno
            next_obs, reward, terminated, truncated, _ = env.step(action)
            scalar_reward = get_scalar_reward(reward)
            total_reward_ep += scalar_reward

            # Si el episodio termina, rompe el bucle
            if terminated or truncated:
                break

            # Procesa la siguiente transición
            next_state = tuple(next_obs)
            scalar_reward = get_scalar_reward(reward)

            # Actualiza la tabla Q usando la ecuación de Q-learning
            best_next = np.max(q_table[next_state])
            td_target = scalar_reward + gamma * best_next
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            # Avanza al siguiente estado
            state = next_state
        episode_rewards.append(total_reward_ep)

    env.close()
    return dict(q_table), episode_rewards

# --------------------------------------------------------------
# Q-LEARNING CON ALPHA DECRECIENTE (INCREMENTAL)
# --------------------------------------------------------------
# Versión de Q-learning donde la tasa de aprendizaje α disminuye según
# el número de visitas a cada par (estado, acción). Esto permite una
# convergencia más suave a largo plazo.
# Parámetros similares a `train_generic`.
# Se agrega:
#   visit_count – conteo de cuántas veces se ha visitado cada par (estado, acción).
# Devuelve:
#   Una tabla Q (diccionario) entrenada con α adaptativo.
def train_incremental(env_name="four-room-v0",
                      episodes=1000,
                      alpha=0.1,
                      gamma=0.99,
                      epsilon=0.7,
                      render_every=100):
    q_table = defaultdict(lambda: np.zeros(mo_gym.make(env_name).action_space.n))
    visit_count = defaultdict(lambda: np.zeros(mo_gym.make(env_name).action_space.n))
    env = mo_gym.make(env_name)
    episode_rewards = []

    for ep in range(episodes):
        # Reconstruye el entorno con o sin render
        if (ep + 1) % render_every == 0:
            env.close()
            env = mo_gym.make(env_name, render_mode="human")
        else:
            env.close()
            env = mo_gym.make(env_name)

        obs, _ = env.reset()
        state = tuple(obs)
        total_reward_ep = 0

        while True:
            # Selecciona acción ε-greedy
            action = epsilon_greedy(q_table[state], epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            scalar_reward = get_scalar_reward(reward)
            total_reward_ep += scalar_reward

            if terminated or truncated:
                break

            next_state = tuple(next_obs)
            scalar_reward = get_scalar_reward(reward)

            # 1. Llevar la cuenta de cuántas veces se ha tomado esta acción desde este estado
            visit_count[state][action] += 1
            # 2. Calcular la nueva tasa de aprendizaje adaptativa para esta actualización específica
            adaptive_alpha = alpha / (1 + 0.01 * visit_count[state][action])

            # 3. Usar esta tasa de aprendizaje reducida para actualizar la Tabla Q
            #    (La actualización es la estándar de Q-Learning)
            best_next = np.max(q_table[next_state])
            td_target = scalar_reward + gamma * best_next
            q_table[state][action] += adaptive_alpha * (td_target - q_table[state][action])

            state = next_state
        episode_rewards.append(total_reward_ep)
    env.close()
    return dict(q_table), episode_rewards

# --------------------------------------------------------------
# Q-LEARNING OPTIMISTA (INICIALIZACIÓN OPTIMISTA)
# --------------------------------------------------------------
# Inicializa todos los valores Q con un valor optimista `init_value` para impulsar
# la exploración temprana. Luego entrena con Q-learning estándar.
# Parámetros adicionales:
#   init_value – valor inicial optimista para la tabla Q.
# Devuelve:
#   Una tabla Q entrenada con inicialización optimista.
def train_optimist(env_name="four-room-v0",
                   episodes=1000,
                   alpha=0.1,
                   gamma=0.99,
                   epsilon=0.1,
                   init_value=1.0,
                   render_every=100):
    #Todos los valores Q se inician con un valor alto 
    # (por ejemplo, 1.0) para los estados y acciones no visitados.
    q_table = defaultdict(lambda: np.full(mo_gym.make(env_name).action_space.n, init_value))
    env = mo_gym.make(env_name)
    episode_rewards = []

    for ep in range(episodes):
        if (ep + 1) % render_every == 0:
            env.close()
            env = mo_gym.make(env_name, render_mode="human")
        else:
            env.close()
            env = mo_gym.make(env_name)

        obs, _ = env.reset()
        state = tuple(obs)
        total_reward_ep = 0

        while True:
            action = epsilon_greedy(q_table[state], epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            scalar_reward = get_scalar_reward(reward)
            total_reward_ep += scalar_reward

            if terminated or truncated:
                break

            next_state = tuple(next_obs)
            scalar_reward = get_scalar_reward(reward)

            # Actualización estándar de Q-learning
            best_next = np.max(q_table[next_state])
            td_target = scalar_reward + gamma * best_next
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
        episode_rewards.append(total_reward_ep)
    env.close()
    return dict(q_table), episode_rewards

# --------------------------------------------------------------
# Q-LEARNING CON UPPER CONFIDENCE BOUND (UCB)
# --------------------------------------------------------------
# Implementa Q-learning donde la selección de acciones utiliza el criterio UCB:
#     Q(s,a) + c * sqrt( log(N(s)) / N(s,a) )
# para balancear exploración y explotación. 
# Parámetros adicionales:
#   c – coeficiente de confianza que controla la magnitud de exploración.
# Devuelve:
#   Una tabla Q entrenada usando UCB para la exploración.
def train_UCB(env_name="four-room-v0",
              episodes=1000,
              alpha=0.1,
              gamma=0.99,
              c=2.0,
              render_every=100):
    n_actions = mo_gym.make(env_name).action_space.n
    q_table = defaultdict(lambda: np.zeros(n_actions))
    visit_count = defaultdict(lambda: np.zeros(n_actions))
    total_visits = defaultdict(int)
    env = mo_gym.make(env_name)
    episode_rewards = []

    for ep in range(episodes):
        if (ep + 1) % render_every == 0:
            env.close()
            env = mo_gym.make(env_name, render_mode="human")
        else:
            env.close()
            env = mo_gym.make(env_name)

        obs, _ = env.reset()
        state = tuple(obs)
        total_reward_ep = 0

        while True:
            # 1. Al inicio de cada paso, si el estado es nuevo, elige acción aleatoria para inicializar.
            if total_visits[state] == 0:
                action = np.random.randint(n_actions)
            else:
                # 2. Si el estado ya ha sido visitado, calcula los valores UCB para todas las acciones
                ucb_values = q_table[state] + c * np.sqrt(
                    np.log(total_visits[state]) / (visit_count[state] + 1e-10)
                )
                # 3. Elige la acción con el valor UCB más alto
                action = np.argmax(ucb_values)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            scalar_reward = get_scalar_reward(reward)
            total_reward_ep += scalar_reward

            if terminated or truncated:
                break

            next_state = tuple(next_obs)
            scalar_reward = get_scalar_reward(reward)

            # Actualiza contadores de visitas
            visit_count[state][action] += 1
            total_visits[state] += 1

            # ... (Luego, el agente ejecuta la acción y actualiza Q-table como en Q-learning estándar)
            best_next = np.max(q_table[next_state])
            td_target = scalar_reward + gamma * best_next
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
        episode_rewards.append(total_reward_ep)
    env.close()
    return dict(q_table), episode_rewards

# --------------------------------------------------------------
# Q-LEARNING BASADO EN GRADIENTES (SOFTMAX)
# --------------------------------------------------------------
# En lugar de ε-greedy, utiliza softmax para seleccionar acciones y
# ajusta la tasa de aprendizaje según el error temporal (TD error),
# escalando α con tanh(|TD_error|). Esto permite actualizaciones más
# suaves cuando el error es grande.
# Devuelve:
#   Una tabla Q entrenada con actualizaciones basadas en gradiente.
def train_gradient(env_name="four-room-v0",
                   episodes=1000,
                   alpha=0.1,
                   gamma=0.99,
                   epsilon=0.1,
                   render_every=100):
    q_table = defaultdict(lambda: np.zeros(mo_gym.make(env_name).action_space.n))
    env = mo_gym.make(env_name)
    episode_rewards = []


    for ep in range(episodes):
        if (ep + 1) % render_every == 0:
            env.close()
            env = mo_gym.make(env_name, render_mode="human")
        else:
            env.close()
            env = mo_gym.make(env_name)

        obs, _ = env.reset()
        state = tuple(obs)
        total_reward_ep = 0

        while True:
            # 1. Selecciona acción usando Softmax
            #    (asume una función softmax_action que devuelve la acción basada en Q-values y epsilon)
            action = softmax_action(q_table[state], epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            scalar_reward = get_scalar_reward(reward)
            total_reward_ep += scalar_reward

            if terminated or truncated:
                break

            next_state = tuple(next_obs)
            scalar_reward = get_scalar_reward(reward)

            # 2. Calcula el Error de Diferencia Temporal (TD Error)
            best_next = np.max(q_table[next_state])
            td_error = scalar_reward + gamma * best_next - q_table[state][action]

            # 3. Ajusta la tasa de aprendizaje (alpha) basada en la magnitud del TD error
            gradient_scale = np.tanh(abs(td_error))
            effective_alpha = alpha * gradient_scale
            
            # 4. Aplica la actualización de Q-learning con el alpha efectivo
            q_table[state][action] += effective_alpha * td_error

            state = next_state
        episode_rewards.append(total_reward_ep)
    env.close()
    return dict(q_table), episode_rewards

# --------------------------------------------------------------
# EVALUACIÓN DE LA POLÍTICA APRENDIDA
# --------------------------------------------------------------
# Ejecuta la política almacenada en `q_table` durante `test_episodes`
# sin renderizado, calculando la recompensa media y su desviación estándar.
# Parámetros:
#   env_name     – nombre del entorno
#   q_table      – diccionario con valores Q aprendidos
#   test_episodes– número de episodios para evaluación
# Devuelve:
#   (media_recompensas, desviación_estándar)
def evaluate_policy(env_name, q_table, test_episodes=10):
    env = mo_gym.make(env_name)
    n_actions = env.action_space.n
    total_rewards = []

    for _ in range(test_episodes):
        obs, _ = env.reset()
        state = tuple(obs)
        episode_reward = 0

        while True:
            # Elige la mejor acción según Q (sin exploración)
            action = np.argmax(q_table.get(state, np.zeros(n_actions)))
            next_obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += get_scalar_reward(reward)

            if terminated or truncated:
                break

            state = tuple(next_obs)

        total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards), np.std(total_rewards)

# --------------------------------------------------------------
# ENTRENAMIENTO Y COMPARACIÓN DE DIFERENTES MÉTODOS
# --------------------------------------------------------------
# Ejecuta todos los métodos de entrenamiento definidos (Generic, Incremental,
# Optimist, UCB, Gradient), evalúa cada política y muestra los resultados.
# Parámetros:
#   env_name     – nombre del entorno
#   episodes     – episodios de entrenamiento por método
#   render_every – frecuencia para renderizar durante entrenamiento
# Devuelve:
#   Un diccionario `results` con métricas y tablas Q de cada método.
def train_and_compare(env_name="four-room-v0",
                      episodes=1000,
                      render_every=100):
    methods = {
        'Generic': lambda: train_generic(env_name, episodes, render_every=render_every),
        'Incremental': lambda: train_incremental(env_name, episodes, render_every=render_every),
        'Optimist': lambda: train_optimist(env_name, episodes, render_every=render_every),
        'UCB': lambda: train_UCB(env_name, episodes, render_every=render_every),
        'Gradient': lambda: train_gradient(env_name, episodes, render_every=render_every)
    }
    
    ''' methods = {
        'Generic': lambda: train_generic(env_name, episodes, alpha=0.5, gamma=0.55,epsilon=0.5, render_every=render_every),
        'Incremental': lambda: train_incremental(env_name, episodes, alpha=0.1, gamma=0.5,epsilon=0.5, render_every=render_every),
        'Optimist': lambda: train_optimist(env_name, episodes, alpha=0.1, gamma=0.6,epsilon=0.5, init_value=5.0, render_every=render_every),
        'UCB': lambda: train_UCB(env_name, episodes,alpha=0.1, gamma=0.5, c=2.0, render_every=render_every),
        'Gradient': lambda: train_gradient(env_name, episodes,alpha=0.1, gamma=0.6,epsilon=0.5, step_size=0.01, render_every=render_every)
    } '''


    results = {}

    print("Entrenando agentes...")
    for name, train_fn in methods.items():
        print(f"  -> {name}...")
        q_table, rewards = train_fn()
        mean_reward, std_reward = evaluate_policy(env_name, q_table)
        results[name] = {
            'q_table': q_table,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'states_visited': len(q_table),
            'rewards_per_episode': rewards
        }

    print("\n=== RESULTADOS ===")
    for name, result in results.items():
        print(f"{name:12}: Reward={result['mean_reward']:.3f}±{result['std_reward']:.3f}, "
              f"Estados={result['states_visited']}")

    best_method = max(results.keys(), key=lambda x: results[x]['mean_reward'])
    print(f"\nMejor método: {best_method}")

    return results

if __name__ == "__main__":
    # Ejemplo de uso: entrena y compara métodos con render cada 500 episodios
    resultados = train_and_compare(env_name="four-room-v0", episodes=1000, render_every=500)

    # Mostrar los primeros 50 estados de la tabla Q del mejor método
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['mean_reward'])
    mejor_q = resultados[mejor]['q_table']    
    print(f"\nPrimeros 5 estados del mejor método ({mejor}):")
    for i, (state, q_vals) in enumerate(list(mejor_q.items())[:50]):
        print(f"  Estado {state}: {q_vals}")
    plt.figure(figsize=(10, 6))

    for name, data in resultados.items():
        rewards = data['rewards_per_episode']
        epochs = np.arange(1, len(rewards) + 1)
        # Opcional: calcular media móvil para suavizar la curva
        window = 20
        if len(rewards) >= window:
            cumsum = np.cumsum(np.insert(rewards, 0, 0)) 
            moving_avg = (cumsum[window:] - cumsum[:-window]) / window
            plt.plot(epochs[window - 1:], moving_avg, label=f"{name} (media móvil {window})")
        else:
            plt.plot(epochs, rewards, label=name)

    plt.title("Recompensa por Episodio vs. Época para cada Método")
    plt.xlabel("Época (episodio)")
    plt.ylabel("Recompensa acumulada en el episodio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()