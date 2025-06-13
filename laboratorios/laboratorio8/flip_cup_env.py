import pybullet as p
import pybullet_data
import gym
import numpy as np
from gym import spaces

class FlipCupEnv(gym.Env):
    def __init__(self, render=False, fixed_cup=True):
        super().__init__()

        # ¿Queremos ver la simulación con PyBullet GUI?
        self.render = render

        # Si True, el vaso comienza siempre en la misma posición. Ideal para entrenamiento progresivo.
        self.fixed_cup = fixed_cup

        # Inicializamos la conexión con el motor físico
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)

        # Añadimos las rutas por defecto de PyBullet (incluye URDFs básicos como plano o cubos)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Establecemos gravedad estándar hacia abajo
        p.setGravity(0, 0, -9.8)

        # Tiempo de simulación por paso (240 Hz)
        self.time_step = 1. / 240.
        p.setTimeStep(self.time_step)

        # Número máximo de pasos por episodio
        self.max_steps = 500
        self.step_counter = 0  # Contador interno de pasos

        self.cup_id = None     # ID del cuerpo físico del vaso
        self.robot_id = None   # ID del modelo del robot
        self.last_dist = None  # Se usa para calcular el cambio en distancia (reward shaping)

        self._load_env()  # Carga la escena inicial
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=50,
                             cameraPitch=-35,
                             cameraTargetPosition=[0, 0.6, 0])

        self.num_joints = 7  # Número de grados de libertad del brazo KUKA

        # Espacio de observación: posición 3D del vaso + 7 ángulos de articulación del brazo
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Espacio de acción: 7 valores continuos (-1 a 1), uno para cada articulación
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def _load_env(self):
        # Reseteamos la simulación para empezar desde cero
        p.resetSimulation()
        p.loadURDF("plane.urdf")  # Suelo plano

        # Cargamos el brazo robótico KUKA iiwa en posición fija
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Definimos la posición del vaso: fija o aleatoria en X si está activado el curriculum
        cup_pos = [0.0, 0.6, 0.1] if self.fixed_cup else [
            np.random.uniform(-0.2, 0.2), 0.6, 0.1]

        # Cargamos el vaso desde un URDF cilíndrico (asegúrate de que cylinder.urdf exista en tu carpeta)
        self.cup_id = p.loadURDF("cylinder.urdf", cup_pos)

    def reset(self):
        # Reestablece la simulación y devuelve la observación inicial
        self.step_counter = 0
        self._load_env()

        # Calculamos distancia inicial entre efector final y vaso
        cup_pos, _ = p.getBasePositionAndOrientation(self.cup_id)
        ee_pos = p.getLinkState(self.robot_id, self.num_joints - 1)[0]
        self.last_dist = np.linalg.norm(np.array(ee_pos) - np.array(cup_pos))

        return self._get_obs()

    def _get_obs(self):
        # Recuperamos la posición actual del vaso
        cup_pos, _ = p.getBasePositionAndOrientation(self.cup_id)

        # Extraemos ángulo actual de cada junta
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]

        # Creamos el vector de observación para la red neuronal: [x,y,z] del vaso + 7 ángulos
        return np.array(list(cup_pos) + joint_angles, dtype=np.float32)

    def step(self, action):
        # Aplicamos la acción: desplazamiento relativo a cada articulación
        for i in range(self.num_joints):
            current = p.getJointState(self.robot_id, i)[0]  # Ángulo actual
            delta = float(action[i]) * 0.3                  # Cambios angulares escalados
            target = np.clip(current + delta, -2.5, 2.5)    # Limitamos el rango de movimientos
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                    targetPosition=target, force=500)

        p.stepSimulation()  # Avanzamos la simulación física
        self.step_counter += 1  # Actualizamos el contador

        obs = self._get_obs()              # Obtenemos el nuevo estado
        reward, done = self._compute_reward()  # Calculamos recompensa y si termina el episodio

        return obs, reward, done, {}

    def _compute_reward(self):
        # Posición del vaso y del efector final
        cup_pos, _ = p.getBasePositionAndOrientation(self.cup_id)
        ee_pos = p.getLinkState(self.robot_id, self.num_joints - 1)[0]

        # Calculamos distancia euclidiana entre el efector y el vaso
        dist = np.linalg.norm(np.array(ee_pos) - np.array(cup_pos))

        # Diferencia con la distancia anterior (reward shaping)
        improvement = self.last_dist - dist
        shaping = improvement * 10.0       # Amplificamos la mejora con factor 10
        self.last_dist = dist              # Guardamos nueva distancia para siguiente paso

        # Bonus si el vaso fue empujado lo suficiente hacia arriba (se considera "volteado")
        bonus = 10.0 if cup_pos[2] > 0.25 else 0.0

        # Recompensa total: mejora + bonus
        reward = shaping + bonus

        # Terminamos episodio si se logra el objetivo o se alcanza el límite de pasos
        done = self.step_counter >= self.max_steps or bonus > 0.0
        return reward, done

    def close(self):
        # Cerramos la simulación para liberar recursos
        p.disconnect()
