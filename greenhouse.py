import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class GreenhouseEnv(gym.Env):
    """
    A custom environment for managing a greenhouse, which balances plant growth with environmental impact.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(GreenhouseEnv, self).__init__()
        # Define action and observation space
        # Actions: amount of water, light, and fertilizer
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        # Observations: plant health, soil health, water quality
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Initial conditions
        self.plant_health = 0.5
        self.soil_health = 0.5
        self.water_quality = 0.5

    def step(self, action):
        water, light, fertilizer = action

        # Update the environment
        self.plant_health += 0.05 * light - 0.02 * (fertilizer - 0.3)**2
        self.soil_health -= 0.01 * fertilizer
        self.water_quality -= 0.03 * fertilizer

        # Calculate reward
        reward = self.plant_health + self.soil_health + 10 * self.water_quality - 5 * fertilizer

        # Check if any condition is below a threshold
        done = self.plant_health < 0.1 or self.soil_health < 0.1 or self.water_quality < 0.1

        # Normalize for stability
        self.plant_health = np.clip(self.plant_health, 0, 1)
        self.soil_health = np.clip(self.soil_health, 0, 1)
        self.water_quality = np.clip(self.water_quality, 0, 1)

        info = {}

        return np.array([self.plant_health, self.soil_health, self.water_quality]), reward, done, info

    def reset(self):
        self.plant_health = 0.5
        self.soil_health = 0.5
        self.water_quality = 0.5
        return np.array([self.plant_health, self.soil_health, self.water_quality])

    def render(self, mode='console'):
        if mode == 'console':
            print(f'Plant Health: {self.plant_health}, Soil Health: {self.soil_health}, Water Quality: {self.water_quality}')

def main():
    env = GreenhouseEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
