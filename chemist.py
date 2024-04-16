import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class ChemicalExperimentEnv(gym.Env):
    """
    A custom environment for chemical experiments, which simulates interactions between chemicals.
    The environment ensures that only safe chemical combinations are explored.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(ChemicalExperimentEnv, self).__init__()
        # Assuming a simplified model where there are 10 different chemicals
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Discrete(10)

        # Safe combinations matrix (1 for safe, 0 for unsafe)
        self.safe_combinations = np.random.choice([0, 1], size=(10, 10), p=[0.2, 0.8])

        # Initialize state randomly
        self.current_state = np.random.randint(0, 10)

    def step(self, action):
        # Check if the combination is safe
        if self.safe_combinations[self.current_state][action] == 0:
            reward = -100  # Penalize unsafe actions heavily
            done = True
        else:
            reward = 1  # Found a safe action
            done = False

        # Update the state
        self.current_state = action
        info = {}

        return self.current_state, reward, done, info

    def reset(self):
        # Reset to a random state
        self.current_state = np.random.randint(0, 10)
        return self.current_state

    def render(self, mode='console'):
        if mode == 'console':
            print(f'Current state: {self.current_state}')

def main():
    # Create the environment
    env = ChemicalExperimentEnv()

    # Define the model: use PPO algorithm with safe exploration
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent for 5000 steps
    model.learn(total_timesteps=5000)

    # Test the trained agent
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
