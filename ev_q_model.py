import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

class EVChargingQModel:
    def __init__(self, hours=24, demand_bins=3, solar_bins=4, action_space_size=4, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.hours = hours
        self.demand_bins = demand_bins
        self.solar_bins = solar_bins
        self.action_space_size = action_space_size

        self.state_space_size = hours * demand_bins * solar_bins
        self.q_table = np.zeros((self.state_space_size, action_space_size))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.rewards_per_episode = []
        self.actions_per_episode = []

    def get_state_index(self, hour, demand, solar):
        return hour * self.demand_bins * self.solar_bins + demand * self.solar_bins + solar

    def choose_action(self, state_index):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Explore
        return np.argmax(self.q_table[state_index])  # Exploit

    def train(self, env_function, episodes=10000):
        for episode in range(episodes):
            state = env_function.reset()
            state_index = self.get_state_index(*state)
            total_reward = 0
            actions = []

            done = False
            while not done:
                action = self.choose_action(state_index)
                actions.append(action)

                next_state, reward, done = env_function.step(action)
                next_state_index = self.get_state_index(*next_state)

                # Q-Learning update
                old_value = self.q_table[state_index, action]
                next_max = np.max(self.q_table[next_state_index])
                self.q_table[state_index, action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

                state_index = next_state_index
                total_reward += reward

            self.rewards_per_episode.append(total_reward)
            self.actions_per_episode.append(actions)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def predict(self, state):
        state_index = self.get_state_index(*state)
        return np.argmax(self.q_table[state_index])

    def save_model(self, path='q_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, path='q_model.pkl'):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def plot_rewards(self):
        plt.plot(self.rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")
        plt.grid(True)
        plt.show()
