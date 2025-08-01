import numpy as np

class DummyEnv:
    def __init__(self):
        self.hour = 0

    def reset(self):
        self.hour = 0
        demand = np.random.randint(0, 3)
        solar = np.random.randint(0, 4)
        return (self.hour, demand, solar)

    def step(self, action):
        self.hour = (self.hour + 1) % 24
        next_demand = np.random.randint(0, 3)
        next_solar = np.random.randint(0, 4)
        reward = np.random.choice([-1, 0, 1])
        done = self.hour == 0
        return (self.hour, next_demand, next_solar), reward, done
