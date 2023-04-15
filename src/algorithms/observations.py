import numpy as np

class Observations():
    def __init__(self):
        self.observed_points = np.array([[], []])
    
    def reset(self):
        self.observed_points = np.array([[], []])
    
    def update(self, observation):
        self.observed_points = np.append(self.observed_points, observation, axis=-1)
