import lidar
import numpy as np

class Drone(object):
    id = 0
    pos = None
    sight_range = 0
    sight_env = None
    lidar = None

    def __init__(self, id, sight_range, sight_env, pos, goal, values):
        self.id = id
        self.pos = np.array(pos, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.sight_range = sight_range
        self.sight_env = sight_env
        self.values = values
    
    def get_id(self):
        return self.id
    
    def get_sight_range(self):
        return self.sight_range
    
    def get_sight_env(self):
        return self.sight_env
    
    def set_id(self, id):
        self.id = id

    def set_sight_range(self, sight_range):
        self.sight_range = sight_range

    def set_sight_env(self, sight_env):
        self.sight_env = sight_env

    def get_obs(self):
        scan = lidar.get_lidar_surroundings(self.values, self.pos, self.sight_range)
        direction, distance = lidar.get_goal_vector(self.pos, self.goal, self.sight_range)

        return np.concatenate([scan, direction, [distance]])  # (n_rays + 4,)