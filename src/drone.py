import lidar
import numpy as np

class Drone(object):
    id = 0
    sight_range = 0
    lidar = None

    def __init__(self, id, sight_range, values, pos, goal):
        self.id = id
        self.pos = np.array(pos, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.sight_range = sight_range
        self.values = values

    def move(self, direction, amount=1):
            direction = np.array(direction, dtype=np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)  # normalise
    
            new_pos = self.pos + direction * amount
            new_pos = np.clip(new_pos, 0, np.array(self.values.shape) - 1)  # stay in bounds
    
            # clear old pos, set new pos in grid
            self.values[int(self.pos[0]), int(self.pos[1]), int(self.pos[2])] = 0
            self.pos = new_pos
            self.values[int(self.pos[0]), int(self.pos[1]), int(self.pos[2])] = 5
    
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

    def get_obs(self):
        scan = lidar.get_lidar_surroundings(self.values, self.pos, self.sight_range)
        direction, distance = lidar.get_goal_vector(self.pos, self.goal, self.sight_range)

        return np.concatenate([scan, direction, [distance]])  # (n_rays + 4,)