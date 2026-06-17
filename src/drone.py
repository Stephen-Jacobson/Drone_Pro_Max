class Drone(object):
    id = 0
    sight_range = 0;
    sight_env = None;

    def __init__(self, id, sight_range, sight_env):
        self.id = id
        self.sight_range = sight_range
        self.sight_env = sight_env
    
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