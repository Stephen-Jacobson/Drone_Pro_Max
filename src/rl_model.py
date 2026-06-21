# TODO: create model in here to be used by drone
import torch
from torch import nn
from torch import multiprocessing

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)        # GPU
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)                               # decides if torch will go on gpu or cpu, gpu better
num_cells = 256                 # cells per hidden layer
lr = 3e-4
max_grad_norm = 1.0             # prevents too violent actions early in training

print(f"training on {device}")

steps_per_batch = 1000          # how many moves will make before model learns from it, so model isnt updating until steps_per_batch moves have been done, 1 step = 1 move
total_steps = 50_000            # how many total steps until done training
#therefore if 1000 steps in batch and 50000 steps total, will learn 50000/1000 = 50 times

# PPO Parameters (Proximal Policy Optimization)
    # At each steps_per_batch we will run to optimise model, this is done by taking a sub batch size, eg 64,
    # then taking 64 random steps(each step stored as action and reward as well current and next lidar scans, 
    # and if done or hit object) from the 1000 which just occured, and will do that until steps_per_batch=1000 random values have been taken. 
    # Will do that num_epoch number of times, eg 10, therefore will have (1000/64)*10 = ~156 gradient updates. At each epoch gradient/model 
    # updates ~15 times, so model updates every 64 taken from 1000, ie ~156
sun_batch_size = 64
num_epochs = 10
clip_epsilon = 0.2              # stops policy from updating too much in one steps, 0.2 will stop from updating when change is more than 20%
gamma = 0.99                    # between 0-1, closer to 1, worries more about future rewards, closer to 0, worries more about immediate rewards
lmbda = 0.95                    # used to compute advantage of move, ie was it better or worse than the expaected reward which our critic calculates
entropy_eps = 1e-4              # rewards exploration at beginning of training so model doesnt commit to badd moves, less important later in training


