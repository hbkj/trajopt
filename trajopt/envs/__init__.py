from gym.envs.registration import register

# ----------------------------------------
# trajopt environments
# ----------------------------------------
# max_episode_steps is not used internally in trajopt
# value picked based on requirements of RL

register(
    id='trajopt_reacher-v0',
    entry_point='trajopt.envs:Reacher7DOFEnv',
    max_episode_steps=75,
)

register(
    id='trajopt_continual_reacher-v0',
    entry_point='trajopt.envs:ContinualReacher7DOFEnv',
    max_episode_steps=250,
)

register(
    id='trajopt_hdt_reacher-v0',
    entry_point='trajopt.envs:HDTAnglerEnv',
    max_episode_steps=250,
)

register(
    id='trajopt_continual_hdt-v0',
    entry_point='trajopt.envs:ContinualHDTAnglerEnv',
    max_episode_steps=250,
)

from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from trajopt.envs.reacher_env import Reacher7DOFEnv, ContinualReacher7DOFEnv
from trajopt.envs.hdt_angler_env import HDTAnglerEnv, ContinualHDTAnglerEnv