import argparse
import os
import time
from glob import glob

import numpy as np

import robosuite as robosuite
from robosuite.wrappers import DataCollectionWrapper


from traj import playback_trajectory

env = robosuite.make(
    "TwoArmLift",
    robots=["Panda", "Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    # controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="opposed",            # (two-arm envs only) arms face each other
    has_renderer=True,                     # no on-screen rendering
    has_offscreen_renderer=False,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=False,                   # provide image observations to agent
    camera_names=["all-robotview","all-robotview"],               # use "agentview" camera for observations
    camera_heights=84,                      # image height
    camera_widths=84,                       # image width
    reward_shaping=True,                    # use a dense reward signal for learning
)
data_directory = "haha/ep_1731724246_8065286"

env = DataCollectionWrapper(env, data_directory)

playback_trajectory(env, data_directory, 20)