import robosuite
# from robosuite.controllers import load_part_controller_config
import argparse
import os
import time
from glob import glob

import numpy as np
# from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from PIL import Image  # Import PIL for saving images
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from robosuite.wrappers import DataCollectionWrapper
from traj import playback_trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", type=str, default="haha")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    args = parser.parse_args()

    #init OPENVLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")
    INSTRUCTION = "lift the box"
    prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"


    # create an environment to visualize on-screen
   # create an environment for policy learning from pixels
    env = robosuite.make(
    "TwoArmLift",
    robots=["Panda", "Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    # controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="opposed",            # (two-arm envs only) arms face each other
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                   # provide image observations to agent
    camera_names=["all-robotview","all-robotview"],               # use "agentview" camera for observations
    camera_heights=84,                      # image height
    camera_widths=84,                       # image width
    reward_shaping=True,                    # use a dense reward signal for learning
)

    data_directory = args.directory
    env = DataCollectionWrapper(env, data_directory)
    # reset the environmentcd

    #currently 1 policy for both arm
    def policy(obs):
        # a trained policy could be used here, but we choose a random action
        low, high = env.action_spec
        camera_image0 = obs['robot0_robotview_image'] 
        camera_image1 = obs['robot1_robotview_image'] 
        image1 = Image.fromarray(camera_image0)  # Convert the NumPy array to an image object
        image2 = Image.fromarray(camera_image1)  # Convert the NumPy array to an image object

        inputs1 = processor(prompt, image1).to("cuda:0", dtype=torch.bfloat16)
        action1 = vla.predict_action(**inputs1, unnorm_key="bridge_orig", do_sample=False)

        inputs2 = processor(prompt, image2).to("cuda:0", dtype=torch.bfloat16)
        action2 = vla.predict_action(**inputs2, unnorm_key="bridge_orig", do_sample=False)

        print(action1)
        print(action2)

        return action1, action2

    
    # reset the environment to prepare for a rollout
    obs = env.reset()
    print(obs.keys())
    # robot_1_camera_image = obs['robot1_robotview_image']
    # print(type(robot_1_camera_image))
    # print(robot_1_camera_image)
    # image1 = Image.fromarray(robot_1_camera_image)  # Convert the NumPy array to an image object

    # print(f"img size:{np.size(robot_1_camera_image)}")
    # robot_0_camera_image = obs['robot0_robotview_image']
    # image0 = Image.fromarray(robot_0_camera_image)  # Convert the NumPy array to an image object
    # image0.save("initial_robot0_image.png")  # Save as PNG (you can choose other formats like .jpg)
    # image1.save("initial_robot1_image.png")  # Save as PNG (you can choose other formats like .jpg)


    done = False
    ret = 0.
    dof = env.action_dim

    while not done:
        action1,action2 = policy(obs)         # use observation to decide on an action
        obs, reward, done, _ = env.step(np.concatenate([action1, action2])) # play action
        # env.render()
        
        # limit frame rate if necessary
        max_fr = None
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)


        ret += reward

    print("rollout completed with return {}".format(ret))

    _ = input("Press any key to begin the playback...")
    print("Playing back the data...")
    data_directory = env.ep_directory
    playback_trajectory(env, data_directory, args.max_fr)