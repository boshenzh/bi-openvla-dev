import robosuite
# from robosuite.controllers import load_part_controller_config
import argparse
import os
import time
from glob import glob
import imageio
import numpy as np
# from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from PIL import Image  # Import PIL for saving images
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from robosuite.wrappers import DataCollectionWrapper
import tensorflow as tf


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img

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
        # "openvla/openvla-7b", 
        "openvla/openvla-7b-finetuned-libero-spatial",
        # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")
    INSTRUCTION = "lift the box"
    prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
    
    controller_configs = robosuite.load_controller_config(default_controller="OSC_POSE")

    
    env = robosuite.make(
        # "TwoArmLift",
        # robots=["Panda", "Panda"],             # load a Sawyer robot and a Panda robot
        "Lift",
        robots=["Panda"],             # load a Sawyer robot and a Panda robot
        gripper_types="default",                # use default grippers per robot arm
        # controller_configs=controller_config,   # each arm is controlled using OSC
        env_configuration="single-arm-opposed",            # (two-arm envs only) arms face each other
        has_renderer=False,                     # no on-screen rendering
        has_offscreen_renderer=True,            # off-screen rendering needed for image obs
        controller_configs=controller_configs,
        control_freq=20,                        # 20 hz control for applied actions
        horizon=100,                            # each episode terminates after 200 steps
        use_object_obs=False,                   # don't provide object observations to agent
        use_camera_obs=True,                   # provide image observations to agent
        camera_names=['frontview', 
                      'birdview', 
                      'agentview', 
                    #   'sideview', 
                      'all-robotview', 
                      'all-eye_in_hand',],               # use "agentview" camera for observations
        camera_heights=256,                      # image height
        camera_widths=256,                       # image width
        reward_shaping=True,                    # use a dense reward signal for learning
    )

    def policy(obs):
        # a trained policy could be used here, but we choose a random action
        low, high = env.action_spec
        img = np.flipud(np.fliplr(obs['robot0_robotview_image']))
        camera_image0 = resize_image(img, (224, 224))
        image1 = Image.fromarray(camera_image0)  # Convert the NumPy array to an image object
        
        inputs1 = processor(prompt, image1).to("cuda:0", dtype=torch.bfloat16)
        action1 = vla.predict_action(**inputs1, unnorm_key="libero_spatial", do_sample=False)
        action1 = normalize_gripper_action(action1)
        
        # img = np.flipud(np.fliplr(obs['robot1_robotview_image']))
        # camera_image0 = resize_image(img, (224, 224))
        # image1 = Image.fromarray(camera_image0)  # Convert the NumPy array to an image object
        
        # inputs1 = processor(prompt, image1).to("cuda:0", dtype=torch.bfloat16)
        # action2 = vla.predict_action(**inputs1, unnorm_key="libero_spatial", do_sample=False)
        # action2 = normalize_gripper_action(action1)
        # action = np.concatenate([action1, action2])
        
        return action1
    
    obs = env.reset()
    print(obs.keys())
    
    done = False
    ret = 0.
    dof = env.action_dim
    output_file = os.path.join(args.directory, "output.mp4")
    writer = imageio.get_writer(output_file, fps=20)
    
    while not done:
        action = policy(obs)         # use observation to decide on an action
        obs, reward, done, _ = env.step(action) # play action
        
        
        grid_image = np.vstack([
            # 第一行：水平拼接 agentview 和 frontview 图像
            np.hstack([
                np.flipud(np.fliplr(obs['agentview_image'])),  # agentview 图像旋转 180 度
                np.flipud(np.fliplr(obs['frontview_image']))   # frontview 图像旋转 180 度
            ]),
            # 第二行：水平拼接 robotview 和 eye-in-hand 图像
            np.hstack([
                np.flipud(np.fliplr(obs['robot0_robotview_image'])),  # robotview 图像旋转 180 度
                np.flipud(np.fliplr(obs['robot0_robotview_image'])) # eye-in-hand 图像旋转 180 度
            ])
        ])
        writer.append_data(grid_image)
        ret += reward
    writer.close()
    print("rollout completed with return {}".format(ret))
