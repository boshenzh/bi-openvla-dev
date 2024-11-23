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

def image_preprocess(image, center_crop=True):
    image = Image.fromarray(image)
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")
    return image

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
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
    INSTRUCTION = "lift the red box"
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
        
        img = get_libero_image(obs, 224)
        img = image_preprocess(img)
        inputs1 = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
        action1 = vla.predict_action(**inputs1, unnorm_key="libero_spatial", do_sample=False)
        action1 = normalize_gripper_action(action1)
        # action1 = invert_gripper_action(action1)
        return action1, img
    
    obs = env.reset()
    print(obs.keys())
    
    done = False
    ret = 0.
    dof = env.action_dim
    output_file = os.path.join(args.directory, "output.mp4")
    writer = imageio.get_writer(output_file, fps=20)
    
    output_file2 = os.path.join(args.directory, "output2.mp4")
    writer2 = imageio.get_writer(output_file2, fps=20)
    
    while not done:
        action, img = policy(obs)         # use observation to decide on an action
        
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
        writer2.append_data(np.flipud(np.fliplr(img)))
        writer.append_data(grid_image)
        ret += reward
    writer.close()
    writer2.close()
    print("rollout completed with return {}".format(ret))
