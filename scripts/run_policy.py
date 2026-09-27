#!/usr/bin/env python3
"""
Run a SmolVLA policy on a real SO-101 follower arm.
Live preview of cameras 0 and 1 + SmolVLA inference.

POLICY_PATH options:
  - "lerobot/smolvla_base"                                                 : HF base model (SO-100 data)
  - "outputs/train/toothpaste_from_20k/checkpoints/last/pretrained_model"  : the fine-tuned model from this repo
"""
import sys
sys.path.insert(0, "src")

import time
import os
import datetime
import cv2
import torch
import numpy as np

# ─── Model ────────────────────────────────────────────────────────────────────
# base: chamborgir/smolvla_pickplace_20k, fine-tuned 30K steps on toothpaste_grasp (51 ep)
# (train_expert_only=True, frozen vision encoder; no LoRA/PEFT -- see configs/train_config.json)
# camera keys: "camera1" (up, index=1) + "camera2" (side, index=0)
POLICY_PATH     = "outputs/train/toothpaste_from_20k/checkpoints/last/pretrained_model"
NORM_STATS_PATH = "outputs/train/toothpaste_from_20k/checkpoints/last/pretrained_model"
# ─────────────────────────────────────────────────────────────────────────────

# ─── Cameras ─────────────────────────────────────────────────────────────────
# training-data camera keys (after rename_map): "camera1" + "camera2"
CAMERA1_INDEX = 1   # dataset key "up"    -> model input "camera1"
CAMERA2_INDEX = 0   # dataset key "side"  -> model input "camera2"

# per-camera flip correction (start with False; set True if the image is mirrored)
FLIP_CAM1_HORIZONTAL = False
FLIP_CAM1_VERTICAL   = False
FLIP_CAM2_HORIZONTAL = False
FLIP_CAM2_VERTICAL   = False

# ─── Gripper ────────────────────────────────────────────────────────────────
# multiplier applied to the predicted gripper action before sending (1.2 in the demo run)
GRIPPER_SCALE = 1.2
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_PORT = "/dev/tty.usbmodemXXXXXXXXXXX"  # set to your follower arm's serial port
ROBOT_ID = "my_awesome_follower_arm"
TASK = "Grasp the toothpaste box and lift it up"
N_EPISODES = 3
EPISODE_TIME_S = 100
FPS = 30
DEVICE = "mps"

MOTOR_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

PREVIEW_W, PREVIEW_H = 640, 480  # preview size per camera

# ─── Recording ──────────────────────────────────────────────────────────────
RECORD_VIDEO = True          # set False to disable recording
RECORD_DIR   = "recordings"  # output folder, relative to the working directory
# ─────────────────────────────────────────────────────────────────────────────


def flip_image(img: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Flip an RGB HxWxC numpy image according to the flags."""
    if flip_h:
        img = img[:, ::-1, :].copy()
    if flip_v:
        img = img[::-1, :, :].copy()
    return img


def make_frame(img_cam1, img_cam2, action_1d, step, elapsed):
    """Build the side-by-side two-camera frame (BGR; used for display and recording)."""
    cam1 = cv2.resize(img_cam1, (PREVIEW_W, PREVIEW_H))
    cam2 = cv2.resize(img_cam2, (PREVIEW_W, PREVIEW_H))

    if cam1.shape[2] == 3:
        cam1 = cv2.cvtColor(cam1, cv2.COLOR_RGB2BGR)
    if cam2.shape[2] == 3:
        cam2 = cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR)

    cv2.putText(cam1, f"camera1 cam#{CAMERA1_INDEX} -> Model",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(cam2, f"camera2 cam#{CAMERA2_INDEX} -> Model",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    if action_1d is not None:
        for i, (name, val) in enumerate(zip(MOTOR_NAMES, action_1d)):
            short = name.replace(".pos", "")
            cv2.putText(cam1, f"{short}: {val:+.1f}",
                        (10, PREVIEW_H - 110 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

    cv2.putText(cam1, f"step={step}  t={elapsed:.1f}s",
                (10, PREVIEW_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

    return np.hstack([cam1, cam2])


def main():
    print("=" * 60)
    print("SmolVLA policy on robot  (camera 0+1 live preview)")
    print("=" * 60)

    # 1. load policy
    print(f"\n[1/3] loading policy: {POLICY_PATH}")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
    policy = policy.to(DEVICE)
    policy.eval()
    print(f"  policy loaded (device={DEVICE})")

    print(f"\n  loading pre/post-processing pipeline...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=NORM_STATS_PATH,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    print(f"  pipeline loaded (norm stats: {NORM_STATS_PATH})")

    # 2. connect robot + cameras
    print(f"\n[2/3] connecting robot: {ROBOT_PORT}")
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    # training-data camera keys (after rename_map): "camera1" + "camera2"
    camera_cfg = {
        "camera1": OpenCVCameraConfig(
            index_or_path=CAMERA1_INDEX,
            fps=FPS,
            width=640,
            height=480,
            fourcc="MJPG",
        ),
        "camera2": OpenCVCameraConfig(
            index_or_path=CAMERA2_INDEX,
            fps=FPS,
            width=640,
            height=480,
            fourcc="MJPG",
        ),
    }
    robot_cfg = SOFollowerRobotConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=camera_cfg,
    )
    robot = SOFollower(robot_cfg)
    robot.connect()
    print(f"  robot connected (camera1=#{CAMERA1_INDEX}, camera2=#{CAMERA2_INDEX})")

    # 3. inference loop
    print(f"\n[3/3] starting inference  (press q in the preview window to stop)")
    device = torch.device(DEVICE)
    blank  = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)

    # create recording folder
    if RECORD_VIDEO:
        os.makedirs(RECORD_DIR, exist_ok=True)

    # video size: two cameras side by side
    video_w = PREVIEW_W * 2
    video_h = PREVIEW_H
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        for ep in range(N_EPISODES):
            print(f"\n{'='*60}")
            print(f"episode {ep+1}/{N_EPISODES}: {TASK}")
            print("  starting in 3 s... (Ctrl+C or q to abort)")

            # one VideoWriter per episode
            writer = None
            if RECORD_VIDEO:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(RECORD_DIR, f"ep{ep+1:02d}_{ts}.mp4")
                writer = cv2.VideoWriter(video_path, fourcc, FPS, (video_w, video_h))
                print(f"  recording: {video_path}")

            # 3 s countdown preview
            for countdown in range(3, 0, -1):
                for _ in range(FPS):
                    obs_pre = robot.get_observation()
                    img_c1 = flip_image(obs_pre.get("camera1", blank), FLIP_CAM1_HORIZONTAL, FLIP_CAM1_VERTICAL)
                    img_c2 = flip_image(obs_pre.get("camera2", blank), FLIP_CAM2_HORIZONTAL, FLIP_CAM2_VERTICAL)
                    c1_bgr = cv2.cvtColor(cv2.resize(img_c1, (PREVIEW_W, PREVIEW_H)), cv2.COLOR_RGB2BGR)
                    c2_bgr = cv2.cvtColor(cv2.resize(img_c2, (PREVIEW_W, PREVIEW_H)), cv2.COLOR_RGB2BGR)
                    cv2.putText(c1_bgr, f"START IN {countdown}s", (160, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                    frame = np.hstack([c1_bgr, c2_bgr])
                    cv2.imshow("SmolVLA Live (q: quit)", frame)
                    if writer:
                        writer.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

            policy.reset()
            preprocessor.reset()

            start_time = time.time()
            step = 0

            while time.time() - start_time < EPISODE_TIME_S:
                loop_start = time.perf_counter()

                obs = robot.get_observation()
                state  = np.array([obs[name] for name in MOTOR_NAMES], dtype=np.float32)
                img_c1 = flip_image(obs["camera1"], FLIP_CAM1_HORIZONTAL, FLIP_CAM1_VERTICAL)
                img_c2 = flip_image(obs["camera2"], FLIP_CAM2_HORIZONTAL, FLIP_CAM2_VERTICAL)

                observation_frame = {
                    "observation.state":          state,
                    "observation.images.camera1": img_c1,
                    "observation.images.camera2": img_c2,
                }

                with torch.inference_mode():
                    for key in list(observation_frame.keys()):
                        val = observation_frame[key]
                        t = torch.from_numpy(val)
                        if "image" in key:
                            t = t.float() / 255.0
                            t = t.permute(2, 0, 1)  # HWC → CHW
                        t = t.unsqueeze(0).to(device)
                        observation_frame[key] = t

                    observation_frame["task"]       = TASK
                    observation_frame["robot_type"] = ""

                    processed_obs = preprocessor(observation_frame)
                    action_tensor = policy.select_action(processed_obs)
                    action_tensor = postprocessor(action_tensor)

                action_1d = action_tensor.squeeze(0).cpu().numpy()
                # gripper (last joint) scale correction (original note: base-policy max ~33 deg vs this robot max ~57 deg)
                action_1d[-1] = action_1d[-1] * GRIPPER_SCALE
                robot_action = {name: float(action_1d[i]) for i, name in enumerate(MOTOR_NAMES)}
                robot.send_action(robot_action)

                elapsed = time.time() - start_time
                frame = make_frame(img_c1, img_c2, action_1d, step, elapsed)
                cv2.imshow("SmolVLA Live (q: quit)", frame)
                if writer:
                    writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

                step += 1
                dt_s = time.perf_counter() - loop_start
                time.sleep(max(0.0, 1.0 / FPS - dt_s))

                if step % 10 == 0:
                    print(f"  step={step:3d}, t={elapsed:.1f}s, "
                          f"action=[{', '.join(f'{v:+.1f}' for v in action_1d)}]")

            if writer:
                writer.release()
                print(f"  recording saved: {video_path}")

            print(f"  episode {ep+1} done ({step} steps)")
            if ep < N_EPISODES - 1:
                print("  waiting 5 s before next episode...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if writer:
            writer.release()
        robot.disconnect()
        cv2.destroyAllWindows()
        print("robot/cameras disconnected")


if __name__ == "__main__":
    main()
