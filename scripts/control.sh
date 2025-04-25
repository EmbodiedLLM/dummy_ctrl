python third_party/lerobot/lerobot/scripts/control_robot.py --robot.type=dummy \
        --robot.cameras='{}' \
        --control.type=teleoperate \
        --robot.inference_time=false

python third_party/lerobot/lerobot/scripts/control_robot.py --robot.type=dummy \
        --control.type=teleoperate \
        --robot.inference_time=false

python third_party/lerobot/lerobot/common/robot_devices/cameras/network.py \
    --images-dir outputs/images_from_network_cameras \
    --camera-urls "http://192.168.65.124:8080/?action=stream" "http://192.168.65.138:8080/?action=stream"

python lerobot/scripts/control_robot.py \
  --robot.type=dummy \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick Cube" \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=1 \
  --control.num_episodes=2 \
  --control.push_to_hub=false \
  --control.display_data=true \
  --control.repo_id=JackYuuuu/dummy_test \
  --control.root=data/dummy_test