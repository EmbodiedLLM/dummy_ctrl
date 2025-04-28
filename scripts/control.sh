# python third_party/lerobot/lerobot/scripts/control_robot.py --robot.type=dummy \
#         --robot.cameras='{}' \
#         --control.type=teleoperate \
#         --robot.inference_time=false

# python third_party/lerobot/lerobot/common/robot_devices/cameras/network.py \
#     --images-dir outputs/images_from_network_cameras \
#     --camera-urls "http://192.168.237.100:8080/?action=stream"

sudo python third_party/lerobot/lerobot/scripts/control_robot.py \
  --robot.type=dummy \ # 使用dummy机器人
  --control.type=record \ # 使用记录模式，teleoperate为控制模式， record为记录模式
  --control.fps=20 \ # 设置相机帧率
  --control.single_task="Press the button" \ # 任务描述（language instruction）
  --control.warmup_time_s=5 \ # 设置预热时间
  --control.episode_time_s=15 \ # 设置每个任务的时间
  --control.reset_time_s=1 \ # 设置重置时间
  --control.num_episodes=1 \ # 设置任务数量
  --control.push_to_hub=false \ # 是否推送到hub（这里设置为true的话会卡住，所以暂时设为false，数据采集完手动推）
  --control.root=data/press_0425 \ # 数据集保存路径
  --control.repo_id=JackYuuuu/press_buttons \ # 数据集仓库id
  --control.display_camera=true \ # 是否在采集时显示相机画面
  --robot.inference_time=false \ 
  --control.resume=false # 是否恢复上次记录