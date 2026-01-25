import subprocess
from pathlib import Path
import os
import sys

# Configuration
REPO_ID = "Ziegelll/RAPID_dummy_50"
LOCAL_DIR = Path("my_local_data") / REPO_ID
TARGET_FPS = 20

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_video_fps(file_path):
    """Get the FPS of a video file using ffprobe."""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=r_frame_rate", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den
        return float(fps_str)
    except Exception as e:
        print(f"Error reading FPS for {file_path}: {e}")
        return None

def convert_video_fps(video_path, target_fps):
    """Convert video to target FPS using ffmpeg."""
    temp_path = video_path.with_suffix(".temp.mp4")
    
    # ffmpeg command:
    # -i input
    # -r target_fps (forces output frame rate, duplicating frames if necessary)
    # -c:v libx264 (re-encode using x264)
    # -crf 18 (high quality)
    # -preset fast
    # -y (overwrite output)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-r", str(target_fps),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-loglevel", "error",
        str(temp_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        # If successful, replace original file
        temp_path.replace(video_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {video_path}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def main():
    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not in PATH. Please install ffmpeg first.")
        sys.exit(1)

    videos_dir = LOCAL_DIR / "videos"
    if not videos_dir.exists():
        print(f"Error: Videos directory not found at {videos_dir}")
        sys.exit(1)

    print(f"Scanning videos in {videos_dir}...")
    video_files = list(videos_dir.rglob("*.mp4"))
    
    print(f"Found {len(video_files)} video files.")
    print(f"Target FPS: {TARGET_FPS}")
    
    converted_count = 0
    skipped_count = 0
    
    for i, video_path in enumerate(video_files):
        current_fps = get_video_fps(video_path)
        
        # Determine if conversion is needed (allow small float tolerance)
        if current_fps and abs(current_fps - TARGET_FPS) < 0.1:
            print(f"[{i+1}/{len(video_files)}] Skipping {video_path.name} (already {current_fps:.2f} FPS)")
            skipped_count += 1
            continue
            
        print(f"[{i+1}/{len(video_files)}] Converting {video_path.name} ({current_fps:.2f} FPS -> {TARGET_FPS} FPS)...")
        if convert_video_fps(video_path, TARGET_FPS):
            converted_count += 1
        else:
            print(f"Failed to convert {video_path.name}")

    print("\nProcessing complete!")
    print(f"Converted: {converted_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total: {len(video_files)}")

if __name__ == "__main__":
    main()
