import os
import subprocess
import math

def download_video_ffmpeg(url:str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    filename = "input_video.mp4" # Standardized input filename
    output_path = os.path.join(output_dir, filename)
    
    # If it's already a local file, we might just want to copy it or use it directly
    # For now, we assume URL or path that ffmpeg can handle
    try:
        subprocess.run(["ffmpeg", "-y", "-i", url, "-c", "copy", output_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading/copying video: {e}")
        raise e

def merge_audio_with_video(processed_video, original_video, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", processed_video,
        "-i", original_video,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-strict", "experimental",
        "-y", output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def apply_fit_with_blur(input_video, dir_path, width, height, new_width, new_height, blur_radius=20, blur_power=4):
    os.makedirs(dir_path, exist_ok=True)
    output_video = os.path.join(dir_path, "output.mp4")
    video_blur = os.path.join(dir_path, "video_blur.mp4")
    video_scaled = os.path.join(dir_path, "video_scaled.mp4")

    # --- Step 1: Calculate proportional scale ---
    aspect_original = width / height
    aspect_new = new_width / new_height

    if aspect_original > aspect_new:
        # Video is wider -> limit by width
        scale_w = new_width
        scale_h = int(new_width / aspect_original)
    else:
        # Video is taller -> limit by height
        scale_h = new_height
        scale_w = int(new_height * aspect_original)

    # --- Ensure width and height are even ---
    scale_w = max(2, math.floor(scale_w / 2) * 2)
    scale_h = max(2, math.floor(scale_h / 2) * 2)

    # --- Step 2: Background blur ---
    cmd_blur = (
        f'ffmpeg -y -i "{input_video}" '
        f'-vf "scale={new_width}:{new_height},boxblur=luma_radius={blur_radius}:luma_power={blur_power}" '
        f'-c:a copy "{video_blur}"'
    )
    subprocess.run(cmd_blur, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- Step 3: Proportionally scale video ---
    cmd_scale = (
        f'ffmpeg -y -i "{input_video}" '
        f'-vf "scale={scale_w}:{scale_h}" '
        f'-c:a copy "{video_scaled}"'
    )
    subprocess.run(cmd_scale, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- Step 4: Centered overlay ---
    cmd_overlay = (
        f'ffmpeg -y -i "{video_blur}" -i "{video_scaled}" '
        f'-filter_complex "[0:v][1:v]overlay=(W-w)/2:(H-h)/2" '
        f'-c:a copy "{output_video}"'
    )
    subprocess.run(cmd_overlay, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up intermediate files
    if os.path.exists(video_blur): os.remove(video_blur)
    if os.path.exists(video_scaled): os.remove(video_scaled)

    return output_video
