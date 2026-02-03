import uuid
import time, os
import math
import cv2
import numpy as np
import pathlib
import json

from detectors.face_detector import FaceDetector
from detectors.scene_detector import SceneDetector
from utils.ffmpeg import merge_audio_with_video, apply_fit_with_blur, download_video_ffmpeg
from config import config, CACHE_DIR

class VideoProcessor:
    def __init__(self):
        self.crop_folder_base = "AUTOCROP/VIDEO"
        
    def _get_crop_folder(self):
        return f"{self.crop_folder_base}/{uuid.uuid4()}"

    def _handle_fit_mode(self, video_input, dir_path, original_width, original_height, new_width, new_height, file_name, crop_folder):
        """
        Handles the logic when is_fit is True.
        Applies blur and fit without using AI models.
        """
        print("Mode: Fit with Blur")

        # Apply fit with blur using ffmpeg
        file_path = apply_fit_with_blur(video_input, dir_path, original_width, original_height, new_width, new_height)
        
        # In local mode, we just return the local file path
        return { 
            "success": True, 
            "video_path": file_path,
            "json_path": ""
        }

    def _handle_crop_mode(self, video_input, dir_path, cap, original_width, original_height, new_width, new_height, fps, total_frames, file_name, job_input, crop_folder):
        """
        Handles the logic when is_fit is False.
        Uses YOLO and other models to crop and center faces.
        """
        print("Mode: Auto Crop with AI", CACHE_DIR)
        
        YOLO_DIR = pathlib.Path(CACHE_DIR)
        output_path = os.path.join(dir_path, file_name)

        # Setup directories
        pyworkPath = os.path.join(dir_path, 'pywork')
        os.makedirs(pyworkPath, exist_ok=True)

        scene_detector = SceneDetector(video_input, pyworkPath)
        
        # Detect scenes
        scenes = scene_detector.get_frame_by_scene()
        frame_intervals = [s["end_frame"] for s in scenes]
        frame_intervals = frame_intervals if frame_intervals else [total_frames]

        # Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = f"{output_path}_temp.mp4"
        output_video = cv2.VideoWriter(temp_video_path, fourcc, fps, (new_width, new_height))

        # Initialize Face Detector (YOLO)
        model_path = YOLO_DIR / "yolov8n.pt"
        if not model_path.exists():
             print(f"Warning: Model not found at {model_path}")

        face_detector = FaceDetector(str(model_path), original_width, original_height)

        detected_faces, frames = [], []
        
        # Process frames for face detection
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            face_boxes = face_detector.upload_frame(idx, frame)
            frames.append(frame)
            detected_faces.append(face_boxes)

        # Homogenize faces
        aux_faces = []
        previous_frame = 0
        for interval in frame_intervals:
            if previous_frame < len(detected_faces):
                interval = interval - 1 if interval > previous_frame else 0
                face_segment = [frame_data for frame_data in detected_faces if frame_data and frame_data[0].get("frame") in range(previous_frame, interval)]
                
                # 1. Stabilize the number of faces first (prevents flickering count)
                aux_face_segment = face_detector.homogenize_group_sizes(face_segment, fps)

                # 2. Apply EMA smoothing for fluid transitions (slower movement with alpha=0.1)
                aux_faces.extend(face_detector.smooth_face_detections(aux_face_segment, alpha=0.1))
            previous_frame = interval
        detected_faces = aux_faces

        cap.release()

        results = []
        initial_crop_done = False
        num_faces = job_input.get('num_faces') if job_input.get('num_faces') and job_input['num_faces'] > 0 else None

        # Generate output video
        for idx in range(total_frames):
            frame = frames[idx]
            annotated_image = np.array(frame)
            face_boxes = detected_faces[idx] if detected_faces and 0 <= idx < len(detected_faces) else []

            rgb_frame = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            if face_boxes and len(face_boxes) > 0:
                results.append(face_boxes) # Logic that creates the json

                combined_image = face_detector.combine_faces_v2(face_boxes, rgb_frame, new_width, new_height, num_faces)
                resized_image = cv2.resize(combined_image, (new_width, new_height))
                final_frame = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)                    

                initial_crop_done = True

            elif not initial_crop_done and not face_boxes:
                center_x = (original_width - new_width) // 2
                center_y = (original_height - new_height) // 2

                if new_width <= original_width and new_height <= original_height:
                    final_frame = frame[center_y:center_y + new_height, center_x:center_x + new_width]
                else:
                    final_frame = cv2.resize(frame, (new_width, new_height))
            else:
                # If no faces but initial crop was done, use the last valid final_frame (or center crop)
                # For simplicity, we keep the last frame or recalculate center
                pass

            output_video.write(final_frame)

        output_video.release()
        cv2.destroyAllWindows()
        
        # Merge audio
        final_video_path = os.path.join(dir_path, f"{file_name}.mp4")
        merge_audio_with_video(temp_video_path, video_input, final_video_path)
        
        # Remove temp video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        # Save JSON results locally
        json_path = os.path.join(dir_path, f"{file_name}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        return { 
            "success": True, 
            "video_path": final_video_path,
            "json_path": json_path
        }

    def process(self, job_input, dir_path: str):
        start_time_ms = float(time.time() * 1000)
        
        try:
            print(f"Processing video from: {job_input['url']}")
            
            # Download Video
            video_input = download_video_ffmpeg(job_input["url"], dir_path)
            file_name = "output" # Fixed name for final output in directory
            
            # Get Video Properties
            cap = cv2.VideoCapture(video_input)
            if not cap.isOpened():
                raise ValueError(f"Cannot open file: {video_input}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Determine target dimensions
            new_width, new_height = 720, 1280
            if job_input.get('width') and job_input.get('height'):
                new_width = job_input['width']
                new_height = job_input['height']

            # Dispatch based on mode
            if job_input.get('is_fit', False):
                cap.release()
                result = self._handle_fit_mode(
                    video_input, dir_path, 
                    original_width, original_height, 
                    new_width, new_height, 
                    file_name, ""
                )
            else:
                result = self._handle_crop_mode(
                    video_input, dir_path, cap,
                    original_width, original_height, 
                    new_width, new_height, 
                    fps, total_frames, 
                    file_name, job_input, ""
                )

            # Finalize
            end_time_ms = int(time.time() * 1000)
            duration = (end_time_ms - start_time_ms) / 1000
            print(f"Duration: {duration:.2f} seconds")

            return result

        except Exception as e:
            print(f"Error in process: {e}")
            raise e
