import numpy as np
from ultralytics import YOLO
import cv2
from statistics import mean
from collections import Counter

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the intersection area
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    # Calculate the area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Calculate the IoU (Intersection over Union)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

class FaceDetector():
    def __init__(self, path_model: str, width = 200, height = 200, stable_threshold=7):
        self.model = YOLO(path_model) # models/yolov11n-face.pt yolo11n  yolov8n yolov11n-face
        self.width = width
        self.height = height
        self.stable_threshold = stable_threshold
        self.previous_faces = []
        self.list_faces = []
        self.preview_size = None
        self.preview_faces = []

    def upload_frame(self, idx, frame_input, iou_threshold=0.7):  
        results = self.model.track(source=frame_input, classes=0)
        filtered_faces = []
        faces = []
        for result in results:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                print("float(conf): ",float(conf))
                if float(conf) > 0.5:
                    x1, y1, x2, y2 = box.tolist()
                    faces.append({"frame": idx ,"bbox": (float(x1), float(y1), float(x2), float(y2)), "conf": float(conf), "is_speaker": False, "score": 0})
                else: 
                    print("delete delete delete: ",float(conf))
        if len(faces) == 0:
            return []
        else:
            # Sort by face position x1, y1
            faces.sort(key=lambda f: (f["bbox"][0], f["bbox"][1]))
            filtered_faces = []
            for face in faces:
                bbox = face["bbox"]
                if not any(calculate_iou(bbox, other["bbox"]) > iou_threshold for other in filtered_faces):
                    filtered_faces.append(face)

        #return filtered_faces
        return faces

    def smooth_face_detections(self, nested_list, alpha=0.2):
        """
        Smooths face bounding boxes using Exponential Moving Average (EMA).
        Improved version using center-based tracking for better stability.
        
        Args:
            nested_list (List[List[Dict]]): List of frames, each containing a list of detected faces.
            alpha (float): Smoothing factor (0 < alpha <= 1). 
                           Lower values = smoother but more lag.
                           Higher values = more responsive but more jitter.
        
        Returns:
            List[List[Dict]]: Smoothed face detections.
        """
        if not nested_list:
            return []

        smoothed_list = []
        
        # Helper to get center x of a face dict
        def get_center_x(face):
            bbox = face["bbox"]
            return (bbox[0] + bbox[2]) / 2

        # Initialize previous faces with the first frame's faces
        previous_faces = nested_list[0]
        # Sort initial frame by center x
        previous_faces.sort(key=get_center_x)
        smoothed_list.append(previous_faces)

        for i in range(1, len(nested_list)):
            current_frame_faces = nested_list[i]
            smoothed_frame_faces = []

            # If number of faces matches, we can smooth
            if len(current_frame_faces) == len(previous_faces):
                # Sort both lists by center x to ensure we match the same face
                current_frame_faces.sort(key=get_center_x)
                previous_faces.sort(key=get_center_x)

                for curr_face, prev_face in zip(current_frame_faces, previous_faces):
                    # Apply EMA to bbox coordinates
                    curr_bbox = curr_face["bbox"]
                    prev_bbox = prev_face["bbox"]
                    
                    new_bbox = tuple(
                        alpha * c + (1 - alpha) * p 
                        for c, p in zip(curr_bbox, prev_bbox)
                    )
                    
                    # Create new face dict with smoothed bbox
                    smoothed_face = curr_face.copy()
                    smoothed_face["bbox"] = new_bbox
                    smoothed_frame_faces.append(smoothed_face)
            else:
                # If face count changes, reset smoothing for this frame
                # But still sort them for consistency
                current_frame_faces.sort(key=get_center_x)
                smoothed_frame_faces = current_frame_faces

            smoothed_list.append(smoothed_frame_faces)
            previous_faces = smoothed_frame_faces

        return smoothed_list

    def homogenize_nested_dicts(self, nested_list, thresholda=40):
        """
        This method stabilizes face bounding boxes over time by averaging small variations
        across consecutive frames that contain the same number of faces.

        nested_list: List[List[Dict]]
        - Each outer list element represents a frame
        - Each inner list contains detected faces for that frame
        """

        # Get the number of faces detected in each frame
        face_counts = [len(frame) for frame in nested_list]

        # If there are no frames, return empty
        if len(face_counts) == 0:
            return []

        grouped_values = []
        # Start with the number of faces from the first frame
        current_faces = face_counts[0]

        # Group consecutive frames that have the same number of faces
        temp_group = []
        for frame, face_count in zip(nested_list, face_counts):
            # If face count changes, close current group and start a new one
            if face_count != current_faces:
                grouped_values.append(temp_group)
                temp_group = []
                current_faces = face_count
            temp_group.append(frame)

        # Add the last group if it exists
        if temp_group:
            grouped_values.append(temp_group)

        processed_frames = []

        # Process each group separately
        for group in grouped_values:
            # Number of faces in this group
            num_dicts = len(group[0])
            # Number of bbox elements (x1, y1, x2, y2)
            num_elements = len(group[0][0]['bbox']) if num_dicts > 0 else 0

            # Prepare structure to collect bbox values across frames
            grouped_bbox_values = [[[] for _ in range(num_elements)] for _ in range(num_dicts)]

            # Collect bbox coordinates for each face across frames
            for frame in group:
                for i, dct in enumerate(frame):
                    for j, value in enumerate(dct.get('bbox', [])):
                        if i < len(grouped_bbox_values) and j < len(grouped_bbox_values[i]):
                            grouped_bbox_values[i][j].append(value)

            # Compute average bbox values for each face
            averaged_values = [
                tuple(int(round(mean(values))) for values in group)
                for group in grouped_bbox_values
            ]

            # Compute dynamic threshold for each face based on its size
            thresholds = []
            for i, avg_bbox in enumerate(averaged_values):
                # width and height of the average bbox
                bbox_w = avg_bbox[2] - avg_bbox[0]
                bbox_h = avg_bbox[3] - avg_bbox[1]
                size = max(bbox_w, bbox_h)
                # scale threshold by base_ratio and FPS
                fps2 = 30
                base_idx = 0.35
                thresholds.append(int(base_idx * size * (30 / fps2)))
                print("threshold: ",int(base_idx * size * (30 / fps2)), int(size * (30 / fps2)), bbox_w, bbox_h)

            # Apply averaged bbox values if the change is within threshold
            for frame in group:
                processed_frame = []
                for i, (dct, avg_bbox) in enumerate(zip(frame, averaged_values)):
                    threshold = thresholds[i]  # dynamic threshold per face
                    smoothed_bbox = tuple(
                        avg if abs(value - avg) <= threshold else value
                        for value, avg in zip(dct['bbox'], avg_bbox)
                    )
                    processed_frame.append({
                        "frame": dct['frame'],
                        "bbox": smoothed_bbox,
                        "is_speaker": dct['is_speaker'],
                        "score": dct["score"],
                        "conf": dct['conf']
                    })
                processed_frames.append(processed_frame)

        return processed_frames

    def homogenize_group_sizes(self, nested_list, group_size=10):
        """
        Stabilizes the number of detected faces across nearby frames by enforcing
        the most common face count within short frame windows, preventing layout
        flickering caused by inconsistent detections.

        Args:
            nested_list (List[List[Any]]):
                A list of frames, where each frame is a list of detected faces.
                Each inner list length represents the number of faces detected
                in that frame.

            group_size (int, optional):
                The number of consecutive frames to process as a temporal window.
                Within each window, the most common face count is used as reference.
                Default is 10 frames.

        Returns:
            List[List[Any]]:
                A list of frames where inconsistent face counts within each window
                are normalized to the most common value.
        """
        processed_list = []
        total_length = len(nested_list)

        # Process frames in fixed-size groups (windows)
        for i in range(0, total_length, group_size):
            group = nested_list[i:i + group_size]

            # Get sizes of non-empty sublists (number of faces per frame)
            non_empty_sizes = [len(sublist) for sublist in group if sublist]

            # If all frames are empty, keep them as-is
            if not non_empty_sizes:
                processed_list.extend(group)
                continue

            # Count how many times each size appears
            size_counts = Counter(non_empty_sizes)

            # Find the most common number of faces in this group
            most_common_size = max(size_counts, key=size_counts.get)

            # Take one frame as reference with the most common size
            reference_list = next(
                (sublist for sublist in group if len(sublist) == most_common_size),
                None
            )

            # Replace frames that do not match the common size (except empty ones)
            new_group = [
                sublist if len(sublist) == most_common_size or len(sublist) == 0
                else reference_list
                for sublist in group
            ]

            processed_list.extend(new_group)

        return processed_list

    def normalize_coordinates(self, x, y, width, height, crop_width, crop_height):
        """
        Centers a face within a crop slot of size crop_width x crop_height,
        adjusting if it touches the edges of the image.

        Args:
            x, y: top-left coordinates of the original face
            width, height: size of the original face
            crop_width, crop_height: desired slot size to center the face

        Returns:
            x1, y1, x2, y2: final crop coordinates, centered if possible
        """
        # Compute the center of the face
        center_x = x + width / 2
        center_y = y + height / 2

        # Center the crop slot on the face center
        x1 = int(center_x - crop_width / 2)
        y1 = int(center_y - crop_height / 2)
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        # Adjust if the crop goes beyond the left or top edges
        if x1 < 0:
            x2 -= x1  # expand on the other side
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0

        # Adjust if the crop goes beyond the right or bottom edges
        if x2 > self.width:
            x1 -= (x2 - self.width)
            x2 = self.width
        if y2 > self.height:
            y1 -= (y2 - self.height)
            y2 = self.height

        # Ensure coordinates are within valid image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)

        return x1, y1, x2, y2

    def combine_faces(self, data, image, width, height, total_faces: None | int = None):
        """
        Combines faces into a single image. 

        For multiple faces: behaves like original.
        For a single face: scales proportionally to avoid distortion, filling as much as possible
        without exceeding the target width/height.

        Args:
            data (List[Dict]): Detected faces with 'bbox'.
            image (np.array): Original frame/image.
            width (int): Target width of output.
            height (int): Target height of output.
            total_faces (int, optional): Force number of faces to use.

        Returns:
            np.array: Combined image of size roughly width x height.
        """
        # Handle empty image
        if image is None or image.size == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)

        faces = []

        sort_faces = total_faces if total_faces else len(data)

        if height > width:
            data.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
            if sort_faces in {1, 3}:
                best_speaker = max(data, key=lambda x: x["score"]) if any(x["score"] > 0 for x in data) else None
                if best_speaker and best_speaker["score"] > 0:
                    data.remove(best_speaker)
                    data.insert(0, best_speaker)
        else:
            data.sort(key=lambda x: (x["bbox"][0], x["bbox"][1]))
            if sort_faces == 1:
                data.sort(key=lambda x: (-x["score"], x["bbox"][0], x["bbox"][1]))

        if total_faces is not None:
            data = data[:total_faces]

        resized_faces = []
        faces = [d["bbox"] for d in data]
        num_faces = len(faces)

        if num_faces == 1:
            # --- Crear fondo borroso ---
            img_h, img_w = image.shape[:2]

            # --- Compute face slot (width = full width, height proportional) ---
            face = data[0]["bbox"]
            x1, y1, x2, y2 = map(int, face)
            original_w = x2 - x1
            original_h = y2 - y1

            # --- Crop background to exact size ---
            scale_ratio = width / img_w
            new_h_img = int(img_h * scale_ratio)
            n_x1, n_y1, n_x2, n_y2 = self.normalize_coordinates(
                x1, y1, original_w, original_h, width, height
            )

            face_crop = image[n_y1:n_y2, n_x1:n_x2]
            resized_fg = cv2.resize(face_crop, (width, new_h_img))


            return resized_fg

        opt_w = 0
        opt_h = 0
        for i, d in enumerate(faces):
            x1, y1, x2, y2 = d
            original_w = x2 - x1
            original_h = y2 - y1

            if height > width:
                if num_faces == 3:
                    if i == 0:
                        new_w, new_h = width, int(height * 0.35)
                        opt_h = new_h
                    else:
                        new_w, new_h = width // 2, height - opt_h  #int(height * 0.65)
                        opt_w = new_w
                        if i == len(faces) - 1:
                            new_w = width - opt_w
                elif num_faces == 4:
                    new_w, new_h = width // 2, height // 2
                else:
                    new_w, new_h = width, height // num_faces
                    if i == len(faces) - 1:
                        new_h = height - opt_h
                    else:
                        opt_h = opt_h + new_h
            else:
                new_w, new_h = width // num_faces, height 
                if i == len(faces) - 1:
                    new_w = width - opt_w
                else:
                    opt_w = opt_w + new_w

            n_x1, n_y1, n_x2, n_y2 = self.normalize_coordinates(x1, y1, original_w, original_h, new_w, new_h)

            face_crop = image[n_y1:n_y2, n_x1:n_x2]

            if data[i].get("is_speaker", False):
                color = (0, 255, 0)
                cv2.rectangle(image, (n_x1, n_y1), (n_x2, n_y2), color, 2)

            face_resized = cv2.resize(face_crop, (new_w, new_h))
            resized_faces.append({"bbox": face_resized, "x1": x1, "y1": y1})

        if height > width:
            if num_faces == 2:
                combined_image = np.vstack([f["bbox"] for f in resized_faces])
            elif num_faces == 3:
                top = resized_faces[0]["bbox"]
                bottom = np.hstack([resized_faces[1]["bbox"], resized_faces[2]["bbox"]])

                """
                if top.shape[1] != bottom.shape[1]:
                    min_width = min(top.shape[1], bottom.shape[1])
                    top = cv2.resize(top, (min_width, top.shape[0]))
                """

                combined_image = np.vstack([top, bottom])
            elif num_faces == 4:
                resized_faces.sort(key=lambda face: (face["x1"], face["y1"]))
                top = np.hstack([resized_faces[0]["bbox"], resized_faces[1]["bbox"]])
                bottom = np.hstack([resized_faces[2]["bbox"], resized_faces[3]["bbox"]])

                """
                if top.shape[1] != bottom.shape[1]:
                    min_width = min(top.shape[1], bottom.shape[1])
                    top = cv2.resize(top, (min_width, top.shape[0]))
                """

                combined_image = np.vstack([top, bottom])
            else:
                combined_image = np.vstack([f["bbox"] for f in resized_faces])
        else:
            combined_image = np.hstack([f["bbox"] for f in resized_faces])

        return combined_image
    
    def combine_faces_v2(self, data, image, width, height, total_faces: None | int = None):
        """
        Combine faces or center the image on a blurred background.

        If image is None or empty, returns a blank frame of size (width, height).
        If there are 2 or more faces, centers the frame in a blurred background that fills the full width and height.
        """
        # Handle empty or None image
        if image is None or image.size == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)

        num_faces = total_faces if total_faces and total_faces > 0 else len(data)
        print("num_faces: ", num_faces)
        # If only 1 face or less, behave like original
        if num_faces <= 1:
            return self.combine_faces(data, image, width, height, total_faces=1)

        # Create full-size background using the image
        # Resize image to fill the background while keeping aspect ratio
        img_h, img_w = image.shape[:2]
        scale_w = width / img_w
        scale_h = height / img_h
        scale = max(scale_w, scale_h)  # Scale so it covers entire frame
        bg_w = int(img_w * scale)
        bg_h = int(img_h * scale)
        resized_bg = cv2.resize(image, (bg_w, bg_h))
        blurred_bg = cv2.GaussianBlur(resized_bg, (51, 51), 0)  # Big blur

        # Crop or center to exact width x height
        start_x = (bg_w - width) // 2 if bg_w > width else 0
        start_y = (bg_h - height) // 2 if bg_h > height else 0
        final_image = blurred_bg[start_y:start_y+height, start_x:start_x+width].copy()

        # Resize original image to fit width and keep aspect ratio
        scale_ratio = width / img_w
        new_h_img = int(img_h * scale_ratio)
        resized_fg = cv2.resize(image, (width, new_h_img))

        # Center foreground vertically
        y_offset = max((height - new_h_img) // 2, 0)
        final_image[y_offset:y_offset + new_h_img, :, :] = resized_fg

        return final_image
