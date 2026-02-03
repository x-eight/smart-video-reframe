# Auto-Reframe Professional

A powerful CLI tool to automatically reframe videos for mobile (9:16) using AI-driven face detection and scene analysis.

## ✨ Features

- **AI-Powered Cropping**: Uses YOLOv8 to detect and track faces, ensuring they remain centered in the frame.
- **Scene Awareness**: Automatically detects scene changes to maintain smooth transitions and consistent framing.
- **Mobile Optimized**: Defaults to 720x1280 resolution, perfect for TikTok, Reels, and Shorts.
- **Fit Mode with Blur**: Optional mode to scale the video and fill the background with a blurred version instead of AI cropping.
- **Local Processing**: No cloud dependencies or external APIs required. Processes everything on your machine.
- **Standardized Output**: Generates both the reframed video and a JSON file containing metadata of the detected faces.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/x-eight/smart-video-reframe.git
   cd smart-video-reframe
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have FFmpeg installed on your system.

4. Place the YOLO model (`yolov8n.pt`) in the `models/` directory.

## 🚀 Usage

### Basic Command
Run the CLI by providing a video URL or a local path:
```bash
python src/cli.py --url path/to/your/video.mp4
```

### Options
- `--width`: Target width (default: 720).
- `--height`: Target height (default: 1280).
- `--is_fit`: Use blur background mode instead of AI crop.
- `--num_faces`: Number of faces to focus on.
- `--output_dir`: Path to save the results (default: `./output`).

Example with custom output:
```bash
python src/cli.py --url input.mp4 --output_dir ./my_results --is_fit
```

## 📂 Output Structure
The tool generates a structured output in the designated folder:
```text
output/
├── input_video.mp4      # Copy of the original input
├── output.mp4            # The reframed/processed video
└── output.json           # Metadata of detected faces and frames
```

## ⚖️ License
This project is licensed under the MIT License.
# smart-video-reframe
