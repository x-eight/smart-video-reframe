import argparse
import sys
import json
import os
from pathlib import Path

# Ensure src is in the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reframer.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Auto Reframe Video CLI")
    parser.add_argument("--url", required=True, help="Video URL or local path")
    parser.add_argument("--width", type=int, default=720, help="Target width (default: 720)")
    parser.add_argument("--height", type=int, default=1280, help="Target height (default: 1280)")
    parser.add_argument("--is_fit", action="store_true", help="Use fit mode (blur background) instead of AI crop")
    parser.add_argument("--num_faces", type=int, help="Number of faces to focus on")
    parser.add_argument("--output_dir", help="Output directory (default: ./output)")

    args = parser.parse_args()

    # Determine job input
    job_input = {
        "url": args.url,
        "width": args.width,
        "height": args.height,
        "is_fit": args.is_fit,
        "num_faces": args.num_faces
    }

    # Determine output directory
    project_root = Path(__file__).resolve().parent.parent
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(project_root, "output")

    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        processor = VideoProcessor()
        result = processor.process(job_input, output_dir)
        print("\nProcessing completed successfully:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
