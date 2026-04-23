import json
import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import trange

from engine.vision_engine import AIEngine
from utils.frame_handler import draw_bbox, draw_zone, display_frame


def main():

    args = parse_args()

    with open(args.config_mode, "r") as f:
        config_mode = json.load(f)
    active_modules = config_mode.get("active_strategies", {})

    # print(f"Loaded config: {config_mode}")
    # print(f"Active modules from config: {active_modules}")

    # initialize AI engine
    engine = AIEngine()
    engine.activate_by_type(active_modules)

    # print(f"Activated strategies: {list(engine.active_strategies.keys())}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video_path}")
        print(f"Absolute path: {os.path.abspath(args.video_path)}")
        print(f"File exists: {os.path.exists(args.video_path)}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make video writer to save results
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    save_path = Path(args.out_dir) / f"{os.path.splitext(os.path.basename(args.video_path))[0]}_pl{len(os.listdir(args.out_dir))+1}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height)) if args.save_video else None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        print(f"Processing frame {frame_count}...")  # Debug statement to track progress
        if not ret:
            print(f"End of video stream or cannot read the frame at frame {frame_count}.")
            cap.release()
            break
        frame_count += 1

        # Analyze the frame with the AI engine
        results = engine.analyze_frame(frame)
        event = []
        for code, result in results.items():
            event.extend(result.get("event", []))
            frame = draw_bbox(frame, result.get("detections", []))

            if code == "intrusion":
                frame = draw_zone(frame, result.get("corner_points", []))

        display_frame(event, frame, int(1000 / fps))
        for e in event:
            print(e)
        
        if args.save_video and out is not None:
            out.write(frame)

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Engine for video analysis")
    parser.add_argument("--config-mode", type=str,
                        default="config/01.json", help="Path to config file")
    parser.add_argument("--video-path", type=str,
                        default="data/videos/people-detection.mp4", help="Path to input video")
    parser.add_argument("--save-video", action="store_true", help="Whether to save the output video")
    parser.add_argument("--out-dir", type=str, default="data/videos/results", help="Directory to save output videos")

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    main()