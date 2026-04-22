import json
import argparse
import cv2
import numpy as np
from src.engine.vision_engine import AIEngine
from utils.frame_handler import draw_bbox, draw_zone, display_frame


def main():

    args = parse_args()

    with open(args.config_mode, "r") as f:
        config_mode = json.load(f)
    active_modules = config_mode.get("active_strategies", {})

    # initialize AI engine
    engine = AIEngine()
    engine.activate_by_type(active_modules)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read the frame.")
            break

        # Analyze the frame with the AI engine
        results = engine.analyze_frame(frame)
        
        event = []
        for code, result in results.items():
            event.extend(result.get("event", []))
            frame = draw_bbox(frame, result.get("bboxs", []))

            if code == "intrusion":
                frame = draw_zone(frame, result.get("corner_points", []))

        display_frame(event, frame, int(1000 / fps))


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Engine for video analysis")
    parser.add_argument("--config-mode", type=str,
                        default="config/o1.json", help="Path to config file")
    parser.add_argument("--video-path", type=str,
                        default="data/videos/people-detection.mp4", help="Path to input video")
    return parser.parse_args()


if __name__ == "__main__":
    main()
