import argparse
import cv2
import numpy as np

from utils.inference_io import preprocess_batch
from utils.postprocessing import postprocess_yolo
from triton_client import TritonClient

from sort import Sort

class VideoBatchTracker:
    def __init__(self, video_path, batch_size=8):
        self.video_path = video_path
        self.batch_size = batch_size
        self.triton_client = TritonClient()


        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_size = (orig_w, orig_h) 
        
        frames_buffer = []
        trakkers = []

        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                frames_buffer.append(frame)
                
            if len(frames_buffer) == self.batch_size or (not ret and len(frames_buffer) > 0):
                
                batch_input = preprocess_batch(frames_buffer) 
                batch_outputs = self.triton_client.run(batch_input=batch_input)
                
                input_size = (batch_input.shape[3], batch_input.shape[2])  # (W, H) for postprocessing
                for i in range(len(frames_buffer)):
                    current_frame = frames_buffer[i]
                    raw_frame_output = batch_outputs[i]

                    detections = postprocess_yolo(
                    	raw_output=raw_frame_output,
                    	input_size=input_size,
                    	orig_size=orig_size,
                    	confidence_threshold=0.5,
                    	iou_threshold=0.45,
                    	top_k=20,
                    )
                    
                    if detections:
                        dets = np.array(
                            [[det.box[0], det.box[1], det.box[2], det.box[3], det.score] for det in detections],
                            dtype=np.float32,
                        )
                    else:
                        dets = np.empty((0, 5), dtype=np.float32)
                    tracked_objects = self.tracker.update(dets) 

                    for obj in tracked_objects:
                        x1, y1, x2, y2, track_id = obj.astype(int)
                        
                        # Draw bounding box and ID
                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(current_frame, f"ID: {track_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cv2.imshow("Video Batch Tracking", current_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                frames_buffer = []
            
            if not ret:
                break
        cv2.destroyAllWindows()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton Batch Tracking Example")
    parser.add_argument("--video-path", type=str, default="data/videos/people-detection.mp4", help="Path to input video")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")

    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = parse_args()

    tracker_app = VideoBatchTracker(
        video_path=args.video_path,
        batch_size=args.batch_size
    )
    tracker_app.run()