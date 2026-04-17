import argparse
import cv2
import numpy as np
from tqdm import tqdm
import time

from utils.inference_io import preprocess_batch
from utils.postprocessing import postprocess_yolo
from triton_client import TritonClient

from sort import Sort

class VideoBatchTracker:
    def __init__(self):
        args = parse_args()
        self.video_path = args.video_path
        self.batch_size = args.batch_size
        self.output_path = args.output_path + self.video_path.split("/")[-1] if args.output_path else ""

        self.triton_client = TritonClient()

        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        start = time.time()
        while not self.triton_client.check_server_and_model():
            if time.time() - start > 10:
                print("Triton not ready after 10 seconds, stop.")
                return
            print("Waiting for Triton server to be ready...")
            time.sleep(1)
            

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay_ms = int(1000 / fps) if fps > 0 else 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")  
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_size = (orig_w, orig_h) 
        
        frames_buffer = []
        frames_results = []

        avg_time_per_preprocess = 0
        avg_time_per_inference = 0
        avg_time_per_postprocess = 0

        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                frames_buffer.append(frame)
                
            if len(frames_buffer) == self.batch_size or (not ret and len(frames_buffer) > 0):
                
                t0 = time.perf_counter()
                batch_input = preprocess_batch(frames_buffer) 
                t1 = time.perf_counter()
                batch_outputs = self.triton_client.run(batch_input=batch_input)
                t2 = time.perf_counter()
                
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
                    frames_results.append(current_frame)
                t3 = time.perf_counter()
                pbar.update(len(frames_buffer))
                
                frames_buffer = []
            
                avg_time_per_postprocess += (t3 - t2)
                avg_time_per_preprocess += (t1 - t0)
                avg_time_per_inference += (t2 - t1)

            if not ret:
                break
        
        num_batches = pbar.n / self.batch_size

        pbar.close()
        print(f"Average time per preprocess: {avg_time_per_preprocess / num_batches:.4f} s")
        print(f"Average time per inference: {avg_time_per_inference / num_batches:.4f} s")
        print(f"Average time per postprocess: {avg_time_per_postprocess / num_batches:.4f} s")

        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # writer = cv2.VideoWriter(self.output_path, fourcc, fps if fps > 0 else 30, orig_size)
        # if not writer.isOpened():
        #     raise RuntimeError(f"Cannot open VideoWriter for {self.output_path}")
        # for frame in frames_results:
        #     writer.write(frame)
        # writer.release()
        # print(f"Saved video to: {self.output_path}")
        
        # for frame in frames_results:
        #     cv2.imshow("Video Batch Tracking", frame)

        #     if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton Batch Tracking Example")
    parser.add_argument("--video-path", type=str, default="data/videos/people-detection.mp4", help="Path to input video")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output-path", type=str, default="data/videos/results/", help="Path to save output video (optional)")

    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    tracker_app = VideoBatchTracker()
    tracker_app.run()