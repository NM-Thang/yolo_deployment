import argparse

import cv2
import numpy as np

import tritonclient.http as httpclient
from utils.inference_io import preprocess_batch
from utils.image_processing import preprocess_image_single
from utils.postprocessing import postprocess_yolo
from triton_client import run


from sort import Sort

class VideoBatchTracker:
    def __init__(self, video_path, host="localhost", protocol="grpc", model_name="yolov8n_trt", model_version=None, batch_size=8):
        self.video_path = video_path
        self.host = host
        self.protocol = protocol
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_version = model_version

        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    # def preprocess_batch(self, frames):
    #     """
    #     Preprocess a list of frames to a single batch tensor for YOLOv8.
    #     Input: list of BGR images.
    #     Output: numpy array of shape [Batch, Channels, Height, Width].
    #     """
    #     batch_images = []
    #     for frame in frames:
    #         img = cv2.resize(frame, (640, 640))
    #         img = img.astype(np.float32) / 255.0
    #         img = np.transpose(img, (2, 0, 1))  # Convert HWC to CHW
    #         batch_images.append(img)
            
    #     return np.array(batch_images) # Shape: [B, 3, 640, 640]

    # def postprocess_single_frame(self, frame_output, orig_w, orig_h, conf_thresh=0.5):
    #     """
    #     Process raw YOLO output for a single frame.
    #     Scales coordinates back to the original image size.
    #     """
    #     # YOLOv8 output shape is [84, 8400], transpose to [8400, 84]
    #     frame_output = frame_output.transpose()
        
    #     detections = []
    #     x_scale = orig_w / 640.0
    #     y_scale = orig_h / 640.0

    #     for row in frame_output:
    #         conf = np.max(row[4:])
    #         if conf > conf_thresh:
    #             cx, cy, w, h = row[:4]
                
    #             # Calculate coordinates and scale to original video resolution
    #             x1 = int((cx - w/2) * x_scale)
    #             y1 = int((cy - h/2) * y_scale)
    #             x2 = int((cx + w/2) * x_scale)
    #             y2 = int((cy + h/2) * y_scale)
                
    #             detections.append([x1, y1, x2, y2, conf])
                
    #    return np.array(detections) if detections else np.empty((0, 5))



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
                
                batch_input = preprocess_batch(frames_buffer, orig_size) 
                batch_outputs = run(batch_input=batch_input, host=self.host, protocol=self.protocol, model_name=self.model_name, model_version=self.model_version)
                
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

                    dets = np.array([[det.box[0], det.box[1], det.box[2], det.box[3], det.score] for det in detections])
                    tracked_objects = self.tracker.update(dets) 
                    for obj in tracked_objects:
                        x1, y1, x2, y2, track_id = obj.astype(int)
                        
                        # Draw bounding box and ID
                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(current_frame, f"ID: {track_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # cv2.imshow("Video Batch Tracking", current_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                frames_buffer = []
            
            if not ret:
                break

        cap.release()
        cv2.destroyAllWindows()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton Batch Tracking Example")
    parser.add_argument("--video-path", type=str, default="data/videos/person-bicycle-car-detection.mp4", help="Path to input video")
    parser.add_argument("--model-name", type=str, default="yolov8_trt", help="Triton model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--host", type=str, default="localhost", help="Triton server URL")
    parser.add_argument("--protocol", type=str, default="grpc", choices=["grpc", "http"], help="Triton protocol")
    parser.add_argument("--model-version", type=str, default=None, help="Triton model version")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    tracker_app = VideoBatchTracker(
        video_path=args.video_path,
        host=args.host,
        protocol=args.protocol,
        model_name=args.model_name,
        model_version=args.model_version,
        batch_size=args.batch_size
    )
    tracker_app.run()