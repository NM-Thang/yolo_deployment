import cv2
import numpy as np
import tritonclient.http as httpclient
from sort import Sort
from utils.inference_io import preprocess_batch

class VideoBatchTracker:
    def __init__(self, video_path, triton_url="localhost:8000", model_name="yolov8n_trt", batch_size=8):
        self.video_path = video_path
        self.batch_size = batch_size
        self.model_name = model_name
        
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)


    def postprocess_single_frame(self, frame_output, orig_w, orig_h, conf_thresh=0.5):
        """
        Process raw YOLO output for a single frame.
        Scales coordinates back to the original image size.
        """
        # YOLOv8 output shape is [84, 8400], transpose to [8400, 84]
        frame_output = frame_output.transpose()
        
        detections = []
        x_scale = orig_w / 640.0
        y_scale = orig_h / 640.0

        for row in frame_output:
            conf = np.max(row[4:])
            if conf > conf_thresh:
                cx, cy, w, h = row[:4]
                
                # Calculate coordinates and scale to original video resolution
                x1 = int((cx - w/2) * x_scale)
                y1 = int((cy - h/2) * y_scale)
                x2 = int((cx + w/2) * x_scale)
                y2 = int((cy + h/2) * y_scale)
                
                detections.append([x1, y1, x2, y2, conf])
                
        return np.array(detections) if detections else np.empty((0, 5))

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames_buffer = []

        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                frames_buffer.append(frame)
                
            if len(frames_buffer) == self.batch_size or (not ret and len(frames_buffer) > 0):
                
                # 1. Prepare batch input
                input_data = preprocess_batch(frames_buffer)
                inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
                inputs[0].set_data_from_numpy(input_data)
                
                # 2. Triton Inference
                response = self.client.infer(self.model_name, inputs)
                batch_outputs = response.as_numpy("output0") # Shape: [B, 84, 8400]
                
                # 3. Process each frame in the batch SEQUENTIALLY for Tracking
                for i in range(len(frames_buffer)):
                    current_frame = frames_buffer[i]
                    raw_frame_output = batch_outputs[i]
                    
                    # Post-process: Get bounding boxes
                    dets = self.postprocess_single_frame(raw_frame_output, orig_w, orig_h)
                    
                    # Update SORT tracker
                    # Important: This must run sequentially for frame 0, 1, 2...
                    tracked_objects = self.tracker.update(dets)
                    
                    # 4. Visualization
                    for obj in tracked_objects:
                        x1, y1, x2, y2, track_id = obj.astype(int)
                        
                        # Draw bounding box and ID
                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(current_frame, f"ID: {track_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Display the frame
                    cv2.imshow("Video Batch Tracking", current_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                # Clear buffer for the next batch
                frames_buffer = []
            
            if not ret:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure your Triton config.pbtxt has max_batch_size >= 8
    tracker_app = VideoBatchTracker(video_path="sample_video.mp4", batch_size=8)
    tracker_app.run()