import threading
import queue
import time
from typing import Any

# Define type hints for clarity
FrameInfo = dict[str, Any]

class VideoPipeline:
    def __init__(self) -> None:
        # Use Queues to buffer data between threads.
        # Maxsize is important to prevent memory leaks if Triton is too slow.
        self.frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self.result_queue: queue.Queue = queue.Queue(maxsize=30)
        self.is_running: bool = False

    def camera_reader_thread(self) -> None:
        """Reads frames from the camera (I/O Bound)."""
        frame_id = 0
        while self.is_running:
            # Simulating cv2.VideoCapture.read()
            time.sleep(0.033)  # Roughly 30 FPS
            
            frame_data = {"id": frame_id, "image": "raw_image_data"}
            
            # If queue is full, drop the oldest frame to maintain real-time processing
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame_data)
            frame_id += 1

    def triton_worker_thread(self) -> None:
        """Handles Preprocess -> Triton Request -> Postprocess (Mixed Bound)."""
        while self.is_running:
            try:
                # Wait for a frame with a timeout to allow thread to exit gracefully
                frame_data = self.frame_queue.get(timeout=1.0)
                
                # 1. Preprocess (CPU Bound)
                # processed_img = cv2.resize(frame_data["image"], ...)
                time.sleep(0.01) 
                
                # 2. Triton Inference (I/O Bound)
                # result_tensor = triton_client.infer(processed_img)
                time.sleep(0.05) # Simulating network latency
                
                # 3. Postprocess (CPU Bound)
                # output_frame = draw_boxes(frame_data["image"], result_tensor)
                time.sleep(0.01)
                
                result_data = {"id": frame_data["id"], "final_image": "drawn_image_data"}
                
                if not self.result_queue.full():
                    self.result_queue.put(result_data)
                    
            except queue.Empty:
                continue

    def stream_writer_thread(self) -> None:
        """Pushes the processed frames to the restreamer (I/O Bound)."""
        while self.is_running:
            try:
                result_data = self.result_queue.get(timeout=1.0)
                
                # Simulating pushing to RTSP/WebRTC or writing to cv2.VideoWriter
                time.sleep(0.02)
                print(f"Restreamed Frame ID: {result_data['id']}")
                
            except queue.Empty:
                continue

    def start(self) -> None:
        """Initializes and starts all pipeline threads."""
        self.is_running = True
        
        self.t_reader = threading.Thread(target=self.camera_reader_thread, daemon=True)
        self.t_worker = threading.Thread(target=self.triton_worker_thread, daemon=True)
        self.t_streamer = threading.Thread(target=self.stream_writer_thread, daemon=True)
        
        self.t_reader.start()
        self.t_worker.start()
        self.t_streamer.start()

    def stop(self) -> None:
        """Stops the pipeline gracefully."""
        self.is_running = False
        self.t_reader.join()
        self.t_worker.join()
        self.t_streamer.join()

if __name__ == "__main__":
    pipeline = VideoPipeline()
    print("Starting Video AI Pipeline...")
    pipeline.start()
    
    try:
        # Let the pipeline run for a while
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping pipeline...")
        pipeline.stop()
        print("Pipeline stopped.")