from zone_event_detector import Zone, ZoneEventDetector
from detector_adapter import DetectorAdapter
from triton_client import TritonClient
from async_trition_client import AsyncTritonClient
import cv2
import argparse
import asyncio
import queue
import concurrent.futures
import time
import numpy as np


class VideoPipeline:
    def __init__(self, zone_event_detector: ZoneEventDetector, args) -> None:
        self.ZoneEventDetector = zone_event_detector
        self.args = args
        self.frame_queue = queue.Queue(maxsize=60)
        self.result_queue = queue.Queue(maxsize=200)
        self.is_running = False
        cap = cv2.VideoCapture(self.args.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.delay_ms = int(1000 / self.fps) if self.fps > 0 else 30

    def reader_thread(self) -> None:
        """Reads frames from the camera and pushes them to the frame queue (I/O Bound)."""
        cap = cv2.VideoCapture(self.args.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.delay_ms = int(1000 / self.fps) if self.fps > 0 else 30
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame_data = {"image": frame, "fid": int(
                cap.get(cv2.CAP_PROP_POS_FRAMES))}
            if not self.frame_queue.full():
                self.frame_queue.put(frame_data)
        cap.release()

    def worker_thread(self) -> None:
        """Handles Preprocess -> Triton Request -> Postprocess (Mixed Bound)."""
        while self.is_running:
            try:
                frame_data = self.frame_queue.get()
                frame = frame_data["image"]
                fid = frame_data["fid"]
                result_data = self.ZoneEventDetector.process_frame(frame)
                event = result_data["event"]
                tracked_objects = result_data["tracked_objects"]

                out = draw_frame(frame, tracked_objects,
                                 self.ZoneEventDetector.zone.coordinates)
                if not self.result_queue.full():
                    self.result_queue.put(
                        {"frame": out, "event": event, "fid": fid})

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker thread: {e}")

    def restreamer_thread(self) -> None:
        """Consumes results from the result queue and displays them (I/O Bound)."""
        while self.is_running:

            while self.result_queue.qsize() < int(self.fps):
                time.sleep(0.01)
            try:
                result_data = self.result_queue.get()
                frame, event, fid = result_data["frame"], result_data["event"], result_data["fid"]

                for e in event:
                    print(e)

                cv2.imshow("Zone Event Detector Demo", frame)
                if cv2.waitKey(self.delay_ms) & 0xFF == ord('q'):
                    self.is_running = False
                    break

            except queue.Empty:
                continue


def demo_multithread() -> None:
    """Consumes results from the result queue and displays them (I/O Bound)."""
    args = parse_args()
    triton_client = TritonClient()
    adapter = DetectorAdapter(triton_client)
    zone = Zone(name="Entrance", coordinates=[
                (250, 150), (450, 150), (450, 350), (250, 350)])
    zone_event_detector = ZoneEventDetector(
        zone=zone, detector_adapter=adapter, semaphore_limit=2)

    pipeline = VideoPipeline(zone_event_detector, args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        pipeline.is_running = True
        executor.submit(pipeline.reader_thread)
        executor.submit(pipeline.worker_thread)
        executor.submit(pipeline.restreamer_thread)

        try:
            while pipeline.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pipeline.is_running = False


def draw_frame(frame: np.ndarray, tracked_objects: list, corner_points: list) -> None:
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for i in (0, len(corner_points)-2, 1):
        cv2.line(frame, corner_points[i], corner_points[i+1], (255, 0, 0), 2)
    cv2.line(frame, corner_points[-1], corner_points[0], (255, 0, 0), 2)
    return frame


def display_frame(event: list[str], frame: np.ndarray, delay_ms: int) -> None:
    for e in event:
        print(e)
    cv2.imshow("Zone Event Detector Demo", frame)
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        return


async def main():
    args = parse_args()

    triton_client = await AsyncTritonClient.create()
    adapter = DetectorAdapter(triton_client)
    zone = Zone(name="Entrance", coordinates=[
                (250, 150), (450, 150), (450, 350), (250, 350)])
    event_detector = ZoneEventDetector(
        zone=zone, detector_adapter=adapter, semaphore_limit=2)

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    delay_ms = int(1000 / fps) if fps > 0 else 30

    frame_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)
        results = await event_detector.push_frame_async(frame, fid=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        if results is None:
            continue

        frame = draw_frame(frame, results["tracked_objects"], zone.coordinates)
        display_frame(results["event"], frame, delay_ms)

    cap.release()

    async for result in event_detector.flush_remaining_tasks():
        frame = draw_frame(frame, result["tracked_objects"], zone.coordinates)
        display_frame(result["event"], frame, delay_ms)
    return

def reader_worker_thread(video_path: str, zone: Zone, result_queue: queue.Queue, is_running: bool) -> None:

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        triton_client = loop.run_until_complete(AsyncTritonClient.create())
        adapter = DetectorAdapter(triton_client)
        event_detector = ZoneEventDetector(zone=zone, detector_adapter=adapter, semaphore_limit=32)

        cap = cv2.VideoCapture(video_path)
        frames = []
        while is_running:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            results = loop.run_until_complete(event_detector.push_frame_async(frame, fid=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1))
            if results is None:
                continue
            frame = frames[results["frame_id"]]
            result_queue.put({**results, "frame": frame}, block=True)

        for res in event_detector.flush_remaining_tasks():
            result_queue.put(res)
        cap.release()
    except Exception as e:
        print(f"Error in reader worker thread: {e}")


def restreamer_thread(zone: Zone, result_queue: queue.Queue, delay_ms: int, is_running: bool) -> None:
    print("Restreamer thread started, waiting for results...")
    while is_running:
        if result_queue.qsize() < 24:
            time.sleep(0.01)
            continue
        print("restreamer_thread got results, displaying...")
        print(f"Results in queue: {result_queue.qsize()}")

        try:
            result = result_queue.get()
            event, tracked_objects, frame, fid = result["event"], result["tracked_objects"], result["frame"], result["frame_id"]

            drawn_frame = draw_frame(frame, tracked_objects, zone.coordinates)
            display_frame(event, drawn_frame, delay_ms)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in restreamer thread: {e}")


async def main_v2():
    args = parse_args()

    zone = Zone(name="Entrance", coordinates=[
                (250, 150), (450, 150), (450, 350), (250, 350)])

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps) if fps > 0 else 30

    result_queue = queue.Queue(maxsize=200)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        is_running = True
        executor.submit(reader_worker_thread, args.video_path, zone, result_queue, is_running)
        executor.submit(restreamer_thread, zone, result_queue, delay_ms//10, is_running)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zone Event Detector Demo")
    parser.add_argument("--video-path", type=str,
                        default="data/videos/people-detection.mp4", help="Path to input video")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    # demo_multithread()
    asyncio.run(main())
    # asyncio.run(main_v2())