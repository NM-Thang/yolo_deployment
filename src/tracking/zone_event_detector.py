import numpy as np
from zope import event
from detector_adapter import DetectorAdapter
from sort import Sort
import cv2
import asyncio
from collections import deque 
import time

class Zone:
    def __init__(self, name: str = "", coordinates: list[tuple[int, int]] = None):
        self.name = name
        self.coordinates = coordinates if coordinates else []
        self.mask = None

        self.update_mask()

    def add_coordinate(self, w: int, h: int):
        self.coordinates.append((w, h))

    def update_coordinate(self, coordinate: tuple[int, int], index: int):
        self.coordinates[index] = coordinate

    def max_coordinates(self) -> tuple[int, int]:
        w_coords, h_coords = zip(*self.coordinates)
        return (max(h_coords), max(w_coords))

    def update_mask(self):
        self.mask = np.zeros(self.max_coordinates(), dtype=np.uint8)
        cv2.fillPoly(self.mask, [np.array(self.coordinates)], 1)

    def contains_point(self, w1: int, h1: int, w2: int, h2: int) -> bool:
        h1c = max(0, h1)
        w1c = max(0, w1)
        h2c = min(self.mask.shape[0], h2)
        w2c = min(self.mask.shape[1], w2)
        if h1c >= h2c or w1c >= w2c:
            return False
        region = self.mask[h1c:h2c, w1c:w2c]
        return np.any(region)
       
class ZoneEventDetector:
    def __init__(self, zone: Zone, target_class: str = "person", detector_adapter: DetectorAdapter = None, semaphore_limit: int = 32):
        self.zone = zone
        self.objects_in_zone = set()
        self.target_class = target_class
        self.detector_adapter = detector_adapter

        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.pending_tasks = deque()



    def process_frame(self, image: np.ndarray) -> tuple[set, set]:
        detections = self.detector_adapter.detect(image)
        
        return self._process_tracking_and_zones({"detections": detections, "frame_id": 0})
            
    def handle_event(self, results: dict) -> None:
        event = results["event"]
        for e in event:
            print(e)
            

    async def push_frame_async(self, frm: np.ndarray, fid: int):
        async def _infer_task(frm, fid):
            async with self.semaphore:
                return await self.detector_adapter.infer_async(frm, fid)
            
        task = asyncio.create_task(_infer_task(frm, fid))
        self.pending_tasks.append(task)

        if len(self.pending_tasks) >= self.semaphore._value:
            oldest_task = self.pending_tasks.popleft()
            result = await oldest_task
            
            return self._process_tracking_and_zones(result)
        
        return None
    
        
    
    def _process_tracking_and_zones(self, result: dict) -> dict:
        detections = result["detections"]
        frame_id = result["frame_id"]

        if detections:
            dets = np.array(
                [[det.box[0], det.box[1], det.box[2], det.box[3], det.score]
                    for det in detections if det.class_name == self.target_class],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 5), dtype=np.float32)
        if len(dets) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
            

        tracked_objects = self.tracker.update(dets)
        event = []

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            is_in_zone = self.zone.contains_point(x1, y1, x2, y2)

            if is_in_zone and track_id not in self.objects_in_zone:
                self.objects_in_zone.add(track_id)
                event.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {self.target_class} {track_id} entered the zone.")
            elif not is_in_zone and track_id in self.objects_in_zone:
                self.objects_in_zone.discard(track_id)
                event.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {self.target_class} {track_id} left the zone.")
        
        return {"event": event, "tracked_objects": tracked_objects, "frame_id": frame_id}
    
    async def flush_remaining_tasks(self):
        while self.pending_tasks:
            oldest_task = self.pending_tasks.popleft()
            result = await oldest_task
            yield self._process_tracking_and_zones(result)