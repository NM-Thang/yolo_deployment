import numpy as np
from zope import event
from utils.detector_adapter import DetectorAdapter
from utils.sort import Sort
import cv2
import asyncio
from collections import deque 
import time
from shapely.geometry import Polygon


class Zone:
    def __init__(self, name: str = "", coordinates: list[tuple[int, int]] = None):
        self.name = name
        self.coordinates = coordinates if coordinates else []

    def add_coordinate(self, w: int, h: int):
        self.coordinates.append((w, h))

    def update_coordinate(self, coordinate: tuple[int, int], index: int):
        self.coordinates[index] = coordinate



    def contains_point(self, w1: int, h1: int, w2: int, h2: int) -> bool:
        polygon_zone = Polygon(self.coordinates)
        polygon_object = Polygon([(w1, h1), (w2, h1), (w2, h2), (w1, h2)])
        return polygon_zone.intersects(polygon_object)



       
class ZoneEventDetector:
    def __init__(self):
        self.objects_in_zone = set()
        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
        self.pending_tasks = deque()



    def process_frame(self, image: np.ndarray) -> tuple[set, set]:
        detections = self.detector_adapter.detect(image)
        
        return self._process_tracking_and_zones({"detections": detections, "frame_id": 0})

        
    
    def _process_tracking_and_zones(self, result: dict) -> dict:
        detections = result["detections"]
        frame_id = result["frame_id"]

        if detections:
            dets = np.array(
                [[det.box[0], det.box[1], det.box[2], det.box[3], det.score]
                    for det in detections if det.class_name == "person"],
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
    