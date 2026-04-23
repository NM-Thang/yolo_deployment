import numpy as np
import time
from shapely.geometry import Polygon 
from typing import Any, Dict

from core.base_strategy import AIStrategy
from utils.detector_adapter import DetectorAdapter
from utils.sort import Sort



class IntrusionDetection(AIStrategy):

    def __init__(self, adapter: DetectorAdapter, config: Dict[str, Any] = None):
        self.polygon_zone = Polygon(config.get("zone_coordinates", [])) if config else None
        self.adapter = adapter
        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
    
    def process_frame(self, frame, *args, **kwargs):
        detections = self.adapter.detect_intrusion(frame)
        
        if detections:
            dets = np.array(
                [[det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.score]
                    for det in detections if det.class_name == "person"],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 5), dtype=np.float32)
        if len(dets) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        
        print(f"Detections for current frame: {dets}")  # Debug statement to check detections
        tracked_objects = self.tracker.update(dets)
        event = []
        dets_results = []


        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)

            bbox = (x1, y1, x2, y2)

            polygon_object = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            is_in_zone = self.polygon_zone.intersects(polygon_object)

            if is_in_zone and track_id not in self.objects_in_zone:
                self.objects_in_zone.add(track_id)
                event.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {self.target_class} {track_id} entered the zone.")
            elif not is_in_zone and track_id in self.objects_in_zone:
                self.objects_in_zone.discard(track_id)
                event.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {self.target_class} {track_id} left the zone.")

            
        return {"event": event, "detections": dets_results, "corner_points": list(self.polygon_zone.exterior.coords)}