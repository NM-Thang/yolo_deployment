import numpy as np
import time
from shapely.geometry import Polygon 
from typing import Any, Dict

from core.base_strategy import AIStrategy
from utils.detector_adapter import DetectorAdapter



class FireSmokeDetection(AIStrategy):
    """Detect fire and smoke in video frames"""
    def __init__(self, adapter: DetectorAdapter, config: Dict[str, Any] = None):
        self.adapter = adapter


    def process_frame(self, frame):

        dets = self.adapter.detect_fire_smoke(frame)
        event =[]

        return {"event": event, "detections": dets}
    
    # def is_moving