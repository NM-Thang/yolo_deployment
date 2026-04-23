import numpy as np
from utils.triton_client import TritonClient
from utils.img_processing import  img_preprocessing, batch_preprocessing
from utils.postprocessing import postprocess_yolo, Detection
from utils.fire_processing import Detector_Processing
import time




class DetectorAdapter:
    def __init__(self, triton_client: TritonClient = None, agrs: dict = None):
        self.triton_client = triton_client if triton_client else TritonClient()
        self.det_process = Detector_Processing(height=640, width=640)
        

    def detect_intrusion(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect objects in the input image and return a list of detections.
        Each detection is a tuple of (x1, y1, x2, y2, confidence, class_name).
        """
        
        t0 = time.time()
        input = np.expand_dims(img_preprocessing(frame), axis=0)
        t1 = time.time()
        print(f"Preprocessing time: {t1 - t0:.4f} seconds")
        t2 = time.time()
        data = self.triton_client.run(input=input, model_name="yolov8_trt", model_version="2", input_name="input")
        t3 = time.time()
        print(f"Inference time: {t3 - t2:.4f} seconds")
        t4 = time.time()

        detections = postprocess_yolo(
            raw_output=data[0],
            input_size=(input.shape[3], input.shape[2]),
            orig_size=(frame.shape[1], frame.shape[0]),
            confidence_threshold=0.5,
            iou_threshold=0.45,
            top_k=20,
        )
        t5 = time.time()
        print(f"Postprocessing time: {t5 - t4:.4f} seconds")

        return detections
    
    def detect_fire_smoke(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect fire and smoke in the input image and return a list of detections.
        Each detection is a tuple of (x1, y1, x2, y2, confidence, class_name).
        """

        tensor, draw= self.det_process.det_preprocessing(frame)
        data = self.triton_client.run(input=tensor, model_name="firesmoke_detection")
        
        detections = self.det_process.det_postprocessing(frame, response=data, conf_threshold=0.25)

        return detections
        

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """
        Detect objects in a batch of input images and return a list of detections for each image.
        Each detection is a tuple of (x1, y1, x2, y2, confidence, class_name).
        """

        input = batch_preprocessing(images)
        ouput = self.triton_client.run(batch_input=input)

        return [self.detector.detect(image) for image in images]
        