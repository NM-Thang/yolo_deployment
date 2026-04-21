import numpy as np
from triton_client import TritonClient
from async_trition_client import AsyncTritonClient
from dataclasses import dataclass
from utils.inference_io import preprocess_batch
from utils. image_processing import  preprocess_image_single
from utils.postprocessing import postprocess_yolo, Detection




class DetectorAdapter:
    def __init__(self, triton_client: TritonClient | AsyncTritonClient = None):
        self.triton_client = triton_client if triton_client else TritonClient()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect objects in the input image and return a list of detections.
        Each detection is a tuple of (x1, y1, x2, y2, confidence, class_name).
        """

        input = preprocess_image_single(image)
        ouput = self.triton_client.run(batch_input=input)


        detections = postprocess_yolo(
            raw_output=ouput[0],
            input_size=(input.shape[2], input.shape[1]),
            orig_size=(image.shape[1], image.shape[0]),
            confidence_threshold=0.5,
            iou_threshold=0.45,
            top_k=20,
        )
        return detections
    
    async def infer_async(self, frm: np.ndarray, fid: int):
        
        input = preprocess_image_single(frm)
        ouput = await self.triton_client.async_run(batch_input=input)
        
        detections = postprocess_yolo(
            raw_output=ouput[0],
            input_size=(input.shape[2], input.shape[1]),
            orig_size=(frm.shape[1], frm.shape[0]),
            confidence_threshold=0.5,
            iou_threshold=0.45,
            top_k=20,
        )        

        return {"detections": detections, "frame_id": fid}
        

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """
        Detect objects in a batch of input images and return a list of detections for each image.
        Each detection is a tuple of (x1, y1, x2, y2, confidence, class_name).
        """

        input = preprocess_batch(images)
        ouput = self.triton_client.run(batch_input=input)

        return [self.detector.detect(image) for image in images]
        