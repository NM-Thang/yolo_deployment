import cv2
import numpy as np
from numpy import ndarray
from typing import Tuple, Union, List
import torch
# from pkg_resources import split_sections
from torchvision import ops


# Rescale and make bounding boxes
def letterbox(im: ndarray, new_shape: Union[Tuple, List] = (640, 640), color: Union[Tuple, List] = (114, 114, 114)) -> Tuple[ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while preserving aspect ratio
    
    Args:
        im: Input image
        new_shape: Target size (width, height)
        color: Border color for padding
        
    Returns:
        tuple: (processed_image, scale_ratio, (padding_width, padding_height))
    """
    # Get current image dimensions [height, width]
    shape = im.shape[:2]
    
    # Handle single integer input
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0]) 
    # Calculate new unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw /= 2
    dh /= 2   
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Add border padding
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=color
    )
    
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    """
    Convert image to blob format for inference
    
    Args:
        im: Input image in HWC format
        return_seg: Whether to return segmentation mask
        
    Returns:
        Image blob in NCHW format normalized to [0,1]
        Optional segmentation mask if return_seg=True
    """
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
        
    # Convert HWC to NCHW format (batch, channels, height, width)
    im = im.transpose([2, 0, 1])  # HWC to CHW
    im = im[np.newaxis, ...]      # CHW to NCHW
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im

def det_postprocess(data, ratio, dwdh, conf_threshold=0.5, iou_threshold=0.7):
    """
    Post-process detection results and apply NMS
    
    Args:
        data: Detection data tensor with shape [1, 6, N]
        ratio: Scale ratio from preprocessing
        dwdh: Padding values (dw, dh) from letterboxing
        conf_threshold: Confidence threshold for filtering detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Tuple of (bboxes, scores, labels) after NMS
    """
    data = data[0]  # Remove batch dim, change from (1,6,N) to (6,N)
    cx, cy, w, h, score, cls = data
    
    # Transform center coordinates and dimensions to corner coordinates
    # (cx, cy, w, h) → (x1, y1, x2, y2)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    bboxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    # Scale bounding boxes to original image size
    bboxes[:, [0, 2]] -= dwdh[0]  # Remove x padding
    bboxes[:, [1, 3]] -= dwdh[1]  # Remove y padding
    bboxes[:, [0, 2]] /= ratio    # Scale x coordinates
    bboxes[:, [1, 3]] /= ratio    # Scale y coordinates
    
    # Filter by confidence threshold
    keep = score > conf_threshold
    bboxes = bboxes[keep]
    scores = score[keep]
    labels = cls[keep]  # Use actual class predictions from model
    
    # Return empty tensors if no detections remain after filtering
    if len(bboxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))
    
    # Apply Non-Maximum Suppression
    keep_idx = ops.nms(bboxes, scores, iou_threshold)

    return bboxes[keep_idx], scores[keep_idx], labels[keep_idx]


def transform_output(output_tensor, confidence_threshold=0.25):
    """
    Transform and filter model output tensor for YOLOv8 multi-class detection
    Args:
        output_tensor: Model output tensor with shape [batch_size, 6, num_detections]
                      Format: [cx, cy, w, h, class0_conf, class1_conf] for 2-class model
        confidence_threshold: Minimum confidence score to keep detection
    
    Returns:
        Filtered detection tensor with shape [batch_size, 6, filtered_detections]
        Format: [cx, cy, w, h, max_confidence, class_id]
    """
    # print(f"DEBUG: Input tensor shape: {output_tensor.shape}")
    
    # Extract components from output tensor
    bboxes = output_tensor[:, 0:4, :]  # [batch, 4, num_detections] - coordinates (cx, cy, w, h)
    
    # For YOLOv8 multi-class, channels 4 and 5 are class confidences
    if output_tensor.shape[1] == 6:  # Multi-class format [cx, cy, w, h, class0_conf, class1_conf]
        class_confs = output_tensor[:, 4:6, :]  # [batch, 2, num_detections] - class confidences
        
        # Find the class with maximum confidence for each detection
        max_conf_indices = np.argmax(class_confs[0], axis=0)  # Shape: [num_detections]
        max_confs = np.max(class_confs[0], axis=0)  # Shape: [num_detections]
        
        # Reshape for consistency
        scores = max_confs.reshape(1, 1, -1)  # [batch, 1, num_detections]
        predicted_class = max_conf_indices.reshape(1, 1, -1).astype(np.float32)  # [batch, 1, num_detections]
        
    elif output_tensor.shape[1] == 5:  # Single confidence format [cx, cy, w, h, conf]
        scores = output_tensor[:, 4:5, :]  # [batch, 1, num_detections]
        predicted_class = np.zeros((output_tensor.shape[0], 1, output_tensor.shape[2]), dtype=np.float32)
        # print(f"DEBUG: Single confidence format, defaulting to class 0")
    else:
        raise ValueError(f"Unexpected output tensor shape: {output_tensor.shape}")
    
    if scores.size > 0:
        unique_classes = np.unique(predicted_class[0, 0, :])
        # print(f"DEBUG: Predicted classes: {unique_classes}")
    
    # Only keep detections with confidence above threshold
    valid_detections = np.where(scores[0, 0, :] > confidence_threshold)[0]
    # print(f"DEBUG: Total detections before confidence filter: {scores.shape[2]}")
    # print(f"DEBUG: Detections after confidence filter (>{confidence_threshold}): {len(valid_detections)}")
    
    if len(valid_detections) == 0:
        # print(f"DEBUG: No detections passed confidence threshold {confidence_threshold}")
        return np.zeros((output_tensor.shape[0], 6, 0), dtype=np.float32)
    
    # Filter to keep only valid detections
    filtered_bboxes = bboxes[:, :, valid_detections]
    filtered_scores = scores[:, :, valid_detections]
    filtered_classes = predicted_class[:, :, valid_detections]
    
    # Debug: Show what classes survived filtering
    # if filtered_classes.size > 0:
    #     surviving_classes = np.unique(filtered_classes[0, 0, :])
    #     print(f"DEBUG: Classes after confidence filtering: {surviving_classes}")
        
    #     # Show per-class detection counts
    #     class_counts = {}
    #     for cls in surviving_classes:
    #         count = np.sum(filtered_classes[0, 0, :] == cls)
    #         class_name_str = "Fire" if cls == 0 else "Smoke" if cls == 1 else f"Class{int(cls)}"
    #         class_counts[class_name_str] = count
    #     print(f"DEBUG: Per-class detections: {class_counts}")
    
    # Concatenate to create final output tensor
    # Shape: [batch_size, 6, filtered_detections] where 6 = [cx, cy, w, h, max_confidence, class_id]
    new_output = np.concatenate([filtered_bboxes, filtered_scores, filtered_classes], axis=1)
    
    return new_output


class Detector_Processing:
    """
    Processing class for object detection workflow
    Handles pre-processing and post-processing for detection models
    """

    def __init__(self, height, width):
        """
        Initialize detector with target size
        
        Args:
            height: Target height for model input
            width: Target width for model input
        """
        self.output_size = (height, width)
        self.ratio = None
        self.dwdh = None

    def det_preprocessing(self, image: ndarray):
        """
        Preprocess image for detection model
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            tuple: (tensor, original_image_copy)
                - tensor: Processed image tensor ready for model input
                - original_image_copy: Copy of original image for visualization
        """
        # Make a copy of the original image for drawing results later
        draw = image.copy()
        letterbox
        # Resize and pad image to target size while maintaining aspect ratio
        img, self.ratio, self.dwdh = letterbox(image, self.output_size)
        
        # Convert to RGB and create normalized blob
        tensor = blob(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = torch.asarray(tensor)

        return tensor, draw

    def det_postprocessing(self, image: ndarray, response, conf_threshold=0.25):
        """
        Process model output to get final detection results
        
        Args:
            image: Original input image
            response: Raw model output from inference
            conf_threshold: Confidence threshold for filtering detections
            
        Returns:
            list: List of detection dictionaries with keys "bbox", "score", and "label"
        """
        # Transform raw model output
        data = transform_output(response, confidence_threshold=conf_threshold)
        
        # Handle empty detection case
        if data.size == 0 or data.shape[2] == 0:
            return []
            
        # Convert to torch tensor and apply post-processing
        data = torch.from_numpy(data).float()
        bboxes, scores, labels = det_postprocess(
            data, 
            ratio=self.ratio, 
            dwdh=self.dwdh
        )
        detections = []
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox.int().tolist()
            confidence = score.item()
            class_id = int(label.item())
            class_name = "Fire" if class_id == 0 else "Smoke" if class_id == 1 else f"Class{class_id}"  
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "score": confidence,
                "label": class_name
            })
        return detections

        # # Convert tensors to lists for easier handling
        # filtered_bboxes = [bbox for bbox in bboxes]
        # filtered_scores = [score for score in scores]
        # filtered_labels = [label for label in labels]

        # detections =[]
        # for i in range(len(filtered_bboxes)):
        #     detections.append({
        #         "bbox": filtered_bboxes[i],
        #         "score": filtered_scores[i],
        #         "label": filtered_labels[i]
        #     })

        return detections