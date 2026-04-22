import cv2
import numpy as np


def draw_bbox(frame: np.ndarray, tracked_objects: list) -> None:
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.astype(int)

        if "track_id" in obj:
            track_id = int(obj[4])
        else:
            track_id = ""
        
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

def draw_zone(frame: np.ndarray, corner_points: list) -> None:
    for i in range(0, len(corner_points)-1):
        cv2.line(frame, corner_points[i], corner_points[i+1], (255, 0, 0), 2)
    cv2.line(frame, corner_points[-1], corner_points[0], (255, 0, 0), 2)
    return frame

def display_frame(event: list[str], frame: np.ndarray, delay_ms: int) -> None:
    for e in event:
        print(e)
    cv2.imshow("Zone Event Detector Demo", frame)
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        return