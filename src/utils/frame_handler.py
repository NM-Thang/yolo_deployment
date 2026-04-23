import cv2
import numpy as np


def draw_bbox(frame: np.ndarray, dets: list) -> None:
    for det in dets:
        x1, y1, x2, y2 = map(int, det["bbox"])

        if "track_id" not in det: det["track_id"] = ""
        if "label" not in det: det["label"] = "N/a"
        if "score" not in det: det["score"] = ""
        else: det["score"] = f'{det["score"]:.2f}'


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det['label']} - {det['track_id']}: {det['score']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 1)
    return frame

def draw_zone(frame: np.ndarray, corner_points: list) -> None:
    for i in range(0, len(corner_points)-1):
        cv2.line(frame, tuple(int(x) for x in corner_points[i]), tuple(int(x) for x in corner_points[i+1]), (255, 0, 0), 1)
    cv2.line(frame, tuple(int(x) for x in corner_points[-1]), tuple(int(x) for x in corner_points[0]), (255, 0, 0), 1)
    return frame

def display_frame(event: list[str], frame: np.ndarray, delay_ms: int) -> None:
    for e in event:
        print(e)
    cv2.imshow("Zone Event Detector Demo", frame)
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        return