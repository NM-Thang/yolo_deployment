from __future__ import annotations
from pathlib import Path
import cv2
from utils.postprocessing import Detection


def draw_detections(image_path: str | Path, detections: list[Detection]) -> cv2.Mat:
	image_path = Path(image_path)
	image = cv2.imread(str(image_path))
	if image is None:
		raise ValueError(f"Cannot read image from {image_path}")

	for det in detections:
		x1, y1, x2, y2 = map(int, det.box)
		label = f"{det.class_name} {det.score:.2f}"
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(
			image,
			label,
			(x1, max(0, y1 - 5)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(0, 255, 0),
			2,
		)

	return image


def show_detections(
	image_path: str | Path,
	detections: list[Detection],
	window_name: str = "detections",
	wait_ms: int = 0,
) -> None:
	image = draw_detections(image_path, detections)
	cv2.imshow(window_name, image)
	cv2.waitKey(wait_ms)
	cv2.destroyAllWindows()