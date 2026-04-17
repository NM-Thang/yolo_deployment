from __future__ import annotations
from pathlib import Path
import cv2
from utils.postprocessing import Detection

def draw_detections(image: str | Path | cv2.Mat, detections: list[Detection]) -> cv2.Mat:
	
	if not isinstance(image, cv2.Mat):
		img_path_str = str(image)
		image = cv2.imread(img_path_str)

	if image is None:
		raise ValueError(f"Cannot read image from {image}")

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
	try:
		cv2.imshow(window_name, image)
		if wait_ms > 0:
			cv2.waitKey(wait_ms)
		else:
			while True:
				key = cv2.waitKey(20) & 0xFF
				if key == ord("q") or key == 27:
					break
				if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
					break
		cv2.destroyAllWindows()
	except cv2.error as exc:
		raise RuntimeError(
			"OpenCV GUI is unavailable on this environment. "
			"Run without visualization or save the output image instead."
		) from exc