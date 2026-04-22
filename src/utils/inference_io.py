from pathlib import Path
import numpy as np

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_image_paths(image_path: Path, image_dir: Path | None = None) -> list[Path]:
    if image_dir is not None:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

        candidates = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not candidates:
            raise ValueError(f"No supported images found in {image_dir}")
        return candidates

    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    return [image_path]




def resolve_path(path_str: str | None, project_root: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path