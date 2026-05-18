from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class DepthPrediction:
    depth_maps: Sequence[np.ndarray]
    confidence_maps: Optional[Sequence[np.ndarray]] = None
    intrinsics: Optional[np.ndarray] = None
    extrinsics: Optional[np.ndarray] = None


@dataclass
class DepthCacheEntry:
    depth_map: np.ndarray
    confidence_map: Optional[np.ndarray] = None
    intrinsic: Optional[np.ndarray] = None
    extrinsic: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None
    rgb_size: Optional[Tuple[int, int]] = None


class DepthAnything3Estimator:
    def __init__(
        self,
        model_name: str = "depth-anything/DA3-GIANT-1.1",
        device: str = "cuda",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        debug_dir: Optional[str | Path] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self._model = None
        self._frame_cache: Dict[str, DepthCacheEntry] = {}

    def predict(self, image_paths: Sequence[Path | str]) -> DepthPrediction:
        if len(image_paths) == 0:
            raise ValueError("DepthAnything3Estimator.predict requires at least one image path.")

        resolved_paths = [Path(path).expanduser().resolve() for path in image_paths]
        cache_keys = [str(path) for path in resolved_paths]
        if all(cache_key in self._frame_cache for cache_key in cache_keys):
            LOGGER.info("DA3 cache hit for %d frame(s): %s", len(cache_keys), cache_keys)
            return self._build_prediction_from_cache(cache_keys)

        missing_items = [
            (resolved_path, cache_key)
            for resolved_path, cache_key in zip(resolved_paths, cache_keys)
            if cache_key not in self._frame_cache
        ]
        missing_paths = [item[0] for item in missing_items]
        missing_keys = [item[1] for item in missing_items]
        if missing_keys:
            LOGGER.info("DA3 cache miss for %d frame(s): %s", len(missing_keys), missing_keys)

        model = self._load_model()
        rgb_sizes = []
        for image_path in missing_paths:
            with Image.open(image_path) as image:
                rgb_sizes.append(image.size)
                # LOGGER.info("DA3 input image: path=%s rgb_size=%s", image_path, image.size)

        prediction = model.inference(
            image=[str(path) for path in missing_paths],
            process_res=self.process_res,
            process_res_method=self.process_res_method,
            export_dir=None,
        )

        processed_images = getattr(prediction, "processed_images", None)
        # if processed_images is not None:
        #     LOGGER.info("DA3 processed_images shape=%s", getattr(processed_images, "shape", None))
        # LOGGER.info("DA3 depth shape=%s", getattr(prediction.depth, "shape", None))

        confidence_maps = getattr(prediction, "conf", None)
        # if confidence_maps is not None:
        #     LOGGER.info("DA3 confidence shape=%s", getattr(confidence_maps, "shape", None))
        # LOGGER.info(
        #     "DA3 camera shapes: intrinsics=%s extrinsics=%s",
        #     getattr(getattr(prediction, "intrinsics", None), "shape", None),
        #     getattr(getattr(prediction, "extrinsics", None), "shape", None),
        # )

        depth_maps = list(np.asarray(prediction.depth))
        confidence_map_list = list(np.asarray(confidence_maps)) if confidence_maps is not None else [None] * len(depth_maps)
        intrinsics = getattr(prediction, "intrinsics", None)
        extrinsics = getattr(prediction, "extrinsics", None)
        processed_image_list = list(np.asarray(processed_images)) if processed_images is not None else [None] * len(depth_maps)

        for index, cache_key in enumerate(missing_keys):
            confidence_map = confidence_map_list[index] if index < len(confidence_map_list) else None
            intrinsic = intrinsics[index] if intrinsics is not None and index < len(intrinsics) else None
            extrinsic = extrinsics[index] if extrinsics is not None and index < len(extrinsics) else None
            processed_image = processed_image_list[index] if index < len(processed_image_list) else None
            self._frame_cache[cache_key] = DepthCacheEntry(
                depth_map=depth_maps[index],
                confidence_map=confidence_map,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                processed_image=processed_image,
                rgb_size=rgb_sizes[index] if index < len(rgb_sizes) else None,
            )
            self.save_depth_visualization(
                depth_maps[index],
                missing_paths[index],
                suffix=f"raw_depth_{depth_maps[index].shape[0]}x{depth_maps[index].shape[1]}",
            )

        return self._build_prediction_from_cache(cache_keys)

    def save_depth_visualization(self, depth_map: np.ndarray, image_path: Path | str, suffix: str) -> Optional[Path]:
        if self.debug_dir is None:
            return None

        output_dir = self.debug_dir / "depth_maps"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image_path)
        output_path = output_dir / f"{image_path.parent.name}__{image_path.stem}__{suffix}.png"
        _save_depth_png(depth_map, output_path)
        LOGGER.info("Saved depth visualization: %s", output_path)
        return output_path

    def _build_prediction_from_cache(self, cache_keys: Sequence[str]) -> DepthPrediction:
        entries = [self._frame_cache[cache_key] for cache_key in cache_keys]
        confidence_maps = [entry.confidence_map for entry in entries]
        intrinsics = [entry.intrinsic for entry in entries]
        extrinsics = [entry.extrinsic for entry in entries]
        return DepthPrediction(
            depth_maps=[entry.depth_map for entry in entries],
            confidence_maps=confidence_maps if any(item is not None for item in confidence_maps) else None,
            intrinsics=np.stack(intrinsics) if all(item is not None for item in intrinsics) else None,
            extrinsics=np.stack(extrinsics) if all(item is not None for item in extrinsics) else None,
        )

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "Depth Anything 3 is not installed. Install the official package/repo before using "
                "--use-depth-anything."
            ) from exc

        LOGGER.info("Loading Depth Anything 3 checkpoint=%s device=%s", self.model_name, self.device)
        self._model = DepthAnything3.from_pretrained(self.model_name).to(self.device)
        return self._model


def _save_depth_png(depth_map: np.ndarray, output_path: Path) -> None:
    depth = np.asarray(depth_map, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        image = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [2.0, 98.0])
        if high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
        if high <= low:
            image = np.zeros(depth.shape, dtype=np.uint8)
        else:
            normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
            image = (normalized * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(output_path)
