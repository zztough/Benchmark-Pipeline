from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    source_name = "depth_anything_3"

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
            # self.save_depth_visualization(
            #     depth_maps[index],
            #     missing_paths[index],
            #     suffix=f"raw_depth_{depth_maps[index].shape[0]}x{depth_maps[index].shape[1]}",
            # )

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


@dataclass
class _PreprocessMetadata:
    original_size: Tuple[int, int]
    crop_box: Tuple[int, int, int, int]
    target_shape: Tuple[int, int]
    common_shape: Tuple[int, int]
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


class VGGTOmegaEstimator:
    source_name = "vggt_omega"
    use_span_context = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        image_resolution: int = 512,
        preprocess_mode: str = "balanced",
        patch_size: int = 16,
        max_context_frames: int = 10,
        enable_alignment: bool = False,
        debug_dir: Optional[str | Path] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.device = device
        self.image_resolution = image_resolution
        self.preprocess_mode = preprocess_mode
        self.patch_size = patch_size
        self.max_context_frames = max_context_frames
        self.enable_alignment = enable_alignment
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self._model = None
        self._prediction_cache: Dict[Tuple[str, ...], DepthPrediction] = {}

        if self.preprocess_mode not in {"balanced", "max_size"}:
            raise ValueError("VGGT-Omega preprocess_mode must be 'balanced' or 'max_size'.")
        if self.image_resolution <= 0:
            raise ValueError("VGGT-Omega image_resolution must be positive.")
        if self.patch_size <= 0 or self.image_resolution % self.patch_size != 0:
            raise ValueError("VGGT-Omega image_resolution must be divisible by patch_size.")
        if self.max_context_frames <= 0:
            raise ValueError("VGGT-Omega max_context_frames must be positive.")

    def select_frame_indices(
        self,
        valid_frame_indices: Sequence[int],
        required_indices: Sequence[int],
    ) -> List[int]:
        available = [int(index) for index in valid_frame_indices]
        if not available:
            return sorted({int(index) for index in required_indices})

        required = {int(index) for index in required_indices}
        target_count = min(len(available), max(self.max_context_frames, len(required)))
        selected = set(required)

        if target_count > len(selected):
            positions = np.linspace(0, len(available) - 1, num=target_count)
            for position in positions:
                selected.add(available[int(round(float(position)))])

        for index in required:
            if index not in selected:
                selected.add(index)
        return sorted(selected)

    def predict(self, image_paths: Sequence[Path | str]) -> DepthPrediction:
        if len(image_paths) == 0:
            raise ValueError("VGGTOmegaEstimator.predict requires at least one image path.")

        resolved_paths = [Path(path).expanduser().resolve() for path in image_paths]
        cache_key = tuple(str(path) for path in resolved_paths)
        if cache_key in self._prediction_cache:
            LOGGER.info("VGGT-Omega cache hit for %d frame(s)", len(cache_key))
            return self._prediction_cache[cache_key]

        model = self._load_model()

        try:
            import torch
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            from vggt_omega.utils.pose_enc import encoding_to_camera
        except ImportError as exc:
            raise ImportError(
                "VGGT-Omega is not installed. Clone facebookresearch/vggt-omega, install its requirements, "
                "then run `pip install -e .` before using --depth-provider vggt-omega."
            ) from exc

        metadata = self._build_preprocess_metadata(resolved_paths)
        LOGGER.info(
            "VGGT-Omega cache miss for %d frame(s): %s",
            len(resolved_paths),
            [str(path) for path in resolved_paths],
        )

        images = load_and_preprocess_images(
            [str(path) for path in resolved_paths],
            mode=self.preprocess_mode,
            image_resolution=self.image_resolution,
            patch_size=self.patch_size,
        ).to(self.device)

        with torch.inference_mode():
            predictions = model(images)

        raw_depth_maps = self._split_frame_maps(predictions["depth"], len(resolved_paths), "depth")
        # for raw_depth_map, path in zip(raw_depth_maps, resolved_paths):
        #     self.save_depth_visualization(
        #         raw_depth_map,
        #         path,
        #         suffix=f"vggt_raw_depth_{raw_depth_map.shape[0]}x{raw_depth_map.shape[1]}",
        #     )

        depth_maps = [
            self._restore_map_to_original_shape(depth_map, frame_metadata)
            for depth_map, frame_metadata in zip(raw_depth_maps, metadata)
        ]

        confidence_maps = None
        if "depth_conf" in predictions:
            confidence_maps = self._split_frame_maps(predictions["depth_conf"], len(resolved_paths), "depth_conf")
            confidence_maps = [
                self._restore_map_to_original_shape(confidence_map, frame_metadata)
                for confidence_map, frame_metadata in zip(confidence_maps, metadata)
            ]

        intrinsics = None
        extrinsics = None
        if "pose_enc" in predictions:
            try:
                extrinsics_tensor, intrinsics_tensor = encoding_to_camera(
                    predictions["pose_enc"],
                    predictions["images"].shape[-2:],
                )
                intrinsics = self._to_numpy(intrinsics_tensor)
                extrinsics = self._to_numpy(extrinsics_tensor)
                if intrinsics.ndim >= 4 and intrinsics.shape[0] == 1:
                    intrinsics = intrinsics[0]
                if extrinsics.ndim >= 4 and extrinsics.shape[0] == 1:
                    extrinsics = extrinsics[0]
            except Exception as exc:
                LOGGER.warning("VGGT-Omega camera decoding failed: %s", exc)

        prediction = DepthPrediction(
            depth_maps=depth_maps,
            confidence_maps=confidence_maps,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        self._prediction_cache[cache_key] = prediction

        # for depth_map, path in zip(depth_maps, resolved_paths):
        #     self.save_depth_visualization(
        #         depth_map,
        #         path,
        #         suffix=f"vggt_rgb_depth_{depth_map.shape[0]}x{depth_map.shape[1]}",
        #     )

        return prediction

    def save_depth_visualization(self, depth_map: np.ndarray, image_path: Path | str, suffix: str) -> Optional[Path]:
        if self.debug_dir is None:
            return None

        output_dir = self.debug_dir / "depth_maps"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image_path)
        output_path = output_dir / f"{image_path.parent.name}__{image_path.stem}__{suffix}.png"
        _save_depth_png(depth_map, output_path)
        LOGGER.info("Saved VGGT-Omega depth visualization: %s", output_path)
        return output_path

    def _load_model(self):
        if self._model is not None:
            return self._model

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {self.checkpoint_path}")

        try:
            import torch
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:
            raise ImportError(
                "VGGT-Omega is not installed. Clone facebookresearch/vggt-omega, install its requirements, "
                "then run `pip install -e .` before using --depth-provider vggt-omega."
            ) from exc

        LOGGER.info(
            "Loading VGGT-Omega checkpoint=%s device=%s resolution=%d mode=%s max_context_frames=%d",
            self.checkpoint_path,
            self.device,
            self.image_resolution,
            self.preprocess_mode,
            self.max_context_frames,
        )
        model = VGGTOmega(enable_alignment=self.enable_alignment).to(self.device).eval()
        state = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        model.load_state_dict(state)
        self._model = model
        return self._model

    def _build_preprocess_metadata(self, image_paths: Sequence[Path]) -> List[_PreprocessMetadata]:
        partial: List[Tuple[Tuple[int, int], Tuple[int, int, int, int], Tuple[int, int]]] = []
        target_shapes = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                original_size = image.size

            crop_box = self._supported_aspect_crop_box(*original_size)
            crop_width = max(crop_box[2] - crop_box[0], 1)
            crop_height = max(crop_box[3] - crop_box[1], 1)
            aspect_ratio = crop_height / max(crop_width, 1)
            if self.preprocess_mode == "balanced":
                target_shape = self._balanced_target_shape(aspect_ratio)
            else:
                target_shape = self._max_size_target_shape(aspect_ratio)
            partial.append((original_size, crop_box, target_shape))
            target_shapes.append(target_shape)

        common_height = max(shape[0] for shape in target_shapes)
        common_width = max(shape[1] for shape in target_shapes)
        metadata = []
        for original_size, crop_box, target_shape in partial:
            target_height, target_width = target_shape
            h_padding = common_height - target_height
            w_padding = common_width - target_width
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            metadata.append(
                _PreprocessMetadata(
                    original_size=original_size,
                    crop_box=crop_box,
                    target_shape=target_shape,
                    common_shape=(common_height, common_width),
                    pad_top=pad_top,
                    pad_bottom=pad_bottom,
                    pad_left=pad_left,
                    pad_right=pad_right,
                )
            )
        return metadata

    def _supported_aspect_crop_box(
        self,
        width: int,
        height: int,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
    ) -> Tuple[int, int, int, int]:
        aspect_ratio = height / max(width, 1)
        if aspect_ratio < min_aspect_ratio:
            crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
            left = max((width - crop_width) // 2, 0)
            return left, 0, left + crop_width, height
        if aspect_ratio > max_aspect_ratio:
            crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
            top = max((height - crop_height) // 2, 0)
            return 0, top, width, top + crop_height
        return 0, 0, width, height

    def _balanced_target_shape(self, aspect_ratio: float) -> Tuple[int, int]:
        token_number = (self.image_resolution // self.patch_size) ** 2
        width_patches = np.sqrt(token_number / aspect_ratio)
        height_patches = token_number / width_patches
        width_patches = max(1, int(np.round(width_patches)))
        height_patches = max(1, int(np.round(height_patches)))
        return height_patches * self.patch_size, width_patches * self.patch_size

    def _max_size_target_shape(self, aspect_ratio: float) -> Tuple[int, int]:
        if aspect_ratio >= 1.0:
            height = self.image_resolution
            width = self._round_to_patch_multiple(self.image_resolution / aspect_ratio)
        else:
            width = self.image_resolution
            height = self._round_to_patch_multiple(self.image_resolution * aspect_ratio)
        return height, width

    def _round_to_patch_multiple(self, value: float) -> int:
        return max(self.patch_size, int(np.round(float(value) / self.patch_size)) * self.patch_size)

    def _restore_map_to_original_shape(self, frame_map: np.ndarray, metadata: _PreprocessMetadata) -> np.ndarray:
        frame_map = np.asarray(frame_map, dtype=np.float32)
        if metadata.pad_bottom > 0:
            bottom = frame_map.shape[0] - metadata.pad_bottom
        else:
            bottom = frame_map.shape[0]
        if metadata.pad_right > 0:
            right = frame_map.shape[1] - metadata.pad_right
        else:
            right = frame_map.shape[1]
        frame_map = frame_map[metadata.pad_top:bottom, metadata.pad_left:right]

        crop_left, crop_top, crop_right, crop_bottom = metadata.crop_box
        crop_height = max(crop_bottom - crop_top, 1)
        crop_width = max(crop_right - crop_left, 1)
        crop_map = _resize_float_array(frame_map, (crop_height, crop_width))

        original_width, original_height = metadata.original_size
        restored = np.full((original_height, original_width), np.nan, dtype=np.float32)
        restored[crop_top:crop_bottom, crop_left:crop_right] = crop_map
        return restored

    def _split_frame_maps(self, value: Any, frame_count: int, field_name: str) -> List[np.ndarray]:
        array = self._to_numpy(value)
        if array.ndim == 5 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[0] == 1 and array.shape[1] == frame_count:
            array = array[0]

        if array.ndim == 4 and array.shape[0] == frame_count and array.shape[-1] == 1:
            array = array[..., 0]
        elif array.ndim == 4 and array.shape[0] == frame_count and array.shape[1] == 1:
            array = array[:, 0]

        if array.ndim == 3 and array.shape[0] == frame_count:
            return [np.asarray(array[index], dtype=np.float32) for index in range(frame_count)]
        if array.ndim == 2 and frame_count == 1:
            return [np.asarray(array, dtype=np.float32)]
        if array.ndim == 3 and frame_count == 1:
            return [np.asarray(array.squeeze(), dtype=np.float32)]

        raise ValueError(f"Could not split VGGT-Omega {field_name} output with shape {array.shape}.")

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)


def _resize_float_array(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


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
