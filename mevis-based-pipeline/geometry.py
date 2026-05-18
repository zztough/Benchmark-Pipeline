from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class MaskStatistics:
    centroid_x: float
    centroid_y: float
    area: float
    height: int
    width: int


@dataclass
class MaskDepthStatistics:
    mean_depth: float
    median_depth: float
    valid_ratio: float
    confidence_mean: Optional[float] = None


def _decode_compressed_counts(encoded: str) -> List[int]:
    # 把 COCO RLE 中压缩后的字符串 "ea_d21..." 恢复成真正的 run-length 序列 [3,1,5,2,...]，COCO采用的是两步压缩，mask->RLE counts->压缩字符串
    counts: List[int] = []
    index = 0
    length = len(encoded)
    while index < length:
        shift = 0
        value = 0
        more = True
        while more:
            current = ord(encoded[index]) - 48
            index += 1
            value |= (current & 0x1F) << shift 
            more = bool(current & 0x20)
            shift += 5
            if not more and (current & 0x10):
                value |= -1 << shift
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def _normalize_counts(counts_value: object) -> List[int]:
    if isinstance(counts_value, list):
        return [int(item) for item in counts_value]
    if isinstance(counts_value, str):
        return _decode_compressed_counts(counts_value)
    if isinstance(counts_value, bytes):
        return _decode_compressed_counts(counts_value.decode("utf-8"))
    raise TypeError(f"Unsupported RLE counts type: {type(counts_value)!r}")


def decode_rle_mask(rle: Optional[dict]) -> Optional[List[List[int]]]:
    # RLE格式的mask解码为二维矩阵，矩阵元素为0或1，1表示该像素被mask覆盖，
    # 输入rle = {"size": [4,5], "counts": [3,1,3,3,2,2,6]}，输出：mask = [
                                                                        #     [0,0,1,1,0],
                                                                        #     [0,0,1,1,0],
                                                                        #     [0,0,0,0,0],
                                                                        #     [1,1,0,0,0]
                                                                        # ]，注意原始RLE（counts）是按列存储的，分析上述示例可以看到
    if rle is None:
        return None
    size = rle.get("size")
    counts = _normalize_counts(rle.get("counts")) 
    # RLE：Run Length Encoding，统计连续相同数字长度，示例：[3,1,3,3,2,2,6]，含义：3个0、1个1、3个0、3个1、2个0、2个1、6个0，
    # 默认从0开始计数
    if not size or len(size) != 2:
        raise ValueError(f"Invalid RLE size: {size!r}")
    height, width = int(size[0]), int(size[1])

    flat_mask = [0] * (height * width)
    cursor = 0
    fill_value = 0
    for run_length in counts:
        if run_length < 0:
            raise ValueError(f"Negative RLE run length: {run_length}")
        if fill_value == 1:
            for offset in range(run_length):
                flat_mask[cursor + offset] = 1
        cursor += run_length
        fill_value = 1 - fill_value

    matrix = [[0 for _ in range(width)] for _ in range(height)]
    for flat_index, value in enumerate(flat_mask):
        if not value:
            continue
        row = flat_index % height
        col = flat_index // height
        matrix[row][col] = 1
    return matrix


def compute_mask_statistics(mask: List[List[int]]) -> Optional[MaskStatistics]:
    if mask is None:
        return None
    height = len(mask)
    width = len(mask[0]) if height else 0
    if height == 0 or width == 0:
        return None

    total_x = 0.0
    total_y = 0.0
    area = 0.0
    for row_index, row in enumerate(mask):
        for col_index, value in enumerate(row):
            if value:
                total_x += float(col_index)
                total_y += float(row_index)
                area += 1.0

    if area == 0:
        return None

    centroid_x = total_x / area
    centroid_y = total_y / area
    return MaskStatistics(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        area=area,
        height=height,
        width=width,
    )


def compute_mask_depth_statistics(
    mask: List[List[int]],
    depth_map: Any,
    confidence_map: Optional[Any] = None,
    image_path: Optional[Path] = None,
    depth_estimator: Optional[object] = None,
) -> Optional[MaskDepthStatistics]:
    import numpy as np

    mask_array = np.asarray(mask, dtype=bool)
    depth_array = np.asarray(depth_map, dtype=np.float32)
    if mask_array.size == 0 or depth_array.size == 0:
        return None

    original_depth_shape = depth_array.shape
    original_confidence_shape = getattr(confidence_map, "shape", None) if confidence_map is not None else None
    if mask_array.shape != depth_array.shape:
        depth_array = _resize_array(depth_array, mask_array.shape)
        if confidence_map is not None:
            confidence_map = _resize_array(np.asarray(confidence_map, dtype=np.float32), mask_array.shape)
    # LOGGER.info(
    #     "Mask/depth alignment: mask_shape=%s original_depth_shape=%s aligned_depth_shape=%s "
    #     "original_confidence_shape=%s aligned_confidence_shape=%s",
    #     mask_array.shape,
    #     original_depth_shape,
    #     depth_array.shape,
    #     original_confidence_shape,
    #     getattr(confidence_map, "shape", None) if confidence_map is not None else None,
    # )
    if image_path is not None and depth_estimator is not None and hasattr(depth_estimator, "save_depth_visualization"):
        depth_estimator.save_depth_visualization(
            depth_array,
            image_path,
            suffix=f"aligned_depth_{depth_array.shape[0]}x{depth_array.shape[1]}",
        )

    masked_depth = depth_array[mask_array]
    finite_depth = masked_depth[np.isfinite(masked_depth)]
    if finite_depth.size == 0:
        return None

    valid_ratio = float(finite_depth.size) / max(float(masked_depth.size), 1.0)
    confidence_mean = None
    if confidence_map is not None:
        confidence_array = np.asarray(confidence_map, dtype=np.float32)
        if confidence_array.shape == mask_array.shape:
            masked_confidence = confidence_array[mask_array]
            finite_confidence = masked_confidence[np.isfinite(masked_confidence)]
            if finite_confidence.size:
                confidence_mean = float(np.mean(finite_confidence))

    return MaskDepthStatistics(
        mean_depth=float(np.mean(finite_depth)),
        median_depth=float(np.median(finite_depth)),
        valid_ratio=valid_ratio,
        confidence_mean=confidence_mean,
    )


def _resize_array(array: Any, shape: Tuple[int, int]) -> Any:
    import numpy as np
    from PIL import Image

    image = Image.fromarray(array.astype(np.float32), mode="F")
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def extract_stats_from_rle(rle: Optional[dict]) -> Optional[MaskStatistics]:
    mask = decode_rle_mask(rle)
    if mask is None:
        return None
    return compute_mask_statistics(mask)


def _infer_depth_direction(
    first_rle: dict,
    last_rle: dict,
    first_frame_path: Path,
    last_frame_path: Path,
    depth_estimator: object,
    depth_threshold: float,
) -> Tuple[str, Optional[dict]]:
    first_mask = decode_rle_mask(first_rle)
    last_mask = decode_rle_mask(last_rle)
    if first_mask is None or last_mask is None:
        return "", None

    prediction = depth_estimator.predict([first_frame_path, last_frame_path])
    if len(prediction.depth_maps) < 2:
        return "", None

    confidence_maps = prediction.confidence_maps or []
    first_confidence = confidence_maps[0] if len(confidence_maps) > 0 else None
    last_confidence = confidence_maps[1] if len(confidence_maps) > 1 else None
    first_depth_stats = compute_mask_depth_statistics(
        first_mask,
        prediction.depth_maps[0],
        first_confidence,
        image_path=first_frame_path,
        depth_estimator=depth_estimator,
    )
    last_depth_stats = compute_mask_depth_statistics(
        last_mask,
        prediction.depth_maps[1],
        last_confidence,
        image_path=last_frame_path,
        depth_estimator=depth_estimator,
    )
    if first_depth_stats is None or last_depth_stats is None:
        return "", None

    depth_delta = last_depth_stats.median_depth - first_depth_stats.median_depth
    depth_delta_ratio = depth_delta / max(abs(first_depth_stats.median_depth), 1e-6)

    depth_dir = ""
    if depth_delta_ratio <= -depth_threshold:
        depth_dir = "forward"
    elif depth_delta_ratio >= depth_threshold:
        depth_dir = "backward"

    return depth_dir, {
        "depth_source": "depth_anything_3",
        "first_depth_mean": first_depth_stats.mean_depth,
        "last_depth_mean": last_depth_stats.mean_depth,
        "first_depth_median": first_depth_stats.median_depth,
        "last_depth_median": last_depth_stats.median_depth,
        "depth_delta": depth_delta,
        "depth_delta_ratio": depth_delta_ratio,
        "first_depth_valid_ratio": first_depth_stats.valid_ratio,
        "last_depth_valid_ratio": last_depth_stats.valid_ratio,
        "first_confidence_mean": first_depth_stats.confidence_mean,
        "last_confidence_mean": last_depth_stats.confidence_mean,
        "has_intrinsics": prediction.intrinsics is not None,
        "has_extrinsics": prediction.extrinsics is not None,
    }


def _pick_valid_frames(mask_sequence: Sequence[Optional[dict]], span: Tuple[int, int]) -> List[Tuple[int, dict]]:
    start, end = span
    valid_frames: List[Tuple[int, dict]] = []
    for frame_index in range(max(0, start), min(len(mask_sequence) - 1, end) + 1):
        rle = mask_sequence[frame_index]
        if rle is not None:
            valid_frames.append((frame_index, rle))
    return valid_frames


def infer_direction_from_mask_span(
    mask_sequence: Sequence[Optional[dict]],
    span: Tuple[int, int],
    lateral_threshold: float = 0.02,
    depth_threshold: float = 0.08,
    frame_path_resolver: Optional[Callable[[int], Optional[Path]]] = None,
    depth_estimator: Optional[object] = None,
) -> Tuple[str, Optional[dict]]:
    valid_frames = _pick_valid_frames(mask_sequence, span)
    if not valid_frames:
        return "unknown", None

    first_index, first_rle = valid_frames[0]
    last_index, last_rle = valid_frames[-1]
    first_stats = extract_stats_from_rle(first_rle)
    last_stats = extract_stats_from_rle(last_rle)
    if first_stats is None or last_stats is None:
        return "unknown", None

    delta_x = (last_stats.centroid_x - first_stats.centroid_x) / max(float(first_stats.width), 1.0)
    area_ratio = (last_stats.area - first_stats.area) / max(float(first_stats.area), 1.0)

    # 分别判断横向和纵深方向
    lateral_dir = ""
    if delta_x <= -lateral_threshold:
        lateral_dir = "left"
    elif delta_x >= lateral_threshold:
        lateral_dir = "right"

    depth_dir = ""
    depth_evidence = None
    if depth_estimator is not None and frame_path_resolver is not None:
        first_frame_path = frame_path_resolver(first_index)
        last_frame_path = frame_path_resolver(last_index)
        if first_frame_path is not None and last_frame_path is not None:
            try:
                depth_dir, depth_evidence = _infer_depth_direction(
                    first_rle,
                    last_rle,
                    first_frame_path,
                    last_frame_path,
                    depth_estimator,
                    depth_threshold,
                )
            except Exception as exc:
                depth_evidence = {
                    "depth_source": "depth_anything_3",
                    "depth_error": str(exc),
                }

    if not depth_dir and depth_estimator is None:
        if area_ratio >= depth_threshold:
            depth_dir = "forward"
        elif area_ratio <= -depth_threshold:
            depth_dir = "backward"

    # 组合 8 个方向
    if depth_dir and lateral_dir:
        direction = f"Moving {depth_dir}-{lateral_dir}"
    elif depth_dir:
        direction = f"Moving {depth_dir}"
    elif lateral_dir:
        direction = f"Moving {lateral_dir}"
    else:
        direction = "Stationary"

    evidence = {
        "start_frame": first_index,
        "end_frame": last_index,
        "delta_x": delta_x,
        "area_ratio": area_ratio,
        "depth_method": "depth_anything_3" if depth_estimator is not None else "mask_area_proxy",
    }
    if depth_evidence is not None:
        evidence.update(depth_evidence)

    return direction, evidence
