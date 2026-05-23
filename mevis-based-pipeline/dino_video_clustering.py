from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None


LOGGER = logging.getLogger("dinov3_video_clustering")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NORMALIZATION_STATS = {
    "lvd1689m": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "DINOv3 LVD-1689M / ImageNet evaluation normalization",
    },
    "sat493m": {
        "mean": (0.430, 0.411, 0.296),
        "std": (0.213, 0.156, 0.143),
        "description": "DINOv3 SAT-493M satellite normalization",
    },
}


@dataclass(frozen=True)
class QARow:
    index: int
    line_number: int
    video_id: str
    data: Dict[str, Any]


@dataclass(frozen=True)
class VideoInput:
    video_id: str
    frame_dir: Path
    frame_count: int
    sampled_frames: List[Path]
    qa_count: int


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    silhouette: Optional[float]


class FallbackDinoTransform:
    """Torchvision-free version of the official DINOv3 square resize + normalize transform."""

    def __init__(self, resize_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.resize_size = resize_size
        self.mean = mean
        self.std = std

    def __call__(self, image):
        require_numpy()
        require_pillow()
        import torch

        if not hasattr(image, "convert"):
            image = Image.open(image)
        image = image.convert("RGB")
        image = image.resize((self.resize_size, self.resize_size), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)


class DINOv3Embedder:
    def __init__(
        self,
        backend: str,
        repo_dir: Optional[Path],
        weights: Optional[str],
        hf_model_name: str,
        device: str,
        resize_size: int,
        batch_size: int,
        frame_pooling: str,
        dino_arch: str,
        pretrain_dataset: str,
        trust_remote_code: bool,
        hf_local_files_only: bool,
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("DINOv3 feature extraction requires PyTorch.") from exc

        self.torch = torch
        self.backend = backend
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.frame_pooling = frame_pooling
        self.transform = make_dinov3_transform(resize_size=resize_size, pretrain_dataset=pretrain_dataset)
        self.model = self._load_model(
            backend=backend,
            repo_dir=repo_dir,
            weights=weights,
            hf_model_name=hf_model_name,
            dino_arch=dino_arch,
            trust_remote_code=trust_remote_code,
            hf_local_files_only=hf_local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_model(
        self,
        backend: str,
        repo_dir: Optional[Path],
        weights: Optional[str],
        hf_model_name: str,
        dino_arch: str,
        trust_remote_code: bool,
        hf_local_files_only: bool,
    ):
        if backend == "torchhub":
            if repo_dir is None:
                raise ValueError("--repo-dir is required when --model-backend torchhub.")
            if weights is None:
                raise ValueError("--weights is required for DINOv3 torchhub loading.")
            LOGGER.info("Loading DINOv3 torchhub model arch=%s weights=%s", dino_arch, weights)
            return self.torch.hub.load(str(repo_dir), dino_arch, source="local", weights=weights)

        if backend == "hf":
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise RuntimeError("The hf backend requires transformers>=4.56.0 for DINOv3.") from exc
            model_name_or_path = weights or hf_model_name
            LOGGER.info("Loading DINOv3 Hugging Face model %s", model_name_or_path)
            return AutoModel.from_pretrained(
                model_name_or_path,
                local_files_only=hf_local_files_only,
                trust_remote_code=trust_remote_code,
            )

        raise ValueError(f"Unsupported model backend: {backend}")

    def encode_paths(self, image_paths: Sequence[Path]) -> np.ndarray:
        require_numpy()
        require_pillow()
        if not image_paths:
            raise ValueError("Cannot encode an empty frame list.")

        features: List[np.ndarray] = []
        torch = self.torch
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            tensors = []
            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                tensors.append(self.transform(image))
            batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
            with torch.inference_mode():
                output = self._forward(batch)
                pooled = self._pool_frame_output(output)
            features.append(pooled.detach().float().cpu().numpy())
        return np.concatenate(features, axis=0)

    def _forward(self, batch):
        if self.backend == "hf":
            return self.model(pixel_values=batch)
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(batch)
        return self.model(batch)

    def _pool_frame_output(self, output):
        torch = self.torch

        if isinstance(output, dict):
            if self.frame_pooling in {"auto", "cls"}:
                for key in ("x_norm_clstoken", "cls_token", "pooler_output"):
                    value = output.get(key)
                    if torch.is_tensor(value):
                        return value
            for key in ("x_norm_patchtokens", "patch_tokens"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value.mean(dim=1)
            for value in output.values():
                if torch.is_tensor(value):
                    return pool_tensor_tokens(value, self.frame_pooling)

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            if self.frame_pooling in {"auto", "cls"}:
                return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return pool_tensor_tokens(output.last_hidden_state, self.frame_pooling)

        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if torch.is_tensor(first):
                return pool_tensor_tokens(first, self.frame_pooling)

        if torch.is_tensor(output):
            return pool_tensor_tokens(output, self.frame_pooling)

        raise RuntimeError(f"Could not pool DINOv3 output of type {type(output)!r}.")


def pool_tensor_tokens(tensor, frame_pooling: str):
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        if frame_pooling == "patch_mean" and tensor.shape[1] > 1:
            return tensor[:, 1:].mean(dim=1)
        return tensor[:, 0]
    raise RuntimeError(f"Expected 2D or 3D model output, got shape={tuple(tensor.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster MeViS videos referenced by qa_samples.jsonl with DINOv3 frame features, "
            "then write selected QA samples per cluster."
        )
    )
    parser.add_argument("--qa-path", type=Path, default=Path("outputs/mevis_demo_6/qa_samples.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/Datasets/MeViSv2/train"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-backend", choices=["torchhub", "hf"], default="hf")
    parser.add_argument("--repo-dir", "--torchhub-repo", dest="repo_dir", type=Path, default=None)
    parser.add_argument("--weights", "--model-path", dest="weights", type=str, default="/home/zhaobing/.cache/modelscope/hub/models/facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--dino-arch", type=str, default="dinov3_vitl16")
    parser.add_argument("--hf-model-name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--hf-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--pretrain-dataset", choices=["lvd1689m", "sat493m"], default="lvd1689m")
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Defaults to cuda if available.")
    parser.add_argument("--cuda-visible-devices", type=str, default="0", help="Set CUDA_VISIBLE_DEVICES before loading DINOv3, e.g. 0 or 0,1.")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-pooling", choices=["auto", "cls", "patch_mean"], default="auto")
    parser.add_argument("--video-pooling", choices=["mean", "max", "mean_std", "mean_std_diff"], default="mean")
    parser.add_argument("--temporal-std-weight", type=float, default=0.5)
    parser.add_argument("--temporal-diff-weight", type=float, default=0.5)
    parser.add_argument("--num-clusters", type=int, default=0, help="Use 0 to choose k automatically with silhouette score.")
    parser.add_argument("--min-clusters", type=int, default=2) # 自动搜索时从 2 类开始试
    parser.add_argument("--max-clusters", type=int, default=12) # 自动搜索时最多试 12 类，超过这个数量可能会过拟合或计算成本过高
    parser.add_argument("--kmeans-n-init", type=int, default=10) # 每个类别数重复初始化几次，选最好的结果
    parser.add_argument("--kmeans-max-iter", type=int, default=100) # KMeans 的最大迭代次数，允许提前收敛
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--samples-per-cluster", type=int, default=None, help="Fixed number of videos to keep per cluster. Defaults to 1 when no ratio is set.")
    parser.add_argument("--sample-ratio-per-cluster", type=float, default=0.50, help="Fraction of videos to keep per cluster, e.g. 0.3 keeps ceil(30%%). If both ratio and fixed count are set, the fixed count is a cap.")
    parser.add_argument("--write-filtered-qa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--annotate-filtered-qa", action="store_true")
    parser.add_argument("--missing-video-policy", choices=["skip", "error"], default="skip")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_selection_args(args: argparse.Namespace) -> None:
    if args.samples_per_cluster is not None and args.samples_per_cluster <= 0:
        raise ValueError("--samples-per-cluster must be positive when it is set.")
    if args.sample_ratio_per_cluster is not None:
        if args.sample_ratio_per_cluster <= 0 or args.sample_ratio_per_cluster > 1:
            raise ValueError("--sample-ratio-per-cluster must be in the range (0, 1].")


def load_qa_rows(path: Path) -> Tuple[List[QARow], List[str]]:
    rows: List[QARow] = []
    video_ids: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            video_id = str(data.get("video_id", "")).strip()
            if not video_id:
                LOGGER.warning("Skipping QA row %d without video_id.", line_number)
                continue
            rows.append(QARow(index=len(rows), line_number=line_number, video_id=video_id, data=data))
            if video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)
    return rows, video_ids


def collect_video_inputs(
    video_ids: Sequence[str],
    qa_rows: Sequence[QARow],
    data_root: Path,
    frames_per_video: int,
    missing_video_policy: str,
) -> Tuple[List[VideoInput], List[Dict[str, str]]]:
    qa_count_by_video = Counter(row.video_id for row in qa_rows)
    video_inputs: List[VideoInput] = []
    skipped: List[Dict[str, str]] = []
    for video_id in video_ids:
        frame_dir = resolve_frame_dir(data_root, video_id)
        if frame_dir is None:
            reason = "frame_dir_not_found"
            handle_missing_video(video_id, reason, missing_video_policy)
            skipped.append({"video_id": video_id, "reason": reason})
            continue
        frame_paths = list_frame_paths(frame_dir)
        if not frame_paths:
            reason = "no_image_frames"
            handle_missing_video(video_id, reason, missing_video_policy)
            skipped.append({"video_id": video_id, "reason": reason})
            continue
        sampled_frames = sample_uniform(frame_paths, frames_per_video)
        video_inputs.append(
            VideoInput(
                video_id=video_id,
                frame_dir=frame_dir,
                frame_count=len(frame_paths),
                sampled_frames=sampled_frames,
                qa_count=qa_count_by_video[video_id],
            )
        )
    return video_inputs, skipped


def handle_missing_video(video_id: str, reason: str, policy: str) -> None:
    message = f"Video {video_id} skipped: {reason}."
    if policy == "error":
        raise FileNotFoundError(message)
    LOGGER.warning(message)


def resolve_frame_dir(data_root: Path, video_id: str) -> Optional[Path]:
    for candidate in (data_root / video_id, data_root / "JPEGImages" / video_id):
        if candidate.is_dir():
            return candidate
    return None


def list_frame_paths(frame_dir: Path) -> List[Path]:
    paths = [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=natural_sort_key)


def natural_sort_key(path: Path) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def sample_uniform(paths: Sequence[Path], count: int) -> List[Path]:
    if count <= 0:
        raise ValueError("--frames-per-video must be positive.")
    if len(paths) <= count:
        return list(paths)
    if count == 1:
        return [paths[0]]
    last_index = len(paths) - 1
    indices = [round(index * last_index / (count - 1)) for index in range(count)]
    return [paths[int(index)] for index in indices]


def make_dinov3_transform(resize_size: int, pretrain_dataset: str):
    stats = NORMALIZATION_STATS[pretrain_dataset]
    mean = stats["mean"]
    std = stats["std"]
    try:
        import torch
        from torchvision.transforms import v2
    except ImportError:
        LOGGER.warning("torchvision.transforms.v2 is unavailable; using a PIL fallback with the same resize/normalize settings.")
        return FallbackDinoTransform(resize_size=resize_size, mean=mean, std=std)

    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def require_numpy() -> None:
    if np is None:
        raise RuntimeError("This step requires numpy. Install numpy>=1.24.0 in the active Python environment.")


def require_pillow() -> None:
    if Image is None:
        raise RuntimeError("This step requires Pillow. Install Pillow>=10.0.0 in the active Python environment.")


def load_or_extract_features(args: argparse.Namespace, feature_cache: Path, video_inputs: Sequence[VideoInput]) -> np.ndarray:
    require_numpy()
    if feature_cache.exists() and not args.overwrite_cache:
        cached = np.load(feature_cache, allow_pickle=False)
        cached_ids = [str(value) for value in cached["video_ids"].tolist()]
        cached_features = cached["features"].astype(np.float32)
        index_by_id = {video_id: index for index, video_id in enumerate(cached_ids)}
        missing = [item.video_id for item in video_inputs if item.video_id not in index_by_id]
        if not missing:
            LOGGER.info("Loaded cached DINOv3 video features from %s.", feature_cache)
            return np.stack([cached_features[index_by_id[item.video_id]] for item in video_inputs], axis=0)
        LOGGER.warning("Ignoring feature cache because %d videos are missing from it.", len(missing))

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    embedder = DINOv3Embedder(
        backend=args.model_backend,
        repo_dir=args.repo_dir,
        weights=args.weights,
        hf_model_name=args.hf_model_name,
        device=device,
        resize_size=args.resize_size,
        batch_size=args.batch_size,
        frame_pooling=args.frame_pooling,
        dino_arch=args.dino_arch,
        pretrain_dataset=args.pretrain_dataset,
        trust_remote_code=args.trust_remote_code,
        hf_local_files_only=args.hf_local_files_only,
    )

    features: List[np.ndarray] = []
    for index, item in enumerate(video_inputs, start=1):
        LOGGER.info(
            "[%d/%d] Extracting DINOv3 frame features for video=%s from %d uniformly sampled frames.",
            index,
            len(video_inputs),
            item.video_id,
            len(item.sampled_frames),
        )
        frame_features = embedder.encode_paths(item.sampled_frames)
        features.append(
            aggregate_frame_features(
                frame_features,
                video_pooling=args.video_pooling,
                std_weight=args.temporal_std_weight,
                diff_weight=args.temporal_diff_weight,
            ).astype(np.float32)
        )

    feature_matrix = np.stack(features, axis=0)
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_cache,
        video_ids=np.asarray([item.video_id for item in video_inputs]),
        features=feature_matrix,
    )
    LOGGER.info("Saved DINOv3 video feature cache to %s.", feature_cache)
    return feature_matrix


def aggregate_frame_features(
    frame_features: np.ndarray,
    video_pooling: str,
    std_weight: float,
    diff_weight: float,
) -> np.ndarray:
    require_numpy()
    frame_features = l2_normalize(frame_features.astype(np.float32), axis=1)

    if video_pooling == "max":
        return l2_normalize_vector(frame_features.max(axis=0))

    mean_feature = l2_normalize_vector(frame_features.mean(axis=0))
    if video_pooling == "mean":
        return mean_feature

    parts = [mean_feature]
    if video_pooling in {"mean_std", "mean_std_diff"}:
        parts.append(std_weight * l2_normalize_vector(frame_features.std(axis=0)))
    if video_pooling == "mean_std_diff" and len(frame_features) > 1:
        diff_feature = np.abs(np.diff(frame_features, axis=0)).mean(axis=0)
        parts.append(diff_weight * l2_normalize_vector(diff_feature))
    return l2_normalize_vector(np.concatenate(parts, axis=0))


def l2_normalize(values: np.ndarray, axis: int, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, eps)


def l2_normalize_vector(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm < eps:
        return np.zeros_like(values)
    return values / norm


def choose_or_run_kmeans(args: argparse.Namespace, features: np.ndarray) -> KMeansResult:
    require_numpy()
    video_count = len(features)
    if video_count == 0:
        raise ValueError("No video features were extracted.")
    if video_count == 1:
        return KMeansResult(np.zeros(1, dtype=np.int64), features.copy(), 0.0, None)

    if args.num_clusters > 0:
        k = min(args.num_clusters, video_count)
        result = run_kmeans(features, k, args.seed, args.kmeans_n_init, args.kmeans_max_iter)
        silhouette = compute_silhouette(features, result.labels)
        LOGGER.info("KMeans finished with k=%d silhouette=%.4f.", k, silhouette)
        return KMeansResult(result.labels, result.centers, result.inertia, silhouette)

    min_k = max(2, args.min_clusters)
    max_k = min(max(args.max_clusters, min_k), video_count)
    best_result: Optional[KMeansResult] = None
    best_k: Optional[int] = None
    for k in range(min_k, max_k + 1):
        result = run_kmeans(features, k, args.seed + k, args.kmeans_n_init, args.kmeans_max_iter)
        silhouette = compute_silhouette(features, result.labels)
        result = KMeansResult(result.labels, result.centers, result.inertia, silhouette)
        LOGGER.info("Tried k=%d inertia=%.4f silhouette=%.4f.", k, result.inertia, silhouette)
        if best_result is None or silhouette > (best_result.silhouette or -math.inf) + 1e-8:
            best_result = result
            best_k = k
    if best_result is None or best_k is None:
        raise RuntimeError("Failed to select a cluster count.")
    LOGGER.info("Auto-selected k=%d with silhouette=%.4f.", best_k, best_result.silhouette or 0.0)
    return best_result


def run_kmeans(features: np.ndarray, k: int, seed: int, n_init: int, max_iter: int) -> KMeansResult:
    rng = np.random.default_rng(seed)
    best_labels: Optional[np.ndarray] = None
    best_centers: Optional[np.ndarray] = None
    best_inertia = math.inf
    for _ in range(max(1, n_init)):
        centers = init_kmeans_plus_plus(features, k, rng)
        labels = np.zeros(len(features), dtype=np.int64)
        for _iteration in range(max_iter):
            distances = squared_distances(features, centers)
            new_labels = distances.argmin(axis=1)
            new_centers = recompute_centers(features, new_labels, k, distances)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                centers = new_centers
                break
            labels = new_labels
            centers = new_centers
        distances = squared_distances(features, centers)
        inertia = float(distances[np.arange(len(features)), labels].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    if best_labels is None or best_centers is None:
        raise RuntimeError("KMeans failed to initialize.")
    return KMeansResult(best_labels, best_centers, best_inertia, None)


def init_kmeans_plus_plus(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, features.shape[1]), dtype=np.float32)
    centers[0] = features[int(rng.integers(0, len(features)))]
    closest_distances = squared_distances(features, centers[:1]).reshape(-1)
    for center_index in range(1, k):
        total = float(closest_distances.sum())
        if total <= 1e-12:
            next_index = int(rng.integers(0, len(features)))
        else:
            next_index = int(rng.choice(len(features), p=closest_distances / total))
        centers[center_index] = features[next_index]
        new_distances = squared_distances(features, centers[center_index : center_index + 1]).reshape(-1)
        closest_distances = np.minimum(closest_distances, new_distances)
    return centers


def recompute_centers(features: np.ndarray, labels: np.ndarray, k: int, previous_distances: np.ndarray) -> np.ndarray:
    centers = np.empty((k, features.shape[1]), dtype=np.float32)
    farthest_order = np.argsort(previous_distances.min(axis=1))[::-1]
    used_fallbacks = set()
    for cluster_id in range(k):
        members = features[labels == cluster_id]
        if len(members) > 0:
            centers[cluster_id] = members.mean(axis=0)
        else:
            fallback_index = next(index for index in farthest_order if int(index) not in used_fallbacks)
            used_fallbacks.add(int(fallback_index))
            centers[cluster_id] = features[int(fallback_index)]
        centers[cluster_id] = l2_normalize_vector(centers[cluster_id])
    return centers


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    distances = left_norm + right_norm - 2.0 * np.matmul(left, right.T)
    return np.maximum(distances, 0.0)


def compute_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = sorted(int(label) for label in np.unique(labels))
    if len(unique_labels) < 2 or len(unique_labels) >= len(features):
        return 0.0
    cosine_distance = 1.0 - np.clip(np.matmul(features, features.T), -1.0, 1.0)
    scores: List[float] = []
    for index, label in enumerate(labels):
        same_indices = np.where(labels == label)[0]
        if len(same_indices) <= 1:
            scores.append(0.0)
            continue
        a_distance = float(cosine_distance[index, same_indices[same_indices != index]].mean())
        b_distance = math.inf
        for other_label in unique_labels:
            if other_label == int(label):
                continue
            other_indices = np.where(labels == other_label)[0]
            b_distance = min(b_distance, float(cosine_distance[index, other_indices].mean()))
        denominator = max(a_distance, b_distance)
        scores.append(0.0 if denominator <= 1e-12 else (b_distance - a_distance) / denominator)
    return float(np.mean(scores))


def build_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    feature_cache: Path,
    video_inputs: Sequence[VideoInput],
    qa_rows: Sequence[QARow],
    features: np.ndarray,
    kmeans: KMeansResult,
    skipped_videos: Sequence[Dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cluster_summary.json"
    assignments_path = output_dir / "video_cluster_assignments.jsonl"
    filtered_qa_path = output_dir / "qa_samples_cluster_filtered.jsonl"

    labels = kmeans.labels
    centers = kmeans.centers
    distances = np.sqrt(squared_distances(features, centers)[np.arange(len(features)), labels])
    rows_by_video: Dict[str, List[QARow]] = defaultdict(list)
    for row in qa_rows:
        rows_by_video[row.video_id].append(row)

    selected_videos_by_cluster = select_videos_per_cluster(
        args=args,
        video_inputs=video_inputs,
        rows_by_video=rows_by_video,
        labels=labels,
        distances=distances,
    )
    selected_video_ids = {
        video_id for video_ids in selected_videos_by_cluster.values() for video_id in video_ids
    }
    selected_rows_by_cluster: Dict[int, List[QARow]] = {}
    for cluster_id, selected_video_ids_for_cluster in selected_videos_by_cluster.items():
        selected_rows: List[QARow] = []
        for video_id in selected_video_ids_for_cluster:
            selected_rows.extend(rows_by_video.get(video_id, []))
        selected_rows_by_cluster[cluster_id] = selected_rows

    cluster_records: List[Dict[str, Any]] = []
    total_selected_video_count = sum(len(video_ids) for video_ids in selected_videos_by_cluster.values())
    total_selected_qa_count = sum(len(rows) for rows in selected_rows_by_cluster.values())

    with assignments_path.open("w", encoding="utf-8") as assignment_file:
        for cluster_id in sorted(int(label) for label in np.unique(labels)):
            member_indices = np.where(labels == cluster_id)[0]
            member_indices = np.asarray(
                sorted(member_indices, key=lambda idx: (float(distances[idx]), video_inputs[idx].video_id))
            )
            representative_index = int(member_indices[0])
            cluster_selected_video_ids = selected_videos_by_cluster.get(cluster_id, [])
            cluster_selected_video_id_set = set(cluster_selected_video_ids)
            cluster_selected_rows = selected_rows_by_cluster.get(cluster_id, [])
            target_video_count = compute_cluster_video_sample_target(args, len(member_indices))
            candidate_qa_count = sum(
                len(rows_by_video.get(video_inputs[int(index)].video_id, [])) for index in member_indices
            )
            videos = []
            for index in member_indices:
                item = video_inputs[int(index)]
                assignment = {
                    "video_id": item.video_id,
                    "cluster_id": cluster_id,
                    "qa_count": item.qa_count,
                    "frame_dir": str(item.frame_dir),
                    "frame_count": item.frame_count,
                    "sampled_frames": [str(path) for path in item.sampled_frames],
                    "distance_to_centroid": float(distances[int(index)]),
                    "is_representative": int(index) == representative_index,
                    "is_selected": item.video_id in cluster_selected_video_id_set,
                }
                videos.append(assignment)
                assignment_file.write(json.dumps(assignment, ensure_ascii=False) + "\n")

            record: Dict[str, Any] = {
                "cluster_id": cluster_id,
                "size": int(len(member_indices)),
                "representative_video_id": video_inputs[representative_index].video_id,
                "candidate_qa_count": int(candidate_qa_count),
                "target_video_count": int(target_video_count),
                "selected_video_count": int(len(cluster_selected_video_ids)),
                "selected_qa_count": int(len(cluster_selected_rows)),
                "selected_video_ids": cluster_selected_video_ids,
                "videos": videos,
            }
            if cluster_selected_rows:
                first = cluster_selected_rows[0]
                record.update(
                    {
                        "kept_qa_index": first.index,
                        "kept_qa_line_number": first.line_number,
                        "kept_qa_video_id": first.video_id,
                        "kept_qa": first.data,
                    }
                )
            cluster_records.append(record)

    summary = {
        "source_qa_path": str(args.qa_path),
        "data_root": str(args.data_root),
        "feature_cache": str(feature_cache),
        "video_count": len(video_inputs),
        "qa_count": len(qa_rows),
        "skipped_video_count": len(skipped_videos),
        "skipped_videos": list(skipped_videos),
        "model": {
            "family": "DINOv3",
            "backend": args.model_backend,
            "repo_dir": str(args.repo_dir) if args.repo_dir else None,
            "weights": args.weights,
            "dino_arch": args.dino_arch,
            "hf_model_name": args.hf_model_name,
            "pretrain_dataset": args.pretrain_dataset,
        },
        "transform": {
            "resize": [args.resize_size, args.resize_size],
            "mean": NORMALIZATION_STATS[args.pretrain_dataset]["mean"],
            "std": NORMALIZATION_STATS[args.pretrain_dataset]["std"],
            "source": NORMALIZATION_STATS[args.pretrain_dataset]["description"],
        },
        "frame_sampling": {
            "frames_per_video": args.frames_per_video,
            "method": "uniform_over_full_frame_sequence",
        },
        "embedding": {
            "frame_pooling": args.frame_pooling,
            "video_pooling": args.video_pooling,
            "dimension": int(features.shape[1]) if len(features) else 0,
        },
        "clustering": {
            "algorithm": "kmeans_cosine_like_l2_on_l2_normalized_embeddings",
            "num_clusters": int(len(cluster_records)),
            "requested_num_clusters": args.num_clusters,
            "auto_selected": args.num_clusters == 0,
            "inertia": kmeans.inertia,
            "silhouette": kmeans.silhouette,
        },
        "selection": {
            "unit": "video",
            "filtered_qa_path": str(filtered_qa_path) if args.write_filtered_qa else None,
            "policy": "per_cluster_video_sampling_then_keep_all_qas_for_selected_videos",
            "samples_per_cluster": args.samples_per_cluster,
            "sample_ratio_per_cluster": args.sample_ratio_per_cluster,
            "default_samples_per_cluster": 1,
            "total_selected_video_count": int(total_selected_video_count),
            "total_selected_qa_count": int(total_selected_qa_count),
        },
        "clusters": cluster_records,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    if args.write_filtered_qa:
        with filtered_qa_path.open("w", encoding="utf-8") as file:
            for cluster_id in sorted(selected_rows_by_cluster):
                selected_rows = selected_rows_by_cluster[cluster_id]
                for row in selected_rows:
                    data = dict(row.data)
                    if args.annotate_filtered_qa:
                        data["_cluster_id"] = cluster_id
                        data["_selected_video_id"] = row.video_id
                        data["_selection_reason"] = "video_selected_from_dinov3_cluster"
                    file.write(json.dumps(data, ensure_ascii=False) + "\n")

    LOGGER.info(
        "Selected %d videos and kept %d QA rows across %d clusters.",
        total_selected_video_count,
        total_selected_qa_count,
        len(cluster_records),
    )
    LOGGER.info("Wrote cluster summary to %s.", summary_path)
    LOGGER.info("Wrote video cluster assignments to %s.", assignments_path)
    if args.write_filtered_qa:
        LOGGER.info("Wrote filtered QA JSONL to %s.", filtered_qa_path)


def compute_cluster_video_sample_target(args: argparse.Namespace, candidate_video_count: int) -> int:
    if candidate_video_count <= 0:
        return 0
    if args.sample_ratio_per_cluster is not None:
        target = max(1, int(math.ceil(candidate_video_count * args.sample_ratio_per_cluster)))
        if args.samples_per_cluster is not None:
            target = min(target, args.samples_per_cluster)
    elif args.samples_per_cluster is not None:
        target = args.samples_per_cluster
    else:
        target = 1
    return min(candidate_video_count, target)


def select_videos_per_cluster(
    args: argparse.Namespace,
    video_inputs: Sequence[VideoInput],
    rows_by_video: Dict[str, List[QARow]],
    labels: np.ndarray,
    distances: np.ndarray,
) -> Dict[int, List[str]]:
    selected_by_cluster: Dict[int, List[str]] = {}
    signature_counts: Counter = Counter()
    template_counts: Counter = Counter()

    for cluster_id in sorted(int(label) for label in np.unique(labels)):
        member_indices = np.where(labels == cluster_id)[0]
        target_count = compute_cluster_video_sample_target(args, len(member_indices))
        candidates = []
        for video_index in member_indices:
            item = video_inputs[int(video_index)]
            qa_rows = rows_by_video.get(item.video_id, [])
            signatures = []
            templates = []
            for row in qa_rows:
                template = str(row.data.get("template_name", ""))
                direction_source = str(row.data.get("direction_source", ""))
                answer = str(row.data.get("answer", ""))
                signatures.append((template, direction_source, answer))
                templates.append(template)
            candidates.append(
                {
                    "video_id": item.video_id,
                    "distance": float(distances[int(video_index)]),
                    "signatures": signatures,
                    "templates": templates,
                    "qa_count": len(qa_rows),
                }
            )

        selected_video_ids: List[str] = []
        selected_id_set = set()
        while len(selected_video_ids) < target_count:
            remaining = [candidate for candidate in candidates if candidate["video_id"] not in selected_id_set]
            if not remaining:
                break
            remaining.sort(
                key=lambda item: (
                    sum(signature_counts[signature] for signature in item["signatures"]),
                    sum(template_counts[template] for template in item["templates"]),
                    item["distance"],
                    item["video_id"],
                )
            )
            selected = remaining[0]
            selected_video_ids.append(selected["video_id"])
            selected_id_set.add(selected["video_id"])
            for signature in selected["signatures"]:
                signature_counts[signature] += 1
            for template in selected["templates"]:
                template_counts[template] += 1

        selected_by_cluster[cluster_id] = selected_video_ids
    return selected_by_cluster

def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    setup_logging(args.log_level)
    validate_selection_args(args)
    qa_path = args.qa_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else qa_path.parent / "dinov3_video_clusters"
    feature_cache = args.feature_cache.resolve() if args.feature_cache else output_dir / "dinov3_video_features.npz"

    qa_rows, video_ids = load_qa_rows(qa_path)
    LOGGER.info("Loaded %d QA rows for %d unique videos from %s.", len(qa_rows), len(video_ids), qa_path)
    video_inputs, skipped_videos = collect_video_inputs(
        video_ids=video_ids,
        qa_rows=qa_rows,
        data_root=args.data_root,
        frames_per_video=args.frames_per_video,
        missing_video_policy=args.missing_video_policy,
    )
    LOGGER.info("Resolved %d videos with frames; skipped %d.", len(video_inputs), len(skipped_videos))

    features = load_or_extract_features(args, feature_cache, video_inputs)
    features = l2_normalize(features.astype(np.float32), axis=1)
    kmeans = choose_or_run_kmeans(args, features)
    build_outputs(args, output_dir, feature_cache, video_inputs, qa_rows, features, kmeans, skipped_videos)


if __name__ == "__main__":
    main()
