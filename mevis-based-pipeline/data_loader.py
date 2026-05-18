from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from schemas import ExpressionRecord, VideoRecord

LOGGER = logging.getLogger(__name__)


class MeViSDataLoader:
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.meta_path = self._resolve_meta_path()
        self.mask_path = self.data_root / "mask_dict.json"
        self.image_root = self.data_root / "JPEGImages"
        self.image_tar = self.data_root / "JPEGImages.tar"

    def _resolve_meta_path(self) -> Path:
        candidates = [
            # self.data_root / "meta_expressions_demo.json",
            self.data_root / "meta_expressions.json",
            self.data_root / "meta_expressions_v2.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find meta_expressions.json or meta_expressions_v2.json under {self.data_root}"
        )

    def load(self) -> Dict[str, VideoRecord]:
        if not self.meta_path.exists():
            raise FileNotFoundError(self.meta_path)
        if not self.mask_path.exists():
            raise FileNotFoundError(self.mask_path)

        with self.meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        with self.mask_path.open("r", encoding="utf-8") as f:
            mask_dict = json.load(f)

        videos: Dict[str, VideoRecord] = {}
        raw_videos = meta.get("videos", {})
        for video_key, video_payload in raw_videos.items():
            expressions = self._parse_expressions(video_payload.get("expressions", {}))
            video_id = int(video_payload.get("vid_id", -1))
            frame_names = list(video_payload.get("frames", []))
            video_record = VideoRecord(
                video_key=video_key,
                video_id=video_id,
                frame_names=frame_names,
                expressions=expressions,
                image_root=self._resolve_image_root(video_key),
                image_tar=self.image_tar if self.image_tar.exists() else None,
            )

            for expression in expressions.values():
                for anno_id in expression.anno_ids:
                    mask_key = str(anno_id)
                    if mask_key not in mask_dict:
                        LOGGER.warning("Missing mask entry for anno_id=%s in video=%s", anno_id, video_key)

            videos[video_key] = video_record

        LOGGER.info("Loaded %d videos from %s", len(videos), self.data_root)
        return videos

    def load_mask_sequence(self, anno_id: int) -> List[Optional[dict]]:
        with self.mask_path.open("r", encoding="utf-8") as f:
            mask_dict = json.load(f)
        sequence = mask_dict.get(str(anno_id))
        if sequence is None:
            return []
        return sequence

    def resolve_frame_path(self, video: VideoRecord, frame_index: int) -> Optional[Path]:
        if video.image_root is None:
            return None
        if frame_index < 0 or frame_index >= len(video.frame_names):
            return None

        frame_name = video.frame_names[frame_index]
        candidate_names = [frame_name]
        if not Path(frame_name).suffix:
            candidate_names.extend([f"{frame_name}.jpg", f"{frame_name}.jpeg", f"{frame_name}.png"])

        for candidate_name in candidate_names:
            candidate_path = video.image_root / candidate_name
            if candidate_path.exists():
                return candidate_path
        return None

    def _parse_expressions(self, expressions_payload: Dict[str, dict]) -> Dict[str, ExpressionRecord]:
        expressions: Dict[str, ExpressionRecord] = {}
        for exp_id, exp_payload in expressions_payload.items():
            expressions[str(exp_id)] = ExpressionRecord(
                exp_id=str(exp_id),
                expression=str(exp_payload.get("exp", "")),
                obj_ids=[int(obj_id) for obj_id in exp_payload.get("obj_id", [])],
                anno_ids=[int(anno_id) for anno_id in exp_payload.get("anno_id", [])],
            )
        return expressions

    def _resolve_image_root(self, video_key: str) -> Optional[Path]:
        candidate = self.image_root / video_key
        if candidate.exists():
            return candidate
        return None
