from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from api_clients import DeepSeekExtractionInterface, QwenVTGInterface
from data_loader import MeViSDataLoader
from geometry import infer_direction_from_mask_span
from schemas import ActionSpan, EntityLog, StructuredLog, VideoRecord

LOGGER = logging.getLogger(__name__)

ENTITY_HINTS = {
    "elephant",
    "elephants",
    "lizard",
    "lizards",
    "rabbit",
    "rabbits",
    "monkey",
    "monkeys",
    "horse",
    "horses",
    "car",
    "cars",
    "truck",
    "trucks",
    "bear",
    "bears",
    "dog",
    "dogs",
    "cat",
    "cats",
    "bird",
    "birds",
    "turtle",
    "turtles",
    "fish",
    "goat",
    "goats",
    "panda",
    "pandas",
    "cow",
    "cows",
    "sheep",
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "yak",
    "yaks",
    "bicycle",
    "bike",
    "plane",
    "airplane",
    "aircraft",
}

ACTION_KEYWORDS = {
    "move",
    "moving",
    "walk",
    "walking",
    "turn",
    "turning",
    "face",
    "facing",
    "eat",
    "eating",
    "jump",
    "jumping",
    "run",
    "running",
    "stand",
    "standing",
    "sit",
    "sitting",
    "lie",
    "lying",
    "crouch",
    "crouching",
    "climb",
    "climbing",
    "hold",
    "holding",
    "pull",
    "pulling",
    "push",
    "pushing",
    "follow",
    "following",
    "chase",
    "chasing",
    "swim",
    "swimming",
    "drive",
    "driving",
    "fly",
    "flying",
    "pick",
    "picking",
    "carry",
    "carrying",
    "press",
    "pressing",
    "guide",
    "guiding",
    "play",
    "playing",
}

ACTION_TAILS = {
    "around",
    "away",
    "back",
    "backward",
    "backwards",
    "forward",
    "left",
    "right",
    "up",
    "down",
    "toward",
    "towards",
    "us",
    "the",
    "in",
    "out",
    "circularly",
    "circle",
    "circles",
}


@dataclass
class ExtractionResult:
    name: str
    actions: List[str]


class MockLLMInterface:
    """Heuristic placeholder for a real LLM extraction service."""

    def extract_name_and_actions(self, expression: str, video_context: Optional[VideoRecord] = None) -> ExtractionResult:
        cleaned = self._normalize_expression(expression)
        name = self._extract_name(cleaned)
        actions = self._extract_actions(cleaned, name)
        return ExtractionResult(name=name, actions=actions)

    def _normalize_expression(self, expression: str) -> str:
        return re.sub(r"\s+", " ", expression).strip()

    def _extract_name(self, expression: str) -> str:
        tokens = re.findall(r"[A-Za-z']+", expression.lower())
        if not tokens:
            return "entity"

        first_hit = None
        for index, token in enumerate(tokens):
            if token in ENTITY_HINTS:
                first_hit = index
                break

        if first_hit is None:
            fallback = " ".join(tokens[:3]) if len(tokens) >= 3 else tokens[0]
            return self._ensure_article(fallback)

        start = first_hit
        while start > 0 and tokens[start - 1] not in {"and", "then", "while", "after", "before", "or"}:
            candidate = tokens[start - 1]
            if candidate in {"the", "a", "an", "of", "to", "from", "with", "without", "in", "on", "at", "by"}:
                break
            start -= 1
            if first_hit - start >= 3:
                break

        name_tokens = tokens[start:first_hit + 1]
        if not name_tokens:
            name_tokens = tokens[first_hit:first_hit + 1]
        return self._ensure_article(" ".join(name_tokens))

    def _ensure_article(self, name: str) -> str:
        if not name:
            return "entity"
        stripped = name.strip()
        if stripped.startswith(("the ", "a ", "an ")):
            return stripped
        return f"the {stripped}"

    def _extract_actions(self, expression: str, name: str) -> List[str]:
        normalized = expression.lower().strip().rstrip(".")
        raw_name = name.lower().removeprefix("the ").strip()
        if raw_name and raw_name in normalized:
            normalized = normalized.replace(raw_name, "", 1).strip()

        normalized = normalized.lstrip(" ,:-")
        if not normalized:
            return ["moving"]

        clauses = self._split_clauses(normalized)
        actions: List[str] = []
        for clause in clauses:
            clause = clause.strip(" ,.-")
            if clause:
                actions.append(self._compact_action(clause))

        if not actions:
            actions = [normalized]
        return actions

    def _split_clauses(self, text: str) -> List[str]:
        pattern = re.compile(r"(?:\band then\b|\bthen\b|\bwhile\b|\bafter\b|\bbefore\b|,|;)", re.IGNORECASE)
        parts = [part.strip() for part in pattern.split(text) if part.strip()]
        if len(parts) > 1:
            return parts

        if " and " in text:
            candidates = [segment.strip() for segment in text.split(" and ") if segment.strip()]
            if len(candidates) > 1:
                return candidates

        return [text]

    def _compact_action(self, clause: str) -> str:
        tokens = re.findall(r"[A-Za-z']+", clause.lower())
        if not tokens:
            return clause.strip()

        first_verb = None
        for index, token in enumerate(tokens):
            if token in ACTION_KEYWORDS:
                first_verb = index
                break

        if first_verb is None:
            return " ".join(tokens[:4])

        compacted = [tokens[first_verb]]
        for token in tokens[first_verb + 1 :]:
            if token in ACTION_TAILS:
                compacted.append(token)
                continue
            break

        return " ".join(compacted)


class MockVTGInterface:
    """Heuristic placeholder for a real temporal grounding model."""

    def ground_action_spans(
        self,
        mask_sequence: Sequence[Optional[dict]],
        actions: Sequence[str],
    ) -> List[Tuple[int, int]]:
        valid_indices = [index for index, rle in enumerate(mask_sequence) if rle is not None]
        if not valid_indices:
            return []

        start = valid_indices[0]
        end = valid_indices[-1]
        if not actions:
            return [(start, end)]

        total = max(end - start + 1, 1)
        span_length = max(int(math.ceil(total / max(len(actions), 1))), 1)
        spans: List[Tuple[int, int]] = []
        cursor = start
        for index, _action in enumerate(actions):
            next_cursor = end if index == len(actions) - 1 else min(end, cursor + span_length - 1)
            spans.append((cursor, next_cursor))
            cursor = min(end, next_cursor + 1)

        if spans and spans[-1][1] < end:
            spans[-1] = (spans[-1][0], end)
        return spans


class StructuredLogBuilder:
    def __init__(
        self,
        data_loader: MeViSDataLoader,
        llm: Optional[DeepSeekExtractionInterface] = None,
        vtg: Optional[QwenVTGInterface] = None,
        depth_estimator: Optional[object] = None,
        mask_debug_dir: Optional[str | Path] = None,
    ):
        self.data_loader = data_loader
        self.llm = llm or DeepSeekExtractionInterface()
        self.vtg = vtg or QwenVTGInterface()
        self.depth_estimator = depth_estimator
        self.mask_debug_dir = Path(mask_debug_dir) if mask_debug_dir is not None else None
        self._active_video: Optional[VideoRecord] = None

    def build_for_video(self, video: VideoRecord) -> Optional[StructuredLog]:
        self._active_video = video
        entities: Dict[str, EntityLog] = {}
        seen_anno_ids = set()
        for exp_id, expression_record in video.expressions.items():
            if not expression_record.anno_ids:
                LOGGER.warning(
                    "Skip expression without anno_id: video=%s exp_id=%s exp=%s",
                    video.video_key,
                    exp_id,
                    expression_record.expression,
                )
                continue

            if len(expression_record.anno_ids) != 1:
                LOGGER.warning(
                    "Skip expression with multiple anno_ids: video=%s exp_id=%s anno_ids=%s exp=%s",
                    video.video_key,
                    exp_id,
                    expression_record.anno_ids,
                    expression_record.expression,
                )
                continue

            anno_id = expression_record.anno_ids[0]
            if anno_id in seen_anno_ids:
                LOGGER.warning(
                    "Skip duplicated anno_id in same video: video=%s exp_id=%s anno_id=%s exp=%s",
                    video.video_key,
                    exp_id,
                    anno_id,
                    expression_record.expression,
                )
                continue

            try:
                extraction = self.llm.extract_name_and_actions(expression_record.expression, video)
            except Exception as exc:
                LOGGER.warning(
                    "LLM extraction failed: video=%s exp_id=%s exp=%s err=%s",
                    video.video_key,
                    exp_id,
                    expression_record.expression,
                    exc,
                )
                continue

            mask_sequence = self.data_loader.load_mask_sequence(anno_id)
            if not mask_sequence:
                LOGGER.warning("Missing or empty mask sequence: video=%s anno_id=%s", video.video_key, anno_id)
                continue

            try:
                spans = self.vtg.ground_action_spans(
                    video,
                    extraction.actions,
                    query_text=expression_record.expression,
                    entity_name=extraction.name,
                )
            except Exception as exc:
                LOGGER.warning(
                    "VTG grounding failed: video=%s anno_id=%s exp=%s err=%s",
                    video.video_key,
                    anno_id,
                    expression_record.expression,
                    exc,
                )
                continue

            if not spans:
                LOGGER.warning(
                    "No grounded span found: video=%s anno_id=%s exp=%s",
                    video.video_key,
                    anno_id,
                    expression_record.expression,
                )
                continue

            entity_key = f"anno_id_{anno_id}"
            actions = [ActionSpan(action=action, span=span) for action, span in zip(extraction.actions, spans)]
            entities[entity_key] = EntityLog(
                name=extraction.name,
                actions=actions,
                mask_sequence=mask_sequence,
                source_exp=expression_record.expression,
                source_exp_id=exp_id,
            )
            seen_anno_ids.add(anno_id)

        if not entities:
            return None

        return StructuredLog(video_id=video.video_key, entities=entities)

    def infer_direction(
        self,
        mask_sequence: Sequence[Optional[dict]],
        span: Tuple[int, int],
        video: Optional[VideoRecord] = None,
    ) -> Tuple[str, Optional[dict]]:
        active_video = video or self._active_video
        frame_path_resolver = None
        if active_video is not None:
            frame_path_resolver = lambda frame_index: self.data_loader.resolve_frame_path(active_video, frame_index)
        mask_debug_prefix = ""
        if active_video is not None:
            mask_debug_prefix = f"{active_video.video_key}__span_{span[0]}_{span[1]}"
        return infer_direction_from_mask_span(
            mask_sequence,
            span,
            frame_path_resolver=frame_path_resolver,
            depth_estimator=self.depth_estimator,
            mask_debug_dir=self.mask_debug_dir,
            mask_debug_prefix=mask_debug_prefix,
        )
