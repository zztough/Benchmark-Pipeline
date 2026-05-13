from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

from extractor import StructuredLogBuilder
from schemas import EntityLog, QAExample, StructuredLog, VideoRecord

LOGGER = logging.getLogger(__name__)


class QAGenerator:
    def __init__(self, log_builder: StructuredLogBuilder, seed: int = 42):
        self.log_builder = log_builder
        self.rng = random.Random(seed)

    def generate_for_video(self, video: VideoRecord, structured_log: StructuredLog) -> List[QAExample]:
        entity_items = list(structured_log.entities.items())
        if not entity_items:
            return []

        examples: List[QAExample] = []
        template_one = self._build_template_one(video, structured_log, entity_items)
        if template_one is not None:
            examples.append(template_one)

        if len(entity_items) == 1:
            return examples

        template_two = self._build_template_two(video, structured_log, entity_items)
        if template_two is not None:
            examples.append(template_two)

        return examples

    def _build_template_one(
        self,
        video: VideoRecord,
        structured_log: StructuredLog,
        entity_items: List[Tuple[str, EntityLog]],
    ) -> Optional[QAExample]:
        viable = [item for item in entity_items if item[1].actions]
        if not viable:
            return None

        entity_id, entity = self.rng.choice(viable)
        action = self.rng.choice(entity.actions)
        direction, _stats = self.log_builder.infer_direction(entity.mask_sequence, action.span)
        verb = "finish" if self._is_plural(entity.name) else "finishes"
        possessive = "their" if self._is_plural(entity.name) else "its"
        question = f"After {entity.name} {verb} {action.action}, what is {possessive} movement direction relative to the camera?"
        answer = self._normalize_answer(direction)
        return QAExample(
            video_id=video.video_key,
            question=question,
            answer=answer,
            template_name="template_after_action",
            entity_a=entity_id,
            action=action.action,
            span=list(action.span),
            direction_source="self",
        )

    def _build_template_two(
        self,
        video: VideoRecord,
        structured_log: StructuredLog,
        entity_items: List[Tuple[str, EntityLog]],
    ) -> Optional[QAExample]:
        if len(entity_items) < 2:
            return None

        viable = [item for item in entity_items if item[1].actions]
        if len(viable) < 2:
            return None

        unique_names = {entity.name for _, entity in viable}
        if len(unique_names) < 2:
            return None

        entity_a_id, entity_a = self.rng.choice(viable)
        remaining = [item for item in viable if item[0] != entity_a_id and item[1].name != entity_a.name]
        if not remaining:
            return None

        entity_b_id, entity_b = self.rng.choice(remaining)
        if entity_a.name == entity_b.name:
            return None
        action = self.rng.choice(entity_a.actions)
        direction, _stats = self.log_builder.infer_direction(entity_b.mask_sequence, action.span)
        be_verb = "are" if self._is_plural(entity_a.name) else "is"
        question = f"While {entity_a.name} {be_verb} {action.action}, what is the movement direction of {entity_b.name} relative to the camera?"
        answer = self._normalize_answer(direction)
        return QAExample(
            video_id=video.video_key,
            question=question,
            answer=answer,
            template_name="template_while_action",
            entity_a=entity_a_id,
            entity_b=entity_b_id,
            action=action.action,
            span=list(action.span),
            direction_source="other",
        )

    def _normalize_answer(self, direction: str) -> str:
        if not direction:
            return "unknown"
        return direction.replace("_", " ")

    def _is_plural(self, name: str) -> bool:
        tokens = [token for token in name.lower().split() if token not in {"the", "a", "an"}]
        if not tokens:
            return False
        if any(token in {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "pair", "trio", "group", "duo", "many", "several"} for token in tokens):
            return True
        last_token = tokens[-1]
        if last_token.endswith("s") and last_token not in {"us", "is"}:
            return True
        return False
