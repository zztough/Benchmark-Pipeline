from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExpressionRecord:
    exp_id: str
    expression: str
    obj_ids: List[int] = field(default_factory=list)
    anno_ids: List[int] = field(default_factory=list)


@dataclass
class VideoRecord:
    video_key: str
    video_id: int
    frame_names: List[str]
    expressions: Dict[str, ExpressionRecord]
    image_root: Optional[Path] = None
    image_tar: Optional[Path] = None


@dataclass
class ActionSpan:
    action: str
    span: Tuple[int, int]


@dataclass
class EntityLog:
    name: str
    actions: List[ActionSpan] = field(default_factory=list)
    mask_sequence: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    source_exp: str = ""
    source_exp_id: str = ""


@dataclass
class StructuredLog:
    video_id: str
    entities: Dict[str, EntityLog]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "entities": {
                entity_id: {
                    "name": entity.name,
                    "actions": [
                        {"action": action.action, "span": list(action.span)}
                        for action in entity.actions
                    ],
                    "mask_sequence": entity.mask_sequence,
                    "source_exp": entity.source_exp,
                    "source_exp_id": entity.source_exp_id,
                }
                for entity_id, entity in self.entities.items()
            },
        }


@dataclass
class QAExample:
    video_id: str
    question: str
    answer: str
    template_name: str
    entity_a: str
    entity_b: Optional[str] = None
    action: str = ""
    span: Optional[List[int]] = None
    direction_source: str = ""
    evidence: Optional[Dict[str, Any]] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)
