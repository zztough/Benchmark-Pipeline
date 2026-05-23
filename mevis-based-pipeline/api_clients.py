from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

LOGGER = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    name: str
    actions: List[str]


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return stripped


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text!r}")

    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON is not an object")
    return payload


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_action_list(actions: Any) -> List[str]:
    if not isinstance(actions, list):
        raise ValueError(f"Expected actions to be a list, got {type(actions)!r}")

    normalized: List[str] = []
    for item in actions:
        text = _normalize_text(item)
        if text:
            normalized.append(text.lower())
    return normalized


def _write_vtg_response_log(response: Any, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(response.model_dump_json(indent=2))
        handle.write("\n")


class DeepSeekExtractionInterface:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-ai/DeepSeek-V4-Flash",
        client: Optional[OpenAI] = None,
    ):
        api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        base_url = base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        if client is None:
            if not api_key:
                raise ValueError(
                    "Missing SiliconFlow API key. Set SILICONFLOW_API_KEY or pass api_key to DeepSeekExtractionInterface."
                )
            client = OpenAI(api_key=api_key, base_url=base_url)

        self.client = client
        self.model = model

    def extract_name_and_actions(self, expression: str, video_context: Optional[object] = None) -> ExtractionResult:
        system_prompt = (
            "You are a professional video referring-expression parser. "
            "Your job is to extract one unique target entity and an ordered list of atomic action phrases from a single expression. "
            "Return JSON only, with no markdown, no explanation, and no extra keys. "
            "Use English only. "
            'The JSON schema must be: {"name": string, "actions": [string, ...]}. '
            "The entity name should be a noun phrase that RETAINS its descriptive modifiers "
            "(such as adjectives, prepositional phrases like 'in the distance', or relative clauses) "
            "to preserve spatial and descriptive context. "
            "For example, extract 'the elephant in the distance' instead of just 'the elephant'. "
            "Keep plurality when needed. "
            "The actions list should preserve temporal order and contain short verb phrases; split compound or sequential actions into multiple items when needed. "
            "If no explicit action is given, infer the most salient motion phrase."
        )
        context_payload: Dict[str, Any] = {"expression": expression}
        if video_context is not None:
            context_payload["video_key"] = getattr(video_context, "video_key", "unknown")
            context_payload["video_id"] = getattr(video_context, "video_id", -1)
            frame_names = getattr(video_context, "frame_names", [])
            context_payload["frame_count"] = len(frame_names) if frame_names is not None else 0

        user_prompt = (
            "Parse the following referring expression under the constraints above. "
            "Prefer the main referenced entity only. "
            "If the expression is already scoped to one unique annotation, keep the name aligned with that target.\n\n"
            f"Context: {json.dumps(context_payload, ensure_ascii=False)}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content or ""
        payload = _extract_json_object(content)
        name = _normalize_text(payload.get("name", "entity"))
        if name and not name.lower().startswith(("the ", "a ", "an ")):
            name = f"the {name}"

        actions = _normalize_action_list(payload.get("actions", []))
        if not actions:
            actions = ["moving"]
        return ExtractionResult(name=name or "entity", actions=actions)


class QwenVTGInterface:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "Qwen/Qwen3.6-35B-A3B",
        max_frames: int = 12,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        extra_body: Optional[Dict[str, Any]] = None,
        response_log_path: Optional[Path | str] = None,
        client: Optional[OpenAI] = None,
    ):
        api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        base_url = base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        if client is None:
            if not api_key:
                raise ValueError(
                    "Missing SiliconFlow API key. Set SILICONFLOW_API_KEY or pass api_key to QwenVTGInterface."
                )
            client = OpenAI(api_key=api_key, base_url=base_url)

        self.client = client
        self.model = model
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.response_log_path = Path(response_log_path) if response_log_path else None

    def ground_action_spans(
        self,
        video_context: Optional[object],
        actions: Sequence[str],
        query_text: Optional[str] = None,
        entity_name: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        frame_records = self._build_frame_records(video_context, max_frames=self.max_frames)
        if not frame_records:
            return []

        if not actions:
            return [(frame_records[0]["frame_idx"], frame_records[-1]["frame_idx"])]

        system_prompt = (
            "You are a professional temporal grounding model for video referring expressions. "
            "You will receive sampled raw video frames in chronological order. "
            "Use only the visual content of these frames to infer when each action starts and ends; do not rely on masks, box tracks, or other auxiliary annotations. "
            "Assign one inclusive frame span to each action. "
            "Return JSON only, with no explanation, no markdown, no prose, and no reasoning text. "
            'The JSON schema must be: {"spans": [[start, end], ...]}. '
            "Spans must use original frame indices, be ordered, non-overlapping, and aligned to the actions order. "
            "Use integers only. "
            "The number of spans must exactly equal the number of actions. "
            "If uncertain, choose the most stable and plausible interval, and prefer contiguous spans. "
            "Keep the answer as short as possible; usually one compact JSON object is enough."
        )
        user_prompt = (
            "Ground the actions against the sampled raw frames below. "
            "The frames are ordered by time and each entry includes its original frame index. "
            "Use only the visual content in the images to judge the motion interval of each action. "
            "If a referring expression or entity name is provided, use it only as a brief textual anchor, not as additional visual evidence. "
            "Output exactly one JSON object and stop immediately after the closing brace. "
            'Do not include keys other than "spans". '
            "Do not include confidence, comments, analysis, or frame descriptions. "
            'Example output for two actions: {"spans":[[0,12],[13,47]]}\n\n'
            f"{json.dumps({'actions': list(actions), 'action_count': len(actions), 'sampled_frame_count': len(frame_records), 'query_text': query_text, 'entity_name': entity_name}, ensure_ascii=False)}"
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_multimodal_user_content(user_prompt, frame_records)},
            ]
            request_kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": messages,
                "max_tokens": self.max_tokens,
            }
            if self.extra_body:
                request_kwargs["extra_body"] = self.extra_body
            response = self.client.chat.completions.create(**request_kwargs)
            if self.response_log_path:
                _write_vtg_response_log(response, self.response_log_path)
            content = response.choices[0].message.content or ""
            payload = _extract_json_object(content)
            spans = self._normalize_spans(payload.get("spans", payload), len(actions), frame_records)
            if spans:
                return spans
        except Exception as exc:
            LOGGER.warning("VTG API grounding failed, falling back to heuristic spans: %s", exc)

        return self._fallback_spans(frame_records, len(actions))

    def _build_multimodal_user_content(self, user_prompt: str, frame_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for record in frame_records:
            frame_idx = record["frame_idx"]
            content.append({"type": "text", "text": f"Frame index {frame_idx}."})
            image_data_uri = record.get("image_data_uri")
            if image_data_uri:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri, "detail": "low"},
                    }
                )
        return content

    def _build_frame_records(
        self,
        video_context: Optional[object],
        max_frames: int = 12,
    ) -> List[Dict[str, Any]]:
        if video_context is None:
            return []

        image_root = getattr(video_context, "image_root", None)
        frame_names = list(getattr(video_context, "frame_names", []) or [])
        if image_root is None or not frame_names:
            return []

        selected_indices = self._select_frame_indices(len(frame_names), max_frames=max_frames)
        records: List[Dict[str, Any]] = []
        for frame_idx in selected_indices:
            if frame_idx < 0 or frame_idx >= len(frame_names):
                continue
            frame_path = self._resolve_frame_path(Path(image_root), frame_names[frame_idx])
            if frame_path is None:
                continue

            record: Dict[str, Any] = {
                "frame_idx": frame_idx,
                "image_data_uri": self._encode_image_to_data_uri(frame_path),
            }
            records.append(record)

        return records

    def _select_frame_indices(self, frame_count: int, max_frames: int = 12) -> List[int]:
        if frame_count <= 0:
            return []

        source_indices = list(range(frame_count))
        if frame_count <= max_frames:
            return source_indices

        selected = {source_indices[0], source_indices[-1]}
        if max_frames > 2:
            span = frame_count - 1
            for index in range(1, max_frames - 1):
                position = round(index * span / (max_frames - 1))
                selected.add(source_indices[position])

        return sorted(selected)

    def _resolve_frame_path(self, image_root: Path, frame_name: str) -> Optional[Path]:
        candidate_names = [frame_name]
        if not Path(frame_name).suffix:
            candidate_names.extend([f"{frame_name}.jpg", f"{frame_name}.jpeg", f"{frame_name}.png"])

        for candidate_name in candidate_names:
            candidate_path = image_root / candidate_name
            if candidate_path.exists():
                return candidate_path
        return None

    def _encode_image_to_data_uri(self, frame_path: Path) -> str:
        image_bytes = frame_path.read_bytes()
        suffix = frame_path.suffix.lower()
        mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png" if suffix == ".png" else "application/octet-stream"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _normalize_spans(
        self,
        raw_spans: Any,
        action_count: int,
        valid_frames: Sequence[Dict[str, Any]],
    ) -> List[Tuple[int, int]]:
        spans_candidate = raw_spans
        if isinstance(raw_spans, dict):
            for key in ("spans", "span", "frames"):
                if key in raw_spans:
                    spans_candidate = raw_spans[key]
                    break

        if action_count == 1 and isinstance(spans_candidate, dict):
            spans_candidate = [spans_candidate]

        if not isinstance(spans_candidate, list):
            return []

        if len(spans_candidate) != action_count:
            if action_count == 1 and len(spans_candidate) == 1:
                pass
            else:
                return []

        normalized: List[Tuple[int, int]] = []
        valid_frame_indices = [int(frame["frame_idx"]) for frame in valid_frames]
        frame_min = min(valid_frame_indices)
        frame_max = max(valid_frame_indices)

        for span in spans_candidate:
            start, end = self._coerce_span(span)
            if start is None or end is None:
                return []
            start = max(frame_min, min(frame_max, start))
            end = max(start, min(frame_max, end))
            normalized.append((start, end))

        return normalized

    def _coerce_span(self, span: Any) -> Tuple[Optional[int], Optional[int]]:
        if isinstance(span, dict):
            start = span.get("start", span.get("begin"))
            end = span.get("end", span.get("stop"))
            try:
                return int(start), int(end)
            except (TypeError, ValueError):
                return None, None

        if isinstance(span, (list, tuple)) and len(span) == 2:
            try:
                return int(span[0]), int(span[1])
            except (TypeError, ValueError):
                return None, None

        if isinstance(span, str):
            match = re.search(r"(-?\d+)\D+(-?\d+)", span)
            if match:
                return int(match.group(1)), int(match.group(2))

        return None, None

    def _fallback_spans(self, valid_frames: Sequence[Dict[str, Any]], action_count: int) -> List[Tuple[int, int]]:
        valid_indices = [int(frame["frame_idx"]) for frame in valid_frames]
        if not valid_indices:
            return []

        start = valid_indices[0]
        end = valid_indices[-1]
        if action_count <= 1:
            return [(start, end)]

        total = max(end - start + 1, 1)
        span_length = max(int((total + action_count - 1) / action_count), 1)
        spans: List[Tuple[int, int]] = []
        cursor = start
        for index in range(action_count):
            next_cursor = end if index == action_count - 1 else min(end, cursor + span_length - 1)
            spans.append((cursor, next_cursor))
            cursor = min(end, next_cursor + 1)

        if spans and spans[-1][1] < end:
            spans[-1] = (spans[-1][0], end)
        return spans
