from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from utils import VisemeSpan, WordViseme, round_timestamp
from viseme_mapper import COMPACT_VISEMES


def _viseme_to_dict(span: VisemeSpan) -> dict[str, Any]:
    return {
        "viseme": span.viseme,
        "start": round_timestamp(span.start),
        "end": round_timestamp(span.end),
    }


def build_main_payload(
    audio_file: Path,
    language: str,
    transcript: str,
    words: Sequence[WordViseme],
    timeline: Sequence[VisemeSpan],
    model_name: str,
) -> dict[str, Any]:
    words_payload: list[dict[str, Any]] = []
    for item in words:
        words_payload.append(
            {
                "word": item.word,
                "start": round_timestamp(item.start),
                "end": round_timestamp(item.end),
                "visemes": [_viseme_to_dict(span) for span in item.visemes],
            }
        )

    return {
        "metadata": {
            "generator": "spanish-viseme-pipeline",
            "timestamp_precision_decimals": 3,
            "viseme_inventory": list(COMPACT_VISEMES),
            "model": model_name,
        },
        "audio_file": str(audio_file),
        "language": language,
        "transcript": transcript.strip(),
        "words": words_payload,
        "timeline": [_viseme_to_dict(span) for span in timeline],
    }


def build_mouth_cues_payload(timeline: Sequence[VisemeSpan]) -> dict[str, Any]:
    return {
        "mouthCues": [
            {
                "value": span.viseme,
                "start": round_timestamp(span.start),
                "end": round_timestamp(span.end),
            }
            for span in timeline
        ]
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
