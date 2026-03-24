from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
_DIACRITIC_MAP = str.maketrans({"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"})


class PipelineError(Exception):
    """Raised when the viseme pipeline cannot continue safely."""


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


@dataclass
class VisemeSpan:
    viseme: str
    start: float
    end: float


@dataclass
class WordViseme:
    word: str
    start: float
    end: float
    visemes: list[VisemeSpan]


def round_timestamp(value: float) -> float:
    return round(max(0.0, float(value)) + 1e-9, 3)


def normalize_word(word: str) -> str:
    text = word.strip().lower()
    if not text:
        return ""

    text = text.replace("’", "'").replace("`", "'")
    text = text.translate(_DIACRITIC_MAP)
    text = re.sub(r"^[^a-z0-9ñü]+|[^a-z0-9ñü]+$", "", text)
    text = re.sub(r"[^a-z0-9ñü'-]+", "", text)
    return text.strip("-'")


def join_words(words: Iterable[WordTiming]) -> str:
    return " ".join(item.word for item in words if item.word).strip()


def ensure_audio_input(audio_path: Path) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise PipelineError(f"Input path is not a file: {audio_path}")

    extension = audio_path.suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise PipelineError(
            f"Unsupported audio extension '{extension}'. Allowed extensions: {allowed}"
        )
