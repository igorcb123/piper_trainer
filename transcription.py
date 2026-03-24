from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils import PipelineError, WordTiming, join_words

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    WhisperModel = None  # type: ignore[assignment]


@dataclass
class TranscriptionResult:
    language: str
    transcript: str
    words: list[WordTiming]


def transcribe_spanish_audio(
    audio_path: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> TranscriptionResult:
    if WhisperModel is None:
        raise PipelineError(
            "Missing dependency: faster-whisper. Install with `pip install -r requirements.txt`."
        )

    model = WhisperModel(model_size_or_path=model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(audio_path),
        language="es",
        word_timestamps=True,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=True,
        temperature=0.0,
    )

    words: list[WordTiming] = []
    segment_text: list[str] = []

    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            segment_text.append(text)

        for word in segment.words or []:
            token = (word.word or "").strip()
            if not token or word.start is None or word.end is None:
                continue
            start = float(word.start)
            end = float(word.end)
            if end <= start:
                continue
            words.append(WordTiming(word=token, start=start, end=end))

    if not words:
        raise PipelineError(
            "No word-level timestamps were produced by faster-whisper. "
            "Try cleaner audio or a larger model."
        )

    words.sort(key=lambda item: (item.start, item.end))
    transcript = " ".join(segment_text).strip()
    if not transcript:
        transcript = join_words(words)

    language = (getattr(info, "language", None) or "es").lower()
    return TranscriptionResult(language=language, transcript=transcript, words=words)
