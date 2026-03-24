import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    torch = None  # type: ignore[assignment]

try:
    import whisper
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    whisper = None  # type: ignore[assignment]


DEFAULT_MODEL = "large-v3"
PHRASE_END_RE = re.compile(r"[.!?]+[\"')\]]*$")


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: Optional[str],
    normalize_phrases_flag: bool = True,
) -> Dict[str, Any]:
    if whisper is None:
        raise RuntimeError(
            "Missing dependency: openai-whisper. Install with `pip install -r requirements.txt`."
        )
    model = whisper.load_model(model_name)
    options: Dict[str, Any] = {
        "temperature": 0.0,
        "beam_size": 5,
        "condition_on_previous_text": True,
        "fp16": bool(torch and torch.cuda.is_available()),
        "word_timestamps": True,
    }
    if language:
        options["language"] = language

    result = model.transcribe(str(audio_path), **options)

    segments: List[Dict[str, Any]] = []
    for segment in result.get("segments", []):
        segments.append(
            {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip(),
                "words": segment.get("words", []),
            }
        )

    phrases = build_phrases(segments)
    if normalize_phrases_flag:
        phrases = normalize_phrases(phrases, min_dur=1.5, max_dur=7.0)

    return {
        "audio_file": str(audio_path.resolve()),
        "model": model_name,
        "language": result.get("language", language),
        "text": result.get("text", "").strip(),
        "segments": segments,
        "phrases": phrases,
    }


def build_phrases(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    phrases: List[Dict[str, Any]] = []
    current_words: List[Dict[str, Any]] = []

    def flush_phrase() -> None:
        if not current_words:
            return
        start = current_words[0].get("start")
        end = current_words[-1].get("end")
        text = join_words(current_words)
        phrases.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": list(current_words),
            }
        )
        current_words.clear()

    for segment in segments:
        words = segment.get("words") or []
        if words:
            for word in words:
                current_words.append(word)
                token = (word.get("word") or word.get("text") or "").strip()
                if token and PHRASE_END_RE.search(token):
                    flush_phrase()
            continue

        segment_text = (segment.get("text") or "").strip()
        if segment_text:
            phrases.append(
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment_text,
                    "words": [],
                }
            )

    flush_phrase()
    return phrases


def _phrase_duration(phrase: Dict[str, Any]) -> float:
    start = phrase.get("start")
    end = phrase.get("end")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return 0.0
    return max(0.0, float(end) - float(start))


def _rebuild_phrase_from_words(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    start = float(words[0].get("start", 0.0))
    end = float(words[-1].get("end", start))
    return {
        "start": start,
        "end": end,
        "text": join_words(words),
        "words": list(words),
    }


def _split_phrase_if_long(phrase: Dict[str, Any], max_dur: float) -> List[Dict[str, Any]]:
    words = phrase.get("words") or []
    if not words:
        return [phrase]

    duration = _phrase_duration(phrase)
    if duration <= max_dur:
        return [phrase]

    chunks: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    for word in words:
        current.append(word)
        candidate = _rebuild_phrase_from_words(current)
        if _phrase_duration(candidate) >= max_dur and len(current) > 1:
            last = current.pop()
            chunks.append(_rebuild_phrase_from_words(current))
            current = [last]

    if current:
        chunks.append(_rebuild_phrase_from_words(current))

    return chunks


def normalize_phrases(
    phrases: List[Dict[str, Any]],
    min_dur: float = 1.5,
    max_dur: float = 7.0,
) -> List[Dict[str, Any]]:
    """
    Normalize phrase durations for downstream dataset building.
    - Split very long phrases (when word timings are available).
    - Merge short adjacent phrases to reach `min_dur` when possible.
    """
    if not phrases:
        return []

    split_phrases: List[Dict[str, Any]] = []
    for phrase in phrases:
        split_phrases.extend(_split_phrase_if_long(phrase, max_dur=max_dur))

    normalized: List[Dict[str, Any]] = []
    i = 0

    while i < len(split_phrases):
        current = split_phrases[i]
        words = list(current.get("words") or [])
        start = current.get("start")
        end = current.get("end")
        text = (current.get("text") or "").strip()

        while _phrase_duration({"start": start, "end": end}) < min_dur and i + 1 < len(split_phrases):
            i += 1
            nxt = split_phrases[i]
            next_words = list(nxt.get("words") or [])
            words.extend(next_words)
            end = nxt.get("end", end)
            next_text = (nxt.get("text") or "").strip()
            if text and next_text:
                text = f"{text} {next_text}"
            elif next_text:
                text = next_text

        if words:
            normalized.append(_rebuild_phrase_from_words(words))
        else:
            normalized.append(
                {
                    "start": start,
                    "end": end,
                    "text": text.strip(),
                    "words": [],
                }
            )
        i += 1

    return normalized


def join_words(words: List[Dict[str, Any]]) -> str:
    tokens: List[str] = []
    for word in words:
        token = (word.get("word") or word.get("text") or "").strip()
        if not token:
            continue
        if not tokens:
            tokens.append(token)
            continue
        if token[:1] in ".,!?;:)]}":
            tokens[-1] = tokens[-1] + token
        elif tokens[-1].endswith(("(", "[", "{", "\"", "'")):
            tokens[-1] = tokens[-1] + token
        else:
            tokens.append(token)
    return " ".join(tokens).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio with Whisper and save the result as JSON."
    )
    parser.add_argument("audio", type=Path, help="Ruta al archivo de audio")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Ruta del archivo JSON de salida. Por defecto usa el nombre del audio con .json",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="Modelo de Whisper a usar: tiny, base, small, medium, large, large-v3",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Idioma del audio, por ejemplo: es, en, pt. Si no se indica, Whisper lo detecta.",
    )
    parser.add_argument(
        "--normalize-phrases",
        dest="normalize_phrases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Asegura que las frases estén entre 1.5s y 7s (por defecto: True).",
    )
    args = parser.parse_args()

    audio_path = args.audio
    if not audio_path.exists():
        raise FileNotFoundError(f"No existe el archivo de audio: {audio_path}")

    output_path = args.output or audio_path.with_suffix(".json")
    data = transcribe_audio(audio_path, args.model, args.language, args.normalize_phrases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Transcripcion guardada en: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
