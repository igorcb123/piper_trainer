#!/usr/bin/env python3
"""
build_ljspeech_dataset.py

Convert a Whisper transcription JSON into an LJSpeech-style dataset for Piper TTS.

Requirements:
- Python 3.10+
- pydub (and ffmpeg available on PATH for pydub to work)

Key behaviors:
- Loads a transcription dict (or JSON file).
- Splits the source audio per segment and exports WAVs (mono, 16-bit PCM).
- Produces `metadata.csv` in either classic LJSpeech or Piper-simple format.
- Deterministic 4-letter audio prefix derived from original audio filename.
- Skips invalid/too-short/too-long segments and logs reasons.
- Optional trimming of leading/trailing silence.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import shutil
import subprocess

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_ljspeech_dataset")

# --- Utilities ---


def load_transcription(source: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    """Load transcription from a JSON file path or return passed dict."""
    if isinstance(source, dict):
        return source
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Input JSON not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _remove_accents(text: str) -> str:
    """Remove accents for file-safe operations (use for prefix derivation)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def derive_audio_prefix(audio_path: str | Path) -> str:
    """
    Deterministic prefix derived from the audio filename.

    - Remove accents, keep letters/digits only, lowercase.
    - Use first 4 characters, padded with 'x' if too short.
    - Fall back to 'audx' if invalid.
    """
    stem = Path(audio_path).stem
    if not stem:
        return "audx"
    # Normalize: remove accents, non-alnum -> removed
    no_acc = _remove_accents(stem)
    cleaned = re.sub(r"[^A-Za-z0-9]", "", no_acc).lower()
    if not cleaned:
        return "audx"
    prefix = (cleaned[:4] + "xxxx")[:4]
    # ensure letters only if possible; if not, fallback
    if not re.search(r"[a-z0-9]", prefix):
        return "audx"
    return prefix


def normalize_text_es(text: str) -> str:
    """
    Conservative Spanish normalization:
    - Trim whitespace, collapse repeated spaces, remove line breaks.
    - Normalize Unicode quotes/dashes to simple equivalents.
    - Preserve accents and useful punctuation.
    - Remove obvious artifacts at the edges.
    """
    if text is None:
        return ""
    s = str(text)
    # Normalize quotes/dashes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("—", "-").replace("–", "-")
    # Remove/normalize newlines
    s = s.replace("\r", " ").replace("\n", " ")
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    # Trim leading/trailing odd artifacts (like leading/trailing '--', '... ')
    s = re.sub(r"^[\s\-\.\,\;:]+", "", s)
    s = re.sub(r"[\s\-\.\,\;:]+$", "", s)
    return s


def _ensure_wavs_dir(out_dir: Path) -> Path:
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    return wavs


def _existing_max_index(prefix: str, wavs_dir: Path) -> int:
    """Find the max numeric suffix already present for this prefix in the wavs dir."""
    max_idx = 0
    pattern = re.compile(re.escape(prefix) + r"_(\d{6})\.wav$")
    for p in wavs_dir.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    pass
    return max_idx


def _check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def export_segment(
    audio_path: Path,
    start_s: float,
    end_s: float,
    out_path: Path,
    sample_rate: int = 22050,
    trim_silence: bool = False,
    words: Optional[List[Dict[str, Any]]] = None,
    end_padding: float = 0.12,
) -> Tuple[bool, str]:
    """
    Use ffmpeg to extract a segment [start_s, end_s) from `audio_path` to `out_path`.
    Ensures mono, 16-bit PCM WAV and resamples to `sample_rate`.

    `trim_silence` is not implemented with ffmpeg here (kept for CLI compatibility).
    Returns (success, message).
    """
    if end_s <= start_s:
        return False, "invalid timestamps"
    if not _check_ffmpeg():
        return False, "ffmpeg not found in PATH"

    # Build ffmpeg command: -y overwrite, -i input, -ss START -to END (more accurate than -t),
    # resample to sample_rate, mono, s16
    # If word-level timestamps are provided, prefer their boundaries
    start = float(start_s)
    end = float(end_s)
    if words:
        try:
            w0 = words[0]
            wl = words[-1]
            # prefer word boundaries when available
            start = float(w0.get("start", start_s))
            end = float(wl.get("end", end_s))
        except Exception:
            start = float(start_s)
            end = float(end_s)

    # Add a small padding at the end to avoid chopping final syllables
    end += float(end_padding)
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        return True, f"exported ({duration:.3f}s)"
    except subprocess.CalledProcessError as exc:
        return False, f"ffmpeg failed: {exc}"


def generate_metadata(
    entries: List[Tuple[str, str, Optional[str]]],  # (filename, raw_text, normalized_text)
    metadata_path: Path,
    metadata_format: str = "classic",
) -> None:
    """
    Write metadata.csv in requested format.
    - classic: filename|raw_text|normalized_text
    - piper: filename|normalized_text
    """
    mode = metadata_format.lower()
    with metadata_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for filename, raw_text, norm_text in entries:
            if mode == "piper":
                writer.writerow([filename, norm_text])
            else:
                writer.writerow([filename, raw_text, norm_text])


# --- Main dataset builder ---


def build_ljspeech_dataset(
    transcription: Dict[str, Any],
    output_dir: Path,
    sample_rate: int = 22050,
    metadata_format: str = "classic",
    min_duration: float = 0.4,
    max_duration: Optional[float] = None,
    trim_silence: bool = False,
    end_padding: float = 0.25,
) -> Dict[str, Any]:
    """
    Build dataset from transcription dict.

    Returns summary dict.
    """
    summary = {
        "total_segments": 0,
        "exported_segments": 0,
        "skipped_segments": 0,
        "output_dir": str(output_dir.resolve()),
        "metadata_path": str((output_dir / "metadata.csv").resolve()),
        "audio_prefix": None,
        "sample_rate": sample_rate,
    }

    audio_file = transcription.get("audio_file")
    if not audio_file:
        raise ValueError("transcription['audio_file'] missing")

    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Verify ffmpeg is available (we use ffmpeg for slicing)
    logger.info("Using ffmpeg to extract segments (ffmpeg must be in PATH)")
    if not _check_ffmpeg():
        raise FileNotFoundError("ffmpeg not found in PATH. Install ffmpeg and try again.")

    # Prepare output dirs
    wavs_dir = _ensure_wavs_dir(output_dir)
    summary["audio_prefix"] = derive_audio_prefix(audio_path)

    # determine starting index to avoid collisions in wavs_dir
    start_idx = _existing_max_index(summary["audio_prefix"], wavs_dir) + 1
    idx = start_idx

    metadata_entries: List[Tuple[str, str, str]] = []

    segments = transcription.get("segments", [])
    for seg in segments:
        summary["total_segments"] += 1
        seg_id = seg.get("id")
        s = seg.get("start")
        e = seg.get("end")
        raw_text = (seg.get("text") or "").strip()

        # Validation
        if s is None or e is None:
            logger.warning("Skipping segment %s: missing timestamps", seg_id)
            summary["skipped_segments"] += 1
            continue
        if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
            logger.warning("Skipping segment %s: non-numeric timestamps", seg_id)
            summary["skipped_segments"] += 1
            continue
        if e <= s:
            logger.warning("Skipping segment %s: end <= start (%.3f <= %.3f)", seg_id, float(e), float(s))
            summary["skipped_segments"] += 1
            continue

        # Text filtering
        if not raw_text:
            logger.info("Skipping segment %s: empty text after strip", seg_id)
            summary["skipped_segments"] += 1
            continue

        dur = float(e) - float(s)
        if dur < min_duration:
            logger.info("Skipping segment %s: duration %.3fs < min %.3fs", seg_id, dur, min_duration)
            summary["skipped_segments"] += 1
            continue
        if max_duration is not None and dur > max_duration:
            logger.info("Skipping segment %s: duration %.3fs > max %.3fs", seg_id, dur, max_duration)
            summary["skipped_segments"] += 1
            continue

        # Prepare filename (unique)
        fname = f"{summary['audio_prefix']}_{idx:06d}.wav"
        out_path = wavs_dir / fname

        success, msg = export_segment(
            audio_path,
            float(s),
            float(e),
            out_path,
            sample_rate=sample_rate,
            trim_silence=trim_silence,
            words=seg.get("words"),
            end_padding=end_padding,
        )
        if not success:
            logger.warning("Failed to export segment %s: %s", seg_id, msg)
            summary["skipped_segments"] += 1
            continue

        norm_text = normalize_text_es(raw_text)
        metadata_entries.append((fname, raw_text, norm_text))
        logger.debug("Exported %s: %s", fname, msg)
        summary["exported_segments"] += 1
        idx += 1

    # Write metadata
    metadata_path = output_dir / "metadata.csv"
    generate_metadata(metadata_entries, metadata_path, metadata_format=metadata_format)
    logger.info("Wrote metadata to %s", metadata_path)

    return summary


# --- CLI ---


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LJSpeech-style dataset from Whisper JSON transcription.")
    parser.add_argument("--input-json", "-i", required=True, type=Path, help="Whisper transcription JSON file")
    parser.add_argument("--output-dir", "-o", required=True, type=Path, help="Output dataset directory")
    parser.add_argument(
        "--sample-rate",
        "-r",
        type=int,
        choices=[16000, 22050],
        default=22050,
        help="Output WAV sample rate (default 22050). 16000 supported.",
    )
    parser.add_argument(
        "--metadata-format",
        "-m",
        choices=["classic", "piper"],
        default="classic",
        help="Metadata format: classic => filename|raw_text|normalized_text; piper => filename|normalized_text",
    )
    parser.add_argument("--min-duration", type=float, default=0.4, help="Minimum allowed segment duration in seconds")
    parser.add_argument("--max-duration", type=float, default=None, help="Maximum allowed segment duration in seconds (optional; default: no maximum)")
    parser.add_argument("--trim-silence", action="store_true", help="Trim leading/trailing silence from exported segments")
    parser.add_argument(
        "--end-padding",
        type=float,
        default=0.25,
        help="Padding in seconds added after last word to avoid cutting final syllables (default 0.25)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    transcription = load_transcription(args.input_json)
    try:
        summary = build_ljspeech_dataset(
            transcription,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            metadata_format=args.metadata_format,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            trim_silence=args.trim_silence,
            end_padding=args.end_padding,
        )
    except Exception as exc:
        logger.exception("Failed to build dataset: %s", exc)
        return 2

    logger.info("Summary: total=%d exported=%d skipped=%d", summary["total_segments"], summary["exported_segments"], summary["skipped_segments"])
    logger.info("Output dir: %s", summary["output_dir"])
    logger.info("Metadata: %s", summary["metadata_path"])
    logger.info("Prefix: %s", summary["audio_prefix"])
    logger.info("Sample rate: %s", summary["sample_rate"])
    # Print JSON summary for programmatic consumption
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
