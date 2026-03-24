from __future__ import annotations

import argparse
import sys
from pathlib import Path

from export import (
    build_main_payload,
    build_mouth_cues_payload,
    write_json,
)
from phonemizer_es import word_to_units
from timing import build_global_timeline, distribute_word_visemes
from transcription import transcribe_spanish_audio
from utils import PipelineError, WordViseme, ensure_audio_input, normalize_word
from viseme_mapper import units_to_visemes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Spanish viseme timings for avatar lip-sync using faster-whisper."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input audio file")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output viseme timeline JSON file",
    )
    parser.add_argument(
        "--mouth-cues",
        type=Path,
        default=None,
        help="Optional output JSON file for mouth cues format",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="faster-whisper model size/path (e.g. tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type for faster-whisper (auto, int8, float16, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size for transcription decoding",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.06,
        help="Minimum gap (seconds) to insert SIL in global timeline",
    )
    parser.add_argument(
        "--no-vad-filter",
        action="store_true",
        help="Disable VAD filter in faster-whisper",
    )
    parser.add_argument(
        "--skip-silence-cues",
        action="store_true",
        help="Do not insert SIL blocks for pauses in the global timeline",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_audio_input(args.input)
    transcription = transcribe_spanish_audio(
        audio_path=args.input,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad_filter,
    )

    words_with_visemes: list[WordViseme] = []
    for item in transcription.words:
        normalized = normalize_word(item.word)
        if not normalized:
            continue

        units = word_to_units(normalized)
        visemes = units_to_visemes(units)
        spans = distribute_word_visemes(item.start, item.end, visemes)
        words_with_visemes.append(
            WordViseme(word=normalized, start=item.start, end=item.end, visemes=spans)
        )

    if not words_with_visemes:
        raise PipelineError(
            "No usable words found after normalization. Check the audio quality and language."
        )

    timeline = build_global_timeline(
        words=words_with_visemes,
        include_silence=not args.skip_silence_cues,
        silence_threshold=max(0.0, args.silence_threshold),
    )

    payload = build_main_payload(
        audio_file=args.input,
        language=transcription.language,
        transcript=transcription.transcript,
        words=words_with_visemes,
        timeline=timeline,
        model_name=args.model,
    )
    write_json(args.output, payload)

    if args.mouth_cues is not None:
        cues_payload = build_mouth_cues_payload(timeline)
        write_json(args.mouth_cues, cues_payload)


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(args)
    except FileNotFoundError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1
    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 3

    print(f"Viseme JSON exported to: {args.output.resolve()}")
    if args.mouth_cues:
        print(f"Mouth cues JSON exported to: {args.mouth_cues.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
