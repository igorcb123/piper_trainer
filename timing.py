from __future__ import annotations

from typing import Sequence

from utils import VisemeSpan, WordViseme


VISEME_WEIGHTS: dict[str, float] = {
    "SIL": 1.0,
    "PP": 1.0,
    "FF": 1.0,
    "TD": 1.0,
    "SS": 1.0,
    "CH": 1.0,
    "KG": 1.0,
    "YL": 1.0,
    "AA": 1.25,
    "EE": 1.2,
    "II": 1.1,
    "OO": 1.2,
    "UU": 1.1,
}


def merge_adjacent_visemes(
    spans: Sequence[VisemeSpan],
    tolerance: float = 1e-3,
) -> list[VisemeSpan]:
    if not spans:
        return []

    ordered = sorted(spans, key=lambda item: (item.start, item.end))
    merged: list[VisemeSpan] = []

    for span in ordered:
        if span.end <= span.start:
            continue

        if not merged:
            merged.append(VisemeSpan(viseme=span.viseme, start=span.start, end=span.end))
            continue

        current = merged[-1]
        if span.viseme == current.viseme and span.start <= current.end + tolerance:
            current.end = max(current.end, span.end)
            continue

        start = max(span.start, current.end)
        if span.end <= start:
            continue
        merged.append(VisemeSpan(viseme=span.viseme, start=start, end=span.end))

    return merged


def distribute_word_visemes(
    word_start: float,
    word_end: float,
    visemes: Sequence[str],
) -> list[VisemeSpan]:
    duration = max(0.0, word_end - word_start)
    if duration <= 0.0:
        return []

    sequence = list(visemes) if visemes else ["SIL"]
    weights = [VISEME_WEIGHTS.get(vis, 1.0) for vis in sequence]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        total_weight = float(len(sequence))
        weights = [1.0 for _ in sequence]

    spans: list[VisemeSpan] = []
    consumed = 0.0

    for idx, (viseme, weight) in enumerate(zip(sequence, weights)):
        start = word_start + (consumed / total_weight) * duration
        consumed += weight
        if idx == len(sequence) - 1:
            end = word_end
        else:
            end = word_start + (consumed / total_weight) * duration
        spans.append(VisemeSpan(viseme=viseme, start=start, end=end))

    return merge_adjacent_visemes(spans)


def build_global_timeline(
    words: Sequence[WordViseme],
    include_silence: bool = True,
    silence_threshold: float = 0.06,
) -> list[VisemeSpan]:
    base_spans: list[VisemeSpan] = []
    for word in words:
        base_spans.extend(word.visemes)

    if not base_spans:
        return []

    base_spans.sort(key=lambda item: (item.start, item.end))
    timeline: list[VisemeSpan] = []
    cursor = 0.0

    for span in base_spans:
        if include_silence and span.start - cursor >= silence_threshold:
            timeline.append(VisemeSpan(viseme="SIL", start=cursor, end=span.start))

        start = max(span.start, cursor if timeline else span.start)
        if span.end <= start:
            cursor = max(cursor, span.end)
            continue

        timeline.append(VisemeSpan(viseme=span.viseme, start=start, end=span.end))
        cursor = max(cursor, span.end)

    return merge_adjacent_visemes(timeline)
