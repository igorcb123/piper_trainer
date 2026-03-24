from __future__ import annotations

from utils import normalize_word


_VOWEL_MAP = {
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "ü": "u",
}


def _norm_vowel(ch: str) -> str:
    return _VOWEL_MAP.get(ch, ch)


def word_to_units(word: str) -> list[str]:
    """
    Practical Spanish grapheme-to-pronunciation units for lip-sync.
    This is intentionally compact and animation-oriented, not linguistic-grade G2P.
    """
    text = normalize_word(word)
    if not text:
        return []

    units: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        chunk3 = text[i : i + 3]
        chunk2 = text[i : i + 2]
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        nxt2 = text[i + 2] if i + 2 < n else ""

        if chunk3 in {"güe", "güi"}:
            units.extend(["g", "u", chunk3[-1]])
            i += 3
            continue

        if chunk2 in {"ch", "ll", "rr"}:
            units.append(chunk2)
            i += 2
            continue

        if ch in _VOWEL_MAP:
            units.append(_norm_vowel(ch))
            i += 1
            continue

        if ch in {"'", "-"}:
            i += 1
            continue

        if ch == "h":
            i += 1
            continue

        if ch in {"b", "v"}:
            units.append("b")
            i += 1
            continue

        if ch == "q":
            if nxt == "u" and nxt2 in "ei":
                units.extend(["k", nxt2])
                i += 3
                continue
            units.append("k")
            i += 1
            continue

        if ch == "g":
            if nxt == "u" and nxt2 in "ei":
                units.extend(["g", nxt2])
                i += 3
                continue
            if nxt in "ei":
                units.extend(["j", nxt])
                i += 2
                continue
            units.append("g")
            i += 1
            continue

        if ch == "c":
            if nxt in "ei":
                units.extend(["s", nxt])
                i += 2
                continue
            units.append("k")
            i += 1
            continue

        if ch in {"z", "s", "x"}:
            units.append("s")
            i += 1
            continue

        if ch == "j":
            units.append("j")
            i += 1
            continue

        if ch == "y":
            if n == 1 or i == n - 1:
                units.append("i")
            else:
                units.append("y")
            i += 1
            continue

        if ch == "ñ":
            units.append("ny")
            i += 1
            continue

        if ch == "w":
            units.append("u")
            i += 1
            continue

        if ch.isdigit():
            i += 1
            continue

        units.append(ch)
        i += 1

    return units
