from __future__ import annotations

from typing import Sequence


UNIT_TO_VISEME: dict[str, str] = {
    "sil": "SIL",
    "p": "PP",
    "b": "PP",
    "m": "PP",
    "f": "FF",
    "t": "TD",
    "d": "TD",
    "n": "TD",
    "l": "TD",
    "r": "TD",
    "rr": "TD",
    "s": "SS",
    "z": "SS",
    "j": "SS",
    "x": "SS",
    "ch": "CH",
    "sh": "CH",
    "k": "KG",
    "g": "KG",
    "q": "KG",
    "y": "YL",
    "ll": "YL",
    "ny": "YL",
    "ñ": "YL",
    "a": "AA",
    "e": "EE",
    "i": "II",
    "o": "OO",
    "u": "UU",
}

COMPACT_VISEMES: tuple[str, ...] = (
    "SIL",
    "PP",
    "FF",
    "TD",
    "SS",
    "CH",
    "KG",
    "YL",
    "AA",
    "EE",
    "II",
    "OO",
    "UU",
)


def unit_to_viseme(unit: str) -> str:
    key = unit.strip().lower()
    if not key:
        return "SIL"
    if key in UNIT_TO_VISEME:
        return UNIT_TO_VISEME[key]

    if key[0] in {"a", "e", "i", "o", "u"}:
        return UNIT_TO_VISEME[key[0]]
    if key[0].isalpha():
        return "TD"
    return "SIL"


def units_to_visemes(units: Sequence[str]) -> list[str]:
    if not units:
        return ["SIL"]
    return [unit_to_viseme(unit) for unit in units if unit]
