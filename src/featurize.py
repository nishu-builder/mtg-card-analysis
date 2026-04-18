from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

RARITY_ORDER = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3, "special": 2, "bonus": 2}
COLORS = ["W", "U", "B", "R", "G"]


def _coerce_pt(x: object) -> tuple[float, int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan, 0
    s = str(x)
    try:
        return float(s), 0
    except ValueError:
        if any(ch in s for ch in ["*", "X", "?"]):
            return np.nan, 1
        return np.nan, 0


def _first_face(row: pd.Series) -> pd.Series:
    faces = row.get("card_faces")
    if isinstance(faces, (list, np.ndarray)) and len(faces) > 0:
        face = faces[0]
        for k in ("oracle_text", "power", "toughness", "type_line", "mana_cost"):
            if (pd.isna(row.get(k)) or row.get(k) is None or row.get(k) == "") and face.get(k) is not None:
                row[k] = face.get(k)
    return row


def _structured(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(_first_face, axis=1)
    out = pd.DataFrame(index=df.index)
    out["name"] = df["name"]
    out["cmc"] = pd.to_numeric(df["cmc"], errors="coerce")

    pts = df["power"].apply(_coerce_pt).tolist()
    tts = df["toughness"].apply(_coerce_pt).tolist()
    out["power"] = [p[0] for p in pts]
    out["toughness"] = [t[0] for t in tts]
    out["is_variable_pt"] = [max(p[1], t[1]) for p, t in zip(pts, tts)]

    ci = df["color_identity"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
    out["num_colors"] = ci.apply(len)
    for c in COLORS:
        out[f"ci_{c}"] = ci.apply(lambda xs, c=c: int(c in xs))

    out["rarity_ord"] = df["rarity"].str.lower().map(RARITY_ORDER).fillna(0).astype(int)

    tl = df["type_line"].fillna("").str.lower()
    out["is_creature"] = tl.str.contains("creature").astype(int)
    out["is_instant"] = tl.str.contains(r"\binstant\b", regex=True).astype(int)
    out["is_sorcery"] = tl.str.contains(r"\bsorcery\b", regex=True).astype(int)
    out["is_enchantment"] = tl.str.contains("enchantment").astype(int)
    out["is_artifact"] = tl.str.contains("artifact").astype(int)
    out["is_land"] = tl.str.contains(r"\bland\b", regex=True).astype(int)
    out["is_planeswalker"] = tl.str.contains("planeswalker").astype(int)
    return out


def _keyword_features(df: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    kws = df["keywords"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
    counter: Counter[str] = Counter()
    for row in kws:
        for k in row:
            counter[k] += 1
    keep = [k for k, v in counter.items() if v >= min_count]
    out = pd.DataFrame(index=df.index)
    for k in keep:
        slug = re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")
        out[f"kw_{slug}"] = kws.apply(lambda xs, k=k: int(k in xs))
    print(f"Keyword features kept ({len(keep)}): {sorted(keep)}")
    return out


_DRAW_RE = re.compile(r"draw[s]? (a|an|one|two|three|four|five|\d+) card", re.I)
_DMG_RE = re.compile(r"deal[s]? (\d+) damage", re.I)
_LIFE_RE = re.compile(r"gain[s]? (\d+) life", re.I)
_BUFF_RE = re.compile(r"\+(\d+)/\+(\d+)", re.I)
_TARGET_RE = re.compile(r"target [a-z0-9,\- ]+", re.I)
_WORD_TO_INT = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _parse_int(tok: str) -> int:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return _WORD_TO_INT.get(tok, 0)


def _oracle_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["oracle_text"].fillna("").str.lower()
    out = pd.DataFrame(index=df.index)

    def sum_draws(s: str) -> int:
        return sum(_parse_int(m.group(1)) for m in _DRAW_RE.finditer(s))

    def max_dmg(s: str) -> int:
        vals = [int(m.group(1)) for m in _DMG_RE.finditer(s)]
        return max(vals) if vals else 0

    def sum_life(s: str) -> int:
        return sum(int(m.group(1)) for m in _LIFE_RE.finditer(s))

    def max_buff(s: str) -> tuple[int, int]:
        matches = list(_BUFF_RE.finditer(s))
        if not matches:
            return 0, 0
        ps = [int(m.group(1)) for m in matches]
        ts = [int(m.group(2)) for m in matches]
        return max(ps), max(ts)

    out["draw_cards"] = text.apply(sum_draws)
    out["damage_dealt"] = text.apply(max_dmg)
    out["life_gain"] = text.apply(sum_life)
    buffs = text.apply(max_buff)
    out["pt_buff_power"] = [b[0] for b in buffs]
    out["pt_buff_toughness"] = [b[1] for b in buffs]

    out["has_etb"] = text.str.contains(r"enters(?: the battlefield)?", regex=True).astype(int)
    out["has_attack_trigger"] = text.str.contains(r"whenever [^\.]*attacks", regex=True).astype(int)
    out["has_death_trigger"] = text.str.contains(r"when[^\.]*dies|whenever [^\.]*dies", regex=True).astype(int)
    out["has_activated_ability"] = text.str.contains(r"[^\n]*\{[^\}]+\}[^\n]*:", regex=True).astype(int)
    out["is_removal"] = text.str.contains(
        r"destroy target creature|exile target creature|deals \d+ damage to target creature|destroy target (?:artifact|enchantment|permanent|nonland)",
        regex=True,
    ).astype(int)
    out["is_counterspell"] = text.str.contains(r"counter target (?:spell|ability)", regex=True).astype(int)
    out["is_ramp"] = text.str.contains(
        r"search your library for (?:a|up to [a-z]+|\d+) (?:basic )?land|add \{[wubrgc]\}|add one mana",
        regex=True,
    ).astype(int)
    out["is_tutor"] = text.str.contains(
        r"search your library for (?:a|up to [a-z]+|\d+) (?!basic )", regex=True
    ).astype(int)
    out["is_bounce"] = text.str.contains(
        r"return target (?:creature|permanent|nonland permanent)[^\.]*to (?:its|their) owner'?s hand",
        regex=True,
    ).astype(int)
    out["creates_token"] = text.str.contains(r"create[s]? .* token", regex=True).astype(int)
    out["has_x_cost"] = df["mana_cost"].fillna("").str.contains(r"\{X\}").astype(int)

    out["targeting_flexibility"] = text.apply(
        lambda s: len(set(m.group(0) for m in _TARGET_RE.finditer(s)))
    )
    return out


def featurize(set_code: str) -> pd.DataFrame:
    code = set_code.lower()
    cards = pd.read_parquet(PROCESSED / f"cards_with_ratings_{code}.parquet")

    struct = _structured(cards)
    kws = _keyword_features(cards, min_count=3)
    oracle = _oracle_features(cards)

    feats = pd.concat(
        [struct.reset_index(drop=True), kws.reset_index(drop=True), oracle.reset_index(drop=True)],
        axis=1,
    )

    extras = cards[
        [c for c in ["name", "gih_wr", "n_gih", "n_oh", "alsa", "iwd"] if c in cards.columns]
    ].copy()
    feats = feats.merge(extras, on="name", how="left")
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_code", required=True)
    args = ap.parse_args()

    feats = featurize(args.set_code)
    out = PROCESSED / f"features_{args.set_code.lower()}.parquet"
    feats.to_parquet(out, index=False)
    print(f"Wrote {out} with {feats.shape[0]} rows, {feats.shape[1]} cols")
    print("Columns:", list(feats.columns))


if __name__ == "__main__":
    main()
