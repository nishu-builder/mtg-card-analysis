from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "outputs" / "models"
IMG_CACHE = RAW / "card_images"
ORACLE_JSON = RAW / "scryfall_oracle.json"

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.fetch_cards import ASSOCIATED_SETS  # noqa: E402


VARIANTS = ["no_rarity", "with_rarity"]
VARIANT_LABELS = {"no_rarity": "Without rarity", "with_rarity": "With rarity"}


def discover_sets() -> list[str]:
    out: list[str] = []
    for p in sorted(MODELS.glob("ebm_*_with_rarity.pkl")):
        code = p.stem[len("ebm_"):-len("_with_rarity")]
        if not code:
            continue
        if not (MODELS / f"ebm_{code}_no_rarity.pkl").exists():
            continue
        if (PROCESSED / f"features_{code}.parquet").exists() and (
            PROCESSED / f"cards_with_ratings_{code}.parquet"
        ).exists():
            out.append(code)
    return out


def current_set() -> str:
    sets = discover_sets()
    fallback = sets[0] if sets else "tmt"
    return st.session_state.get("set_code", fallback)


def current_variant() -> str:
    return st.session_state.get("variant", "no_rarity")


def scryfall_sets_for(set_code: str) -> set[str]:
    return set(ASSOCIATED_SETS.get(set_code.lower(), [set_code.lower()]))

# Key Findings thresholds (tunable)
MIN_BINARY_MINORITY = 10
MIN_BIN_COUNT = 5
MIN_EFFECT = 0.005  # 0.5pp of GIH WR
TOP_K_IMPORTANCE = 15
CORR_THRESHOLD = 0.6

# Feature display metadata ----------------------------------------------
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "cmc": "Mana value",
    "power": "Power",
    "toughness": "Toughness",
    "is_variable_pt": "Variable P/T (*/X)",
    "num_colors": "Number of colors",
    "ci_W": "White",
    "ci_U": "Blue",
    "ci_B": "Black",
    "ci_R": "Red",
    "ci_G": "Green",
    "rarity_ord": "Rarity (ordinal: C→M)",
    "is_creature": "Creature",
    "is_instant": "Instant",
    "is_sorcery": "Sorcery",
    "is_enchantment": "Enchantment",
    "is_artifact": "Artifact",
    "is_land": "Land",
    "is_planeswalker": "Planeswalker",
    "kw_deathtouch": "Deathtouch",
    "kw_menace": "Menace",
    "kw_food": "Food",
    "kw_sneak": "Sneak",
    "kw_flash": "Flash",
    "kw_flying": "Flying",
    "kw_trample": "Trample",
    "kw_disappear": "Disappear",
    "kw_alliance": "Alliance",
    "kw_scry": "Scry",
    "kw_haste": "Haste",
    "kw_landcycling": "Landcycling",
    "kw_typecycling": "Typecycling",
    "kw_cycling": "Cycling",
    "kw_mill": "Mill",
    "kw_vigilance": "Vigilance",
    "kw_equip": "Equip",
    "kw_reach": "Reach",
    "kw_enchant": "Enchant",
    "draw_cards": "Cards drawn",
    "damage_dealt": "Damage dealt (max)",
    "life_gain": "Life gained",
    "pt_buff_power": "P/T buff: power",
    "pt_buff_toughness": "P/T buff: toughness",
    "has_etb": "Enters-the-battlefield trigger",
    "has_attack_trigger": "Attack trigger",
    "has_death_trigger": "Death/dies trigger",
    "has_activated_ability": "Activated ability",
    "is_removal": "Removal",
    "is_counterspell": "Counterspell",
    "is_ramp": "Ramp",
    "is_tutor": "Tutor",
    "is_bounce": "Bounce",
    "creates_token": "Creates a token",
    "has_x_cost": "Has X in mana cost",
    "targeting_flexibility": "Targeting flexibility",
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "cmc": "Converted mana cost.",
    "power": "Printed power (variable coerced to NaN, then median-imputed).",
    "toughness": "Printed toughness (variable coerced to NaN).",
    "is_variable_pt": "1 if power or toughness uses */X.",
    "num_colors": "Size of the color identity (0–5).",
    "ci_W": "Color identity includes white.",
    "ci_U": "Color identity includes blue.",
    "ci_B": "Color identity includes black.",
    "ci_R": "Color identity includes red.",
    "ci_G": "Color identity includes green.",
    "rarity_ord": "Common=0, Uncommon=1, Rare=2, Mythic=3.",
    "is_creature": "Type line contains 'creature'.",
    "is_instant": "Type line is instant.",
    "is_sorcery": "Type line is sorcery.",
    "is_enchantment": "Type line contains 'enchantment'.",
    "is_artifact": "Type line contains 'artifact'.",
    "is_land": "Type line contains 'land'.",
    "is_planeswalker": "Type line contains 'planeswalker'.",
    "kw_deathtouch": "Has the Deathtouch keyword.",
    "kw_menace": "Has the Menace keyword.",
    "kw_food": "Creates Food tokens or references the Food mechanic.",
    "kw_sneak": "Has the Sneak keyword.",
    "kw_flash": "Has the Flash keyword.",
    "kw_flying": "Has the Flying keyword.",
    "kw_trample": "Has the Trample keyword.",
    "kw_disappear": "Has the Disappear keyword.",
    "kw_alliance": "Has the Alliance keyword.",
    "kw_scry": "References Scry.",
    "kw_haste": "Has the Haste keyword.",
    "kw_landcycling": "Has Landcycling.",
    "kw_typecycling": "Has Typecycling.",
    "kw_cycling": "Has Cycling.",
    "kw_mill": "Has the Mill keyword.",
    "kw_vigilance": "Has the Vigilance keyword.",
    "kw_equip": "Has the Equip keyword (equipment).",
    "kw_reach": "Has the Reach keyword.",
    "kw_enchant": "Has the Enchant keyword (auras).",
    "draw_cards": "Sum of 'draw N card(s)' in oracle text.",
    "damage_dealt": "Max single 'deals N damage' instance.",
    "life_gain": "Sum of 'gain N life' amounts.",
    "pt_buff_power": "Largest +X in +X/+Y on oracle text.",
    "pt_buff_toughness": "Largest +Y in +X/+Y on oracle text.",
    "has_etb": "Has an enters-the-battlefield trigger.",
    "has_attack_trigger": "Has a 'whenever ... attacks' trigger.",
    "has_death_trigger": "Has a 'when ... dies' trigger.",
    "has_activated_ability": "Has at least one activated ability with a mana cost.",
    "is_removal": "Destroys or exiles target creature, or deals damage to target creature.",
    "is_counterspell": "Counters target spell or ability.",
    "is_ramp": "Searches for a land or directly adds mana.",
    "is_tutor": "Searches library for non-basic cards.",
    "is_bounce": "Returns a creature/permanent to its owner's hand.",
    "creates_token": "Creates one or more tokens.",
    "has_x_cost": "Mana cost contains {X}.",
    "targeting_flexibility": "Count of distinct 'target X' phrases.",
}


def display_name(term: str) -> str:
    if " & " in term:
        a, b = term.split(" & ", 1)
        return f"{FEATURE_DISPLAY_NAMES.get(a, a)} × {FEATURE_DISPLAY_NAMES.get(b, b)}"
    return FEATURE_DISPLAY_NAMES.get(term, term)


RARITY_ORDINAL_LABELS = {0: "common", 1: "uncommon", 2: "rare", 3: "mythic"}

FEATURE_VALUE_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "rarity_ord": lambda v: RARITY_ORDINAL_LABELS.get(int(round(float(v))), str(int(round(float(v))))),
    "ci_W": lambda v: "W" if float(v) >= 0.5 else "",
    "ci_U": lambda v: "U" if float(v) >= 0.5 else "",
    "ci_B": lambda v: "B" if float(v) >= 0.5 else "",
    "ci_R": lambda v: "R" if float(v) >= 0.5 else "",
    "ci_G": lambda v: "G" if float(v) >= 0.5 else "",
}


def format_feature_value(name: str, value: Any, X: pd.DataFrame) -> str:
    if name in FEATURE_VALUE_FORMATTERS:
        return FEATURE_VALUE_FORMATTERS[name](value)
    kind = feature_kind(X, name)
    if kind == "binary":
        return "yes" if float(value) >= 0.5 else "no"
    v = float(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"


def format_term_value(term: str, X_row: pd.Series, X: pd.DataFrame) -> str:
    if " & " in term:
        a, b = term.split(" & ", 1)
        return f"{format_feature_value(a, X_row[a], X)} × {format_feature_value(b, X_row[b], X)}"
    return format_feature_value(term, X_row[term], X)


st.set_page_config(page_title="MTG EBM Explorer", layout="wide")


# --- Loaders ------------------------------------------------------------
@st.cache_resource
def _load_ebm_cached(set_code: str, variant: str) -> tuple[Any, list[str], list[str]]:
    path = MODELS / f"ebm_{set_code}_{variant}.pkl"
    if not path.exists():
        st.error(
            f"EBM artifact not found at {path}. "
            f"Run the pipeline first: `uv run python -m src.run_pipeline --set {set_code.upper()}`."
        )
        st.stop()
    with path.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names: list[str] = bundle["feature_names"]
    term_names: list[str] = model.explain_global().data()["names"]
    return model, feature_names, term_names


def load_ebm() -> tuple[Any, list[str], list[str]]:
    return _load_ebm_cached(current_set(), current_variant())


@st.cache_data
def _load_features_cached(set_code: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"features_{set_code}.parquet")


def load_features() -> pd.DataFrame:
    return _load_features_cached(current_set())


@st.cache_data
def _load_cards_cached(set_code: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"cards_with_ratings_{set_code}.parquet")


def load_cards() -> pd.DataFrame:
    return _load_cards_cached(current_set())


@st.cache_data
def _load_image_url_map_cached(set_code: str) -> dict[str, str]:
    pre = PROCESSED / f"image_map_{set_code}.json"
    if pre.exists():
        return json.loads(pre.read_text())
    if not ORACLE_JSON.exists():
        return {}
    sets = scryfall_sets_for(set_code)
    with ORACLE_JSON.open() as f:
        data = json.load(f)
    out: dict[str, str] = {}
    for c in data:
        if c.get("set") not in sets:
            continue
        url = None
        if isinstance(c.get("image_uris"), dict):
            url = c["image_uris"].get("normal") or c["image_uris"].get("large")
        elif isinstance(c.get("card_faces"), list) and c["card_faces"]:
            face = c["card_faces"][0]
            if isinstance(face.get("image_uris"), dict):
                url = face["image_uris"].get("normal") or face["image_uris"].get("large")
        if url:
            out[c["name"]] = url
    return out


def load_image_url_map() -> dict[str, str]:
    return _load_image_url_map_cached(current_set())


@st.cache_data(show_spinner=False)
def _get_card_image_bytes_cached(name: str, set_code: str) -> bytes | None:
    IMG_CACHE.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    cache_path = IMG_CACHE / f"{safe}.jpg"
    if cache_path.exists():
        return cache_path.read_bytes()
    url_map = _load_image_url_map_cached(set_code)
    url = url_map.get(name)
    if not url:
        return None
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None
    cache_path.write_bytes(r.content)
    return r.content


def get_card_image_bytes(name: str) -> bytes | None:
    return _get_card_image_bytes_cached(name, current_set())


def load_ebm_cached_tuple() -> tuple[Any, list[str], list[str]]:
    return load_ebm()


@st.cache_data
def _get_X_cached(set_code: str, variant: str) -> pd.DataFrame:
    feats = _load_features_cached(set_code)
    _, feat_cols, _ = _load_ebm_cached(set_code, variant)
    eligible = feats[
        (feats["is_land"] == 0) & (feats["n_gih"].fillna(0) >= 200) & feats["gih_wr"].notna()
    ].copy()
    X = eligible[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in "fi" else s.fillna(0))
    X.index = eligible["name"].values
    return X


def get_X() -> pd.DataFrame:
    return _get_X_cached(current_set(), current_variant())


@st.cache_data
def _get_model_table_cached(set_code: str, variant: str) -> pd.DataFrame:
    feats = _load_features_cached(set_code)
    model, feat_cols, _ = _load_ebm_cached(set_code, variant)
    eligible = feats[
        (feats["is_land"] == 0) & (feats["n_gih"].fillna(0) >= 200) & feats["gih_wr"].notna()
    ].copy()
    X = eligible[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in "fi" else s.fillna(0))
    preds = model.predict(X)
    eligible = eligible.reset_index(drop=True).copy()
    eligible["predicted_gih_wr"] = preds
    eligible["residual"] = eligible["gih_wr"].values - preds
    cards = _load_cards_cached(set_code)
    meta = cards[
        [c for c in ["name", "mana_cost", "type_line", "power", "toughness", "rarity",
                     "oracle_text", "color_identity"] if c in cards.columns]
    ].copy()
    return eligible.merge(meta, on="name", how="left", suffixes=("", "_c"))


def get_model_table() -> pd.DataFrame:
    return _get_model_table_cached(current_set(), current_variant())


@st.cache_data
def _get_corr_matrix_cached(set_code: str, variant: str) -> pd.DataFrame:
    return _get_X_cached(set_code, variant).corr().abs()


def get_corr_matrix() -> pd.DataFrame:
    return _get_corr_matrix_cached(current_set(), current_variant())


@st.cache_data
def _get_term_ranges_cached(set_code: str, variant: str) -> dict[str, tuple[float, float]]:
    model, _, term_names = _load_ebm_cached(set_code, variant)
    g = model.explain_global()
    out: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(term_names):
        scores = np.asarray(g.data(i).get("scores", []), dtype=float)
        if scores.size == 0:
            out[name] = (0.0, 0.0)
        else:
            out[name] = (float(scores.min()), float(scores.max()))
    return out


def get_term_ranges() -> dict[str, tuple[float, float]]:
    return _get_term_ranges_cached(current_set(), current_variant())


# --- Feature kind detection --------------------------------------------
def feature_kind(X: pd.DataFrame, name: str) -> str:
    """Return 'binary', 'ordinal', or 'continuous' based on values in X[name]."""
    if name not in X.columns:
        return "continuous"
    vals = X[name].dropna().unique()
    n_unique = len(vals)
    as_set = set(float(v) for v in vals)
    if n_unique <= 2 and as_set.issubset({0.0, 1.0}):
        return "binary"
    if n_unique <= 5:
        return "ordinal"
    return "continuous"


# --- Local / global shape helpers --------------------------------------
def _local_scores_for_card(name: str) -> tuple[list[str], list[float], list[Any], float]:
    model, _, _ = load_ebm_cached_tuple()
    X = get_X()
    if name not in X.index:
        return [], [], [], float("nan")
    row = X.loc[[name]]
    loc = model.explain_local(row, np.array([0.0]))
    d = loc.data(0)
    scores = [float(s) for s in d["scores"]]
    values = list(d["values"])
    names = list(d["names"])
    intercept = float(d.get("extra", {}).get("scores", [0.0])[0])
    return names, scores, values, intercept


def _univariate_shape(term_idx: int) -> dict[str, Any]:
    model, _, _ = load_ebm_cached_tuple()
    g = model.explain_global()
    return g.data(term_idx)


def _bin_supports(X: pd.DataFrame, name: str, xs: np.ndarray) -> list[int]:
    """For a continuous/ordinal feature, count cards falling into each EBM bin."""
    col = X[name] if name in X.columns else pd.Series(dtype=float)
    supports = []
    for i in range(len(xs) - 1):
        lo, hi = float(xs[i]), float(xs[i + 1])
        if i == len(xs) - 2:
            mask = (col >= lo) & (col <= hi)
        else:
            mask = (col >= lo) & (col < hi)
        supports.append(int(mask.sum()))
    return supports


# --- Rendering helpers -------------------------------------------------
def _render_card_header(card_row: pd.Series) -> None:
    img = get_card_image_bytes(card_row["name"])
    col_img, col_meta = st.columns([1, 2])
    with col_img:
        if img:
            st.image(img, width=260)
        else:
            st.caption("(no image available)")
    with col_meta:
        st.markdown(f"### {card_row['name']}")
        bits = []
        if pd.notna(card_row.get("mana_cost")) and card_row.get("mana_cost"):
            bits.append(f"**{card_row['mana_cost']}**")
        if pd.notna(card_row.get("type_line")) and card_row.get("type_line"):
            bits.append(str(card_row["type_line"]))
        if bits:
            st.markdown(" · ".join(bits))
        pt = ""
        if pd.notna(card_row.get("power")) and pd.notna(card_row.get("toughness")):
            pt = f"{card_row['power']}/{card_row['toughness']}"
        rarity = card_row.get("rarity", "")
        meta_line = []
        if pt:
            meta_line.append(f"P/T: {pt}")
        if rarity:
            meta_line.append(f"Rarity: {rarity}")
        if meta_line:
            st.caption(" · ".join(meta_line))
        if pd.notna(card_row.get("oracle_text")) and card_row.get("oracle_text"):
            st.markdown(f"> {card_row['oracle_text'].replace(chr(10), '  \n> ')}")


def _render_big_numbers(actual: float, predicted: float, residual: float) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Actual GIH WR", f"{actual:.1%}",
        help="Observed 17Lands game-in-hand win rate. Fraction of games won when this card was drawn.",
    )
    c2.metric(
        "Predicted GIH WR", f"{predicted:.1%}",
        help="The EBM's prediction from this card's features. Baseline + sum of per-feature contributions.",
    )
    c3.metric(
        "Residual", f"{residual*100:+.2f}pp",
        help="Actual − predicted, in percentage points. Positive = card outperforms what its features suggest.",
    )


def _contribution_bar(
    names: list[str],
    scores: list[float],
    X_row: pd.Series,
    title: str = "Per-feature EBM contributions",
) -> go.Figure:
    X = get_X()
    ranges = get_term_ranges()
    df = pd.DataFrame({"term": names, "score": scores})
    df["display"] = df["term"].apply(display_name)
    df["value_str"] = df["term"].apply(lambda t: format_term_value(t, X_row, X))
    df["label"] = df["display"] + ": " + df["value_str"]
    df["contrib_pp"] = df["score"] * 100.0
    df["lo_pp"] = df["term"].map(lambda t: ranges.get(t, (0.0, 0.0))[0] * 100.0)
    df["hi_pp"] = df["term"].map(lambda t: ranges.get(t, (0.0, 0.0))[1] * 100.0)
    df["abs"] = df["score"].abs()
    df = df[df["abs"] > 1e-9].sort_values("abs", ascending=True)
    colors = ["#2ca02c" if s > 0 else "#d62728" for s in df["score"]]
    customdata = np.stack(
        [
            df["display"].to_numpy(),
            df["value_str"].to_numpy(),
            df["contrib_pp"].to_numpy(),
            df["lo_pp"].to_numpy(),
            df["hi_pp"].to_numpy(),
        ],
        axis=-1,
    )
    fig = go.Figure(
        go.Bar(
            y=df["label"],
            x=df["score"],
            orientation="h",
            marker_color=colors,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "value: %{customdata[1]}<br>"
                "contribution: %{customdata[2]:+.2f}pp<br>"
                "Across the set, this feature contributes between "
                "%{customdata[3]:+.2f}pp and %{customdata[4]:+.2f}pp."
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="contribution to GIH WR",
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(300, 22 * len(df) + 80),
    )
    return fig


# --- Feature Shapes renderers ------------------------------------------
def _render_binary_shape(X: pd.DataFrame, name: str, fd: dict[str, Any]) -> tuple[go.Figure, str, tuple[int, int]]:
    scores = np.asarray(fd["scores"], dtype=float)
    y_without = float(scores[0]) if len(scores) >= 1 else 0.0
    y_with = float(scores[1]) if len(scores) >= 2 else 0.0
    n_with = int((X[name] == 1).sum()) if name in X.columns else 0
    n_without = int((X[name] == 0).sum()) if name in X.columns else 0

    gap = y_with - y_without
    disp = display_name(name)
    labels = [f"Without {disp}<br>(n={n_without})", f"With {disp}<br>(n={n_with})"]
    vals = [y_without, y_with]
    colors = ["#cccccc", "#cccccc"]
    if abs(gap) > 1e-12:
        hi = 0 if y_without > y_with else 1
        lo = 1 - hi
        colors[hi] = "#2ca02c"
        colors[lo] = "#d62728"

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=vals,
            marker_color=colors,
            text=[f"{v:+.4f}" for v in vals],
            textposition="outside",
            hovertemplate="%{x}<br>contribution: %{y:+.4f} WR<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="gray")
    fig.update_layout(
        title=f"EBM shape: {disp}",
        yaxis_title="contribution to GIH WR",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
        showlegend=False,
    )
    caption = (
        f"Cards WITH *{disp}* contribute {y_with:+.4f} vs cards WITHOUT at {y_without:+.4f} — "
        f"a gap of {gap*100:+.2f}pp of predicted win rate."
    )
    return fig, caption, (n_with, n_without)


def _render_ordinal_shape(X: pd.DataFrame, name: str, fd: dict[str, Any]) -> tuple[go.Figure, str]:
    xs = np.asarray(fd["names"], dtype=float)
    ys = np.asarray(fd["scores"], dtype=float)
    centers = 0.5 * (xs[:-1] + xs[1:])
    if name in X.columns:
        col = X[name]
        supports = _bin_supports(X, name, xs)
    else:
        supports = [0] * len(ys)
    label_vals = [int(round(c)) if abs(c - round(c)) < 1e-6 else round(float(c), 2) for c in centers]
    labels = [f"{v} (n={s})" for v, s in zip(label_vals, supports)]
    disp = display_name(name)

    colors = []
    if len(ys):
        mx, mn = float(ys.max()), float(ys.min())
        for y in ys:
            if abs(y - mx) < 1e-12 and mx > 0:
                colors.append("#2ca02c")
            elif abs(y - mn) < 1e-12 and mn < 0:
                colors.append("#d62728")
            else:
                colors.append("#6baed6")

    fig = go.Figure(
        go.Bar(
            x=labels, y=ys, marker_color=colors,
            text=[f"{y:+.4f}" for y in ys], textposition="outside",
            hovertemplate="%{x}<br>contribution: %{y:+.4f} WR<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="gray")
    fig.update_layout(
        title=f"EBM shape: {disp}",
        xaxis_title=disp, yaxis_title="contribution to GIH WR",
        margin=dict(l=10, r=10, t=40, b=10), height=380, showlegend=False,
    )
    low_v, high_v = label_vals[int(np.argmin(ys))], label_vals[int(np.argmax(ys))]
    caption = (
        f"Contribution ranges from {float(ys.min()):+.4f} at value {low_v} to "
        f"{float(ys.max()):+.4f} at value {high_v}."
    )
    return fig, caption


def _render_continuous_shape(X: pd.DataFrame, name: str, fd: dict[str, Any]) -> tuple[go.Figure, str]:
    xs = np.asarray(fd["names"], dtype=float)
    ys = np.asarray(fd["scores"], dtype=float)
    if len(xs) == len(ys) + 1:
        centers = 0.5 * (xs[:-1] + xs[1:])
        fig = go.Figure(
            go.Scatter(
                x=centers, y=ys, mode="lines+markers", line_shape="hv",
                hovertemplate="x=%{x:.2f}<br>contribution=%{y:+.4f}<extra></extra>",
            )
        )
    else:
        fig = go.Figure(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                hovertemplate="x=%{x}<br>contribution=%{y:+.4f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_width=1, line_color="gray")
    disp = display_name(name)
    fig.update_layout(title=f"EBM shape: {disp}", xaxis_title=disp, yaxis_title="contribution to GIH WR",
                      margin=dict(l=10, r=10, t=40, b=10), height=380)

    if len(ys) > 1:
        diffs = np.abs(np.diff(ys))
        jump_idx = int(np.argmax(diffs))
        if len(xs) == len(ys) + 1:
            a, b = float(xs[jump_idx]), float(xs[jump_idx + 1])
            jump_txt = f"between {a:g} and {b:g}"
        else:
            jump_txt = f"between {xs[jump_idx]} and {xs[jump_idx+1]}"
    else:
        jump_txt = "n/a"
    caption = (
        f"Contribution ranges from {float(ys.min()):+.4f} to {float(ys.max()):+.4f}. "
        f"Biggest jump is {jump_txt}."
    )
    return fig, caption


# --- Key Findings ------------------------------------------------------
def _correlated_features(name: str, threshold: float = CORR_THRESHOLD) -> list[tuple[str, float]]:
    corr = get_corr_matrix()
    if name not in corr.index:
        return []
    series = corr.loc[name].drop(name, errors="ignore")
    hits = series[series >= threshold].sort_values(ascending=False)
    return [(k, float(v)) for k, v in hits.items()]


def _key_findings() -> tuple[list[dict], list[dict]]:
    model, _, term_names = load_ebm_cached_tuple()
    X = get_X()
    g = model.explain_global()
    importances = np.asarray(g.data()["scores"], dtype=float)

    univ = [(t, imp) for t, imp in zip(term_names, importances) if " & " not in t]
    univ_sorted = sorted(univ, key=lambda x: -x[1])
    top_k = {t for t, _ in univ_sorted[:TOP_K_IMPORTANCE]}

    kept: list[dict] = []
    rejected: list[dict] = []

    for term_idx, name in enumerate(term_names):
        if " & " in name:
            continue
        in_top = name in top_k
        kind = feature_kind(X, name)
        fd = g.data(term_idx)
        xs = np.asarray(fd["names"], dtype=float)
        ys = np.asarray(fd["scores"], dtype=float)

        reasons: list[str] = []
        info: dict[str, Any] = {
            "name": name,
            "display": display_name(name),
            "kind": kind,
            "importance": float(importances[term_idx]),
            "importance_rank": next((i + 1 for i, (t, _) in enumerate(univ_sorted) if t == name), -1),
            "term_idx": term_idx,
            "fd": fd,
        }

        if not in_top:
            reasons.append(f"importance rank {info['importance_rank']} (only top {TOP_K_IMPORTANCE} shown)")

        if kind == "binary":
            n_with = int((X[name] == 1).sum()) if name in X.columns else 0
            n_without = int((X[name] == 0).sum()) if name in X.columns else 0
            minority = min(n_with, n_without)
            info.update({"n_with": n_with, "n_without": n_without, "minority": minority})
            if minority < MIN_BINARY_MINORITY:
                reasons.append(f"minority class only {minority} cards (need ≥{MIN_BINARY_MINORITY})")
            effect_signed = float(ys[1] - ys[0]) if len(ys) >= 2 else 0.0
            effect = abs(effect_signed)
            info.update({"effect": effect, "effect_signed": effect_signed,
                         "y_with": float(ys[1]) if len(ys) >= 2 else 0.0,
                         "y_without": float(ys[0]) if len(ys) >= 1 else 0.0})
            if effect < MIN_EFFECT:
                reasons.append(f"effect {effect*100:.2f}pp below {MIN_EFFECT*100:.2f}pp floor")
        else:
            supports = _bin_supports(X, name, xs)
            supported_mask = [s >= MIN_BIN_COUNT for s in supports]
            info["bin_supports"] = supports
            info["supported_mask"] = supported_mask
            if not any(supported_mask) or sum(supported_mask) < 2:
                reasons.append(f"<2 well-supported bins (need ≥{MIN_BIN_COUNT} cards/bin)")
                effect = 0.0
            else:
                sup_ys = ys[np.array(supported_mask)]
                effect = float(sup_ys.max() - sup_ys.min())
            info["effect"] = effect
            info["effect_signed"] = effect
            if effect < MIN_EFFECT:
                reasons.append(f"effect {effect*100:.2f}pp below {MIN_EFFECT*100:.2f}pp floor")

        info["reasons"] = reasons
        if reasons:
            rejected.append(info)
        else:
            kept.append(info)

    kept.sort(key=lambda d: -d["effect"])
    rejected.sort(key=lambda d: -d["effect"])
    return kept, rejected


def _key_finding_takeaway(info: dict) -> str:
    disp = info["display"]
    if info["kind"] == "binary":
        gap = info["effect_signed"] * 100
        direction = "gained" if gap > 0 else "lost"
        return (
            f"Cards WITH *{disp}* {direction} roughly **{gap:+.2f}pp** of predicted win rate vs "
            f"cards without, controlling for other features."
        )
    fd = info["fd"]
    xs = np.asarray(fd["names"], dtype=float)
    ys = np.asarray(fd["scores"], dtype=float)
    supp = np.array(info.get("supported_mask", [True] * len(ys)))
    sup_ys = ys[supp]
    sup_xs_c = (0.5 * (xs[:-1] + xs[1:]))[supp]
    lo_idx = int(np.argmin(sup_ys))
    hi_idx = int(np.argmax(sup_ys))
    lo_val = sup_xs_c[lo_idx]
    hi_val = sup_xs_c[hi_idx]
    diffs = np.abs(np.diff(ys))
    jump_i = int(np.argmax(diffs))
    a, b = float(xs[jump_i]), float(xs[jump_i + 1])
    return (
        f"Contribution **rises from {float(sup_ys.min()):+.4f} at {lo_val:g} to "
        f"{float(sup_ys.max()):+.4f} at {hi_val:g}**, with the sharpest jump "
        f"between {a:g} and {b:g}."
    )


def _key_finding_card(info: dict) -> None:
    disp = info["display"]
    name = info["name"]
    st.markdown(f"#### {disp}")
    st.caption(FEATURE_DESCRIPTIONS.get(name, ""))
    st.markdown(_key_finding_takeaway(info))

    # small shape plot
    X = get_X()
    fd = info["fd"]
    if info["kind"] == "binary":
        fig, _, _ = _render_binary_shape(X, name, fd)
    elif info["kind"] == "ordinal":
        fig, _ = _render_ordinal_shape(X, name, fd)
    else:
        fig, _ = _render_continuous_shape(X, name, fd)
    fig.update_layout(height=260, title_text="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if info["kind"] == "binary":
        st.caption(
            f"Support: {info['n_with']} cards with / {info['n_without']} cards without "
            f"(importance rank #{info['importance_rank']})"
        )
    else:
        total = sum(s for s, ok in zip(info["bin_supports"], info["supported_mask"]) if ok)
        st.caption(
            f"Support: {total} cards across well-supported bins "
            f"(bin counts: {info['bin_supports']}, importance rank #{info['importance_rank']})"
        )

    corrs = _correlated_features(name)
    if corrs:
        hit_names = ", ".join(f"{display_name(n)} (r={r:.2f})" for n, r in corrs[:3])
        st.warning(f"Correlated with {hit_names} — interpret jointly.")
    st.divider()


# --- Help text ----------------------------------------------------------
def _help_expander_key_findings() -> None:
    with st.expander("How to read this", expanded=False):
        st.markdown(
            "These are the features where the model found a signal large enough and "
            "well-supported enough to be worth interpreting. Features excluded from this "
            "view either affected too few cards, had effects too small to distinguish from noise, "
            "or ranked low in overall importance. See the **Feature Shapes** tab for the unfiltered view.\n\n"
            f"**Thresholds:** top {TOP_K_IMPORTANCE} by importance, ≥{MIN_EFFECT*100:.2f}pp effect, "
            f"≥{MIN_BINARY_MINORITY} cards in the minority class (binary) or ≥{MIN_BIN_COUNT} cards/bin "
            "(continuous). Contributions are in win-rate units (0.01 = 1 percentage point)."
        )


def _help_expander_card_explorer() -> None:
    with st.expander("How to read this", expanded=False):
        st.markdown(
            "The model predicts this card's win rate by summing a baseline plus a contribution "
            "from each feature. **Positive bars pushed the prediction up, negative bars pushed it down.** "
            "The card's actual win rate minus the predicted one is the **residual** — large positive "
            "residuals are candidates for \"better than its stat line suggests.\"\n\n"
            "Contributions are in win-rate units: 0.01 = 1 percentage point of GIH WR."
        )


def _help_expander_residuals() -> None:
    with st.expander("How to read this", expanded=False):
        st.markdown(
            "**Residual = actual WR − model-predicted WR.** A large positive residual means the card "
            "won more than its features predicted — possibly because it has a hard-to-featurize "
            "quality (evasion synergy, key archetype role, etc.) or because the model is underweighting "
            "something about it. Large negative residuals are the opposite.\n\n"
            "Small sample sizes inflate residuals — use the **games-played filter** in the sidebar."
        )


def _help_expander_feature_shapes() -> None:
    with st.expander("How to read this", expanded=False):
        st.markdown(
            "The EBM learns a separate function for each feature that maps the feature's value "
            "to a **contribution** (in win-rate units). Summing all features' contributions plus a "
            "baseline gives the predicted win rate. The shape shows what the model learned about this "
            "one feature, holding all others constant.\n\n"
            "- **Y-axis** is contribution to predicted GIH WR. Zero means \"average for this feature.\" "
            "Positive means the feature value pushes win rate up relative to cards with other values.\n"
            "- **Check the distribution below.** Where the histogram is empty or thin, the shape function "
            "is extrapolating from few or no examples and should not be trusted.\n"
            "- **Binary features** only have two meaningful points (0 and 1). For them, the app shows "
            "a two-bar chart instead of a line."
        )


def _help_expander_compare() -> None:
    with st.expander("How to read this", expanded=False):
        st.markdown(
            "The divergence bars show where the two cards' feature contributions differ most. This "
            "answers **\"what does the model think makes these cards different?\"** rather than \"what "
            "actually makes them different.\""
        )


# --- Tab 0: Key Findings -----------------------------------------------
def tab_key_findings() -> None:
    _help_expander_key_findings()
    st.markdown(
        "These are the features where the model found a signal large enough and well-supported "
        "enough to be worth interpreting. Features excluded from this view either affected too few "
        "cards, had effects too small to distinguish from noise, or ranked low in overall importance. "
        "See the *Feature Shapes* tab for the unfiltered view."
    )
    kept, rejected = _key_findings()
    st.caption(f"{len(kept)} features kept · {len(rejected)} filtered out")

    for info in kept:
        _key_finding_card(info)

    if rejected:
        with st.expander(f"Filtered out ({len(rejected)})"):
            rows = []
            for r in rejected:
                rows.append(
                    {
                        "feature": r["display"],
                        "kind": r["kind"],
                        "effect (WR)": f"{r.get('effect', 0.0):.4f}",
                        "importance_rank": r["importance_rank"],
                        "why excluded": "; ".join(r["reasons"]),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --- Tab 1: Card Explorer ------------------------------------------------
def tab_card_explorer() -> None:
    _help_expander_card_explorer()
    table = get_model_table()
    names = sorted(table["name"].tolist())
    default = st.session_state.get("selected_card", names[0])
    if default not in names:
        default = names[0]
    selected = st.selectbox("Card", names, index=names.index(default), key="explorer_select")
    st.session_state["selected_card"] = selected

    row = table[table["name"] == selected].iloc[0]
    _render_card_header(row)
    st.divider()
    _render_big_numbers(row["gih_wr"], row["predicted_gih_wr"], row["residual"])

    term_names, scores, values, _ = _local_scores_for_card(selected)
    if not term_names:
        st.warning("No local explanation available.")
        return

    X_row = get_X().loc[selected]
    st.plotly_chart(_contribution_bar(term_names, scores, X_row), use_container_width=True)

    tbl = pd.DataFrame(
        {
            "term": [display_name(t) for t in term_names],
            "value": values,
            "contribution": scores,
        }
    ).sort_values("contribution", key=lambda s: s.abs(), ascending=False)
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# --- Tab 2: Residual Leaderboard ----------------------------------------
def _filter_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Leaderboard filters")
    rarities = sorted([r for r in df["rarity"].dropna().unique()])
    sel_rarities = st.sidebar.multiselect("Rarity", rarities, default=rarities)
    color_opts = ["W", "U", "B", "R", "G", "Colorless"]
    sel_colors = st.sidebar.multiselect("Color identity includes", color_opts, default=color_opts)
    min_games = int(df["n_gih"].min())
    max_games = int(df["n_gih"].max())
    games_floor = st.sidebar.slider(
        "Min games played (GIH)", min_value=min_games, max_value=max_games,
        value=max(min_games, 500), step=50,
    )
    filt = df[df["rarity"].isin(sel_rarities) & (df["n_gih"] >= games_floor)].copy()

    def color_ok(ci: Any) -> bool:
        if not isinstance(ci, (list, np.ndarray)):
            ci = []
        ci_list = list(ci)
        if not ci_list:
            return "Colorless" in sel_colors
        return any(c in sel_colors for c in ci_list)

    filt = filt[filt["color_identity"].apply(color_ok)]
    return filt


_LB_COL_HELP = {
    "name": "Card name",
    "rarity": "Common / Uncommon / Rare / Mythic",
    "cmc": "Converted mana cost",
    "actual_wr": "Observed GIH win rate (%)",
    "predicted_wr": "EBM-predicted GIH win rate (%)",
    "residual": "Actual − predicted, in percentage points. Positive = overperforms features.",
    "games": "Number of games where this card was in hand (GIH)",
}


def tab_residual_leaderboard() -> None:
    _help_expander_residuals()
    table = get_model_table()
    filt = _filter_leaderboard(table)

    cols = ["name", "rarity", "cmc", "gih_wr", "predicted_gih_wr", "residual", "n_gih"]
    view = filt[cols].rename(
        columns={"gih_wr": "actual_wr", "predicted_gih_wr": "predicted_wr", "n_gih": "games"}
    )

    top_pos = view.sort_values("residual", ascending=False).head(25).reset_index(drop=True)
    top_neg = view.sort_values("residual", ascending=True).head(25).reset_index(drop=True)

    col_help_md = "  \n".join(f"- **{k}** — {v}" for k, v in _LB_COL_HELP.items())
    st.caption(f"{len(view)} cards after filters. Click a row to load it in Card Explorer.")
    with st.expander("Column definitions", expanded=False):
        st.markdown(col_help_md)

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["actual_wr"] = (out["actual_wr"] * 100).round(2)
        out["predicted_wr"] = (out["predicted_wr"] * 100).round(2)
        out["residual"] = (out["residual"] * 100).round(2)
        return out

    col_config = {
        "name": st.column_config.TextColumn("name", help=_LB_COL_HELP["name"]),
        "rarity": st.column_config.TextColumn("rarity", help=_LB_COL_HELP["rarity"]),
        "cmc": st.column_config.NumberColumn("cmc", help=_LB_COL_HELP["cmc"]),
        "actual_wr": st.column_config.NumberColumn("actual_wr", help=_LB_COL_HELP["actual_wr"], format="%.2f%%"),
        "predicted_wr": st.column_config.NumberColumn("predicted_wr", help=_LB_COL_HELP["predicted_wr"], format="%.2f%%"),
        "residual": st.column_config.NumberColumn("residual", help=_LB_COL_HELP["residual"], format="%+.2f pp"),
        "games": st.column_config.NumberColumn("games", help=_LB_COL_HELP["games"]),
    }

    left, right = st.columns(2)
    with left:
        st.subheader("Top 25 overperformers")
        ev = st.dataframe(
            _fmt(top_pos), use_container_width=True, hide_index=True, on_select="rerun",
            selection_mode="single-row", key="lb_pos", column_config=col_config,
        )
        if ev and ev.selection and ev.selection["rows"]:
            idx = ev.selection["rows"][0]
            st.session_state["selected_card"] = top_pos.iloc[idx]["name"]
            st.session_state["_go_tab1"] = True

    with right:
        st.subheader("Top 25 underperformers")
        ev2 = st.dataframe(
            _fmt(top_neg), use_container_width=True, hide_index=True, on_select="rerun",
            selection_mode="single-row", key="lb_neg", column_config=col_config,
        )
        if ev2 and ev2.selection and ev2.selection["rows"]:
            idx = ev2.selection["rows"][0]
            st.session_state["selected_card"] = top_neg.iloc[idx]["name"]
            st.session_state["_go_tab1"] = True

    if st.session_state.get("_go_tab1"):
        st.info(f"Selection set to **{st.session_state['selected_card']}** — open the *Card Explorer* tab.")


# --- Tab 3: Feature Shapes ----------------------------------------------
def tab_feature_shapes() -> None:
    _help_expander_feature_shapes()
    model, feat_cols, term_names = load_ebm_cached_tuple()
    X = get_X()
    feats = load_features()

    def _term_label(t: str) -> str:
        if " & " in t:
            return f"{display_name(t)}  (interaction)"
        kind = feature_kind(X, t)
        return f"{display_name(t)}  [{kind}]"

    choice = st.selectbox(
        "Feature", term_names, key="shape_feature", format_func=_term_label,
        help=(
            "Pick any EBM term. For binary (0/1) features the app shows a two-bar chart since "
            "only the two endpoints are meaningful. For pairwise interactions, a heatmap is shown."
        ),
    )
    term_idx = term_names.index(choice)
    fd = _univariate_shape(term_idx)
    is_interaction = " & " in choice

    if is_interaction:
        st.info("Pairwise interaction term — rendered as a heatmap.")
        left_bins = np.asarray(fd.get("left_names", fd.get("names", [[0], [0]])[0]), dtype=float)
        right_bins = np.asarray(fd.get("right_names", fd.get("names", [[0], [0]])[1]), dtype=float)
        scores = np.asarray(fd["scores"], dtype=float)
        left_ctr = 0.5 * (left_bins[:-1] + left_bins[1:]) if len(left_bins) == scores.shape[0] + 1 else left_bins
        right_ctr = 0.5 * (right_bins[:-1] + right_bins[1:]) if len(right_bins) == scores.shape[1] + 1 else right_bins
        hmap = go.Figure(
            go.Heatmap(
                z=scores.T, x=left_ctr, y=right_ctr, colorscale="RdBu_r", zmid=0,
                hovertemplate="%{x:.2f}, %{y:.2f}: %{z:+.4f}<extra></extra>",
            )
        )
        hmap.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10),
                           title=f"Interaction: {display_name(choice)}")
        st.plotly_chart(hmap, use_container_width=True)
        return

    kind = feature_kind(X, choice)

    if kind == "binary":
        fig, caption, (n_with, n_without) = _render_binary_shape(X, choice, fd)
        st.plotly_chart(fig, use_container_width=True)
        bars = go.Figure(
            go.Bar(
                x=["Without", "With"], y=[n_without, n_with],
                marker_color="#6baed6", text=[n_without, n_with], textposition="outside",
            )
        )
        bars.update_layout(title="Cards in set", height=220, margin=dict(l=10, r=10, t=40, b=10),
                           yaxis_title="# cards", showlegend=False)
        st.plotly_chart(bars, use_container_width=True)
        st.caption(caption)
        return

    if kind == "ordinal":
        fig, caption = _render_ordinal_shape(X, choice, fd)
        st.plotly_chart(fig, use_container_width=True)
        # histogram below
        if choice in feats.columns:
            vals = feats[choice].dropna()
            hist = go.Figure(go.Histogram(x=vals, nbinsx=min(30, max(5, int(vals.nunique())))))
            hist.update_layout(title=f"Distribution of {display_name(choice)}",
                               xaxis_title=display_name(choice), yaxis_title="# cards",
                               margin=dict(l=10, r=10, t=40, b=10), height=240)
            st.plotly_chart(hist, use_container_width=True)
        st.caption(caption)
        return

    fig, caption = _render_continuous_shape(X, choice, fd)
    st.plotly_chart(fig, use_container_width=True)
    if choice in feats.columns:
        vals = feats[choice].dropna()
        hist = go.Figure(go.Histogram(x=vals, nbinsx=min(30, max(5, int(vals.nunique())))))
        hist.update_layout(title=f"Distribution of {display_name(choice)}",
                           xaxis_title=display_name(choice), yaxis_title="# cards",
                           margin=dict(l=10, r=10, t=40, b=10), height=240)
        st.plotly_chart(hist, use_container_width=True)
    st.caption(caption)


# --- Tab 4: Compare Cards ----------------------------------------------
def tab_compare() -> None:
    _help_expander_compare()
    table = get_model_table()
    names = sorted(table["name"].tolist())
    col1, col2 = st.columns(2)
    default_a = st.session_state.get("cmp_a", names[0])
    default_b = st.session_state.get("cmp_b", names[1] if len(names) > 1 else names[0])
    a = col1.selectbox("Card A", names, index=names.index(default_a) if default_a in names else 0, key="cmp_a")
    b = col2.selectbox("Card B", names, index=names.index(default_b) if default_b in names else 0, key="cmp_b")

    with col1:
        _render_card_header(table[table["name"] == a].iloc[0])
        ra = table[table["name"] == a].iloc[0]
        _render_big_numbers(ra["gih_wr"], ra["predicted_gih_wr"], ra["residual"])
    with col2:
        _render_card_header(table[table["name"] == b].iloc[0])
        rb = table[table["name"] == b].iloc[0]
        _render_big_numbers(rb["gih_wr"], rb["predicted_gih_wr"], rb["residual"])

    names_a, scores_a, _, _ = _local_scores_for_card(a)
    names_b, scores_b, _, _ = _local_scores_for_card(b)
    if not names_a or not names_b:
        st.warning("Missing contributions.")
        return

    X = get_X()
    ranges = get_term_ranges()
    row_a = X.loc[a]
    row_b = X.loc[b]

    df_a = pd.DataFrame({"term": names_a, "A": scores_a})
    df_b = pd.DataFrame({"term": names_b, "B": scores_b})
    df = df_a.merge(df_b, on="term", how="outer").fillna(0.0)
    df["display"] = df["term"].apply(display_name)
    df["val_a"] = df["term"].apply(lambda t: format_term_value(t, row_a, X))
    df["val_b"] = df["term"].apply(lambda t: format_term_value(t, row_b, X))
    df["label"] = df["display"] + ": A=" + df["val_a"] + ", B=" + df["val_b"]
    df["diff"] = (df["A"] - df["B"]).abs()
    df["lo_pp"] = df["term"].map(lambda t: ranges.get(t, (0.0, 0.0))[0] * 100.0)
    df["hi_pp"] = df["term"].map(lambda t: ranges.get(t, (0.0, 0.0))[1] * 100.0)
    df = df.sort_values("diff", ascending=False)
    df_sig = df[(df["A"].abs() + df["B"].abs()) > 1e-6].head(30).iloc[::-1]

    cd_a = np.stack(
        [
            df_sig["display"].to_numpy(),
            df_sig["val_a"].to_numpy(),
            (df_sig["A"] * 100.0).to_numpy(),
            df_sig["val_b"].to_numpy(),
            (df_sig["B"] * 100.0).to_numpy(),
            df_sig["lo_pp"].to_numpy(),
            df_sig["hi_pp"].to_numpy(),
        ],
        axis=-1,
    )
    cd_b = np.stack(
        [
            df_sig["display"].to_numpy(),
            df_sig["val_b"].to_numpy(),
            (df_sig["B"] * 100.0).to_numpy(),
            df_sig["val_a"].to_numpy(),
            (df_sig["A"] * 100.0).to_numpy(),
            df_sig["lo_pp"].to_numpy(),
            df_sig["hi_pp"].to_numpy(),
        ],
        axis=-1,
    )
    hover_a = (
        f"<b>%{{customdata[0]}}</b> (Card A = {a})<br>"
        "value: %{customdata[1]}<br>"
        "contribution: %{customdata[2]:+.2f}pp<br>"
        f"Card B = {b} value: %{{customdata[3]}} (contribution %{{customdata[4]:+.2f}}pp)<br>"
        "Across the set, this feature contributes between "
        "%{customdata[5]:+.2f}pp and %{customdata[6]:+.2f}pp."
        "<extra></extra>"
    )
    hover_b = (
        f"<b>%{{customdata[0]}}</b> (Card B = {b})<br>"
        "value: %{customdata[1]}<br>"
        "contribution: %{customdata[2]:+.2f}pp<br>"
        f"Card A = {a} value: %{{customdata[3]}} (contribution %{{customdata[4]:+.2f}}pp)<br>"
        "Across the set, this feature contributes between "
        "%{customdata[5]:+.2f}pp and %{customdata[6]:+.2f}pp."
        "<extra></extra>"
    )

    fig = go.Figure()
    fig.add_bar(
        y=df_sig["label"], x=df_sig["A"], orientation="h", name=a, marker_color="#1f77b4",
        customdata=cd_a, hovertemplate=hover_a,
    )
    fig.add_bar(
        y=df_sig["label"], x=df_sig["B"], orientation="h", name=b, marker_color="#ff7f0e",
        customdata=cd_b, hovertemplate=hover_b,
    )
    fig.update_layout(
        barmode="group",
        title="Per-term contributions (A vs B)",
        xaxis_title="contribution to GIH WR",
        height=max(350, 22 * len(df_sig) + 100),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    top_div = df.head(3)
    st.subheader("Top 3 divergences")
    for _, r in top_div.iterrows():
        gap_pp = (r["A"] - r["B"]) * 100.0
        st.markdown(
            f"- **{r['display']}**: card A={r['val_a']}, card B={r['val_b']} "
            f"(contribution gap: {gap_pp:+.2f}pp)"
        )


def main() -> None:
    sets = discover_sets()
    if not sets:
        st.error(
            "No trained sets found under outputs/models/. "
            "Run `uv run python -m src.run_pipeline --set <CODE>` first."
        )
        st.stop()
    default_idx = sets.index(st.session_state.get("set_code", sets[0])) if st.session_state.get("set_code", sets[0]) in sets else 0
    variant_default = st.session_state.get("variant", "no_rarity")
    variant_idx = VARIANTS.index(variant_default) if variant_default in VARIANTS else 0
    with st.sidebar:
        st.selectbox(
            "Set", sets, index=default_idx, key="set_code",
            format_func=lambda s: s.upper(),
            help="Switch between trained sets.",
        )
        st.radio(
            "Model", VARIANTS, index=variant_idx, key="variant",
            format_func=lambda v: VARIANT_LABELS[v],
            help=(
                "rarity_ord dominates every set's EBM (it's a crude proxy for card power level). "
                "'Without rarity' drops it so finer mechanical signals — stat thresholds, ETB, "
                "color identity — surface in Key Findings and Feature Shapes. 'With rarity' "
                "predicts slightly better but interprets worse."
            ),
        )
    set_code = current_set()
    variant = current_variant()
    st.title(f"{set_code.upper()} Limited — EBM Explorer ({VARIANT_LABELS[variant]})")
    sf_sets = "/".join(sorted(scryfall_sets_for(set_code)))
    st.caption(f"Drafted set: {set_code.upper()} (Scryfall codes: {sf_sets}). Target: 17Lands GIH WR.")

    tabs = st.tabs(["Key Findings", "Card Explorer", "Residual Leaderboard", "Feature Shapes", "Compare Cards"])
    with tabs[0]:
        tab_key_findings()
    with tabs[1]:
        tab_card_explorer()
    with tabs[2]:
        tab_residual_leaderboard()
    with tabs[3]:
        tab_feature_shapes()
    with tabs[4]:
        tab_compare()


if __name__ == "__main__":
    main()
