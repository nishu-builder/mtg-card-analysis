from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

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
EBM_PATH = MODELS / "ebm.pkl"


st.set_page_config(page_title="TMT EBM Explorer", layout="wide")


@st.cache_resource
def load_ebm() -> tuple[Any, list[str], list[str]]:
    if not EBM_PATH.exists():
        st.error(
            f"EBM artifact not found at {EBM_PATH}. "
            "Run the pipeline first: `uv run python -m src.train`."
        )
        st.stop()
    with EBM_PATH.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names: list[str] = bundle["feature_names"]
    term_names: list[str] = model.explain_global().data()["names"]
    return model, feature_names, term_names


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "features.parquet")


@st.cache_data
def load_cards() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "cards_with_ratings.parquet")


@st.cache_data
def load_image_url_map() -> dict[str, str]:
    if not ORACLE_JSON.exists():
        return {}
    with ORACLE_JSON.open() as f:
        data = json.load(f)
    out: dict[str, str] = {}
    for c in data:
        if c.get("set") not in {"tmt", "pza"}:
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


@st.cache_data(show_spinner=False)
def get_card_image_bytes(name: str) -> bytes | None:
    IMG_CACHE.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    cache_path = IMG_CACHE / f"{safe}.jpg"
    if cache_path.exists():
        return cache_path.read_bytes()
    url_map = load_image_url_map()
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


@st.cache_data
def build_model_rows() -> pd.DataFrame:
    feats = load_features()
    model, feat_cols, _ = load_ebm_cached_tuple()
    eligible = feats[
        (feats["is_land"] == 0) & (feats["n_gih"].fillna(0) >= 200) & feats["gih_wr"].notna()
    ].copy()
    X = eligible[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in "fi" else s.fillna(0))
    preds = model.predict(X)
    eligible = eligible.reset_index(drop=True)
    X = X.reset_index(drop=True)
    eligible["predicted_gih_wr"] = preds
    eligible["residual"] = eligible["gih_wr"].values - preds
    return eligible.merge(X.reset_index(drop=True).add_prefix(""), left_index=True, right_index=True,
                          suffixes=("", "_x"))


def load_ebm_cached_tuple() -> tuple[Any, list[str], list[str]]:
    return load_ebm()


@st.cache_data
def get_X() -> pd.DataFrame:
    feats = load_features()
    model, feat_cols, _ = load_ebm_cached_tuple()
    eligible = feats[
        (feats["is_land"] == 0) & (feats["n_gih"].fillna(0) >= 200) & feats["gih_wr"].notna()
    ].copy()
    X = eligible[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in "fi" else s.fillna(0))
    X.index = eligible["name"].values
    return X


@st.cache_data
def get_model_table() -> pd.DataFrame:
    feats = load_features()
    model, feat_cols, _ = load_ebm_cached_tuple()
    eligible = feats[
        (feats["is_land"] == 0) & (feats["n_gih"].fillna(0) >= 200) & feats["gih_wr"].notna()
    ].copy()
    X = eligible[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in "fi" else s.fillna(0))
    preds = model.predict(X)
    eligible = eligible.reset_index(drop=True).copy()
    eligible["predicted_gih_wr"] = preds
    eligible["residual"] = eligible["gih_wr"].values - preds
    cards = load_cards()
    meta = cards[
        [c for c in ["name", "mana_cost", "type_line", "power", "toughness", "rarity",
                     "oracle_text", "color_identity"] if c in cards.columns]
    ].copy()
    return eligible.merge(meta, on="name", how="left", suffixes=("", "_c"))


def _local_scores_for_card(name: str) -> tuple[list[str], list[float], list[Any], float]:
    model, feat_cols, term_names = load_ebm_cached_tuple()
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
    fd = g.data(term_idx)
    return fd


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
    c1.metric("Actual GIH WR", f"{actual:.1%}")
    c2.metric("Predicted GIH WR", f"{predicted:.1%}")
    c3.metric("Residual", f"{residual*100:+.2f}pp")


def _contribution_bar(names: list[str], scores: list[float], title: str = "Per-feature EBM contributions") -> go.Figure:
    df = pd.DataFrame({"term": names, "score": scores})
    df["abs"] = df["score"].abs()
    df = df[df["abs"] > 1e-9].sort_values("abs", ascending=True)
    colors = ["#2ca02c" if s > 0 else "#d62728" for s in df["score"]]
    fig = go.Figure(
        go.Bar(
            y=df["term"],
            x=df["score"],
            orientation="h",
            marker_color=colors,
            customdata=df[["score"]],
            hovertemplate="<b>%{y}</b><br>contribution: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="contribution to GIH WR",
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(300, 22 * len(df) + 80),
    )
    return fig


def _feature_values_table(feat_names_univariate: list[str], names_local: list[str],
                          values_local: list[Any], scores_local: list[float]) -> pd.DataFrame:
    rows = []
    for n, v, s in zip(names_local, values_local, scores_local):
        rows.append({"term": n, "value": v, "contribution": s})
    return pd.DataFrame(rows).sort_values("contribution", key=lambda s: s.abs(), ascending=False)


# --- Tab 1: Card Explorer ------------------------------------------------
def tab_card_explorer() -> None:
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

    term_names, scores, values, intercept = _local_scores_for_card(selected)
    if not term_names:
        st.warning("No local explanation available.")
        return

    st.plotly_chart(_contribution_bar(term_names, scores), use_container_width=True)

    tbl = _feature_values_table(term_names, term_names, values, scores)
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


def tab_residual_leaderboard() -> None:
    table = get_model_table()
    filt = _filter_leaderboard(table)

    cols = ["name", "rarity", "cmc", "gih_wr", "predicted_gih_wr", "residual", "n_gih"]
    view = filt[cols].rename(
        columns={
            "gih_wr": "actual_wr",
            "predicted_gih_wr": "predicted_wr",
            "n_gih": "games",
        }
    )

    top_pos = view.sort_values("residual", ascending=False).head(25).reset_index(drop=True)
    top_neg = view.sort_values("residual", ascending=True).head(25).reset_index(drop=True)

    st.caption(f"{len(view)} cards after filters. Click a row to load it in Card Explorer.")

    left, right = st.columns(2)

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["actual_wr"] = (out["actual_wr"] * 100).round(2)
        out["predicted_wr"] = (out["predicted_wr"] * 100).round(2)
        out["residual"] = (out["residual"] * 100).round(2)
        return out

    with left:
        st.subheader("Top 25 overperformers")
        ev = st.dataframe(
            _fmt(top_pos), use_container_width=True, hide_index=True, on_select="rerun",
            selection_mode="single-row", key="lb_pos",
        )
        if ev and ev.selection and ev.selection["rows"]:
            idx = ev.selection["rows"][0]
            st.session_state["selected_card"] = top_pos.iloc[idx]["name"]
            st.session_state["_go_tab1"] = True

    with right:
        st.subheader("Top 25 underperformers")
        ev2 = st.dataframe(
            _fmt(top_neg), use_container_width=True, hide_index=True, on_select="rerun",
            selection_mode="single-row", key="lb_neg",
        )
        if ev2 and ev2.selection and ev2.selection["rows"]:
            idx = ev2.selection["rows"][0]
            st.session_state["selected_card"] = top_neg.iloc[idx]["name"]
            st.session_state["_go_tab1"] = True

    if st.session_state.get("_go_tab1"):
        st.info(
            f"Selection set to **{st.session_state['selected_card']}** — open the *Card Explorer* tab."
        )


# --- Tab 3: Feature Shapes ----------------------------------------------
def tab_feature_shapes() -> None:
    model, feat_cols, term_names = load_ebm_cached_tuple()
    feats = load_features()
    choice = st.selectbox("Feature", term_names, key="shape_feature")
    term_idx = term_names.index(choice)
    fd = _univariate_shape(term_idx)
    is_interaction = " & " in choice

    if is_interaction:
        st.info("This is a pairwise interaction term; use the heatmap plot in `outputs/figures/interactions/`. "
                "Showing contribution grid below.")
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
                           title=f"Interaction: {choice}")
        st.plotly_chart(hmap, use_container_width=True)
        return

    xs = np.asarray(fd["names"])
    ys = np.asarray(fd["scores"], dtype=float)

    if len(xs) == len(ys) + 1:
        centers = 0.5 * (xs[:-1].astype(float) + xs[1:].astype(float))
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
    fig.update_layout(title=f"EBM shape: {choice}",
                      xaxis_title=choice, yaxis_title="contribution to GIH WR",
                      margin=dict(l=10, r=10, t=40, b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

    if choice in feats.columns:
        vals = feats[choice].dropna()
        hist = go.Figure(go.Histogram(x=vals, nbinsx=min(30, max(5, int(vals.nunique())))))
        hist.update_layout(title=f"Distribution of {choice} in the set",
                           xaxis_title=choice, yaxis_title="# cards",
                           margin=dict(l=10, r=10, t=40, b=10), height=260)
        st.plotly_chart(hist, use_container_width=True)

    # caption
    if choice in feats.columns:
        vmin, vmax = float(feats[choice].min()), float(feats[choice].max())
    else:
        vmin, vmax = float(np.min(xs)), float(np.max(xs))
    cmin, cmax = float(np.min(ys)), float(np.max(ys))
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
    st.caption(
        f"**{choice}** ranges from {vmin:g} to {vmax:g} in the set. "
        f"The model's contribution ranges from {cmin:+.4f} to {cmax:+.4f}. "
        f"Biggest jump is {jump_txt}."
    )


# --- Tab 4: Compare Cards ----------------------------------------------
def tab_compare() -> None:
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

    df_a = pd.DataFrame({"term": names_a, "A": scores_a})
    df_b = pd.DataFrame({"term": names_b, "B": scores_b})
    df = df_a.merge(df_b, on="term", how="outer").fillna(0.0)
    df["diff"] = (df["A"] - df["B"]).abs()
    df = df.sort_values("diff", ascending=False)
    df_sig = df[(df["A"].abs() + df["B"].abs()) > 1e-6].head(30).iloc[::-1]

    fig = go.Figure()
    fig.add_bar(y=df_sig["term"], x=df_sig["A"], orientation="h", name=a, marker_color="#1f77b4")
    fig.add_bar(y=df_sig["term"], x=df_sig["B"], orientation="h", name=b, marker_color="#ff7f0e")
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
        st.markdown(
            f"- **{r['term']}**: A={r['A']:+.4f}, B={r['B']:+.4f} (|Δ|={r['diff']:.4f})"
        )


def main() -> None:
    st.title("TMT Limited — EBM Explorer")
    st.caption("Drafted set: Teenage Mutant Ninja Turtles (tmt/pza). Target: 17Lands GIH WR.")

    tabs = st.tabs(["Card Explorer", "Residual Leaderboard", "Feature Shapes", "Compare Cards"])
    with tabs[0]:
        tab_card_explorer()
    with tabs[1]:
        tab_residual_leaderboard()
    with tabs[2]:
        tab_feature_shapes()
    with tabs[3]:
        tab_compare()


if __name__ == "__main__":
    main()
