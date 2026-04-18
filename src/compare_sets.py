from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

from src.train import MIN_GIH, NON_FEATURES, TARGET

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures" / "cross_set"


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    eligible = df[(df["is_land"] == 0) & (df["n_gih"].fillna(0) >= MIN_GIH) & df[TARGET].notna()].copy()
    feat_cols = [c for c in df.columns if c not in NON_FEATURES and c != "is_land"]
    X = eligible[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in "fi" else s.fillna(0))
    y = eligible[TARGET].astype(float)
    return X, y


def _train_ebm(set_code: str) -> ExplainableBoostingRegressor:
    code = set_code.lower()
    feats = pd.read_parquet(PROCESSED / f"features_{code}.parquet")
    X, y = _prep(feats)
    print(f"[{code.upper()}] training EBM on {len(X)} cards, {X.shape[1]} features")
    ebm = ExplainableBoostingRegressor(interactions=10, max_bins=32, random_state=0)
    ebm.fit(X, y)
    return ebm


def _shape(ebm: ExplainableBoostingRegressor, feature: str) -> tuple[np.ndarray, np.ndarray]:
    exp = ebm.explain_global()
    names = exp.data()["names"]
    if feature not in names:
        raise KeyError(f"feature {feature!r} not in model terms: {names[:20]}...")
    fd = exp.data(names.index(feature))
    xs = np.asarray(fd["names"])
    ys = np.asarray(fd["scores"], dtype=float)
    if fd.get("type") == "univariate" and len(xs) == len(ys) + 1:
        xs = 0.5 * (xs[:-1].astype(float) + xs[1:].astype(float))
    return xs, ys


def compare(set_a: str, set_b: str, feature: str) -> Path:
    a = set_a.lower()
    b = set_b.lower()
    ebm_a = _train_ebm(a)
    ebm_b = _train_ebm(b)
    xs_a, ys_a = _shape(ebm_a, feature)
    xs_b, ys_b = _shape(ebm_b, feature)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.step(xs_a, ys_a, where="mid", label=a.upper(), color="#1f77b4")
    ax.scatter(xs_a, ys_a, s=16, color="#1f77b4")
    ax.step(xs_b, ys_b, where="mid", label=b.upper(), color="#d62728")
    ax.scatter(xs_b, ys_b, s=16, color="#d62728")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"EBM shape: {feature} — {a.upper()} vs {b.upper()}")
    ax.set_xlabel(feature)
    ax.set_ylabel("contribution to GIH WR")
    ax.legend()
    fig.tight_layout()

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{feature}_{a}_vs_{b}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help="Two set codes, comma-separated. e.g. TMT,FIN")
    ap.add_argument("--feature", default="toughness")
    args = ap.parse_args()

    parts = [s.strip() for s in args.sets.split(",") if s.strip()]
    if len(parts) != 2:
        raise SystemExit("--sets requires exactly two codes, e.g. --sets TMT,FIN")
    compare(parts[0], parts[1], args.feature)


if __name__ == "__main__":
    main()
