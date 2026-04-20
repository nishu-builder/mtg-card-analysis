from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"

REQUIRED_SHAPE_FEATURES = ["cmc", "power", "toughness"]


VARIANT = "no_rarity"  # interpretation model — surfaces mechanical signal


def _load(set_code: str) -> tuple:
    code = set_code.lower()
    with (MODELS / f"ebm_{code}_{VARIANT}.pkl").open("rb") as f:
        bundle = pickle.load(f)
    with (MODELS / f"summary_{code}_{VARIANT}.pkl").open("rb") as f:
        summary = pickle.load(f)
    return bundle["model"], bundle["feature_names"], summary


def _plot_univariate(explain, feature: str, out_path: Path) -> None:
    data = explain.data()
    names = data["names"]
    idx = names.index(feature)
    fd = explain.data(idx)
    xs = fd.get("names")
    ys = fd.get("scores")
    if xs is None or ys is None:
        print(f"  !! cannot plot {feature}: no shape data")
        return

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ys_arr = np.asarray(ys, dtype=float)
    xs_arr = np.asarray(xs)
    if fd.get("type") == "univariate" and len(xs_arr) == len(ys_arr) + 1:
        centers = 0.5 * (xs_arr[:-1] + xs_arr[1:])
        ax.step(centers, ys_arr, where="mid")
        ax.scatter(centers, ys_arr, s=12)
    else:
        ax.plot(range(len(ys_arr)), ys_arr, marker="o")
        ax.set_xticks(range(len(xs_arr)))
        ax.set_xticklabels([str(x) for x in xs_arr], rotation=45, ha="right")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"EBM shape: {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("contribution to GIH WR")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_interaction(explain, idx: int, feat_name: str, out_path: Path) -> None:
    fd = explain.data(idx)
    xs_left = np.asarray(fd["left_names"], dtype=float) if "left_names" in fd else np.asarray(fd["names"][0])
    xs_right = np.asarray(fd["right_names"], dtype=float) if "right_names" in fd else np.asarray(fd["names"][1])
    scores = np.asarray(fd["scores"], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(scores.T, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(scores)), vmax=np.max(np.abs(scores)))
    left_ticks = np.arange(scores.shape[0])
    right_ticks = np.arange(scores.shape[1])
    left_labels = xs_left if len(xs_left) == scores.shape[0] else 0.5 * (xs_left[:-1] + xs_left[1:])
    right_labels = xs_right if len(xs_right) == scores.shape[1] else 0.5 * (xs_right[:-1] + xs_right[1:])
    ax.set_xticks(left_ticks)
    ax.set_xticklabels([f"{v:.1f}" if isinstance(v, (int, float, np.floating)) else str(v) for v in left_labels],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(right_ticks)
    ax.set_yticklabels([f"{v:.1f}" if isinstance(v, (int, float, np.floating)) else str(v) for v in right_labels],
                       fontsize=8)
    ax.set_title(f"EBM interaction: {feat_name}")
    fig.colorbar(im, ax=ax, label="contribution")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_outputs(set_code: str) -> None:
    code = set_code.lower()
    ebm, _, summary = _load(code)
    fig_dir = OUTPUTS / "figures" / code
    fig_int = fig_dir / "interactions"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_int.mkdir(parents=True, exist_ok=True)

    global_exp = ebm.explain_global()
    data = global_exp.data()
    term_names: list[str] = data["names"]
    importances = np.asarray(data["scores"], dtype=float)

    imp_df = pd.DataFrame({"term": term_names, "importance": importances})
    imp_df["is_interaction"] = imp_df["term"].str.contains(" & ")
    uni = imp_df[~imp_df["is_interaction"]].sort_values("importance", ascending=False)
    inter = imp_df[imp_df["is_interaction"]].sort_values("importance", ascending=False)

    print("Top 10 univariate terms by importance:")
    print(uni.head(10).to_string(index=False))
    print("\nTop 5 interactions by importance:")
    print(inter.head(5).to_string(index=False))

    required = [f for f in REQUIRED_SHAPE_FEATURES if f in term_names]
    top_five = [t for t in uni["term"].tolist() if t not in required][:5]
    plot_set = required + top_five
    print(f"\nPlotting shape functions: {plot_set}")

    for feat in plot_set:
        out_path = fig_dir / f"shape_{feat}.png"
        _plot_univariate(global_exp, feat, out_path)
        print(f"  wrote {out_path}")

    for name in inter.head(3)["term"].tolist():
        idx = term_names.index(name)
        slug = name.replace(" & ", "__").replace(" ", "_")
        _plot_interaction(global_exp, idx, name, fig_int / f"inter_{slug}.png")
        print(f"  wrote interaction: {name}")

    resid = pd.read_csv(OUTPUTS / f"residuals_{code}_{VARIANT}.csv")
    top_pos = resid.head(10)
    top_neg = resid.tail(10).iloc[::-1]

    def guess(row: pd.Series) -> str:
        a = row["actual_gih_wr"]
        p = row["predicted_gih_wr"]
        if row["residual"] > 0:
            return f"plays better than its stats/text suggest (actual {a:.1%} vs predicted {p:.1%})"
        return f"underperforms its profile (actual {a:.1%} vs predicted {p:.1%})"

    lines = [f"# Outputs — {code.upper()} (no-rarity model)\n"]
    lines.append(
        f"Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped "
        "so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.\n"
    )
    lines.append(f"**Cards in model:** {summary['n_cards_used']} non-land cards with >=200 GIH games "
                 f"out of {summary['n_cards_total']} total.\n")
    lines.append("## Model R^2\n")
    lines.append(f"- OLS in-sample R^2: **{summary['ols_in_sample_r2']:.4f}**, 5-fold CV R^2: **{summary['ols_cv_r2']:.4f}**")
    lines.append(f"- EBM in-sample R^2: **{summary['ebm_in_sample_r2']:.4f}**, 5-fold CV R^2: **{summary['ebm_cv_r2']:.4f}**\n")
    lines.append("## Top 10 positive residuals (overperforming their profile)\n")
    for _, r in top_pos.iterrows():
        lines.append(f"- **{r['name']}**: {guess(r)}")
    lines.append("\n## Top 10 negative residuals (underperforming their profile)\n")
    for _, r in top_neg.iterrows():
        lines.append(f"- **{r['name']}**: {guess(r)}")

    (OUTPUTS / f"README_{code}.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote outputs/README_{code}.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_code", required=True)
    args = ap.parse_args()
    make_outputs(args.set_code)


if __name__ == "__main__":
    main()
