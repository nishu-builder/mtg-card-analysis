from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"

TARGET = "gih_wr"
MIN_GIH = 200

NON_FEATURES = {"name", "gih_wr", "n_gih", "n_oh", "alsa", "iwd"}
RARITY_FEATURES = {"rarity_ord"}


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    eligible = df[(df["is_land"] == 0) & (df["n_gih"].fillna(0) >= MIN_GIH) & df[TARGET].notna()].copy()
    feat_cols = [c for c in df.columns if c not in NON_FEATURES and c != "is_land"]
    X = eligible[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in "fi" else s.fillna(0))
    y = eligible[TARGET].astype(float)
    return X, y, feat_cols, eligible


def _cv_r2(model, X: pd.DataFrame, y: pd.Series, label: str) -> float:
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=1)
    print(f"  {label}: CV R^2 = {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean())


def train(set_code: str, exclude_rarity: bool = False) -> dict:
    code = set_code.lower()
    suffix = "no_rarity" if exclude_rarity else "with_rarity"
    feats = pd.read_parquet(PROCESSED / f"features_{code}.parquet")
    print(f"[{code.upper()}/{suffix}] total cards: {len(feats)}")

    X, y, _, eligible = _prep(feats)
    if exclude_rarity:
        drop = [c for c in RARITY_FEATURES if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
            print(f"  dropped rarity features: {drop}")
    print(f"  cards used: {len(X)}, feature dim = {X.shape[1]}")
    print(f"  target mean = {y.mean():.4f}, std = {y.std():.4f}")

    MODELS.mkdir(parents=True, exist_ok=True)

    print("\n[OLS]")
    Xc = sm.add_constant(X.astype(float))
    ols_fit = sm.OLS(y.values, Xc.values).fit()
    coef_df = pd.DataFrame(
        {
            "feature": ["const"] + list(X.columns),
            "coef": ols_fit.params,
            "std_err": ols_fit.bse,
            "t": ols_fit.tvalues,
            "p_value": ols_fit.pvalues,
        }
    )
    coef_df["abs_t"] = coef_df["t"].abs()
    coef_df = coef_df.sort_values("abs_t", ascending=False).drop(columns=["abs_t"])
    OUTPUTS.mkdir(exist_ok=True)
    coef_df.to_csv(OUTPUTS / f"ols_coefficients_{code}_{suffix}.csv", index=False)
    print(f"  In-sample R^2 = {ols_fit.rsquared:.4f}  (n={len(y)}, k={X.shape[1]+1})")

    ols_cv_r2 = _cv_r2(LinearRegression(), X, y, "OLS (sklearn, for CV)")

    print("\n[EBM]")
    ebm = ExplainableBoostingRegressor(interactions=10, max_bins=32, random_state=0)
    ebm_cv_r2 = _cv_r2(ebm, X, y, "EBM")
    ebm.fit(X, y)
    in_sample_r2 = ebm.score(X, y)
    print(f"  EBM in-sample R^2 = {in_sample_r2:.4f}")

    with (MODELS / f"ebm_{code}_{suffix}.pkl").open("wb") as f:
        pickle.dump({"model": ebm, "feature_names": list(X.columns)}, f)

    preds = ebm.predict(X)
    resid_df = pd.DataFrame(
        {
            "name": eligible["name"].values,
            "n_gih": eligible["n_gih"].values,
            "actual_gih_wr": y.values,
            "predicted_gih_wr": preds,
            "residual": y.values - preds,
        }
    ).sort_values("residual", ascending=False)
    resid_df.to_csv(OUTPUTS / f"residuals_{code}_{suffix}.csv", index=False)

    summary = {
        "set_code": code,
        "variant": suffix,
        "exclude_rarity": exclude_rarity,
        "n_cards_total": int(len(feats)),
        "n_cards_used": int(len(X)),
        "n_features": int(X.shape[1]),
        "ols_in_sample_r2": float(ols_fit.rsquared),
        "ols_cv_r2": ols_cv_r2,
        "ebm_in_sample_r2": float(in_sample_r2),
        "ebm_cv_r2": ebm_cv_r2,
    }
    print("\nSummary:", summary)
    with (MODELS / f"summary_{code}_{suffix}.pkl").open("wb") as f:
        pickle.dump(summary, f)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_code", required=True)
    ap.add_argument("--no-rarity", action="store_true", help="Drop rarity_ord before fitting.")
    args = ap.parse_args()
    train(args.set_code, exclude_rarity=args.no_rarity)


if __name__ == "__main__":
    main()
