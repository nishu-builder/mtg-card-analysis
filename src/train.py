from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"

TARGET = "gih_wr"
MIN_GIH = 200

NON_FEATURES = {"name", "gih_wr", "n_gih", "n_oh", "alsa", "iwd"}


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    eligible = df[(df["is_land"] == 0) & (df["n_gih"].fillna(0) >= MIN_GIH) & df[TARGET].notna()].copy()
    feat_cols = [c for c in df.columns if c not in NON_FEATURES]
    feat_cols = [c for c in feat_cols if c != "is_land"]
    X = eligible[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if s.dtype.kind in "fi" else s.fillna(0))
    y = eligible[TARGET].astype(float)
    return X, y, feat_cols, eligible


def _cv_r2(model, X: pd.DataFrame, y: pd.Series, label: str) -> float:
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=1)
    print(f"  {label}: CV R^2 = {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean())


def main() -> None:
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    print(f"Total cards: {len(feats)}")
    non_land = (feats["is_land"] == 0).sum()
    has_target = feats[TARGET].notna().sum()
    above_thresh = (feats["n_gih"].fillna(0) >= MIN_GIH).sum()
    print(f"Non-land cards: {non_land}, with GIH WR: {has_target}, with >= {MIN_GIH} GIH games: {above_thresh}")

    X, y, feat_cols, eligible = _prep(feats)
    print(f"Cards used for training: {len(X)} (feature dim = {X.shape[1]})")
    print(f"Target mean = {y.mean():.4f}, std = {y.std():.4f}")

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
    (ROOT / "outputs").mkdir(exist_ok=True)
    coef_df.to_csv(ROOT / "outputs" / "ols_coefficients.csv", index=False)
    print(f"  In-sample R^2 = {ols_fit.rsquared:.4f}  (n={len(y)}, k={X.shape[1]+1})")

    from sklearn.linear_model import LinearRegression

    ols_cv_r2 = _cv_r2(LinearRegression(), X, y, "OLS (sklearn, for CV)")

    print("\n[EBM]")
    ebm = ExplainableBoostingRegressor(
        interactions=10,
        max_bins=32,
        random_state=0,
    )
    ebm_cv_r2 = _cv_r2(ebm, X, y, "EBM")

    ebm.fit(X, y)
    in_sample_r2 = ebm.score(X, y)
    print(f"  EBM in-sample R^2 = {in_sample_r2:.4f}")

    with (MODELS / "ebm.pkl").open("wb") as f:
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
    resid_df.to_csv(ROOT / "outputs" / "residuals.csv", index=False)

    summary = {
        "n_cards_total": int(len(feats)),
        "n_cards_used": int(len(X)),
        "ols_in_sample_r2": float(ols_fit.rsquared),
        "ols_cv_r2": ols_cv_r2,
        "ebm_in_sample_r2": float(in_sample_r2),
        "ebm_cv_r2": ebm_cv_r2,
    }
    print("\nSummary:", summary)
    with (MODELS / "summary.pkl").open("wb") as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    main()
