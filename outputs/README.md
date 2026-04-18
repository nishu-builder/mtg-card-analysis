# Outputs

**Interactive explorer:** `uv run streamlit run app.py` — 4-tab UI over the EBM (Card Explorer, Residual Leaderboard, Feature Shapes, Compare Cards). Reuses the artifacts below, no retraining.

**Cards in model:** 177 non-land cards with >=200 GIH games out of 190 total.

## Model R^2

- OLS in-sample R^2: **0.5071**, 5-fold CV R^2: **-0.1528**
- EBM in-sample R^2: **0.6711**, 5-fold CV R^2: **0.1325**

(OLS CV R^2 is negative — the full linear specification overfits on 177 rows; EBM is the better estimator here.)

## Files

- `ols_coefficients.csv`: OLS coefs, std errors, t, p — sorted by |t|.
- `residuals.csv`: actual vs EBM-predicted GIH WR, sorted descending (positive residuals = candidates the model thinks are better than their profile).
- `figures/shape_*.png`: EBM 1-D shape functions for cmc, power, toughness, and top-5 most important terms.
- `figures/interactions/inter_*.png`: EBM heatmaps for top-3 pairwise interactions.
- `models/ebm.pkl`: pickled EBM and feature list.

## Top 10 positive residuals (overperforming their profile)

- **Sally Pride, Lioness Leader**: plays better than its stats/text suggest (actual 69.6% vs predicted 63.6%)
- **Mighty Mutanimals**: plays better than its stats/text suggest (actual 65.8% vs predicted 60.2%)
- **Lessons from Life**: plays better than its stats/text suggest (actual 60.9% vs predicted 56.0%)
- **Metalhead**: plays better than its stats/text suggest (actual 62.5% vs predicted 58.4%)
- **Agent Bishop, Man in Black**: plays better than its stats/text suggest (actual 65.3% vs predicted 61.3%)
- **April O'Neil, Hacktivist**: plays better than its stats/text suggest (actual 64.2% vs predicted 60.2%)
- **Courier of Comestibles**: plays better than its stats/text suggest (actual 62.2% vs predicted 58.3%)
- **Dream Beavers**: plays better than its stats/text suggest (actual 63.5% vs predicted 59.8%)
- **Manhole Missile**: plays better than its stats/text suggest (actual 59.9% vs predicted 56.4%)
- **Frog Butler**: plays better than its stats/text suggest (actual 62.1% vs predicted 58.9%)

## Top 10 negative residuals (underperforming their profile)

- **Hard-Won Jitte**: underperforms its profile (actual 47.2% vs predicted 53.5%)
- **Shredder's Armor**: underperforms its profile (actual 48.5% vs predicted 54.5%)
- **Turtles Forever**: underperforms its profile (actual 50.3% vs predicted 56.1%)
- **Negate**: underperforms its profile (actual 50.0% vs predicted 55.3%)
- **April, Reporter of the Weird**: underperforms its profile (actual 50.9% vs predicted 55.3%)
- **Party Dude**: underperforms its profile (actual 51.2% vs predicted 55.4%)
- **Putrid Pals**: underperforms its profile (actual 50.5% vs predicted 54.5%)
- **Casey Jones, Vigilante**: underperforms its profile (actual 52.6% vs predicted 56.4%)
- **New Generation's Technique**: underperforms its profile (actual 52.1% vs predicted 55.9%)
- **Leonardo, Cutting Edge**: underperforms its profile (actual 56.9% vs predicted 60.6%)
