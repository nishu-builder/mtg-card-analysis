# mtg-card-analysis

Per-feature contribution analysis of MTG *Teenage Mutant Ninja Turtles* (set `tmt`, drafted with `pza`) Limited win rate, using an Explainable Boosting Machine so non-linear effects like stat thresholds emerge without hand-coding.

Target: 17Lands **GIH WR** (game-in-hand win rate) for non-land cards with ≥200 GIH games.

## Quickstart

```bash
uv sync

# 1. Fetch card data from Scryfall (cached for 7 days)
uv run python -m src.fetch_cards

# 2. Load 17Lands card ratings CSV (download from 17lands.com/public_datasets
#    and drop into data/raw/17lands_card_ratings.csv — a copy is already in the repo)
uv run python -m src.load_17lands

# 3. Build feature matrix
uv run python -m src.featurize

# 4. Train OLS + EBM, write residuals + ols coefs
uv run python -m src.train

# 5. Render plots + README writeup
uv run python -m src.make_outputs

# Interactive explorer
uv run streamlit run app.py
```

## Layout

```
data/
  raw/          # Scryfall bulk JSON (gitignored, ~170MB), 17Lands CSV, card images (gitignored)
  processed/    # Parquets: cards, cards_with_ratings, features (gitignored — regeneratable)
src/
  fetch_cards.py      # Scryfall bulk fetch + set filter
  load_17lands.py     # parse CSV, join to cards, report unmatched
  featurize.py        # structured + keyword + oracle-text-regex features
  train.py            # OLS + EBM, 5-fold CV R², residuals
  make_outputs.py     # shape plots, interaction heatmaps, residual writeup
app.py                # Streamlit 4-tab EBM explorer
outputs/              # Run artifacts: models/, figures/, residuals.csv, ols_coefficients.csv, README.md
```

## Features (53 total, non-land TMT cards)

- **Structured** (Scryfall): cmc, power, toughness (with `is_variable_pt`), num_colors, 5 color-identity one-hots, rarity ordinal, 7 card-type flags.
- **Keyword one-hots**: every keyword on ≥3 cards in the set (19 kept: Deathtouch, Flying, Menace, Food, Sneak, Flash, Trample, …).
- **Oracle-text regex**: `draw_cards`, `damage_dealt`, `life_gain`, `pt_buff_{power,toughness}`; binary flags for ETB / attack / death / activated abilities, removal, counterspell, ramp, tutor, bounce, creates_token, has_x_cost; `targeting_flexibility` = distinct "target X" phrases.

## Model

Two models on 177 non-land cards with ≥200 GIH games:

| model | in-sample R² | 5-fold CV R² |
| --- | --- | --- |
| OLS (statsmodels) | 0.507 | **−0.153** (overfits: 53 features / 177 rows) |
| EBM (`interactions=10`) | 0.671 | **0.133** |

EBM is the better estimator here. OLS coefficients still useful as a "price per attribute" readout — see `outputs/ols_coefficients.csv`.

## Streamlit app

`uv run streamlit run app.py` opens a 4-tab explorer on top of the trained EBM:

1. **Card Explorer** — typeahead card picker, Scryfall image, actual/predicted/residual as big numbers, Plotly bar chart of per-term `explain_local()` contributions (green positive, red negative), plus a feature-values table.
2. **Residual Leaderboard** — top 25 overperformers vs top 25 underperformers side-by-side; sidebar filters on rarity, color identity, min games played; clicking a row jumps the Card Explorer to that card.
3. **Feature Shapes** — pick any of the 63 EBM terms; view its 1-D shape function (or interaction heatmap) plus a histogram of the feature's distribution in the set, with an auto-caption noting range and biggest jump.
4. **Compare Cards** — two cards side-by-side with grouped contribution bars and a top-3 divergence list.

## Caveats

- 190 unique oracle-ID cards in TMT after Scryfall dedup (pza promos collapse into tmt). 53 features on 177 rows is data-starved; CV R² of 0.13 is honest but modest.
- `rarity_ord` dominates feature importance — it's a crude proxy for overall power level. Dropping it can surface finer signals.
- 17Lands data snapshot: `card-ratings-2026-04-17.csv`. Rerunning `load_17lands.py` on a fresher snapshot may shift residuals.
