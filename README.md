# mtg-card-analysis

Per-feature contribution analysis of **MTG Limited win rate** across multiple sets, using an Explainable Boosting Machine so non-linear effects like stat thresholds and rarity jumps emerge without hand-coding.

**Live app: https://mtg-ebm.streamlit.app/** — pick a set from the sidebar.

Target: 17Lands **GIH WR** (game-in-hand win rate) for non-land cards with ≥200 GIH games.

## Sets covered

| code | set                          | cards used | EBM CV R² |
| ---- | ---------------------------- | ---------: | --------: |
| TMT  | Teenage Mutant Ninja Turtles |        176 |     0.060 |
| DFT  | Aetherdrift                  |        242 |     0.131 |
| TDM  | Tarkir: Dragonstorm          |        232 |     0.290 |
| FIN  | Final Fantasy (UB)           |        241 |     0.045 |
| EOE  | Edge of Eternities           |        246 |    −0.041 |
| DSK  | Duskmourn: House of Horror   |        245 |     0.065 |
| BLB  | Bloomburrow                  |        245 |     0.105 |
| FDN  | Foundations                  |        236 |     0.129 |

"Cards used" = non-land cards with ≥200 GIH games. TMT is smaller because the set itself is smaller. See `outputs/README_<set>.md` for per-set residual writeups.

## Quickstart

```bash
uv sync

# Run the full pipeline for one set
uv run python -m src.run_pipeline --set TMT

# Or invoke individual stages:
uv run python -m src.fetch_cards       --set TMT   # Scryfall bulk → cards parquet
uv run python -m src.build_image_map   --set TMT   # name → image URL (for the app)
uv run python -m src.load_17lands      --set TMT   # 17Lands API → merge with cards
uv run python -m src.featurize         --set TMT   # structured + keyword + oracle-regex features
uv run python -m src.train             --set TMT   # OLS + EBM, residuals, CV R²
uv run python -m src.make_outputs      --set TMT   # shape plots, interaction heatmaps, writeup

# Cross-set format-drift view (overlays EBM shape function for a feature)
uv run python -m src.compare_sets --sets TMT,DFT --feature toughness

# Interactive explorer
uv run streamlit run app.py
```

## Layout

```
data/
  raw/          # Scryfall bulk JSON + 17Lands JSON caches (gitignored, ~171MB)
  processed/    # Per-set parquets + image_map JSON (committed — tiny, needed by app)
src/
  fetch_cards.py      # Scryfall bulk fetch + set filter (handles associated sets like TMT+PZA)
  fetch_17lands.py    # 17Lands API fetch, polite rate limit, 24h cache
  build_image_map.py  # extract card→image URL for app (avoids shipping 171MB oracle JSON)
  load_17lands.py     # merge 17lands data with Scryfall cards
  featurize.py        # structured + keyword + oracle-text-regex features
  train.py            # OLS + EBM, 5-fold CV R², residuals
  make_outputs.py     # shape plots, interaction heatmaps, per-set residual writeup
  compare_sets.py     # cross-set EBM shape-function overlay
  run_pipeline.py     # orchestrator: `--set TMT` runs all stages
app.py                # Streamlit 5-tab EBM explorer with set selector
outputs/              # Per-set run artifacts: models/ (committed), figures/, residuals, README
```

## Features (~55–60 per set)

- **Structured** (Scryfall): cmc, power, toughness (with `is_variable_pt`), num_colors, 5 color-identity one-hots, rarity ordinal, 7 card-type flags.
- **Keyword one-hots**: every keyword on ≥3 cards in the set (varies: Deathtouch, Flying, Menace, Gift, Manifest, …).
- **Oracle-text regex**: `draw_cards`, `damage_dealt`, `life_gain`, `pt_buff_{power,toughness}`; binary flags for ETB / attack / death / activated abilities, removal, counterspell, ramp, tutor, bounce, creates_token, has_x_cost; `targeting_flexibility` = distinct "target X" phrases.

## Streamlit app

5 tabs on top of the trained EBM (set-switchable via sidebar):

1. **Key Findings** — auto-filtered list of features with large, well-supported effects; each with shape plot + takeaway.
2. **Card Explorer** — typeahead card picker, Scryfall image, actual/predicted/residual as big numbers, per-term contribution bars with **feature value + set-range tooltips**, feature-values table.
3. **Residual Leaderboard** — top overperformers vs top underperformers; filter by rarity, color identity, min games played; clicking a row jumps the Card Explorer.
4. **Feature Shapes** — pick any EBM term; shape function (or interaction heatmap) + histogram of the feature's distribution in the set.
5. **Compare Cards** — two cards side-by-side with grouped contribution bars, both values shown per bar, and a top-3 divergence list in the `"card A=yes, card B=no (contribution gap: +X.XXpp)"` format.

## Caveats

- `rarity_ord` dominates feature importance in most sets — it's a crude proxy for overall power level. Dropping it surfaces finer signals.
- CV R² varies dramatically by set (TDM at 0.29, EOE slightly negative). With ~240 rows and 55+ features the signal/noise is modest; treat the leaderboard residuals as suggestive, not definitive.
- FIN's OLS CV R² is catastrophically negative (−18.4) — near-zero-variance color features make the linear model unstable on that set. EBM handles it fine.
- 17Lands data is fetched from the public API each time; results drift as the format evolves. The in-repo artifacts were last trained 2026-04-18.
