# Outputs — DFT (no-rarity model)

Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.

**Cards in model:** 242 non-land cards with >=200 GIH games out of 259 total.

## Model R^2

- OLS in-sample R^2: **0.4011**, 5-fold CV R^2: **-0.5239**
- EBM in-sample R^2: **0.4205**, 5-fold CV R^2: **0.0287**

## Top 10 positive residuals (overperforming their profile)

- **Lumbering Worldwagon**: plays better than its stats/text suggest (actual 65.3% vs predicted 57.5%)
- **Mu Yanling, Wind Rider**: plays better than its stats/text suggest (actual 65.2% vs predicted 57.4%)
- **Perilous Snare**: plays better than its stats/text suggest (actual 61.9% vs predicted 54.7%)
- **March of the World Ooze**: plays better than its stats/text suggest (actual 63.0% vs predicted 56.1%)
- **Howlsquad Heavy**: plays better than its stats/text suggest (actual 61.5% vs predicted 55.4%)
- **Gloryheath Lynx**: plays better than its stats/text suggest (actual 59.5% vs predicted 53.7%)
- **The Aetherspark**: plays better than its stats/text suggest (actual 60.9% vs predicted 55.4%)
- **Stock Up**: plays better than its stats/text suggest (actual 59.4% vs predicted 54.0%)
- **Draconautics Engineer**: plays better than its stats/text suggest (actual 61.6% vs predicted 56.4%)
- **Sab-Sunen, Luxa Embodied**: plays better than its stats/text suggest (actual 67.5% vs predicted 62.7%)

## Top 10 negative residuals (underperforming their profile)

- **Point the Way**: underperforms its profile (actual 42.6% vs predicted 53.8%)
- **Repurposing Bay**: underperforms its profile (actual 45.6% vs predicted 53.9%)
- **Full Throttle**: underperforms its profile (actual 44.9% vs predicted 53.1%)
- **Ketramose, the New Dawn**: underperforms its profile (actual 47.9% vs predicted 54.9%)
- **Alacrian Armory**: underperforms its profile (actual 46.7% vs predicted 53.2%)
- **Stall Out**: underperforms its profile (actual 48.9% vs predicted 54.4%)
- **Racers' Scoreboard**: underperforms its profile (actual 48.1% vs predicted 53.3%)
- **Spell Pierce**: underperforms its profile (actual 49.6% vs predicted 54.6%)
- **Spire Mechcycle**: underperforms its profile (actual 49.9% vs predicted 54.7%)
- **Alacrian Jaguar**: underperforms its profile (actual 50.3% vs predicted 54.7%)
