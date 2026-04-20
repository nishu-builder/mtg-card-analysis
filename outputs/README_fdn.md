# Outputs — FDN (no-rarity model)

Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.

**Cards in model:** 236 non-land cards with >=200 GIH games out of 434 total.

## Model R^2

- OLS in-sample R^2: **0.4059**, 5-fold CV R^2: **-0.1503**
- EBM in-sample R^2: **0.4329**, 5-fold CV R^2: **0.0950**

## Top 10 positive residuals (overperforming their profile)

- **Sylvan Scavenging**: plays better than its stats/text suggest (actual 62.2% vs predicted 53.8%)
- **Celestial Armor**: plays better than its stats/text suggest (actual 62.1% vs predicted 54.7%)
- **Twinflame Tyrant**: plays better than its stats/text suggest (actual 61.1% vs predicted 54.4%)
- **Liliana, Dreadhorde General**: plays better than its stats/text suggest (actual 64.0% vs predicted 57.4%)
- **Day of Judgment**: plays better than its stats/text suggest (actual 59.4% vs predicted 52.9%)
- **Curator of Destinies**: plays better than its stats/text suggest (actual 62.3% vs predicted 55.8%)
- **Blasphemous Edict**: plays better than its stats/text suggest (actual 59.2% vs predicted 52.8%)
- **Leyline Axe**: plays better than its stats/text suggest (actual 61.0% vs predicted 54.6%)
- **Searslicer Goblin**: plays better than its stats/text suggest (actual 60.3% vs predicted 54.1%)
- **Scavenging Ooze**: plays better than its stats/text suggest (actual 61.5% vs predicted 55.4%)

## Top 10 negative residuals (underperforming their profile)

- **Thousand-Year Storm**: underperforms its profile (actual 35.8% vs predicted 47.0%)
- **Doubling Season**: underperforms its profile (actual 40.0% vs predicted 51.1%)
- **Boltwave**: underperforms its profile (actual 44.1% vs predicted 52.4%)
- **Clinquant Skymage**: underperforms its profile (actual 48.5% vs predicted 55.8%)
- **Inspiring Call**: underperforms its profile (actual 46.7% vs predicted 53.9%)
- **Fishing Pole**: underperforms its profile (actual 43.3% vs predicted 49.8%)
- **Midnight Snack**: underperforms its profile (actual 47.1% vs predicted 53.4%)
- **Painful Quandary**: underperforms its profile (actual 45.0% vs predicted 51.1%)
- **Time Stop**: underperforms its profile (actual 47.1% vs predicted 52.9%)
- **Reclamation Sage**: underperforms its profile (actual 50.0% vs predicted 55.2%)
