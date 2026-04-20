# Outputs — FIN (no-rarity model)

Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.

**Cards in model:** 241 non-land cards with >=200 GIH games out of 307 total.

## Model R^2

- OLS in-sample R^2: **0.3587**, 5-fold CV R^2: **-133.0188**
- EBM in-sample R^2: **0.4661**, 5-fold CV R^2: **0.0451**

## Top 10 positive residuals (overperforming their profile)

- **Nibelheim Aflame**: plays better than its stats/text suggest (actual 64.4% vs predicted 56.6%)
- **Ardyn, the Usurper**: plays better than its stats/text suggest (actual 64.3% vs predicted 57.8%)
- **Sazh Katzroy**: plays better than its stats/text suggest (actual 64.1% vs predicted 57.7%)
- **Seifer Almasy**: plays better than its stats/text suggest (actual 59.3% vs predicted 54.0%)
- **The Lunar Whale**: plays better than its stats/text suggest (actual 59.1% vs predicted 53.9%)
- **Buster Sword**: plays better than its stats/text suggest (actual 60.7% vs predicted 55.6%)
- **Summon: Primal Odin**: plays better than its stats/text suggest (actual 62.1% vs predicted 57.1%)
- **Dragoon's Lance**: plays better than its stats/text suggest (actual 61.6% vs predicted 57.0%)
- **The Regalia**: plays better than its stats/text suggest (actual 57.6% vs predicted 53.1%)
- **Summon: Fenrir**: plays better than its stats/text suggest (actual 61.0% vs predicted 56.6%)

## Top 10 negative residuals (underperforming their profile)

- **The Fire Crystal**: underperforms its profile (actual 40.3% vs predicted 51.2%)
- **Louisoix's Sacrifice**: underperforms its profile (actual 47.7% vs predicted 54.9%)
- **Aettir and Priwen**: underperforms its profile (actual 41.2% vs predicted 48.4%)
- **Vaan, Street Thief**: underperforms its profile (actual 46.5% vs predicted 53.3%)
- **Loporrit Scout**: underperforms its profile (actual 48.9% vs predicted 55.7%)
- **Golbez, Crystal Collector**: underperforms its profile (actual 49.4% vs predicted 55.9%)
- **Galuf's Final Act**: underperforms its profile (actual 45.7% vs predicted 52.0%)
- **Excalibur II**: underperforms its profile (actual 45.1% vs predicted 51.0%)
- **Noctis, Prince of Lucis**: underperforms its profile (actual 48.3% vs predicted 53.4%)
- **Random Encounter**: underperforms its profile (actual 49.1% vs predicted 54.1%)
