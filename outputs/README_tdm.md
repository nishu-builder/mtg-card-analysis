# Outputs — TDM (no-rarity model)

Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.

**Cards in model:** 232 non-land cards with >=200 GIH games out of 267 total.

## Model R^2

- OLS in-sample R^2: **0.4700**, 5-fold CV R^2: **0.0071**
- EBM in-sample R^2: **0.6645**, 5-fold CV R^2: **0.1419**

## Top 10 positive residuals (overperforming their profile)

- **Sage of the Skies**: plays better than its stats/text suggest (actual 64.4% vs predicted 57.9%)
- **Fresh Start**: plays better than its stats/text suggest (actual 57.0% vs predicted 51.5%)
- **Sinkhole Surveyor**: plays better than its stats/text suggest (actual 60.2% vs predicted 55.6%)
- **Surrak, Elusive Hunter**: plays better than its stats/text suggest (actual 60.5% vs predicted 56.0%)
- **Roar of Endless Song**: plays better than its stats/text suggest (actual 65.0% vs predicted 60.9%)
- **Qarsi Revenant**: plays better than its stats/text suggest (actual 61.0% vs predicted 56.9%)
- **Winternight Stories**: plays better than its stats/text suggest (actual 60.4% vs predicted 56.4%)
- **Warden of the Grove**: plays better than its stats/text suggest (actual 61.3% vs predicted 57.4%)
- **Tersa Lightshatter**: plays better than its stats/text suggest (actual 58.1% vs predicted 54.2%)
- **Dragonback Assault**: plays better than its stats/text suggest (actual 67.5% vs predicted 63.7%)

## Top 10 negative residuals (underperforming their profile)

- **Stillness in Motion**: underperforms its profile (actual 44.5% vs predicted 51.3%)
- **Glacierwood Siege**: underperforms its profile (actual 49.7% vs predicted 55.4%)
- **Dragonback Lancer**: underperforms its profile (actual 50.9% vs predicted 56.1%)
- **Mammoth Bellow**: underperforms its profile (actual 54.1% vs predicted 58.3%)
- **Call the Spirit Dragons**: underperforms its profile (actual 45.8% vs predicted 49.9%)
- **Monastery Messenger**: underperforms its profile (actual 52.8% vs predicted 56.7%)
- **Highspire Bell-Ringer**: underperforms its profile (actual 51.3% vs predicted 55.2%)
- **Kin-Tree Nurturer**: underperforms its profile (actual 54.1% vs predicted 57.7%)
- **Summit Intimidator**: underperforms its profile (actual 50.5% vs predicted 54.1%)
- **Worthy Cost**: underperforms its profile (actual 50.7% vs predicted 54.2%)
