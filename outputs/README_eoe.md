# Outputs — EOE (no-rarity model)

Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.

**Cards in model:** 246 non-land cards with >=200 GIH games out of 260 total.

## Model R^2

- OLS in-sample R^2: **0.2714**, 5-fold CV R^2: **-0.3982**
- EBM in-sample R^2: **0.3717**, 5-fold CV R^2: **-0.0188**

## Top 10 positive residuals (overperforming their profile)

- **Ouroboroid**: plays better than its stats/text suggest (actual 66.1% vs predicted 55.1%)
- **Quantum Riddler**: plays better than its stats/text suggest (actual 66.0% vs predicted 57.7%)
- **Cosmogrand Zenith**: plays better than its stats/text suggest (actual 63.7% vs predicted 56.1%)
- **Elegy Acolyte**: plays better than its stats/text suggest (actual 64.5% vs predicted 57.2%)
- **Possibility Technician**: plays better than its stats/text suggest (actual 60.9% vs predicted 55.1%)
- **Beyond the Quiet**: plays better than its stats/text suggest (actual 59.3% vs predicted 53.7%)
- **Thrumming Hivepool**: plays better than its stats/text suggest (actual 57.7% vs predicted 52.6%)
- **Sunset Saboteur**: plays better than its stats/text suggest (actual 61.6% vs predicted 56.7%)
- **Anticausal Vestige**: plays better than its stats/text suggest (actual 59.3% vs predicted 54.4%)
- **Sunstar Chaplain**: plays better than its stats/text suggest (actual 60.5% vs predicted 55.7%)

## Top 10 negative residuals (underperforming their profile)

- **Loading Zone**: underperforms its profile (actual 40.7% vs predicted 52.7%)
- **Moonlit Meditation**: underperforms its profile (actual 40.0% vs predicted 51.5%)
- **Ruinous Rampage**: underperforms its profile (actual 41.8% vs predicted 51.6%)
- **Sinister Cryologist**: underperforms its profile (actual 47.7% vs predicted 56.0%)
- **Bioengineered Future**: underperforms its profile (actual 43.4% vs predicted 51.7%)
- **Terminal Velocity**: underperforms its profile (actual 45.4% vs predicted 53.2%)
- **Pinnacle Starcage**: underperforms its profile (actual 46.7% vs predicted 54.4%)
- **Drill Too Deep**: underperforms its profile (actual 49.1% vs predicted 54.6%)
- **Terrasymbiosis**: underperforms its profile (actual 47.4% vs predicted 52.6%)
- **Red Tiger Mechan**: underperforms its profile (actual 49.4% vs predicted 54.1%)
