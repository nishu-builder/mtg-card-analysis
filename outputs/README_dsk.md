# Outputs — DSK (no-rarity model)

Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.

**Cards in model:** 245 non-land cards with >=200 GIH games out of 269 total.

## Model R^2

- OLS in-sample R^2: **0.3365**, 5-fold CV R^2: **-0.2557**
- EBM in-sample R^2: **0.2723**, 5-fold CV R^2: **-0.0032**

## Top 10 positive residuals (overperforming their profile)

- **Unholy Annex // Ritual Chamber**: plays better than its stats/text suggest (actual 64.7% vs predicted 54.8%)
- **Tyvar, the Pummeler**: plays better than its stats/text suggest (actual 63.1% vs predicted 54.6%)
- **Valgavoth's Onslaught**: plays better than its stats/text suggest (actual 65.8% vs predicted 57.6%)
- **Dollmaker's Shop // Porcelain Gallery**: plays better than its stats/text suggest (actual 62.6% vs predicted 55.1%)
- **Unstoppable Slasher**: plays better than its stats/text suggest (actual 61.6% vs predicted 54.2%)
- **Dissection Tools**: plays better than its stats/text suggest (actual 62.3% vs predicted 55.7%)
- **The Swarmweaver**: plays better than its stats/text suggest (actual 64.6% vs predicted 58.3%)
- **Overlord of the Boilerbilges**: plays better than its stats/text suggest (actual 64.1% vs predicted 58.0%)
- **Screaming Nemesis**: plays better than its stats/text suggest (actual 61.8% vs predicted 55.7%)
- **Overlord of the Mistmoors**: plays better than its stats/text suggest (actual 68.2% vs predicted 62.2%)

## Top 10 negative residuals (underperforming their profile)

- **Dazzling Theater // Prop Room**: underperforms its profile (actual 39.3% vs predicted 54.8%)
- **Grievous Wound**: underperforms its profile (actual 43.2% vs predicted 53.7%)
- **Doomsday Excruciator**: underperforms its profile (actual 44.4% vs predicted 53.9%)
- **The Tale of Tamiyo**: underperforms its profile (actual 46.1% vs predicted 54.8%)
- **Cackling Slasher**: underperforms its profile (actual 46.5% vs predicted 54.4%)
- **Unwanted Remake**: underperforms its profile (actual 47.6% vs predicted 54.9%)
- **Meathook Massacre II**: underperforms its profile (actual 49.1% vs predicted 55.6%)
- **Winter, Misanthropic Guide**: underperforms its profile (actual 48.2% vs predicted 53.4%)
- **Enter the Enigma**: underperforms its profile (actual 50.3% vs predicted 55.5%)
- **Charred Foyer // Warped Space**: underperforms its profile (actual 49.4% vs predicted 54.5%)
