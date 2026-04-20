# Outputs — TMT (no-rarity model)

Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.

**Cards in model:** 176 non-land cards with >=200 GIH games out of 190 total.

## Model R^2

- OLS in-sample R^2: **0.3796**, 5-fold CV R^2: **-0.4922**
- EBM in-sample R^2: **0.4532**, 5-fold CV R^2: **-0.0613**

## Top 10 positive residuals (overperforming their profile)

- **Sally Pride, Lioness Leader**: plays better than its stats/text suggest (actual 69.7% vs predicted 59.9%)
- **The Last Ronin**: plays better than its stats/text suggest (actual 70.0% vs predicted 62.6%)
- **April O'Neil, Hacktivist**: plays better than its stats/text suggest (actual 64.6% vs predicted 57.6%)
- **Mighty Mutanimals**: plays better than its stats/text suggest (actual 64.7% vs predicted 58.0%)
- **Agent Bishop, Man in Black**: plays better than its stats/text suggest (actual 64.0% vs predicted 57.9%)
- **Ravenous Robots**: plays better than its stats/text suggest (actual 61.8% vs predicted 56.2%)
- **Lessons from Life**: plays better than its stats/text suggest (actual 60.4% vs predicted 54.8%)
- **Manhole Missile**: plays better than its stats/text suggest (actual 60.0% vs predicted 55.1%)
- **Improvised Arsenal**: plays better than its stats/text suggest (actual 57.7% vs predicted 52.9%)
- **Courier of Comestibles**: plays better than its stats/text suggest (actual 62.9% vs predicted 58.1%)

## Top 10 negative residuals (underperforming their profile)

- **Negate**: underperforms its profile (actual 46.8% vs predicted 54.3%)
- **Putrid Pals**: underperforms its profile (actual 48.6% vs predicted 55.6%)
- **Hard-Won Jitte**: underperforms its profile (actual 45.1% vs predicted 51.4%)
- **Shredder's Armor**: underperforms its profile (actual 47.5% vs predicted 53.7%)
- **Party Dude**: underperforms its profile (actual 50.7% vs predicted 56.8%)
- **Punk Frogs**: underperforms its profile (actual 50.6% vs predicted 56.4%)
- **Foot Mystic**: underperforms its profile (actual 50.4% vs predicted 56.0%)
- **Mutant Chain Reaction**: underperforms its profile (actual 50.8% vs predicted 56.4%)
- **Henchbots**: underperforms its profile (actual 50.9% vs predicted 56.4%)
- **Crustacean Commando**: underperforms its profile (actual 50.8% vs predicted 55.8%)
