# Outputs — BLB (no-rarity model)

Shape plots and residuals below are from the **no-rarity** EBM variant — rarity_ord is dropped so finer mechanical signals surface. See `outputs/README.md` for the with-rarity comparison.

**Cards in model:** 245 non-land cards with >=200 GIH games out of 265 total.

## Model R^2

- OLS in-sample R^2: **0.3500**, 5-fold CV R^2: **-0.2893**
- EBM in-sample R^2: **0.3720**, 5-fold CV R^2: **-0.0347**

## Top 10 positive residuals (overperforming their profile)

- **Mabel, Heir to Cragflame**: plays better than its stats/text suggest (actual 63.9% vs predicted 55.4%)
- **Season of Loss**: plays better than its stats/text suggest (actual 65.6% vs predicted 57.2%)
- **Dreamdew Entrancer**: plays better than its stats/text suggest (actual 65.1% vs predicted 56.9%)
- **Manifold Mouse**: plays better than its stats/text suggest (actual 63.2% vs predicted 56.0%)
- **Valley Questcaller**: plays better than its stats/text suggest (actual 64.0% vs predicted 57.2%)
- **Innkeeper's Talent**: plays better than its stats/text suggest (actual 62.5% vs predicted 56.0%)
- **Dragonhawk, Fate's Tempest**: plays better than its stats/text suggest (actual 62.1% vs predicted 56.1%)
- **Hired Claw**: plays better than its stats/text suggest (actual 60.2% vs predicted 55.0%)
- **Warren Warleader**: plays better than its stats/text suggest (actual 62.6% vs predicted 57.4%)
- **Fecund Greenshell**: plays better than its stats/text suggest (actual 63.4% vs predicted 58.6%)

## Top 10 negative residuals (underperforming their profile)

- **Wear Down**: underperforms its profile (actual 43.4% vs predicted 53.6%)
- **Stormsplitter**: underperforms its profile (actual 43.1% vs predicted 52.0%)
- **For the Common Good**: underperforms its profile (actual 44.7% vs predicted 53.5%)
- **Hoarder's Overflow**: underperforms its profile (actual 45.6% vs predicted 52.1%)
- **Artist's Talent**: underperforms its profile (actual 45.9% vs predicted 51.7%)
- **Dawn's Truce**: underperforms its profile (actual 49.1% vs predicted 54.8%)
- **Finch Formation**: underperforms its profile (actual 50.8% vs predicted 56.4%)
- **Gossip's Talent**: underperforms its profile (actual 48.9% vs predicted 54.3%)
- **Stocking the Pantry**: underperforms its profile (actual 49.3% vs predicted 54.6%)
- **War Squeak**: underperforms its profile (actual 48.7% vs predicted 53.9%)
