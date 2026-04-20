# Per-set EBM results

**Default model in the app: no-rarity.** `rarity_ord` is a crude proxy for "how strong WotC intended this card to be" — it swamps finer mechanical signals (P/T thresholds, ETB, removal, color identity). The with-rarity model predicts slightly better on average; the no-rarity model interprets better.

Per-set residual writeups for the no-rarity model: `outputs/README_<set>.md`.

| set | full name | cards used | CV R² (with rarity) | CV R² (no rarity) | Δ (no − with) |
| --- | --- | ---: | ---: | ---: | ---: |
| TMT | Teenage Mutant Ninja Turtles | 176 | +0.062 | −0.061 | −0.123 |
| DFT | Aetherdrift | 242 | +0.131 | +0.029 | −0.102 |
| TDM | Tarkir: Dragonstorm | 232 | +0.290 | +0.142 | −0.148 |
| FIN | Final Fantasy (UB) | 241 | +0.045 | +0.045 | +0.000 |
| EOE | Edge of Eternities | 246 | −0.041 | −0.019 | +0.022 |
| DSK | Duskmourn: House of Horror | 245 | +0.065 | −0.003 | −0.068 |
| BLB | Bloomburrow | 245 | +0.105 | −0.035 | −0.139 |
| FDN | Foundations | 236 | +0.129 | +0.095 | −0.034 |

## How to read this

- **CV R² (with rarity)** — 5-fold cross-validated R² for the full-feature EBM.
- **CV R² (no rarity)** — same model, `rarity_ord` dropped before fit. EBM interactions auto-exclude rarity because the feature is absent.
- **Δ** — expected to be negative: dropping a highly-predictive feature loses prediction power. What we care about is that the no-rarity model still retains enough signal to be worth interpreting, and that per-feature contributions tell a richer mechanical story than rarity-vs-rest.

Switch between variants in the Streamlit app via the sidebar **Model** radio.
