from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

CSV_PATH = RAW / "17lands_card_ratings.csv"


def _pct_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.rstrip("%").str.strip(), errors="coerce") / 100.0


def _pp_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.rstrip("pp").str.strip(), errors="coerce") / 100.0


def main() -> None:
    cards = pd.read_parquet(PROCESSED / "cards.parquet")
    ratings = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    ratings["Name"] = ratings["Name"].replace({"Bespoke B?": "Bespoke Bō"})
    print(f"17Lands rows: {len(ratings)}")
    print(f"17Lands cols: {list(ratings.columns)}")

    for col in ["% GP", "GP WR", "OH WR", "GD WR", "GIH WR", "GNS WR"]:
        if col in ratings.columns:
            ratings[col] = _pct_to_float(ratings[col])
    if "IIH" in ratings.columns:
        ratings["IWD"] = _pp_to_float(ratings["IIH"])

    ratings = ratings.rename(
        columns={
            "Name": "name",
            "Color": "color_17l",
            "Rarity": "rarity_17l",
            "# Seen": "n_seen",
            "ALSA": "alsa",
            "# Picked": "n_picked",
            "ATA": "ata",
            "# GP": "n_gp",
            "% GP": "pct_gp",
            "GP WR": "gp_wr",
            "# OH": "n_oh",
            "OH WR": "oh_wr",
            "# GD": "n_gd",
            "GD WR": "gd_wr",
            "# GIH": "n_gih",
            "GIH WR": "gih_wr",
            "# GNS": "n_gns",
            "GNS WR": "gns_wr",
        }
    )

    merged = cards.merge(ratings, on="name", how="left", indicator=True)
    matched = (merged["_merge"] == "both").sum()
    only_cards = (merged["_merge"] == "left_only").sum()
    only_ratings = len(ratings) - matched
    print(f"Matched: {matched}")
    print(f"Cards with no 17Lands data: {only_cards}")
    if only_cards:
        print("  Examples (first 10):")
        for n in cards.loc[~cards["name"].isin(ratings["name"]), "name"].head(10):
            print(f"    {n}")
    print(f"17Lands rows with no card match: {only_ratings}")
    if only_ratings:
        unmatched = set(ratings["name"]) - set(cards["name"])
        for n in list(unmatched)[:10]:
            print(f"    {n}")

    merged = merged.drop(columns=["_merge"])
    out = PROCESSED / "cards_with_ratings.parquet"
    merged.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
