from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.fetch_17lands import fetch_17lands

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

RENAME_MAP = {
    "color": "color_17l",
    "rarity": "rarity_17l",
    "game_count": "n_gp",
    "ever_drawn_win_rate": "gih_wr",
    "ever_drawn_game_count": "n_gih",
    "opening_hand_win_rate": "oh_wr",
    "drawn_improvement_win_rate": "iwd",
    "avg_seen": "alsa",
    "avg_pick": "ata",
}


def merge_ratings(set_code: str) -> pd.DataFrame:
    code = set_code.lower()
    cards_path = PROCESSED / f"cards_{code}.parquet"
    cards = pd.read_parquet(cards_path)
    ratings = fetch_17lands(code).rename(columns=RENAME_MAP)
    print(f"17lands rows for {code.upper()}: {len(ratings)}")

    merged = cards.merge(ratings, on="name", how="left", indicator=True)
    matched = (merged["_merge"] == "both").sum()
    only_cards = (merged["_merge"] == "left_only").sum()
    only_ratings = len(ratings) - matched
    print(f"Matched: {matched}")
    print(f"Cards with no 17lands data: {only_cards}")
    if only_cards:
        for n in cards.loc[~cards["name"].isin(ratings["name"]), "name"].head(10):
            print(f"    {n}")
    print(f"17lands rows with no card match: {only_ratings}")
    if only_ratings:
        unmatched = set(ratings["name"]) - set(cards["name"])
        for n in list(unmatched)[:10]:
            print(f"    {n}")
    return merged.drop(columns=["_merge"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_code", required=True)
    args = ap.parse_args()

    merged = merge_ratings(args.set_code)
    out = PROCESSED / f"cards_with_ratings_{args.set_code.lower()}.parquet"
    merged.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
