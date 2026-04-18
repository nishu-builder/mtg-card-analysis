from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ORACLE_PATH = RAW / "scryfall_oracle.json"
STALE_AFTER_SECONDS = 7 * 24 * 3600
SETS = {"tmt", "pza"}


def fetch_oracle_cards() -> list[dict]:
    if ORACLE_PATH.exists():
        age = time.time() - ORACLE_PATH.stat().st_mtime
        if age < STALE_AFTER_SECONDS:
            print(f"Using cached oracle file ({age/3600:.1f}h old)")
            with ORACLE_PATH.open() as f:
                return json.load(f)

    print("Fetching Scryfall bulk metadata...")
    r = requests.get("https://api.scryfall.com/bulk-data", timeout=30)
    r.raise_for_status()
    bulks = r.json()["data"]
    oracle = next(b for b in bulks if b["type"] == "oracle_cards")
    url = oracle["download_uri"]
    print(f"Downloading oracle_cards from {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()
    RAW.mkdir(parents=True, exist_ok=True)
    ORACLE_PATH.write_text(json.dumps(data))
    print(f"Saved {len(data)} cards to {ORACLE_PATH}")
    return data


def main() -> None:
    cards = fetch_oracle_cards()
    df = pd.DataFrame(cards)
    filtered = df[df["set"].isin(SETS)].copy()
    print(f"Total Scryfall cards: {len(df)}")
    print(f"Filtered to sets {SETS}: {len(filtered)}")
    by_set = filtered.groupby("set").size().to_dict()
    print(f"  by set: {by_set}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "cards.parquet"
    keep_cols = [
        c
        for c in [
            "id",
            "oracle_id",
            "name",
            "set",
            "collector_number",
            "mana_cost",
            "cmc",
            "type_line",
            "oracle_text",
            "power",
            "toughness",
            "colors",
            "color_identity",
            "keywords",
            "rarity",
            "layout",
            "produced_mana",
            "card_faces",
        ]
        if c in filtered.columns
    ]
    filtered[keep_cols].to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
