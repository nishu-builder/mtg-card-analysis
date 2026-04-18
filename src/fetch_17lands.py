from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

USER_AGENT = "mtg-card-analysis/0.1 (github.com/nishu-builder; contact: nishu.builder@gmail.com)"
CACHE_TTL_SECONDS = 24 * 3600
REQUEST_SLEEP_SECONDS = 1.0
MIN_GIH = 200

COLUMNS = [
    "name",
    "color",
    "rarity",
    "game_count",
    "ever_drawn_win_rate",
    "ever_drawn_game_count",
    "opening_hand_win_rate",
    "drawn_improvement_win_rate",
    "avg_seen",
    "avg_pick",
]


def _today() -> str:
    return dt.date.today().isoformat()


def _scryfall_release_date(set_code: str) -> str:
    url = f"https://api.scryfall.com/sets/{set_code.lower()}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    time.sleep(REQUEST_SLEEP_SECONDS)
    return r.json()["released_at"]


def _cache_path(set_code: str, end_date: str) -> Path:
    return RAW / f"17lands_{set_code.lower()}_{end_date}.json"


def _fetch_cached(set_code: str, start_date: str, end_date: str) -> list[dict]:
    path = _cache_path(set_code, end_date)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            print(f"Using cached 17lands file ({age/3600:.1f}h old): {path}")
            with path.open() as f:
                return json.load(f)

    params = {
        "expansion": set_code.upper(),
        "format": "PremierDraft",
        "start_date": start_date,
        "end_date": end_date,
    }
    url = "https://www.17lands.com/card_ratings/data"
    print(f"Fetching {url} {params}")
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    data = r.json()
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    print(f"Cached {len(data)} rows to {path}")
    time.sleep(REQUEST_SLEEP_SECONDS)
    return data


def fetch_17lands(set_code: str, end_date: str | None = None) -> pd.DataFrame:
    end_date = end_date or _today()
    start_date = _scryfall_release_date(set_code)
    print(f"Set {set_code.upper()}: release {start_date} → {end_date}")

    rows = _fetch_cached(set_code, start_date, end_date)
    df = pd.DataFrame(rows)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"17lands response missing columns: {missing}")

    df = df[COLUMNS].copy()
    before = len(df)
    df = df[df["ever_drawn_game_count"].fillna(0) >= MIN_GIH].reset_index(drop=True)
    print(f"Filtered ever_drawn_game_count >= {MIN_GIH}: {before} → {len(df)} rows")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_code", required=True, help="Set code, e.g. TMT, FIN, DFT")
    ap.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD); default today")
    args = ap.parse_args()

    df = fetch_17lands(args.set_code, args.end_date)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"17lands_{args.set_code.lower()}.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
