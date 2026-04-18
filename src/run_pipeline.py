from __future__ import annotations

import argparse
from pathlib import Path

import json

from src.build_image_map import build_image_map
from src.fetch_cards import cards_for_set
from src.featurize import featurize
from src.load_17lands import merge_ratings
from src.make_outputs import make_outputs
from src.train import train

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def run(set_code: str) -> None:
    code = set_code.lower()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [1/6] fetch_cards ({code}) ===")
    cards = cards_for_set(code)
    cards.to_parquet(PROCESSED / f"cards_{code}.parquet", index=False)

    print(f"\n=== [2/6] build_image_map ({code}) ===")
    img_map = build_image_map(code)
    (PROCESSED / f"image_map_{code}.json").write_text(json.dumps(img_map, indent=2))
    print(f"Image map: {len(img_map)} cards")

    print(f"\n=== [3/6] load_17lands ({code}) ===")
    merged = merge_ratings(code)
    merged.to_parquet(PROCESSED / f"cards_with_ratings_{code}.parquet", index=False)

    print(f"\n=== [4/6] featurize ({code}) ===")
    feats = featurize(code)
    feats.to_parquet(PROCESSED / f"features_{code}.parquet", index=False)
    print(f"Features: {feats.shape[0]} rows, {feats.shape[1]} cols")

    print(f"\n=== [5/6] train ({code}) ===")
    train(code)

    print(f"\n=== [6/6] make_outputs ({code}) ===")
    make_outputs(code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_code", required=True, help="Set code, e.g. TMT, FIN, DFT")
    args = ap.parse_args()
    run(args.set_code)


if __name__ == "__main__":
    main()
