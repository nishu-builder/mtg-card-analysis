from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fetch_cards import ASSOCIATED_SETS, fetch_oracle_cards

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def build_image_map(set_code: str) -> dict[str, str]:
    code = set_code.lower()
    sets = set(ASSOCIATED_SETS.get(code, [code]))
    data = fetch_oracle_cards()
    out: dict[str, str] = {}
    for c in data:
        if c.get("set") not in sets:
            continue
        url = None
        if isinstance(c.get("image_uris"), dict):
            url = c["image_uris"].get("normal") or c["image_uris"].get("large")
        elif isinstance(c.get("card_faces"), list) and c["card_faces"]:
            face = c["card_faces"][0]
            if isinstance(face.get("image_uris"), dict):
                url = face["image_uris"].get("normal") or face["image_uris"].get("large")
        if url:
            out[c["name"]] = url
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_code", required=True)
    args = ap.parse_args()

    mp = build_image_map(args.set_code)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"image_map_{args.set_code.lower()}.json"
    out.write_text(json.dumps(mp, indent=2))
    print(f"Wrote {out} ({len(mp)} cards)")


if __name__ == "__main__":
    main()
