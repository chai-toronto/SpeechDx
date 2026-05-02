"""Create a pyannoteAI voiceprint from an audio file and save to JSON."""

import argparse
import json
import os
from pathlib import Path

from pyannoteai.sdk import Client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="JSON output path")
    ap.add_argument("--name", default="ellie", help="label for this voiceprint")
    ap.add_argument("--api-key", default=os.environ.get("PYANNOTEAI_API_KEY"))
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set PYANNOTEAI_API_KEY or pass --api-key")

    client = Client(api_key=args.api_key)
    print(f"Uploading {args.audio}...")
    media_url = client.upload(args.audio)
    print("Running voiceprint job...")
    job_id = client.voiceprint(media_url)
    result = client.retrieve(job_id)
    vp = result.get("output", result).get("voiceprint")
    if not vp:
        raise SystemExit(f"No voiceprint in response: {result}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({args.name: vp}, indent=2))
    print(f"Saved voiceprint '{args.name}' -> {args.out}")


if __name__ == "__main__":
    main()
