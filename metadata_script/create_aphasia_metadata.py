import csv
import shutil
from pathlib import Path

# AphasiaBank (TalkBank): PWA vs Control speech across multiple protocols
# (Adler, Kansas, Kurland, SCALE, Wright). Label: Control=0, PWA=1.
# No canonical split provided in the source metadata — all rows set to 0.

SOURCE_ROOT = Path("/Users/lkieu/Downloads/aphasia")
SOURCE_CSV = SOURCE_ROOT / "metadata.csv"
SOURCE_AUDIO_ROOT = SOURCE_ROOT / "data"

DATA_ROOT = Path("data/aphasia")

LABEL_MAP = {"Control": 0, "PWA": 1}


def main():
    out_audio_dir = DATA_ROOT / "processed" / "audio"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        src_fieldnames = reader.fieldnames
        src_rows = list(reader)

    # Preserve all source columns except wav_path (absolute foreign path — not useful)
    passthrough_cols = [c for c in src_fieldnames if c != "wav_path"]

    out_fieldnames = ["uid", "Participant_ID", "split", "label", "path"] + [
        c for c in passthrough_cols if c not in {"participant_id", "subgroup"}
    ]

    rows = []
    copied = skipped = 0
    seen_fnames = set()

    for r in src_rows:
        subgroup = r["subgroup"]
        if subgroup not in LABEL_MAP:
            raise ValueError(f"Unknown subgroup '{subgroup}' for {r['wav_file']}")

        dataset = r["dataset"]
        fname = r["wav_file"]
        if fname in seen_fnames:
            raise ValueError(f"Duplicate filename: {fname}")
        seen_fnames.add(fname)

        src = SOURCE_AUDIO_ROOT / dataset / subgroup / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing source audio: {src}")

        dst = out_audio_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1

        out = {c: r.get(c, "") for c in passthrough_cols}
        out["Participant_ID"] = r["participant_id"]
        out["subgroup"] = subgroup
        out["split"] = 0
        out["label"] = LABEL_MAP[subgroup]
        out["path"] = fname
        rows.append(out)

    print(f"Audio: copied {copied}, skipped {skipped} (already exist)")

    out_csv = DATA_ROOT / "processed" / "aphasia.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        w.writeheader()
        for uid, row in enumerate(rows):
            row["uid"] = uid
            w.writerow({k: row.get(k, "") for k in out_fieldnames})

    print(f"Created {out_csv} with {len(rows)} rows")
    print(f"  Label 0 (Control): {sum(1 for r in rows if r['label'] == 0)}")
    print(f"  Label 1 (PWA):     {sum(1 for r in rows if r['label'] == 1)}")


if __name__ == "__main__":
    main()