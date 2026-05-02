"""
Create IEMOCAP metadata CSV.

Source: /Users/lkieu/Downloads/IEMOCAP_full_release
Audio symlinked into data/iemocap/processed/audio/<subsession>/<sentence>.wav
CSV written to data/iemocap/processed/iemocap.csv

Participant_ID = subsession.split('_')[0] (e.g. Ses01F) -> 10 total.
Gender from trailing M/F.
Label = 3rd whitespace-separated field of the bracketed header line
(raw code like neu/fru/hap/ang/...).

Between bracketed headers there are annotator lines:
  C-<annotator> -> categorical (semicolon-separated emotions)
  A-<annotator> -> attribute (val X; act X; dom X)
Emotions in parentheses are ignored. We count occurrences per emotion
across the categorical annotators and average val/act/dom across
the attribute annotators.

Split: session-based (Sessions 1-3 train, 4 val, 5 test).
"""
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

import soundfile as sf

SRC = Path("/Users/lkieu/Downloads/IEMOCAP_full_release")
DST_ROOT = Path("data/iemocap/processed")
AUDIO_DST = DST_ROOT / "audio"
CSV_PATH = DST_ROOT / "iemocap.csv"
CORRUPT_REPORT_PATH = DST_ROOT / "corrupt_files.csv"

EMOTIONS = [
    "Neutral", "Happiness", "Sadness", "Anger", "Surprise",
    "Fear", "Disgust", "Frustration", "Excited", "Other",
]

HEADER_RE = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+(?P<turn>\S+)\s+(?P<emo>\S+)\s+\[(?P<v>[\d.]+),\s*(?P<a>[\d.]+),\s*(?P<d>[\d.]+)\]"
)
CAT_RE = re.compile(r"^C-[^:]+:\s*(?P<body>.*)$")
ATTR_RE = re.compile(
    r"^A-[^:]+:\s*val\s*(?P<v>\d+)\s*;\s*act\s*(?P<a>\d+)\s*;\s*dom\s*(?P<d>\d+)"
)


def strip_parens(text: str) -> str:
    """Remove everything inside parentheses."""
    return re.sub(r"\([^)]*\)", "", text)


def parse_emo_file(txt_path: Path) -> dict:
    """Parse one EmoEvaluation txt, return {turn: {label, counts, v, a, d}}."""
    results = {}
    current = None
    with open(txt_path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = HEADER_RE.match(line)
            if m:
                if current is not None:
                    _finalize(current)
                    results[current["turn"]] = current
                current = {
                    "turn": m.group("turn"),
                    "label": m.group("emo"),
                    "ghd_v": float(m.group("v")),
                    "ghd_a": float(m.group("a")),
                    "ghd_d": float(m.group("d")),
                    "counts": Counter(),
                    "_v": [], "_a": [], "_d": [],
                }
                continue
            if current is None:
                continue
            mc = CAT_RE.match(line)
            if mc:
                body = strip_parens(mc.group("body"))
                for tok in body.split(";"):
                    tok = tok.strip()
                    if tok:
                        current["counts"][tok] += 1
                continue
            ma = ATTR_RE.match(line)
            if ma:
                current["_v"].append(int(ma.group("v")))
                current["_a"].append(int(ma.group("a")))
                current["_d"].append(int(ma.group("d")))
    if current is not None:
        _finalize(current)
        results[current["turn"]] = current
    return results


def _finalize(entry: dict) -> None:
    def _avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    entry["val"] = _avg(entry["_v"])
    entry["act"] = _avg(entry["_a"])
    entry["dom"] = _avg(entry["_d"])
    del entry["_v"], entry["_a"], entry["_d"]


def session_split(session_idx: int) -> int:
    if session_idx <= 3:
        return 0
    if session_idx == 4:
        return 1
    return 2


def validate_wav(wav_path: Path) -> float:
    """Fully decode a WAV file and return its duration in seconds."""
    with sf.SoundFile(str(wav_path)) as audio:
        if audio.frames == 0 or audio.samplerate == 0:
            raise ValueError("zero-length audio")

        while audio.read(frames=65536, dtype="float32").size:
            pass

        duration = round(audio.frames / audio.samplerate, 3)
        if duration <= 0.0:
            raise ValueError(f"non-positive duration {duration}")
        return duration


def main() -> None:
    AUDIO_DST.mkdir(parents=True, exist_ok=True)

    rows = []
    corrupt = []
    unknown_emotions = Counter()

    for sess in sorted(SRC.glob("Session*")):
        session_idx = int(sess.name.replace("Session", ""))
        wav_root = sess / "sentences" / "wav"
        emo_root = sess / "dialog" / "EmoEvaluation"

        for sub_dir in sorted(p for p in wav_root.iterdir() if p.is_dir()):
            subsession = sub_dir.name
            if subsession.startswith("."):
                continue
            txt_path = emo_root / f"{subsession}.txt"
            if not txt_path.exists():
                print(f"Missing annotation: {txt_path}")
                continue
            anns = parse_emo_file(txt_path)

            # symlink subsession dir
            link_dir = AUDIO_DST / subsession
            link_dir.mkdir(exist_ok=True)

            for wav in sorted(sub_dir.glob("*.wav")):
                turn = wav.stem
                ann = anns.get(turn)
                if ann is None:
                    print(f"No annotation for {turn}")
                    continue

                # corruption / zero-length check
                try:
                    duration = validate_wav(wav)
                except Exception as e:
                    corrupt.append((str(wav), str(e)))
                    continue

                # copy
                dst_path = link_dir / wav.name
                if not dst_path.exists():
                    shutil.copy2(wav, dst_path)

                pid = subsession.split("_")[0]
                gender = "female" if pid.endswith("F") else "male"
                rel_path = f"{subsession}/{wav.name}"

                row = {
                    "Participant_ID": pid,
                    "gender": gender,
                    "session": session_idx,
                    "split": session_split(session_idx),
                    "label": ann["label"],
                    "val": ann["val"],
                    "act": ann["act"],
                    "dom": ann["dom"],
                    "gold_val": ann["ghd_v"],
                    "gold_act": ann["ghd_a"],
                    "gold_dom": ann["ghd_d"],
                    "duration": duration,
                    "path": rel_path,
                }
                for emo in EMOTIONS:
                    row[f"count_{emo}"] = ann["counts"].get(emo, 0)
                # track any emotion tokens outside the fixed list
                for k in ann["counts"]:
                    if k not in EMOTIONS:
                        unknown_emotions[k] += ann["counts"][k]
                rows.append(row)

    rows.sort(key=lambda r: r["path"])

    fieldnames = (
        ["uid", "Participant_ID", "gender", "session", "split", "label",
         "val", "act", "dom", "gold_val", "gold_act", "gold_dom", "duration"]
        + [f"count_{e}" for e in EMOTIONS]
        + ["path"]
    )
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for uid, row in enumerate(rows):
            row["uid"] = uid
            writer.writerow(row)

    with open(CORRUPT_REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "error"])
        writer.writeheader()
        for path, error in corrupt:
            writer.writerow({"path": path, "error": error})

    print(f"Wrote {CSV_PATH} with {len(rows)} rows.")
    print(f"Wrote {CORRUPT_REPORT_PATH} with {len(corrupt)} rows.")
    print(f"Participants: {sorted({r['Participant_ID'] for r in rows})}")
    label_counts = Counter(r["label"] for r in rows)
    print("Label distribution:", dict(label_counts))
    print("Split counts:", Counter(r["split"] for r in rows))
    if unknown_emotions:
        print("Unknown emotion tokens (not tracked per-row):", dict(unknown_emotions))
    print(f"Corrupt files: {len(corrupt)}")
    for p, e in corrupt:
        print(f"  {p}: {e}")


if __name__ == "__main__":
    main()
