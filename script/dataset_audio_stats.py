import os
import sys
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

DATA_ROOT = "/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/data"
DATASETS = [
    "aphasia", "avfad", "c9s", "coswara", "dbank", "edaic",
    "iemocap", "ksof", "mvdr", "nemours", "ravdess", "torgo", "uaspeech",
]


def probe(path):
    try:
        info = sf.info(path)
        return info.frames / info.samplerate, info.samplerate
    except Exception as e:
        return None, None


def collect_paths(dataset):
    csv_path = os.path.join(DATA_ROOT, dataset, "processed", f"{dataset}.csv")
    audio_root = os.path.join(DATA_ROOT, dataset, "processed", "audio")
    paths = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        subjs = set()
        for row in reader:
            p = row["path"]
            paths.append(os.path.join(audio_root, p))
            subjs.add(row["Participant_ID"])
    return paths, len(subjs)



def main():
    print(f"{'dataset':<14} {'n':>7} {'subj_count':>6} {'mean_s':>9} {'q025_s':>9} {'q975_s':>9} {'sample_rate'}")
    with ProcessPoolExecutor(max_workers=8) as pool:
        for ds in DATASETS:
            paths, num_subj = collect_paths(ds)
            durations = []
            srs = []
            for dur, sr in pool.map(probe, paths, chunksize=64):
                if dur is None:
                    continue
                durations.append(dur)
                srs.append(sr)
            if not durations:
                print(f"{ds:<14} no readable files")
                continue
            durations = np.array(durations)
            mean = durations.mean()
            q025, q975 = np.quantile(durations, [0.025, 0.975])
            sr_unique = sorted(set(srs))
            if len(sr_unique) == 1:
                sr_str = str(sr_unique[0])
            else:
                from collections import Counter
                c = Counter(srs)
                sr_str = ", ".join(f"{k}:{v}" for k, v in c.most_common())
            print(f"{ds:<14} {len(durations):>7d} {num_subj:>6} {mean:>9.2f} {q025:>9.2f} {q975:>9.2f} {sr_str}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()