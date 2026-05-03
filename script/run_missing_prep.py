"""Run prep fns for tasks missing metadata/<dataset>/<task>.csv."""
import sys
from pathlib import Path

ROOT = Path("/Users/lkieu/PycharmProjects/Audio-Health-Benchmark")
sys.path.insert(0, str(ROOT))

import tempfile

from ahb.prep.aphasia import prepare_aphasia_pwaC
from ahb.prep.standard import prepare_data as prepare_default
from ahb.prep.dbank import prepare_dbank_mmseR
from ahb.prep.edaic import prepare_edaic_phqR
from ahb.prep.ksof import prepare_ksof_stutL
from ahb.prep.mvdr import prepare_data as prepare_mvdr
from ahb.prep.utils import save_task_csv
import pandas as pd


RATIO = [70, 10, 20]
SEED = 2026


def src(dataset):
    wav = ROOT / f"data/{dataset}/processed/audio"
    meta = ROOT / f"data/{dataset}/processed/{dataset}.csv"
    return str(wav), str(meta)


def tmp_manifests(name):
    d = Path(tempfile.mkdtemp(prefix=f"prep_{name}_"))
    return str(d/"train.json"), str(d/"valid.json"), str(d/"test.json")


def run_aphasia():
    wav, meta = src("aphasia")
    tr, va, te = tmp_manifests("aphasia_pwaC")
    prepare_aphasia_pwaC(wav, meta, tr, va, te, RATIO, SEED, "aphasia", "pwaC")


def run_dbank_adC():
    wav, meta = src("dbank")
    tr, va, te = tmp_manifests("dbank_adC")
    prepare_default(wav, meta, tr, va, te, RATIO, SEED, "dbank", "adC")


def run_dbank_mmseR():
    wav, meta = src("dbank")
    tr, va, te = tmp_manifests("dbank_mmseR")
    prepare_dbank_mmseR(wav, meta, tr, va, te, RATIO, SEED, "dbank", "mmseR")


def _edaic_with_csv(dataset, task, phq_col):
    """edaic/daic prep fns don't save task CSV — replicate minimal logic."""
    wav, meta = src(dataset)
    df = pd.read_csv(meta)
    df["path"] = Path(wav).resolve() / df["path"]
    df["label"] = df[phq_col].astype(float)
    tr = df[df["split"] == 0]; va = df[df["split"] == 1]; te = df[df["split"] == 2]
    save_task_csv(tr, va, te, dataset, task)


def run_edaic_phqR():
    # PHQ8_Score in EDAIC source CSV
    wav, meta = src("edaic")
    df = pd.read_csv(meta)
    # figure out right col
    for c in ["PHQ_Score", "PHQ8_Score", "PHQ_8_Score"]:
        if c in df.columns:
            _edaic_with_csv("edaic", "phqR", c)
            return
    raise RuntimeError(f"PHQ score column not found. cols={df.columns.tolist()[:30]}")


def run_ksof_stutL():
    wav, meta = src("ksof")
    tr, va, te = tmp_manifests("ksof_stutL")
    prepare_ksof_stutL(wav, meta, tr, va, te, RATIO, SEED, "ksof", "stutL")


def run_mvdr(task, raw_label_key, num_fold=5):
    """MVDR uses k-fold; write a metadata CSV with split_0..split_{k-1} via save_kfold_task_csv."""
    wav, meta = src("mvdr")
    d = Path(tempfile.mkdtemp(prefix=f"prep_mvdr_{task}_"))
    prepare_mvdr(
        wav, meta,
        str(d/"train.json"), str(d/"valid.json"),
        SEED, raw_label_key, num_fold,
        "mvdr", task,
    )


RUNNERS = {
    "aphasia_pwaC": run_aphasia,
    "dbank_adC": run_dbank_adC,
    "dbank_mmseR": run_dbank_mmseR,
    "edaic_phqR": run_edaic_phqR,
    "ksof_stutL": run_ksof_stutL,
    "mvdr_hyC": lambda: run_mvdr("hyC", "hy_rating"),
    "mvdr_parkC": lambda: run_mvdr("parkC", "label"),
    "mvdr_updrs5R": lambda: run_mvdr("updrs5R", "updrs_ii5"),
    "mvdr_updrs18R": lambda: run_mvdr("updrs18R", "updrs_iii18"),
}


def main():
    names = sys.argv[1:] or list(RUNNERS)
    for n in names:
        print(f"\n========== {n} ==========")
        try:
            RUNNERS[n]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!! {n} FAILED: {e}")


if __name__ == "__main__":
    main()
