"""Run every task's prepare_data_fn to regenerate metadata CSVs + SB manifests.

Mirrors the paths used by training/config/main.yaml so outputs land where
training expects them.
"""
import importlib
import inspect
import re
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO / "training" / "config" / "tasks"
DATA_FOLDER = REPO / "data"
EXPS_FOLDER = REPO / "exps"
RATIO = [70, 10, 20]
RANDOM_SEED = 2026


_REQUIRED = ("task", "dataset", "data_io_script", "prepare_data_fn")


def load_task(yaml_path: Path) -> dict:
    """Pull simple `key: scalar` fields (ignores hyperpyyaml tags / complex values)."""
    text = yaml_path.read_text()
    out = {}
    for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*):\s*([^#\n]+?)\s*(?:#.*)?$", text, flags=re.MULTILINE):
        key, val = m.group(1), m.group(2)
        if any(tok in val for tok in ("!new", "!ref", "!include", "{", "[")):
            continue
        # numeric cast
        try:
            out[key] = int(val)
            continue
        except ValueError:
            pass
        out[key] = val
    for key in _REQUIRED:
        if key not in out:
            raise ValueError(f"{yaml_path.name}: missing field '{key}'")
    return out


def run_one(yaml_path: Path) -> tuple[str, bool, str]:
    try:
        cfg = load_task(yaml_path)
    except Exception as e:
        return (yaml_path.stem, False, f"load failed: {e}")
    dataset = cfg["dataset"]
    task = cfg["task"]
    module_name = cfg["data_io_script"]
    fn_name = cfg["prepare_data_fn"]

    wav_folder = DATA_FOLDER / dataset / "processed" / "audio"
    metadata_path = DATA_FOLDER / dataset / "processed" / f"{dataset}.csv"
    manifest_dir = EXPS_FOLDER / f"{dataset}_{task}" / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        return (yaml_path.stem, False, f"missing metadata csv: {metadata_path}")

    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name)

    available = {
        "wav_folder": str(wav_folder),
        "metadata_path": str(metadata_path),
        "manifest_train_path": str(manifest_dir / "train.json"),
        "manifest_val_path": str(manifest_dir / "valid.json"),
        "manifest_test_path": str(manifest_dir / "test.json"),
        "ratio": RATIO,
        "random_seed": RANDOM_SEED,
        "dataset": dataset,
        "task": task,
        # k-fold extras (pulled from task yaml)
        "raw_label_key": cfg.get("raw_label_key"),
        "num_fold": cfg.get("num_fold"),
    }
    sig = inspect.signature(fn)
    kwargs = {k: available[k] for k in sig.parameters if k in available}
    missing = [k for k in sig.parameters if k not in available]
    if missing:
        return (yaml_path.stem, False, f"unsupported params: {missing}")

    try:
        fn(**kwargs)
        return (yaml_path.stem, True, "ok")
    except Exception as e:
        return (yaml_path.stem, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def main():
    yamls = sorted(TASKS_DIR.glob("*.yaml"))
    results = []
    for y in yamls:
        print(f"\n########## {y.stem} ##########")
        results.append(run_one(y))

    print("\n================ SUMMARY ================")
    for name, ok, msg in results:
        status = "OK  " if ok else "FAIL"
        short = msg.splitlines()[0] if msg else ""
        print(f"[{status}] {name}: {short}")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results)-n_fail}/{len(results)} succeeded.")


if __name__ == "__main__":
    main()
