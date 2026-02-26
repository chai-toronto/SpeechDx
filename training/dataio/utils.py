import json
from pathlib import Path
import random
from urllib.request import urlretrieve


def ensure_dir(path: Path):
    if path.is_file:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)

class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def locate_bad(obj, trail="$"):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return None
    if isinstance(obj, Path):
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad = locate_bad(v, f"{trail}.{k}")
            if bad: return bad
        return None
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad = locate_bad(v, f"{trail}[{i}]")
            if bad: return bad
        return None
    try:
        json.dumps(obj)
        return None
    except TypeError:
        return f"{trail} :: {type(obj).__name__}"


def proc_length_vec(*vectors, duration, min_length=100, max_length=1000):
    """ Assume 1D vector"""
    raw_embs = vectors
    rel_min_length = duration / min_length
    if rel_min_length < 1.0:
        # pad
        output_embs = []
        for raw_emb in raw_embs:
            n_repeats = int(1.0 / rel_min_length) + 1
            padded_emb = raw_emb.repeat(n_repeats, 0)[:min_length]
            output_embs.append(padded_emb)
        return *output_embs, min_length

    rel_max_length = duration / max_length
    if rel_max_length > 1.0:
        # randomly crop
        output_embs = []
        for raw_emb in raw_embs:
            T, D = raw_emb.shape
            new_length = int(T / rel_max_length)
            start = random.randint(0, T - new_length)
            cropped_emb = raw_emb[start:start + new_length]
            output_embs.append(cropped_emb)
        return *output_embs, new_length

    return *raw_embs, duration
