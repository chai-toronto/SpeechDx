import json
from pathlib import Path
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


