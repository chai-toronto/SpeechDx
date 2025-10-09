import json
from pathlib import Path

def ensure_dir(path: Path):
    if path.is_file:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)

class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)