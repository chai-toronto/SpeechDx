"""Build a substitute ``encoder_params`` block for reader-only jobs.

Reader-only jobs (cache already warm) skip the heavy encoder entirely by
inlining a stub block in place of the ``!include:`` of the real encoder
yaml. Without this, parsing the real encoder yaml would instantiate the
multi-GB encoder via ``!new:`` at YAML load time even though
``forward()`` is never called. The substitute carries only the metadata
fields the rest of ``main.yaml`` references (``sample_rate``,
``feature_dim``, ...) plus a stub encoder satisfying the
``.output_hidden_states`` attribute reads.

Salvaged from ``bench/encoder_params.py``.
"""

from __future__ import annotations

import re
from pathlib import Path


def build_stub_encoder_params(encoder_yaml_path: Path) -> str:
    text = encoder_yaml_path.read_text()
    m = re.search(r"output_hidden_states:\s*(true|false|True|False)", text)
    ohs = (m.group(1).lower() if m else "false")
    fields: dict[str, str] = {}
    for key in ("sample_rate", "feature_dim", "num_layers", "layer_dim",
                "max_length", "min_length"):
        m = re.search(rf"^{key}:\s*(.+?)\s*(?:#.*)?$", text, re.MULTILINE)
        if m:
            fields[key] = m.group(1).strip()
    lines = ["encoder_params:"]
    for k, v in fields.items():
        lines.append(f"  {k}: {v}")
    lines.append("  encoder: !new:model.stub_encoder.StubEncoder")
    lines.append(f"    output_hidden_states: {ohs}")
    return "\n".join(lines)
