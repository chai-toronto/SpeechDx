"""Extract and visualize learned layer weights from LTProbe experiments."""

import glob
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPS = ROOT / "exps"
OUT_DIR = EXPS / "layer_weight_plots"

# Encoder display names and expected layer counts
ENCODER_INFO = {
    "wavlm-basep-all": 12,
    "wavjepa": 12,
    "qwen3voice": 8,
}

TASK_ORDER = ["c9s_t1", "ravdess", "torgo"]

ENCODER_ORDER = ["qwen3voice"]


def discover_checkpoints():
    """Find all LTProbe checkpoint files, excluding CLTProbe and chunker variants."""
    pattern = str(EXPS / "*" / "*CLTP-2*" / "brain-logs" / "save" / "CKPT*" / "model.ckpt")
    paths = glob.glob(pattern)

    results = {}
    for p in paths:
        parts = Path(p).relative_to(EXPS).parts
        task = parts[0]
        exp_name = parts[1]

        # Skip CLTProbe and chunker experiments
        if "CLTProbe" in exp_name or "chunker" in exp_name:
            continue

        # Extract encoder name: strip the -LTProbe suffix (and any variant suffix)
        # e.g. "wavlm-basep-all-LTProbe" -> "wavlm-basep-all"
        # e.g. "wavlm-basep-all-LTProbe-nonorm" -> "wavlm-basep-all"
        encoder = exp_name.split("-LTProbe")[0]

        # Build a variant label if there's a suffix after -LTProbe
        suffix = exp_name.split("-LTProbe")[1] if "-LTProbe" in exp_name else ""
        variant = suffix.lstrip("-") if suffix else None

        key = (task, encoder, variant)
        # If multiple CKPTs exist, pick the latest (lexicographic sort on timestamp)
        if key not in results or p > results[key]:
            results[key] = p

    return results


def load_layer_weights(ckpt_path):
    """Load logits from checkpoint and return (logits, softmax_weights)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    logits = state["probe.layer_pooler.logits"]
    weights = F.softmax(logits, dim=0)
    return logits, weights


def main():
    ckpts = discover_checkpoints()
    if not ckpts:
        print("No LTProbe checkpoints found!")
        return

    # Load all weights
    data = {}
    for (task, encoder, variant), path in sorted(ckpts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or "")):
        logits, weights = load_layer_weights(path)
        label = f"{encoder}-LTProbe" + (f"-{variant}" if variant else "")
        data[(task, encoder, variant)] = {
            "logits": logits,
            "weights": weights,
            "num_layers": logits.shape[0],
            "label": label,
        }

    # Print summary table
    print(f"\n{'Task':<12} {'Encoder':<25} {'Variant':<10} {'Layers':>6}  Weights (softmax)")
    print("-" * 100)
    for (task, encoder, variant), info in sorted(data.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or "")):
        w = info["weights"].numpy()
        w_str = " ".join(f"{v:.3f}" for v in w)
        var_str = variant or ""
        print(f"{task:<12} {encoder:<25} {var_str:<10} {info['num_layers']:>6}  [{w_str}]")

    # Save to disk
    save_data = {}
    for key, info in data.items():
        save_data[key] = {
            "logits": info["logits"],
            "weights": info["weights"],
            "num_layers": info["num_layers"],
        }
    torch.save(save_data, EXPS / "layer_weights.pt")
    print(f"\nSaved layer weights to {EXPS / 'layer_weights.pt'}")
    print(f"Total experiments: {len(data)}")

    # --- Visualization: heatmap-bar style ---
    OUT_DIR.mkdir(exist_ok=True)

    # Group data by task -> list of (encoder_label, weights_array)
    by_task = defaultdict(list)
    for (task, encoder, variant), info in sorted(data.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or "")):
        # Skip nonorm variant for the main grid plot
        if variant:
            continue
        by_task[task].append((encoder, info["weights"].numpy()))

    # Display names
    TASK_DISPLAY = {
        "c9s_t1": "CS-Res",
        "mvdr": "MVDR",
        "ravdess": "RAVDESS",
        "torgo": "TORGO",
        "uaspeech": "UASpeech",
    }
    ENCODER_DISPLAY = {
        "wavlm-basep-all": "WavLM",
        "wavjepa": "WavJEPA",
        "qwen3voice": "Qwen3-Voice",
    }

    # Colormap — high-contrast diverging: cool (low) to warm (high)
    cmap = plt.cm.RdYlBu_r

    # Exclude MVDR — uniform weights, nothing useful
    SKIP_TASKS = {"mvdr"}

    # Normalize weights: multiply by num_layers so uniform = 1.0 across all models
    all_norm_weights = []
    for task_key, entries in by_task.items():
        if task_key in SKIP_TASKS:
            continue
        for _, w in entries:
            norm_w = w * len(w)
            all_norm_weights.extend(norm_w.tolist())
    vmin, vmax = min(all_norm_weights), max(all_norm_weights)

    # Layout: 2 columns, ceil(n_tasks/2) rows
    tasks = [t for t in TASK_ORDER if t in by_task and t not in SKIP_TASKS]
    n_tasks = len(tasks)
    ncols = 2
    nrows = (n_tasks + 1) // 2

    # Colors
    bg_color = "white"
    text_color = "black"

    # Each task panel has n_encoders rows; figure out spacing
    n_enc = len(ENCODER_ORDER)
    bar_h = 0.35       # height of each heatmap bar
    row_gap = 0.25     # gap between encoder rows (increased for label breathing room)
    panel_gap = 0.8    # gap between dataset panels (vertical, increased)
    panel_w = 5.0      # width of each panel's bar area
    col_gap = 1.2      # horizontal gap between columns

    fig_w = ncols * panel_w + (ncols - 1) * col_gap + 2.0
    panel_h = n_enc * (bar_h + row_gap) + panel_gap
    fig_h = nrows * panel_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg_color)

    for idx, task in enumerate(tasks):
        col = idx % ncols
        row = idx // ncols
        entries = by_task[task]

        # Build a dict for quick lookup
        enc_weights = {enc: w for enc, w in entries}

        # Panel origin (in figure coords we'll use axes manually)
        x0 = (col * (panel_w + col_gap) + 1.0) / fig_w
        y_top = 1.0 - (row * panel_h + 0.6) / fig_h

        # Dataset title
        title_ax = fig.add_axes([x0, y_top, panel_w / fig_w, 0.03], facecolor=bg_color)
        title_ax.set_xlim(0, 1)
        title_ax.set_ylim(0, 1)
        title_ax.text(0.5, 0.5, TASK_DISPLAY.get(task, task),
                      ha="center", va="center", fontsize=16, fontweight="bold", color=text_color)
        title_ax.axis("off")

        for enc_idx, encoder in enumerate(ENCODER_ORDER):
            if encoder not in enc_weights:
                continue
            weights = enc_weights[encoder]
            num_layers = len(weights)

            # Position for this encoder's bar (shifted down with more room)
            y_bar = y_top - (enc_idx + 1) * (bar_h + row_gap) / fig_h
            ax = fig.add_axes([x0, y_bar, panel_w / fig_w, bar_h / fig_h], facecolor=bg_color)

            # Normalize: weight * num_layers so uniform = 1.0
            norm_weights = weights * num_layers

            # Draw each layer as a colored segment
            seg_w = 1.0 / num_layers
            for li in range(num_layers):
                color = cmap((norm_weights[li] - vmin) / (vmax - vmin + 1e-12))
                ax.add_patch(plt.Rectangle((li * seg_w, 0), seg_w, 1,
                                           facecolor=color, edgecolor="white", linewidth=0.5))

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([(i + 0.5) * seg_w for i in range(num_layers)])
            ax.set_xticklabels([str(i + 1) for i in range(num_layers)],
                               fontsize=7, color=text_color)
            ax.tick_params(axis="x", length=0, pad=2)
            ax.set_yticks([])

            # Encoder label on the left
            ax.text(-0.02, 0.5, ENCODER_DISPLAY.get(encoder, encoder),
                    ha="right", va="center", fontsize=10, fontweight="bold",
                    color=text_color, transform=ax.transAxes)

        # Colorbar below the last encoder row
        cb_y = y_top - (n_enc + 0.7) * (bar_h + row_gap) / fig_h
        cb_ax = fig.add_axes([x0 + 0.15 * panel_w / fig_w, cb_y,
                              0.7 * panel_w / fig_w, 0.012], facecolor=bg_color)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cb_ax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7, colors=text_color, length=2)
        cb.outline.set_edgecolor(text_color)
        cb.outline.set_linewidth(0.5)

    fig.savefig(OUT_DIR / "layer_weights_heatmap.png", dpi=200,
                facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'layer_weights_heatmap.png'}")
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
