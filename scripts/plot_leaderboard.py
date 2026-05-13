"""Render leaderboard.csv as a printable PDF with one bar plot per task.

Layout per task row:
  - Left 2/3: horizontal bar plot of the top-k models (scores ×1000),
    where k is chosen so adjacent bars have a perceptible difference
    (the largest natural gap among the first max_k sorted scores).
  - Right 1/3: color-coded list of the remaining models, sorted desc.

Run:
  python scripts/plot_leaderboard.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "leaderboard.csv"
BY_CAT_CSV = ROOT / "leaderboard_by_category.csv"
PDF_OUT = ROOT / "leaderboard_charts.pdf"

META_COLS = {"task", "category", "task_type", "avg", "std"}

# Human-readable task names.
TASK_LABELS = {
    "emo_neg": "Emotion: negative vs. other (binary)",
    "edaic_depC": "Depression detection — E-DAIC (binary)",
    "emo_classify": "Emotion classification (multi-class)",
    "edaic_phqR": "PHQ-8 score regression — E-DAIC",
    "aphasia_pwaC": "Aphasia detection (binary)",
    "dbank_adC": "Alzheimer's detection — DementiaBank (binary)",
    "dbank_mmseR": "MMSE score regression — DementiaBank",
    "dysarthria_det": "Dysarthria detection (binary)",
    "ksof_intC": "Stuttering intensity — KSoF (binary)",
    "mvdr_parkC": "Parkinson's detection — MDVR-KCL (binary)",
    "ksof_stutL": "Stuttering type — KSoF (multilabel)",
    "mvdr_hyR": "Hoehn–Yahr stage regression — MDVR-KCL",
    "torgo_sevR": "Dysarthria severity regression — TORGO",
    "mvdr_updrs18R": "UPDRS-III speech score — MDVR-KCL",
    "mvdr_updrs5R": "UPDRS item-5 regression — MDVR-KCL",
    "avfad_pathC": "Voice pathology — AVFAD (binary)",
    "covid_det": "COVID-19 detection (binary)",
    "symptomatic": "Symptomatic vs. healthy (binary)",
    "symptom_multi": "Respiratory symptoms (multilabel)",
}

# Pretty model names for the legend / list.
MODEL_LABELS = {
    "ast": "AST",
    "audiomae": "AudioMAE",
    "clap": "CLAP",
    "emotion2vec": "Emotion2vec",
    "gemini": "Gemini",
    "hubert": "HuBERT",
    "mms": "MMS",
    "opera_gt": "OPERA-GT",
    "qwen3omni": "Qwen3-Omni",
    "qwen3voice": "Qwen3-Voice",
    "w2v2": "wav2vec 2.0",
    "wavjepa": "Wav-JEPA",
    "wavlm": "WavLM",
    "whisper": "Whisper",
}


def pick_top_k(scores_desc: list[float], min_k: int = 4, max_k: int = 8) -> int:
    """Choose k so adjacent bars in the top-k have visible separation.

    Strategy: among candidate cut points k in [min_k, max_k], pick the k whose
    "outgoing" gap (score[k-1] - score[k]) is largest -- i.e. split at the
    biggest natural break.  This drops outliers that would compress the
    surviving bars on the y-axis (the ".99, .98, .5" pathology).
    """
    n = len(scores_desc)
    if n <= min_k:
        return n
    lo = min(min_k, n)
    hi = min(max_k, n - 1)  # need at least one item below the cut
    if hi < lo:
        return n
    best_k = lo
    best_gap = -1.0
    for k in range(lo, hi + 1):
        gap = scores_desc[k - 1] - scores_desc[k]
        if gap > best_gap:
            best_gap = gap
            best_k = k
    return best_k


def make_palette(model_cols: list[str]) -> dict[str, tuple]:
    """Stable color per model across all subplots."""
    cmap = plt.get_cmap("tab20")
    return {m: cmap(i % 20) for i, m in enumerate(sorted(model_cols))}


def draw_task_row(fig, gs_row, row: pd.Series, model_cols: list[str], palette: dict):
    # Drop missing scores (e.g. Gemini on most tasks).
    scores = {m: row[m] for m in model_cols if pd.notna(row[m])}
    sorted_models = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
    sorted_vals = [scores[m] * 1000.0 for m in sorted_models]

    k = pick_top_k(sorted_vals)
    top_models = sorted_models[:k]
    top_vals = sorted_vals[:k]
    rest_models = sorted_models[k:]
    rest_vals = sorted_vals[k:]

    # --- Bar plot (left 2/3) ---
    ax_bar = fig.add_subplot(gs_row[0])
    y_pos = np.arange(len(top_models))[::-1]  # highest at top
    bar_colors = [palette[m] for m in top_models]
    ax_bar.barh(y_pos, top_vals, color=bar_colors, edgecolor="black", linewidth=0.4)

    # Zoom y-axis so adjacent bars are visually distinct.
    vmin = min(top_vals)
    vmax = max(top_vals)
    span = max(vmax - vmin, 1.0)
    pad = 0.25 * span
    left = max(0.0, vmin - pad)
    right = min(1000.0, vmax + pad)
    if right - left < 20:
        right = min(1000.0, left + 20)
    ax_bar.set_xlim(left, right)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([MODEL_LABELS.get(m, m) for m in top_models], fontsize=8)
    ax_bar.tick_params(axis="x", labelsize=7)
    ax_bar.set_xlabel("score × 1000", fontsize=8)

    # Value labels at bar ends.
    for y, v in zip(y_pos, top_vals):
        ax_bar.text(
            v + 0.02 * (right - left),
            y,
            f"{v:.1f}",
            va="center",
            fontsize=7,
        )

    task = row["task"]
    title = TASK_LABELS.get(task, task)
    ax_bar.set_title(
        f"{title}  ·  {row['category']} / {row['task_type']}",
        fontsize=9,
        loc="left",
        pad=4,
    )
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.grid(axis="x", linestyle=":", alpha=0.4)

    # --- Color-coded list (right 1/3) ---
    ax_list = fig.add_subplot(gs_row[1])
    ax_list.axis("off")
    ax_list.set_xlim(0, 1)
    ax_list.set_ylim(0, 1)
    ax_list.set_title("Also-ran (sorted)", fontsize=8, loc="left", pad=4)

    if rest_models:
        n = len(rest_models)
        # Stack list items top-to-bottom inside the panel.
        row_height = min(0.11, 0.92 / n)
        top_margin = 0.92
        for i, (m, v) in enumerate(zip(rest_models, rest_vals)):
            y = top_margin - (i + 0.5) * row_height
            # color swatch
            ax_list.add_patch(
                plt.Rectangle(
                    (0.02, y - row_height * 0.35),
                    0.08,
                    row_height * 0.7,
                    facecolor=palette[m],
                    edgecolor="black",
                    linewidth=0.4,
                    transform=ax_list.transAxes,
                )
            )
            ax_list.text(
                0.14,
                y,
                f"{MODEL_LABELS.get(m, m)}",
                fontsize=7,
                va="center",
                transform=ax_list.transAxes,
            )
            ax_list.text(
                0.98,
                y,
                f"{v:.1f}",
                fontsize=7,
                va="center",
                ha="right",
                transform=ax_list.transAxes,
            )
    else:
        ax_list.text(
            0.5, 0.5, "(none)",
            fontsize=8, color="gray", ha="center", va="center",
            transform=ax_list.transAxes,
        )


AGGREGATE_PANELS = [
    ("Overall (mean of all tasks)", "overall_score"),
    ("Affective",   "Affective_score"),
    ("Cognitive",   "Cognitive_score"),
    ("Motor",       "Motor_score"),
    ("Respiratory", "Respiratory_score"),
]


def _draw_ranking_axes(ax, encoders: list[str], scores: list[float],
                       palette: dict, title: str):
    y_pos = np.arange(len(encoders))[::-1]
    vals = [s * 1000.0 for s in scores]
    bar_colors = [palette.get(m, (0.5, 0.5, 0.5, 1.0)) for m in encoders]
    ax.barh(y_pos, vals, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in encoders], fontsize=7)
    for tick, m in zip(ax.get_yticklabels(), encoders):
        tick.set_color(palette.get(m, (0, 0, 0, 1)))
        tick.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("score × 1000", fontsize=7)
    ax.set_title(title, fontsize=9, loc="left", pad=3)
    if vals:
        xmax = max(vals)
        ax.set_xlim(0, xmax * 1.12 if xmax > 0 else 1.0)
        for y, v in zip(y_pos, vals):
            ax.text(v + 0.01 * xmax, y, f"{v:.0f}",
                    va="center", fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def draw_aggregate_page(pdf, palette: dict):
    df = pd.read_csv(BY_CAT_CSV)
    df = df.rename(columns={df.columns[0]: "encoder"})
    fig = plt.figure(figsize=(8.5, 11))
    outer = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.10, right=0.97, top=0.92, bottom=0.05,
        height_ratios=[1.2, 1, 1],
    )
    slots = [outer[0, :], outer[1, 0], outer[1, 1], outer[2, 0], outer[2, 1]]
    for (title, col), slot in zip(AGGREGATE_PANELS, slots):
        sub = df[["encoder", col]].dropna().copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna().sort_values(col, ascending=False)
        ax = fig.add_subplot(slot)
        _draw_ranking_axes(
            ax, sub["encoder"].tolist(), sub[col].tolist(), palette, title,
        )
    fig.suptitle(
        "Audio Health Benchmark — overall and per-category rankings",
        fontsize=11, y=0.965,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _load_task_layout() -> pd.DataFrame:
    """Load leaderboard.csv (encoder-rows, task-cols) and transpose into the
    task-rows layout this script was written against. The meta rows
    (category, task_type, avg, std) come through naturally as the first
    columns post-transpose.
    """
    raw = pd.read_csv(CSV).set_index("encoder")
    df = raw.T
    df.index.name = "task"
    # Numeric coercion: model-score columns came in as object dtype because
    # the source frame mixed strings (category/task_type) and floats.
    for col in df.columns:
        if col not in ("category", "task_type"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index()


def main():
    df = _load_task_layout()
    model_cols = [c for c in df.columns if c not in META_COLS]
    palette = make_palette(model_cols)

    tasks_per_page = 4
    n_pages = (len(df) + tasks_per_page - 1) // tasks_per_page

    with PdfPages(PDF_OUT) as pdf:
        draw_aggregate_page(pdf, palette)
        for page in range(n_pages):
            fig = plt.figure(figsize=(8.5, 11))
            outer = gridspec.GridSpec(
                tasks_per_page, 1, figure=fig,
                hspace=0.95, left=0.10, right=0.97, top=0.95, bottom=0.05,
            )
            for i in range(tasks_per_page):
                idx = page * tasks_per_page + i
                if idx >= len(df):
                    break
                inner = gridspec.GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=outer[i], width_ratios=[2, 1], wspace=0.20,
                )
                draw_task_row(fig, inner, df.iloc[idx], model_cols, palette)
            fig.suptitle(
                "Audio Health Benchmark — per-task leaderboard",
                fontsize=11, y=0.985,
            )
            pdf.savefig(fig)
            plt.close(fig)

        meta = pdf.infodict()
        meta["Title"] = "Audio Health Benchmark leaderboard"
        meta["Author"] = "AHB"
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
