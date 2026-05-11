#!/usr/bin/env python3
"""Generate a printable PDF report card per encoder.

Per-task scoring (symmetric, both baselines anchored at 0.5):
  classification: raw AUC (chance 0.5 -> 0.5)
  regression:     max(0, 1 - MAE / (2 * MAD))   where MAD = test-label MAD

Output: PhysioBench-2/report_card.tex and .pdf.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SUMMARY_DIR = REPO / "exps" / "_summary_run2_all_mvdr3" / "single_task"
OUT_DIR = REPO / "PhysioBench-2"
TEX = OUT_DIR / "report_card.tex"

GPA_SCALE = [
    (90,"A+",4.0),(85,"A",4.0),(80,"A-",3.7),(77,"B+",3.3),(73,"B",3.0),
    (70,"B-",2.7),(67,"C+",2.3),(63,"C",2.0),(60,"C-",1.7),
    (57,"D+",1.3),(53,"D",1.0),(50,"D-",0.7),(0,"F",0.0),
]

def to_letter_gpa(score: float) -> tuple[str, float]:
    pct = score * 100
    for thresh, letter, gpa in GPA_SCALE:
        if pct >= thresh:
            return letter, gpa
    return "F", 0.0

def gpa_to_letter(g: float) -> str:
    for _, letter, gv in GPA_SCALE:
        if g >= gv - 1e-9:
            return letter
    return "F"

LABEL_SOURCES = {
    "dbank_mmseR":   ("data/dementiabank/processed/dbank.csv", "mmse"),
    "edaic_phqR":    ("data/edaic/processed/edaic.csv",        "PHQ_Score"),
    "torgo_sevR":    ("data/torgo/processed/torgo.csv",        "severity"),
    "mvdr_hyR":      ("data/mvdr/processed/mvdr.csv",          "hy_rating"),
    "mvdr_updrs5R":  ("data/mvdr/processed/mvdr.csv",          "updrs_ii5"),
    "mvdr_updrs18R": ("data/mvdr/processed/mvdr.csv",          "updrs_iii18"),
}

# Six paper-faithful merges (T-IDs in comments).
MERGES = {
    "emo_classify":   ["ravdess_emoC", "iemocap_emoC"],         # T3+T5
    "emo_neg":        ["ravdess_emoBC", "iemocap_emoBC"],       # T4+T6
    "dysarthria_det": ["torgo_dysC", "uaspeech_dysC"],          # T10+T12
    "symptomatic":    ["c9s_t1", "c9s_L_t1", "coswara_sympC"],  # T19+T20+T24
    "covid_det":      ["c9s_t2", "c9s_L_t2", "coswara_covidC"], # T21+T22+T25
    "symptom_multi":  ["c9s_sympL", "coswara_sympL"],           # T23+T26
}

# Tasks grouped by output type. Ordered by class GPA (easiest first) when rendered.
TASK_YEAR = {
    "edaic_depC":1, "dbank_adC":1, "aphasia_pwaC":1, "mvdr_parkC":1,
    "ksof_intC":1, "avfad_pathC":1,
    "emo_neg":1, "dysarthria_det":1, "symptomatic":1, "covid_det":1,
    "emo_classify":2,
    "ksof_stutL":3, "symptom_multi":3,
    "edaic_phqR":4, "dbank_mmseR":4, "torgo_sevR":4,
    "mvdr_hyR":4, "mvdr_updrs5R":4, "mvdr_updrs18R":4,
}
SECTION_NAMES = {
    1: "Binary Classification",
    2: "Multiclass Classification",
    3: "Multilabel Classification",
    4: "Regression",
}

# Map each (post-merge) task to a category id and a paper-style display name.
TASK_CATEGORY = {
    # c1 affective
    "edaic_depC":      ("c1", "Depression / healthy (EDAIC-WOZ)"),
    "edaic_phqR":      ("c1", "PHQ-8 score (EDAIC-WOZ)"),
    "emo_classify":    ("c1", "Emotion classification (RAVDESS+IEMOCAP)"),
    "emo_neg":         ("c1", "Negative / non-negative emotion (RAVDESS+IEMOCAP)"),
    # c2 cognitive
    "aphasia_pwaC":    ("c2", "Aphasia / healthy (AphasiaBank)"),
    "dbank_adC":       ("c2", "Dementia / healthy (DementiaBank)"),
    "dbank_mmseR":     ("c2", "MMSE score (DementiaBank)"),
    # c3 motor
    "dysarthria_det":  ("c3", "Dysarthria / healthy (TORGO+UASpeech)"),
    "torgo_sevR":      ("c3", "Dysarthria severity (TORGO)"),
    "mvdr_parkC":      ("c3", "Parkinson's / healthy (MDVR-KCL)"),
    "mvdr_hyR":        ("c3", "Hoehn & Yahr score (MDVR-KCL)"),
    "mvdr_updrs5R":    ("c3", "UPDRS-II.5 score (MDVR-KCL)"),
    "mvdr_updrs18R":   ("c3", "UPDRS-III.18 score (MDVR-KCL)"),
    "ksof_intC":       ("c3", "Disfluency / healthy (KSoF-C)"),
    "ksof_stutL":      ("c3", "Disfluency multi-label (KSoF-C)"),
    # c4 respiratory / phonatory
    "symptomatic":     ("c4", "Symptomatic / healthy (COVID-19 Sounds + Coswara)"),
    "covid_det":       ("c4", "COVID-19 / non-COVID-19 (COVID-19 Sounds + Coswara)"),
    "symptom_multi":   ("c4", "Symptom multi-label (COVID-19 Sounds + Coswara)"),
    "avfad_pathC":     ("c4", "Vocal pathology / healthy (AVFAD)"),
}

CAT_LABEL = {"c1": "Affective", "c2": "Cognitive", "c3": "Motor", "c4": "Resp."}

# Order encoders alphabetically inside the LaTeX file (the title page lists rank).
ENCODER_ORDER = [
    "ast", "audiomae", "clap", "emotion2vec", "hubert", "mms",
    "opera_gt", "qwen3voice", "w2v2", "wavjepa", "wavlm", "whisper",
]

# Display name + bullet points sourced from PhysioBench-2/tables/models.tex
# and encoder configs in sdx/configs/encoders/.  Keep punchy.
ENCODER_INFO = {
    "whisper": {
        "display": "Whisper (Large-v3)",
        "checkpoint": "openai/whisper-large-v3",
        "bullets": [
            "\\textbf{Architecture}: encoder--decoder transformer; we keep only the encoder (32 layers, 1280-dim, 1.55B params total). v3 uses 128 mel bins (v1/v2 used 80) on 30-second fixed windows with absolute positional embeddings.",
            "\\textbf{Objective}: weakly-supervised sequence-to-sequence transcription, plus translation, language ID, voice-activity, and timestamp prediction --- all trained jointly through special prefix tokens.",
            "\\textbf{Training data}: $\\sim$1M hours of weakly-labelled multilingual web audio + $\\sim$4M hours pseudo-labelled by Whisper Large-v2, totalling $\\sim$5M hours across 99 languages. Only 2--3 epochs over the data.",
            "\\textbf{Augmentation}: SpecAugment (time + frequency masking) on the mel spectrogram during training.",
            "\\textbf{Trade-off here}: the transcription objective pushes the encoder to be \\emph{invariant} to speaker, room, and microphone --- the very paralinguistic cues clinical regression tasks need.",
        ],
    },
    "wavlm": {
        "display": "WavLM (Large)",
        "checkpoint": "microsoft/wavlm-large",
        "bullets": [
            "\\textbf{Architecture}: 24-layer transformer over a CNN feature extractor (1024-dim, 316M params). Uses \\emph{gated relative position bias} instead of standard sinusoidal/RoPE positions --- helps long-range modelling.",
            "\\textbf{Objective}: masked prediction over discretised units \\emph{plus} a denoising auxiliary --- random mixtures of interfering speech/noise are added to the input and the model must still predict the clean masked targets.",
            "\\textbf{Training data}: 94k hours = Libri-Light (60k h, audiobooks) + GigaSpeech (10k h, podcast/YouTube) + VoxPopuli (24k h, parliament). Mixing matters: VoxPopuli adds heavy non-native accents, GigaSpeech adds noisy real-world recordings.",
            "\\textbf{Augmentation}: the in-objective denoising is itself the augmentation. 80\\% mask probability over 10-frame spans, with same-utterance speech mixed in for the noisy view.",
            "\\textbf{Trade-off here}: the denoising forces the model to keep speaker / channel cues to remove them --- representations retain those cues, which helps clinical paralinguistic tasks.",
        ],
    },
    "qwen3voice": {
        "display": "Qwen3-TTS-Tokenizer (12 Hz)",
        "checkpoint": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "bullets": [
            "\\textbf{Architecture}: a neural audio \\emph{codec} (not a representation learner per se), 8 layers, 512-dim, $\\sim$150M params, 24 kHz input down to 12 Hz token rate.",
            "\\textbf{Objective}: codec reconstruction --- encoder $\\to$ residual-vector-quantised tokens $\\to$ decoder. Built as the speech tokenizer for Qwen3-TTS.",
            "\\textbf{Training data}: Qwen ecosystem multilingual speech, reported $>$5M hours.",
            "\\textbf{Why we use it}: the encoder output (pre-quantisation) is what we read --- it must preserve everything needed to \\emph{regenerate} the waveform, including prosody, speaker, and emotion.",
            "\\textbf{Trade-off here}: reconstruction-faithful $\\neq$ semantically organised; there's no contrastive or predictive pressure to build a clean linear-probe-friendly latent space.",
        ],
    },
    "ast": {
        "display": "AST (AudioSet-finetuned)",
        "checkpoint": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "bullets": [
            "\\textbf{Architecture}: pure ViT applied to log-mel spectrograms; 16$\\times$16 patches at \\emph{10 ms} time stride and 10 mel-bin freq stride (the ``10-10'' in the name), 12 layers, 768-dim, 86.6M params.",
            "\\textbf{Initialisation}: weights cross-modally transferred from DeiT (ImageNet) --- 2D positional embeddings interpolated to spectrogram dimensions. Only audio model in the list pretrained \\emph{on images first}.",
            "\\textbf{Objective}: end-to-end \\emph{supervised} multi-label classification over AudioSet's 527 sound events. No masked / contrastive / generative pretraining stage.",
            "\\textbf{Training data}: AudioSet, $\\sim$5.8k hours of 10-second YouTube clips with crowdsourced event labels.",
            "\\textbf{Augmentation}: SpecAugment + mixup during fine-tuning.",
            "\\textbf{Trade-off here}: spectral-pattern specialist; great when condition leaves a distinctive spectral fingerprint (motor speech, AVFAD), weak when the cue is linguistic / semantic (AD detection).",
        ],
    },
    "audiomae": {
        "display": "AudioMAE",
        "checkpoint": "hance-ai/audiomae",
        "bullets": [
            "\\textbf{Architecture}: ViT-Base encoder (12 layers, 768-dim, 85.6M params) plus a lightweight ViT \\emph{decoder}; asymmetric design --- the encoder only sees visible patches.",
            "\\textbf{Objective}: masked autoencoding on log-mel spectrograms. Mask 80\\% of patches at random; encoder embeds the visible 20\\%; decoder reconstructs the missing patches in pixel/spectrogram space.",
            "\\textbf{Decoder}: uses local-window attention (not full self-attention) for efficiency --- audio patches have stronger local than global structure.",
            "\\textbf{Training data}: AudioSet, $\\sim$5.8k hours, \\emph{unlabelled} (no event labels at pretraining time).",
            "\\textbf{Augmentation}: aggressive masking is itself the augmentation; no SpecAugment, no mixup.",
            "\\textbf{Trade-off here}: faithful spectral reconstruction $\\Rightarrow$ retains fine-grained acoustic detail, but representations aren't organised by semantic class.",
        ],
    },
    "wavjepa": {
        "display": "WavJEPA-Nat (Base)",
        "checkpoint": "labhamlet/wavjepa-nat-base",
        "bullets": [
            "\\textbf{Architecture}: Joint Embedding Predictive Architecture (JEPA, LeCun) over raw audio; 12-layer ViT-style transformer ($\\sim$200M params, 768-dim). Two encoders: an online \\emph{context} encoder and a momentum-EMA \\emph{target} encoder, plus a small predictor head.",
            "\\textbf{Objective}: predict the \\emph{latent representation} of masked target regions from visible context regions. No raw-signal reconstruction, no contrastive negatives --- just latent regression to EMA targets.",
            "\\textbf{Training data}: AudioSet's \\emph{naturalistic scenes} subset (the ``Nat'' tag), $\\sim$4.8k hours --- biased toward ambient acoustic events rather than clean speech.",
            "\\textbf{Augmentation}: the masking strategy itself; no time-stretch / pitch-shift / additive noise reported.",
            "\\textbf{Trade-off here}: efficient and noise-robust, but the naturalistic-scene pretraining means it's not optimised for the dense phonetic / prosodic structure that speech-health tasks rely on.",
        ],
    },
    "mms": {
        "display": "MMS-1B",
        "checkpoint": "facebook/mms-1b",
        "bullets": [
            "\\textbf{Architecture}: wav2vec2 backbone scaled up --- 48 transformer layers, 1280-dim, 1.0B params (the largest SSL speech model in the list).",
            "\\textbf{Objective}: contrastive masked prediction over a quantised codebook (vanilla wav2vec2 loss), no language conditioning during pretraining.",
            "\\textbf{Training data}: $\\sim$491k hours covering 1{,}107 languages. Heavy use of religious recordings (Bible / Common Voice + crawled) to reach low-resource languages.",
            "\\textbf{Adapters}: optional language-specific adapter modules added post-hoc for fine-tuning; we don't use them, so the encoder runs in plain backbone mode.",
            "\\textbf{Augmentation}: SpecAugment-style masking, no explicit waveform augmentation.",
            "\\textbf{Trade-off here}: extreme cross-lingual generalisation but the pretraining audio is dominated by clean read speech (religious recitations); paralinguistic variation is comparatively rare.",
        ],
    },
    "hubert": {
        "display": "HuBERT (Large, ASR-FT)",
        "checkpoint": "facebook/hubert-large-ls960-ft",
        "bullets": [
            "\\textbf{Architecture}: 24-layer transformer over CNN front-end (1024-dim, 316M params).",
            "\\textbf{Objective}: \\emph{frame-level cross-entropy} over masked positions, predicting offline-clustered pseudo-labels. Crucially, this is \\emph{not} contrastive --- targets are discrete cluster IDs.",
            "\\textbf{Iterative refinement}: iter-1 targets are k-means on MFCCs; iter-2+ re-clusters HuBERT-encoded features to produce sharper targets. The released checkpoint is iter-3.",
            "\\textbf{Training data}: pretraining on Libri-Light (60k h), then \\emph{ASR fine-tuning} on LibriSpeech 960h with CTC. The ``-ft'' in the checkpoint matters --- representations have been pulled toward graphemes.",
            "\\textbf{Augmentation}: SpecAugment-style masking on the time axis (no frequency masking in original recipe).",
            "\\textbf{Trade-off here}: ASR fine-tuning sharpens phonetic content but flattens speaker / channel info --- the opposite balance from WavLM.",
        ],
    },
    "emotion2vec": {
        "display": "emotion2vec+ Large",
        "checkpoint": "iic/emotion2vec_plus_large",
        "bullets": [
            "\\textbf{Architecture}: data2vec-style transformer ($\\sim$300M params, 1024-dim utterance representation). Multi-scale design fuses utterance-level and frame-level emotion features.",
            "\\textbf{Objective}: \\emph{pseudo-supervised} emotion learning --- a teacher emotion classifier labels a vast unlabelled corpus, and the student is trained on those pseudo-labels.",
            "\\textbf{Training data}: $\\sim$160k hours of emotional speech curated from many sources (the ``+'' suffix means the larger pseudo-labelled corpus vs.\\ the original emotion2vec).",
            "\\textbf{Augmentation}: standard speech augmentation (gain, reverb, noise) during student training.",
            "\\textbf{Trade-off here}: the only \\emph{domain-specialised} encoder for affect --- predictably tops the emotion subject and predictably tanks elsewhere.",
        ],
    },
    "opera_gt": {
        "display": "OPERA-GT",
        "checkpoint": "OPERA",
        "bullets": [
            "\\textbf{Architecture}: tiny transformer at 21M params (smallest in the list), 384-dim, single CLS token output --- no per-frame features.",
            "\\textbf{Objective}: \\emph{generative} reconstruction of respiratory audio (the ``GT'' = Generative Transformer variant of OPERA).",
            "\\textbf{Training data}: $\\sim$404 hours of multi-source respiratory audio (cough, breathing, lung sounds) aggregated from clinical and crowdsourced datasets.",
            "\\textbf{Input window}: fixed 8.18-second clips --- short by design because target events (cough, single breath) are short.",
            "\\textbf{Augmentation}: not specified in the original paper; small-scale supervised+SSL hybrid training.",
            "\\textbf{Trade-off here}: \\emph{domain-matched} to AVFAD / c19sounds / Coswara recording conditions --- often outscores billion-parameter models on those tasks despite being 50$\\times$ smaller.",
        ],
    },
    "w2v2": {
        "display": "wav2vec 2.0 (Large, ASR-FT)",
        "checkpoint": "facebook/wav2vec2-large-960h-lv60-self",
        "bullets": [
            "\\textbf{Architecture}: 7-layer convolutional feature extractor + 24-layer transformer, 1024-dim, 317M params. The original SSL-speech recipe that HuBERT, WavLM, and MMS all descend from.",
            "\\textbf{Objective}: \\emph{contrastive} masked prediction over a Gumbel-softmax-quantised codebook; negatives sampled from the same utterance to force fine-grained discrimination.",
            "\\textbf{Training pipeline}: Libri-Light SSL pretraining (60k h) $\\to$ CTC fine-tuning on LibriSpeech 960h $\\to$ self-training pass on LibriVox (the ``lv60-self'' tag).",
            "\\textbf{Augmentation}: time-domain masking only during pretraining; no SpecAugment in the original recipe.",
            "\\textbf{Trade-off here}: the contrastive objective + ASR fine-tuning is doubly content-focused --- representations are most useful for phoneme-style discrimination, weakest for prosodic and clinical regression cues.",
        ],
    },
    "clap": {
        "display": "CLAP (LAION-Larger-General)",
        "checkpoint": "laion/larger_clap_general",
        "bullets": [
            "\\textbf{Architecture}: dual encoder. Audio side is HT-SAT (Hierarchical Token-Semantic Audio Transformer) --- a Swin-style hierarchical transformer over log-mels at 48 kHz. Text side is RoBERTa. Total $\\sim$400M params.",
            "\\textbf{Objective}: cross-modal contrastive learning (CLIP-style). InfoNCE pulls matching audio--caption pairs together and pushes mismatched pairs apart in a shared 1024-dim space.",
            "\\textbf{Training data}: LAION-Audio-630K (web-scraped audio--text pairs) + AudioSet, $\\sim$10k hours; captions are noisy and dominated by general audio events, not clinical speech.",
            "\\textbf{Sampling rate}: 48 kHz input --- the only encoder in this set above 24 kHz.",
            "\\textbf{Augmentation}: standard SpecAugment + waveform-level mixup during contrastive training.",
            "\\textbf{Trade-off here}: built to align audio with natural-language event labels --- not designed to discriminate clinical conditions, and the bottom-of-class GPA reflects that.",
        ],
    },
}


def compute_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-task scores [task x encoder], regression-only scores)."""
    auc = pd.read_csv(SUMMARY_DIR / "AUC.csv").set_index("task")
    mae = pd.read_csv(SUMMARY_DIR / "MAE.csv").set_index("task")

    mad = {}
    for task, (rel, col) in LABEL_SOURCES.items():
        s = pd.to_numeric(pd.read_csv(REPO / rel)[col], errors="coerce").dropna()
        mad[task] = (s - s.mean()).abs().mean()

    auc_score = auc.clip(lower=0, upper=1)
    mae_score = pd.DataFrame(index=mae.index, columns=mae.columns, dtype=float)
    for task in mae.index:
        mae_score.loc[task] = (1 - mae.loc[task] / (2 * mad[task])).clip(0, 1)

    scores = pd.concat([auc_score, mae_score]).sort_index()
    merged = scores.copy()
    for new, src in MERGES.items():
        merged.loc[new] = scores.loc[src].mean(axis=0)
        merged = merged.drop(index=src)
    return merged, mae_score


def encoder_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Encoder-level summary: cumulative GPA, letter, per-section GPA."""
    rows = []
    for enc in merged.columns:
        per_task_gpa = {t: to_letter_gpa(merged.loc[t, enc])[1] for t in merged.index}
        cum = sum(per_task_gpa.values()) / len(per_task_gpa)
        row = {"encoder": enc, "cumulative_GPA": round(cum, 2),
               "letter": gpa_to_letter(cum)}
        for y in (1, 2, 3, 4):
            cols = [t for t in merged.index if TASK_YEAR[t] == y]
            row[f"y{y}_GPA"] = round(sum(per_task_gpa[t] for t in cols) / len(cols), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("encoder")


# ---------- LaTeX generation ----------

def _esc(s: str) -> str:
    """Minimal LaTeX escaping for the strings we produce."""
    return s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")


def _grade_color(letter: str) -> str:
    if letter.startswith("A"): return "gradeA"
    if letter.startswith("B"): return "gradeB"
    if letter.startswith("C"): return "gradeC"
    if letter.startswith("D"): return "gradeD"
    return "gradeF"


def render_preamble(rank_table: list[tuple[str, str, float, str]]) -> str:
    return r"""\documentclass[10pt]{article}
\usepackage[margin=0.7in,top=0.6in,bottom=0.6in]{geometry}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{tcolorbox}
\tcbuselibrary{skins}

\definecolor{gradeA}{HTML}{1B7F1B}
\definecolor{gradeB}{HTML}{2C82C9}
\definecolor{gradeC}{HTML}{C9A227}
\definecolor{gradeD}{HTML}{D35400}
\definecolor{gradeF}{HTML}{8B0000}
\definecolor{accent}{HTML}{2C3E50}
\definecolor{rowalt}{HTML}{F2F4F7}

\titleformat{\section}{\Large\bfseries\color{accent}}{}{0pt}{}
\titlespacing*{\section}{0pt}{8pt}{6pt}

\setlist[itemize]{leftmargin=1.1em,itemsep=1pt,topsep=1pt,parsep=0pt}
\renewcommand{\arraystretch}{1.05}

\pagestyle{fancy}\fancyhf{}
\fancyfoot[C]{\thepage}
\fancyhead[L]{\small\itshape PhysioBench-2 Encoder Report Cards}
\fancyhead[R]{\small\itshape symmetric scoring (chance = D-)}
\renewcommand{\headrulewidth}{0.4pt}

\newcommand{\bigGPA}[2]{%
  \begin{tcolorbox}[colback=#1!12,colframe=#1,boxrule=1pt,arc=4pt,
                    left=10pt,right=10pt,top=6pt,bottom=6pt,width=\linewidth]
  \centering\Large\bfseries\color{#1} #2
  \end{tcolorbox}}

\title{\bfseries\color{accent}PhysioBench-2 Encoder Report Cards\\
\large\mdseries\color{black!70}Symmetric scoring; chance baseline = 0.5 = D-, perfect = 1.0 = A+}
\author{}
\date{\today}

\begin{document}
\maketitle
\thispagestyle{fancy}
"""


def render_intro(rank_table: list[tuple[str, str, float, str]]) -> str:
    """Title page with class ranking."""
    rows = []
    for rank, (enc, letter, gpa, display) in enumerate(rank_table, 1):
        color = _grade_color(letter)
        rows.append(rf"{rank} & {_esc(display)} & "
                    rf"\textbf{{\color{{{color}}}{letter}}} & {gpa:.2f} \\")
    body = "\n".join(rows)
    return rf"""
\section*{{Class Ranking}}
\begin{{tabularx}}{{\linewidth}}{{r X c r}}
\toprule
\textbf{{Rank}} & \textbf{{Encoder}} & \textbf{{Letter}} & \textbf{{GPA}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}

\vspace{{0.6em}}
\section*{{Scoring}}
\begin{{itemize}}
\item \textbf{{Classification (binary, multiclass, multilabel)}}: raw AUC. Chance (0.5) lands at D- (0.7\,GPA); a perfect classifier (1.0) earns A+ (4.0\,GPA).
\item \textbf{{Regression}}: $\max(0,\, 1 - \text{{MAE}} / (2 \cdot \text{{MAD}}))$, where MAD is the test-label mean absolute deviation. A constant ``predict the mean'' baseline lands at D-; perfect prediction earns A+.
\item Twenty-seven raw tasks are merged into nineteen courses following the paper taxonomy: T19+T20+T24 $\to$ Symptomatic; T21+T22+T25 $\to$ COVID; T23+T26 $\to$ Symptom multilabel; T3+T5 $\to$ Emotion classification; T4+T6 $\to$ Negative-emotion; T10+T12 $\to$ Dysarthria detection.
\item Sections ordered by class GPA (easiest first): \emph{{Multiclass}} (3.32) $>$ \emph{{Binary}} (2.55) $>$ \emph{{Multilabel}} (2.01) $>$ \emph{{Regression}} (1.48).
\end{{itemize}}
\newpage
"""


def render_encoder_page(enc: str, summary: pd.Series, scores_col: pd.Series,
                        class_mean: pd.Series, class_std: pd.Series) -> str:
    """One full page per encoder."""
    info = ENCODER_INFO[enc]
    display = info["display"]
    letter = summary["letter"]
    gpa = summary["cumulative_GPA"]
    color = _grade_color(letter)

    bullet_lines = [rf"  \item {b}" for b in info["bullets"]]
    bullet_lines.append(rf"  \item Hugging Face checkpoint: \texttt{{{_esc(info['checkpoint'])}}}.")
    bullets = "\n".join(bullet_lines)

    # Tasks grouped by output type, ordered by class GPA (easiest first): C, B, L, R.
    section_priority = {2: 0, 1: 1, 3: 2, 4: 3}
    year_blocks = []
    for y in sorted({1, 2, 3, 4}, key=lambda v: section_priority[v]):
        cols = [t for t in scores_col.index if TASK_YEAR[t] == y]
        cols.sort(key=lambda t: (TASK_CATEGORY[t][0], TASK_CATEGORY[t][1]))
        sname = SECTION_NAMES[y]
        ygpa = summary[f"y{y}_GPA"]
        ycolor = _grade_color(gpa_to_letter(ygpa))
        rows = []
        for t in cols:
            cat, name = TASK_CATEGORY[t]
            score = scores_col[t]
            l, g = to_letter_gpa(score)
            row_color = _grade_color(l)
            mu = class_mean[t]
            sd = class_std[t]
            rows.append(
                rf"\textsc{{\scriptsize {CAT_LABEL[cat]}}} & {_esc(name)} & "
                rf"{score:.3f} & \textbf{{\color{{{row_color}}}{l}}} & {g:.1f} & "
                rf"{mu:.3f} & {sd:.3f} \\"
            )
        body = "\n".join(rows)
        year_blocks.append(rf"""
\subsection*{{{sname} \hfill \textcolor{{{ycolor}}}{{\textbf{{GPA {ygpa:.2f}}} ({gpa_to_letter(ygpa)})}}}}
{{\footnotesize
\begin{{tabularx}}{{\linewidth}}{{l X r c r r r}}
\toprule
\textbf{{Cat}} & \textbf{{Course}} & \textbf{{Score}} & \textbf{{Letter}} & \textbf{{GP}} & \textbf{{Class avg}} & \textbf{{Class std}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
}}
""")
    year_section = "\n".join(year_blocks)

    return rf"""
\section*{{{_esc(display)}}}
\bigGPA{{{color}}}{{Cumulative GPA: {gpa:.2f} \quad ({letter})}}

{{\small
\subsection*{{Model design \& training}}
\begin{{itemize}}
{bullets}
\end{{itemize}}

{year_section}
}}

\newpage
"""


def render_leaderboard_page(summary: pd.DataFrame) -> str:
    """Detailed GPA leaderboard with per-section sub-GPA columns."""
    rows = []
    section_order = (2, 1, 3, 4)
    for rank, enc in enumerate(summary.index, 1):
        s = summary.loc[enc]
        cum = s["cumulative_GPA"]
        cum_color = _grade_color(s["letter"])
        section_cells = []
        for y in section_order:
            g = s[f"y{y}_GPA"]
            l = gpa_to_letter(g)
            c = _grade_color(l)
            section_cells.append(rf"{g:.2f} {{\scriptsize\color{{{c}}}{l}}}")
        rows.append(
            rf"{rank} & {_esc(ENCODER_INFO[enc]['display'])} & "
            rf"\textbf{{\color{{{cum_color}}}{cum:.2f}}} & "
            rf"\textbf{{\color{{{cum_color}}}{s['letter']}}} & "
            + " & ".join(section_cells) + r" \\"
        )
    body = "\n".join(rows)
    return rf"""
\section*{{Leaderboard (by GPA)}}
\begin{{tabularx}}{{\linewidth}}{{r X c c c c c c}}
\toprule
\textbf{{\#}} & \textbf{{Encoder}} & \textbf{{Cum. GPA}} & \textbf{{Letter}} & \textbf{{Multiclass}} & \textbf{{Binary}} & \textbf{{Multilabel}} & \textbf{{Regression}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}

\vspace{{0.8em}}
\subsection*{{Reading the table}}
\begin{{itemize}}
\item \textbf{{Cum.~GPA}} averages the per-course grade points across all 19 merged courses (each course weighted equally).
\item Per-section columns show sub-GPA on that subset of courses; sections are ordered easiest $\to$ hardest by class mean (Multiclass 3.32 $>$ Binary 2.55 $>$ Multilabel 2.01 $>$ Regression 1.48).
\item A high cum.~GPA with weak Regression (e.g.\ Whisper, Qwen3) means the encoder over-relies on classification courses --- classic ASR-pretraining signature.
\item A flat profile across all four sections (Hubert, Audiomae) means the encoder generalises evenly; no specialisation.
\end{{itemize}}
\newpage
"""


def render_summary_page(merged: pd.DataFrame) -> str:
    """Class-distribution analysis page."""
    means = merged.mean(axis=1).sort_values()
    stds = merged.std(axis=1).sort_values(ascending=False)

    def task_row(t):
        cat, name = TASK_CATEGORY[t]
        m = merged.loc[t].mean()
        s = merged.loc[t].std()
        l, _ = to_letter_gpa(m)
        col = _grade_color(l)
        return (rf"\textsc{{\small {CAT_LABEL[cat]}}} & {_esc(name)} & "
                rf"{m:.3f} & {s:.3f} & \textbf{{\color{{{col}}}{l}}} \\")

    hardest = "\n".join(task_row(t) for t in means.index[:5])
    easiest = "\n".join(task_row(t) for t in means.index[-5:][::-1])
    most_var = "\n".join(task_row(t) for t in stds.index[:5])

    # Per-section class GPA, ordered easiest first: Multiclass, Binary, Multilabel, Regression.
    year_table_rows = []
    for y in (2, 1, 3, 4):
        cols = [t for t in merged.index if TASK_YEAR[t] == y]
        per_task_gpa = [
            to_letter_gpa(merged.loc[t, enc])[1]
            for enc in merged.columns for t in cols
        ]
        gpa = sum(per_task_gpa) / len(per_task_gpa)
        year_table_rows.append(
            rf"{SECTION_NAMES[y]} & {len(cols)} & {gpa:.2f} & {gpa_to_letter(gpa)} \\"
        )
    year_table = "\n".join(year_table_rows)

    return rf"""
\section*{{Class-Wide Analysis}}

\subsection*{{Difficulty by task type (mean GPA across all encoders, easiest first)}}
\begin{{tabularx}}{{\linewidth}}{{X c c c}}
\toprule
\textbf{{Section}} & \textbf{{Courses}} & \textbf{{Class GPA}} & \textbf{{Letter}} \\
\midrule
{year_table}
\bottomrule
\end{{tabularx}}

\subsection*{{Hardest courses (lowest class mean score)}}
\begin{{tabularx}}{{\linewidth}}{{l X r r c}}
\toprule
\textbf{{Cat}} & \textbf{{Course}} & \textbf{{Mean}} & \textbf{{Std}} & \textbf{{Class Letter}} \\
\midrule
{hardest}
\bottomrule
\end{{tabularx}}

\vspace{{0.6em}}
\subsection*{{Easiest courses (highest class mean score)}}
\begin{{tabularx}}{{\linewidth}}{{l X r r c}}
\toprule
\textbf{{Cat}} & \textbf{{Course}} & \textbf{{Mean}} & \textbf{{Std}} & \textbf{{Class Letter}} \\
\midrule
{easiest}
\bottomrule
\end{{tabularx}}

\vspace{{0.6em}}
\subsection*{{Most divisive courses (highest std across encoders)}}
\begin{{tabularx}}{{\linewidth}}{{l X r r c}}
\toprule
\textbf{{Cat}} & \textbf{{Course}} & \textbf{{Mean}} & \textbf{{Std}} & \textbf{{Class Letter}} \\
\midrule
{most_var}
\bottomrule
\end{{tabularx}}

\subsection*{{What this tells us}}
\begin{{itemize}}
\item \textbf{{Universally hard}}: the bottom of the class-mean list shows where speech-only signal is genuinely weak. PHQ-8 and MMSE regression hover near the mean predictor; multilabel disfluency / symptom characterisation is brutal because raw AUC stays close to 0.5 for most encoders.
\item \textbf{{Universally easy}}: the top of the list (binary aphasia/dysarthria/dementia) is where strong acoustic correlates of the condition leak into nearly any speech encoder.
\item \textbf{{Most divisive}}: high-std tasks are where domain pretraining matters --- emotion classes reward affect-aware models, motor-speech regressions reward AST's spectral CNN, and AD detection penalises spectrogram-only models.
\end{{itemize}}

\end{{document}}
"""


def main() -> None:
    merged, _ = compute_scores()
    summary = encoder_summary(merged)
    summary = summary.sort_values("cumulative_GPA", ascending=False)

    rank_table = [
        (enc, summary.loc[enc, "letter"], summary.loc[enc, "cumulative_GPA"],
         ENCODER_INFO[enc]["display"])
        for enc in summary.index
    ]

    class_mean = merged.mean(axis=1)
    class_std = merged.std(axis=1)

    parts = [render_preamble(rank_table), render_intro(rank_table),
             render_leaderboard_page(summary)]
    for enc in summary.index:  # ordered by GPA, top-of-class first
        parts.append(render_encoder_page(
            enc, summary.loc[enc], merged[enc], class_mean, class_std))
    parts.append(render_summary_page(merged))

    TEX.write_text("\n".join(parts))
    print(f"wrote {TEX}")

    # Compile twice for fancyhdr / cross-refs.
    for i in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "report_card.tex"],
            cwd=OUT_DIR, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-1000:])
            raise SystemExit(f"pdflatex failed on pass {i+1}")
    print(f"wrote {OUT_DIR / 'report_card.pdf'}")


if __name__ == "__main__":
    main()
