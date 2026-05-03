#!/usr/bin/env python3
"""
Load the final checkpoint for qwen3voice-CLTP-t and get all misclassified test cases.
"""
import os
import json

import soundfile as sf
import torch
import pandas as pd
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from ahb.brain import DiagnosticsBrain
import importlib


def main():
    exp_dir = "exps/single_task/c9s_t1/qwen3voice-CLTP-t/brain-logs/final_model"
    hparams_file = os.path.join(exp_dir, "hyperparams.yaml")

    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)

    # Load data
    data_io_module = importlib.import_module(hparams["data_io_script"])
    dataio_prep_fn = getattr(data_io_module, hparams["dataio_prep_fn"])
    datasets = dataio_prep_fn(hparams)

    test_set = datasets["test_val"]

    # Init brain (chunker is now initialized in Model.__init__)
    brain = DiagnosticsBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": "cpu"},
        checkpointer=hparams["checkpointer"],
    )

    brain.device = torch.device("cpu")

    # Run inference
    brain.modules.eval()
    results = []

    test_loader = sb.dataio.dataloader.make_dataloader(
        test_set, **hparams["test_dataloader_options"], shuffle=False
    )

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(brain.device)
            predictions = brain.compute_forward(batch, sb.Stage.TEST)

            # Get labels
            label_key = getattr(hparams, "label_key", "label_encoded") if not isinstance(hparams, dict) else hparams.get("label_key", "label_encoded")
            labels = getattr(batch, label_key)
            labels = labels.float()

            # Binary: sigmoid > 0.5
            probs = torch.sigmoid(predictions.squeeze(-1))
            preds = (probs > 0.5).long()
            labs = labels.long()

            ids = batch.id

            for i in range(len(ids)):
                results.append({
                    "id": ids[i],
                    "true_label": labs[i].item(),
                    "pred_label": preds[i].item(),
                    "prob": probs[i].item(),
                })

    df = pd.DataFrame(results)
    misclassified = df[df["true_label"] != df["pred_label"]]

    print(f"\nTotal test samples: {len(df)}")
    print(f"Misclassified: {len(misclassified)}")
    print(f"Accuracy: {1 - len(misclassified)/len(df):.4f}")

    # Merge with metadata for more info
    with open(hparams["test_annotation"], "r") as f:
        test_data = json.load(f)

    test_meta = test_data["val"]  # test set is stored as "val" in test.json

    # Build output with all metadata columns + duration
    meta_rows = []
    for _, row in misclassified.iterrows():
        uid = row["id"]
        meta = test_meta.get(uid, test_meta.get(str(uid), {}))
        entry = {
            "id": uid,
            "pred_label": int(row["pred_label"]),
            "prob": round(row["prob"], 4),
        }
        # Copy all metadata columns
        entry.update(meta)
        # Get audio duration
        path = meta.get("path", "")
        if path and os.path.isfile(path):
            info = sf.info(path)
            entry["duration_s"] = round(info.duration, 2)
        else:
            entry["duration_s"] = None
        meta_rows.append(entry)

    out_df = pd.DataFrame(meta_rows)
    out_path = os.path.join(exp_dir, "misclassified.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nCSV saved to: {out_path}")
    print(f"Columns: {list(out_df.columns)}")


if __name__ == "__main__":
    main()