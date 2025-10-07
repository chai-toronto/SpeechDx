#!/usr/bin/env python3
"""
Generic Recipe for training a diagnostics model using SpeechBrain.

This script is designed to be reusable across different health-related
datasets. Dataset-specific functions (like data preparation and
dataio pipeline definition) are loaded dynamically from a script
specified in the YAML file.

To run this recipe, use a specific hparams file:
> python train_generic.py hparams/respiratory.yaml

Authors
--
Yi Zhu 2025
"""
import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from training.brain import DiagnosticsBrain

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp.
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dynamically load the data preparation module specified in the YAML
    # This module contains the 'prepare_data' and 'dataio_prep' functions.
    try:
        data_io_module = sb.import_module(hparams["data_io_script"])
    except KeyError:
        sys.exit("Error: 'data_io_script' path must be defined in the YAML file.")

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        prepare_data_fn = getattr(data_io_module, hparams["prepare_data_fn"])
        
        sb.utils.distributed.run_on_main(
            prepare_data_fn,
            kwargs={
                "wav_folder": hparams["wav_folder"],
                "audio_archive_path": hparams["audio_archive_path"],
                "metadata_path": hparams["metadata_path"],
                "manifest_train_path": hparams["train_annotation"],
                "manifest_val_path": hparams["val_annotation"],
                "manifest_test_path": hparams["test_annotation"],
                "ratio": hparams["ratio"],
                "random_seed": hparams["random_seed"],
                "label_key": hparams["label_key"],
                "new_test": hparams["new_test"],
                "num_fold": hparams["num_fold"],
            },
        )

    # Create dataset objects
    dataio_prep_fn = getattr(data_io_module, hparams["dataio_prep_fn"])
    datasets = dataio_prep_fn(hparams)


    # Initialize the Brain object
    brain = DiagnosticsBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=datasets[f"train_{0}"],
        valid_set=datasets[f"valid_{0}"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
    )

    # Evaluation
    test_stats = brain.evaluate(
        test_set=datasets["test"],
        max_key="F1",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
