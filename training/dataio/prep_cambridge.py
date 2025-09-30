"""
Dataset-specific preparation script and dataio pipelines.

This file contains functions that are unique to the 'CS-Res-L' dataset, 
including manifest generation and label encoding logic.

To adapt to a new dataset, copy this file and modify:
1. prepare_data (to generate the correct train/valid/test JSON manifest files).
2. label_pipeline (to handle the specific labels and label-to-index mapping).
3. The 'output_keys' in dataio_prep (if the label key changes).
"""

import torchaudio
import torchaudio.functional as F
import speechbrain as sb
import torch


def prepare_data(
    wav_folder,
    audio_archive_path,
    metadata_path,
    manifest_train_path,
    manifest_valid_path,
    manifest_test_path,
    ratio,
    random_seed,
):
    """
    This function is dataset-specific.
    It takes raw data info (like metadata_path and wav_folder) and 
    creates the SpeechBrain manifest JSON files (train, valid, test).
    
    Replace the body of this function with the logic needed for your 
    specific dataset (e.g., reading a CSV/TSV file and writing JSONs).
    """

    if not all(
        (
            sb.utils.checkpoints.is_pytorch_object_in_folder(manifest_train_path),
            sb.utils.checkpoints.is_pytorch_object_in_folder(manifest_valid_path),
            sb.utils.checkpoints.is_pytorch_object_in_folder(manifest_test_path)
        )
    ):
        print(f"Creating mock manifest files for demonstration in: {manifest_train_path}")

        mock_data = {
            "utt1": {"file_path": "path/to/wavs/mock1.wav", "symptom-label": "non"},
            "utt2": {"file_path": "path/to/wavs/mock2.wav", "symptom-label": "symptomatic"},
        }
        import json
        with open(manifest_train_path, 'w') as f:
            json.dump(mock_data, f, indent=4)
        with open(manifest_valid_path, 'w') as f:
            json.dump(mock_data, f, indent=4)
        with open(manifest_test_path, 'w') as f:
            json.dump(mock_data, f, indent=4)
        print("Manifests created.")
    print("--- prepare_data finished ---")


def dataio_prep(hparams):
    """
    This function is dataset-specific.
    It defines the data processing pipelines and creates the DynamicItemDatasets.

    For a new task, modify the label_pipeline and the output_keys.
    """

    # Initialization of the label encoder.
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("signal","duration")
    def audio_pipeline(file_path):
        """Load the signal, resample, and pass it and its length."""
        
        signal, sr_og = torchaudio.load(file_path)
        # handle multi-channel
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0)

        if sr_og != 16000:
            signal = F.resample(signal,sr_og,new_freq=16000,
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492
                                )
        signal  = signal.squeeze()
        duration = len(signal)
        return signal, duration

    # Define label pipeline
    @sb.utils.data_pipeline.takes("symptom-label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        """Defines the pipeline to process the input label ('non'/'symptomatic')."""
        # Custom label mapping
        label_encoder.lab2ind = {'non':0, 'symptomatic':1}
        # The key produced here ('label_encoded') must match 
        # the 'label_key' used in train.py and the YAML.
        label_encoded = label_encoder.encode_label_torch(label)
        yield label_encoded

    # Define datasets.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"]
    }

    hparams["dataloader_options"]["shuffle"] = True

    for dataset in data_info:

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],

            output_keys=["id", "signal", "duration", "file_path", "symptom_label_encoded", "symptom"],
        )

    return datasets
