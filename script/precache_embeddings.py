#!/usr/bin/env python3
"""
Pre-cache encoder embeddings before training.

This script runs the encoder on all audio samples and caches the embeddings
to disk. This allows parallel training to use pre-computed embeddings instead
of running the encoder on-the-fly.

Usage:
    python script/precache_embeddings.py ahb/configs/main.yaml

"""
import sys
import os
import importlib
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import hashlib

import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


def get_cache_filename(sample_id):
    """Generate a safe filename from sample_id using hash for long/special IDs."""
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    if len(safe_id) > 100:
        safe_id = hashlib.md5(sample_id.encode()).hexdigest()
    return f"{safe_id}.pt"


def precache_embeddings(hparams_file, overrides=None):
    """Pre-cache all encoder embeddings to disk.
    
    Args:
        hparams_file: Path to YAML hyperparameters file
        overrides: Optional dict or string of parameter overrides
    """
    # Force encoder to load by setting use_precached=False
    # Handle both dict and string overrides (from command line)
    if isinstance(overrides, str):
        # sb.parse_arguments returns YAML-formatted string like "key1: val1\nkey2: val2"
        precache_overrides = {"use_precached": False}
        if overrides:
            for line in overrides.strip().split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    precache_overrides[k] = v
    else:
        precache_overrides = overrides.copy() if overrides else {}
        precache_overrides["use_precached"] = False
    
    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, precache_overrides)
    
    # Setup cache directory
    cache_dir = Path(hparams.get("cache_dir", "cache"))
    embeddings_dir = cache_dir / "embeddings"  # Individual embedding files go here
    index_file = cache_dir / "index.pt"
    cache_file = cache_dir / "cache.pt"
    
    # Check if cache already exists and is valid (either format)
    if index_file.exists() or cache_file.exists():
        print(f"Cache already exists at {cache_dir}")
        print("Skipping pre-caching. Delete cache directory to regenerate.")
        return
    
    # Clear existing cache (only if incomplete/invalid)
    if cache_dir.exists():
        print(f"Clearing incomplete cache at {cache_dir}")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Get encoder
    encoder = hparams["encoder"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()
    
    num_layers = hparams.get("num_layers", 1)
    output_hidden_states = getattr(encoder, "output_hidden_states", False)
    
    print(f"Encoder: {type(encoder).__name__}")
    print(f"Device: {device}")
    print(f"Output hidden states: {output_hidden_states}")
    print(f"Num layers: {num_layers}")
    
    # Run data preparation first (creates manifest files)
    data_io_module = importlib.import_module(hparams["data_io_script"])
    if not hparams.get("skip_prep", False):
        prepare_data_fn = getattr(data_io_module, hparams["prepare_data_fn"])
        print("\nRunning data preparation...")
        prepare_data_fn(
            wav_folder=hparams["wav_folder"],
            audio_archive_path=hparams["audio_archive_path"],
            metadata_path=hparams["metadata_path"],
            manifest_train_path=hparams["train_annotation"],
            manifest_val_path=hparams["val_annotation"],
            manifest_test_path=hparams["test_annotation"],
            ratio=hparams.get("ratio", None),
            random_seed=hparams["random_seed"],
            raw_label_key=hparams["raw_label_key"],
            new_test=hparams["new_test"],
            num_fold=hparams["num_fold"],
            max_length=hparams.get("max_length", None)
        )
    
    # Load all data manifests
    import json
    
    all_ids = set()
    data_files = [
        hparams["train_annotation"],
        hparams["val_annotation"],
        hparams["test_annotation"],
    ]
    
    all_data = {}
    for data_file in data_files:
        with open(data_file) as f:
            data = json.load(f)
        
        # Handle both list of folds and single dict
        if isinstance(data, list):
            for fold in data:
                for sample_id, sample_data in fold.items():
                    if sample_id not in all_data:
                        all_data[sample_id] = sample_data
        elif isinstance(data, dict):
            # Test data has train/val keys
            for split_name, split_data in data.items():
                if isinstance(split_data, dict):
                    for sample_id, sample_data in split_data.items():
                        if sample_id not in all_data:
                            all_data[sample_id] = sample_data
    
    print(f"Total unique samples: {len(all_data)}")
    
    # MEMORY-EFFICIENT: Save each embedding to individual file immediately
    # OLD CODE: cache = {} and cache[sample_id] = emb, then torch.save(cache, cache_file)
    # NEW CODE: Save each embedding to embeddings_dir/{sample_id}.pt immediately
    sample_rate = hparams.get("sample_rate", 16000)
    max_length = hparams.get("max_length", None)
    fallback_max_length = int(hparams.get("fallback_max_length", 320000))  # 20s default fallback
    num_augmentations = hparams.get("num_augmentations", 1)  # Number of random crops per sample
    
    if max_length is None or max_length == "null" or (isinstance(max_length, float) and max_length >= 1e9):
        max_length = None
        print(f"No max_length set - will use full audio, fallback to {fallback_max_length/sample_rate:.1f}s on OOM")
    else:
        max_length = int(max_length)
        print(f"Max audio length: {max_length} samples ({max_length/sample_rate:.1f}s)")
    
    if num_augmentations > 1:
        print(f"Generating {num_augmentations} random crops per sample (augmentation)")
    
    truncated_samples = []
    oom_retries = 0
    
    # Create index file to map sample_id -> filename(s)
    # If num_augmentations > 1: sample_id -> [filename_aug0, filename_aug1, ...]
    # If num_augmentations == 1: sample_id -> filename
    index = {}
    
    with torch.no_grad():
        for sample_id, sample_data in tqdm(all_data.items(), desc="Caching embeddings"):
            # Load audio - check various possible key names
            wav_path = sample_data.get("path") or sample_data.get("wav") or sample_data.get("audio_path")
            if not wav_path:
                print(f"Warning: No audio path for {sample_id}, keys: {list(sample_data.keys())[:5]}")
                continue
            
            # Read audio using torchaudio
            import torchaudio
            signal_full, sr = torchaudio.load(wav_path)
            
            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                signal_full = resampler(signal_full)
            
            # Convert to mono if stereo
            if signal_full.shape[0] > 1:
                signal_full = signal_full.mean(dim=0, keepdim=True)
            
            original_length = signal_full.shape[1]
            
            # Determine how many augmentations to create
            # Only augment if audio is longer than max_length
            needs_crop = max_length is not None and original_length > max_length
            actual_num_augs = num_augmentations if needs_crop else 1
            
            aug_filenames = []
            
            for aug_idx in range(actual_num_augs):
                # Random crop to max_length if needed
                if needs_crop:
                    start = random.randint(0, original_length - max_length)
                    signal = signal_full[:, start:start + max_length].clone()
                else:
                    signal = signal_full.clone()
                
                # Remove channel dimension for encoder
                signal = signal.squeeze(0).to(device)
                
                # Try encoding with OOM fallback
                try:
                    emb = encoder(signal.unsqueeze(0))
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower() or "HIP out of memory" in str(e):
                        # OOM - retry with truncation
                        torch.cuda.empty_cache()
                        oom_retries += 1
                        
                        # Truncate to fallback length
                        truncate_to = min(fallback_max_length, original_length)
                        # Random crop for OOM fallback too
                        if original_length > truncate_to:
                            start = random.randint(0, original_length - truncate_to)
                            signal_truncated = signal_full[:, start:start + truncate_to].clone()
                        else:
                            signal_truncated = signal_full.clone()
                        signal_truncated = signal_truncated.squeeze(0).to(device)
                        
                        if aug_idx == 0:  # Only log once per sample
                            truncated_samples.append({
                                "id": sample_id, 
                                "original": original_length, 
                                "truncated_to": truncate_to
                            })
                            tqdm.write(f"OOM on {sample_id} ({original_length/sample_rate:.1f}s) - retrying with {truncate_to/sample_rate:.1f}s")
                        
                        emb = encoder(signal_truncated.unsqueeze(0))
                    else:
                        raise  # Re-raise non-OOM errors
                
                # Handle output format
                if output_hidden_states and isinstance(emb, (tuple, list)):
                    # Store as tuple of tensors (moved to CPU)
                    emb = tuple(x.squeeze(0).cpu() for x in emb)
                else:
                    emb = emb.squeeze(0).cpu()
                
                # Choose caching strategy based on size
                if output_hidden_states:
                    if actual_num_augs > 1:
                        emb_filename = get_cache_filename(f"{sample_id}_aug{aug_idx}")
                    else:
                        emb_filename = get_cache_filename(sample_id)
                    emb_path = embeddings_dir / emb_filename
                    torch.save(emb, emb_path)
                    aug_filenames.append(emb_filename)
                else:
                    aug_filenames.append(emb)
            
            # Store in index
            if output_hidden_states:
                # For augmented samples, store list; for single, store string (backwards compat)
                index[sample_id] = aug_filenames if len(aug_filenames) > 1 else aug_filenames[0]
            else:
                index[sample_id] = aug_filenames if len(aug_filenames) > 1 else aug_filenames[0]
            
            # Clear GPU cache periodically to prevent memory buildup
            if len(index) % 500 == 0:
                torch.cuda.empty_cache()
    
    if truncated_samples:
        print(f"\n  {len(truncated_samples)} samples needed OOM fallback truncation:")
        for t in truncated_samples[:10]:  # Show first 10
            print(f"   {t['id']}: {t['original']/sample_rate:.1f}s → {t['truncated_to']/sample_rate:.1f}s")
        if len(truncated_samples) > 10:
            print(f"   ... and {len(truncated_samples) - 10} more")
    
    if num_augmentations > 1:
        augmented_count = sum(1 for v in index.values() if isinstance(v, list))
        total_embeddings = sum(len(v) if isinstance(v, list) else 1 for v in index.values())
        print(f"\n  Augmentation stats:")
        print(f"   {augmented_count} samples have {num_augmentations} crops (longer than {max_length/sample_rate:.1f}s)")
        print(f"   {len(index) - augmented_count} samples have 1 crop (shorter, no augmentation needed)")
        print(f"   Total embeddings: {total_embeddings}")
    
    # Save cache
    if output_hidden_states:
        # Save index file for individual files (maps sample_id to filename or list of filenames)
        index_file = cache_dir / "index.pt"
        torch.save(index, index_file)
        print(f"Saved index to {index_file}")
        print(f"Cached {len(index)} samples to {embeddings_dir}")
    else:
        # Save single cache.pt file (old style, faster for small caches)
        cache_file = cache_dir / "cache.pt"
        torch.save(index, cache_file)
        print(f"Saved cache to {cache_file} ({len(index)} samples)")
    
    return cache_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script/precache_embeddings.py <hparams_file> [overrides]")
        sys.exit(1)
    
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    precache_embeddings(hparams_file, overrides)
    print("\nPre-caching complete! You can now run training with use_precached: True")
