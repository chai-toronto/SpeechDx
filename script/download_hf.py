import sys

from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    repo_id = sys.argv[1]  # e.g., "username/repo_name"
    filename = sys.argv[2]

    print(f"Downloading and caching model: {repo_id}")
    # Download fully into HF default cache (~/.cache/huggingface/hub)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=None,
        local_dir=None,
        local_dir_use_symlinks=False,
    )

    print("Success!")
    print(f"Model cached at: {path}")
    print("You can now load it anywhere (even offline) using:")
    print(f'  from transformers import AutoModel\n  AutoModel.from_pretrained("{repo_id}")')