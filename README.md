Installation guide:
- (For torchcodec) Install ffmpeg through conda (unless you're on a HPC and it's already installed)
  - further instruction here: https://github.com/meta-pytorch/torchcodec
  - version 0.7 and before only
- Torch 2.9 nuked a lot from torchaudio. We dont support it
- then pip install -r requirements.txt