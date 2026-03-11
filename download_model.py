"""
Download Hunyuan-GameCraft-1.0 weights from HuggingFace.
Run from the repo root: python download_model.py

Set HF_TOKEN env var if the repo requires authentication:
  set HF_TOKEN=hf_...
  python download_model.py
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

token = os.environ.get("HF_TOKEN") or None

local_dir = Path(__file__).parent / "weights"

# --- GameCraft model checkpoints ---
GAMECRAFT_REPO = "tencent/Hunyuan-GameCraft-1.0"
GAMECRAFT_FILES = [
    "gamecraft_models/mp_rank_00_model_states.pt",
    "gamecraft_models/mp_rank_00_model_states_distill.pt",
]

print(f"Downloading {GAMECRAFT_REPO} -> {local_dir}")
snapshot_download(
    repo_id=GAMECRAFT_REPO,
    local_dir=str(local_dir),
    ignore_patterns=["*.metadata", ".cache/*"],
    token=token,
)

# --- VAE from HunyuanVideo ---
VAE_REPO  = "tencent/HunyuanVideo-I2V"
VAE_FILES = [
    "hunyuan-video-i2v-720p/vae/config.json",
    "hunyuan-video-i2v-720p/vae/pytorch_model.pt",
]
VAE_LOCAL = local_dir / "vae_3d" / "hyvae"
VAE_LOCAL.mkdir(parents=True, exist_ok=True)

print(f"\nDownloading VAE from {VAE_REPO} -> {VAE_LOCAL}")
for vae_file in VAE_FILES:
    dest = VAE_LOCAL / Path(vae_file).name
    if dest.exists():
        print(f"  already exists: {dest.name}")
        continue
    print(f"  fetching {vae_file} ...")
    hf_hub_download(
        repo_id=VAE_REPO,
        filename=vae_file,
        local_dir=str(VAE_LOCAL),
        local_dir_use_symlinks=False,
        token=token,
    )

ALL_FILES = [(local_dir, f) for f in GAMECRAFT_FILES] + \
            [(VAE_LOCAL, Path(f).name) for f in VAE_FILES]

# Verify all files
print()
for base, rel in ALL_FILES:
    ckpt = base / rel
    if ckpt.exists():
        size_gb = ckpt.stat().st_size / 1024**3
        if size_gb < 0.01:
            print(f"ERROR: {ckpt.name} is only {ckpt.stat().st_size} bytes — looks like an LFS pointer.")
        else:
            print(f"OK: {rel}  ({size_gb:.2f} GB)")
    else:
        print(f"ERROR: {rel} still missing.")

print(f"\nAll files in {local_dir}:")
for f in sorted(local_dir.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(local_dir)}  ({f.stat().st_size / 1024**2:.1f} MB)")
