"""
Download Hunyuan-GameCraft-1.0 weights from HuggingFace.
Run from the repo root: python download_model.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

REPO_ID    = "tencent/Hunyuan-GameCraft-1.0"
CKPT_FILES = [
    "gamecraft_models/mp_rank_00_model_states.pt",
    "gamecraft_models/mp_rank_00_model_states_distill.pt",
]
local_dir = Path(__file__).parent / "weights"

print(f"Downloading {REPO_ID} -> {local_dir}")
snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(local_dir),
    ignore_patterns=["*.metadata", ".cache/*"],
)

# Verify checkpoints; retry individually if missing
for CKPT_FILE in CKPT_FILES:
    ckpt = local_dir / CKPT_FILE
    if not ckpt.exists():
        print(f"\nWARNING: {CKPT_FILE} not found after snapshot_download.")
        print("Retrying with direct file download...")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=CKPT_FILE,
            local_dir=str(local_dir),
        )

    if ckpt.exists():
        size_gb = ckpt.stat().st_size / 1024**3
        if size_gb < 0.01:
            print(f"\nERROR: {ckpt.name} is only {ckpt.stat().st_size} bytes — looks like an LFS pointer.")
            print("Install git-lfs (https://git-lfs.com) and run: git lfs pull")
        else:
            print(f"\nOK: {CKPT_FILE}  ({size_gb:.2f} GB)")
    else:
        print(f"\nERROR: {CKPT_FILE} still missing after retry.")

# List all downloaded files
print(f"\nAll files in {local_dir}:")
for f in sorted(local_dir.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(local_dir)}  ({f.stat().st_size / 1024**2:.1f} MB)")
