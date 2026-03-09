"""
Low-memory single-GPU inference for Hunyuan-GameCraft-1.0 on Windows.

Uses:
  - Distilled checkpoint (8 steps instead of 50)
  - CPU offload  (DISABLE_SP + CPU_OFFLOAD env vars)
  - FP8 weights
  - 33 frames (shortest clip)

Usage:
  python run_low_mem.py
  python run_low_mem.py --image asset/village.png --actions w a d s
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# ── configurable defaults ──────────────────────────────────────────────────
CHECKPOINT    = str(ROOT / "weights" / "gamecraft_models" / "mp_rank_00_model_states_distill.pt")
IMAGE         = str(ROOT / "asset" / "village.png")
PROMPT        = "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky."
NEG_PROMPT    = "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border."
SAVE_PATH     = str(ROOT / "results_low_mem")
VIDEO_SIZE    = ["480", "832"]    # height width  (reduce to e.g. ["480","832"] for less VRAM)
INFER_STEPS   = 6
N_FRAMES      = 17
CFG_SCALE     = 1.0
SEED          = 42
ACTIONS       = ["w", "a", "d", "s"]
SPEEDS        = ["0.2", "0.2", "0.2", "0.2"]
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Low-memory Hunyuan-GameCraft inference (Windows)")
    p.add_argument("--image",   default=IMAGE,  help="Path to conditioning image")
    p.add_argument("--prompt",  default=PROMPT, help="Text prompt")
    p.add_argument("--actions", nargs="+", default=ACTIONS,
                   help="Action sequence: any combo of w a s d")
    p.add_argument("--speeds",  nargs="+", default=SPEEDS,
                   help="Speed per action (0-3), must match --actions length")
    p.add_argument("--ckpt",    default=CHECKPOINT, help="Checkpoint path")
    p.add_argument("--save",    default=SAVE_PATH,  help="Output directory")
    p.add_argument("--steps",   type=int, default=INFER_STEPS)
    p.add_argument("--frames",  type=int, default=N_FRAMES)
    p.add_argument("--height",  type=int, default=int(VIDEO_SIZE[0]))
    p.add_argument("--width",   type=int, default=int(VIDEO_SIZE[1]))
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.actions) != len(args.speeds):
        print(f"ERROR: --actions ({len(args.actions)}) and --speeds ({len(args.speeds)}) must be the same length.")
        sys.exit(1)

    if not Path(args.ckpt).exists():
        print(f"ERROR: Checkpoint not found: {args.ckpt}")
        print("Download from: https://huggingface.co/tencent/Hunyuan-GameCraft-1.0")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    os.makedirs(args.save, exist_ok=True)

    # Windows env for low-VRAM single-GPU run
    env = os.environ.copy()
    env["PYTHONPATH"]  = str(ROOT)
    env["MODEL_BASE"]  = str(ROOT / "weights" / "stdmodels")
    env["DISABLE_SP"]  = "1"   # disable sequence parallelism
    # Single-process distributed env vars (used by initialize_distributed)
    env["RANK"]        = "0"
    env["LOCAL_RANK"]  = "0"
    env["WORLD_SIZE"]  = "1"

    cmd = [
        sys.executable,
        str(ROOT / "hymm_sp" / "sample_batch.py"),
        "--image-path",       args.image,
        "--prompt",           args.prompt,
        "--add-neg-prompt",   NEG_PROMPT,
        "--ckpt",             args.ckpt,
        "--video-size",       str(args.height), str(args.width),
        "--cfg-scale",        str(CFG_SCALE),
        "--image-start",
        "--action-list",      *args.actions,
        "--action-speed-list",*args.speeds,
        "--seed",             str(SEED),
        "--sample-n-frames",  str(args.frames),
        "--infer-steps",      str(args.steps),
        "--flow-shift-eval-video", "5.0",
        "--use-fp8",
        "--save-path",        args.save,
    ]

    print("=" * 60)
    print("Hunyuan-GameCraft  |  low-memory single-GPU  |  Windows")
    print("=" * 60)
    print(f"  image   : {args.image}")
    print(f"  actions : {' '.join(args.actions)}  speeds: {' '.join(args.speeds)}")
    print(f"  size    : {args.height}x{args.width}   frames: {args.frames}   steps: {args.steps}")
    print(f"  output  : {args.save}")
    print("=" * 60)

    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
