"""
GameCraft VBench Batch Inference
=================================
Loops over VBench crop images, generates num_samples videos per prompt
(with different seeds), skips already-generated outputs, and writes
VBench-compatible output files: {vbench_output_dir}/{prompt}-{index}.mp4

Usage (from Hunyuan-GameCraft-1.0 root):
    python scripts/gc_vbench_batch.py --help

Example:
    python scripts/gc_vbench_batch.py \\
        --vbench_output_dir results_low_mem\\videos \\
        --num_samples 5 \\
        --image_types abstract,background \\
        --ckpt weights\\gamecraft_models\\mp_rank_00_model_states_distill.pt \\
        --neg_prompt "overexposed, low quality" \\
        --height 704 --width 1216 \\
        --steps 8 --frames 33 --cfg_scale 1.0 \\
        --actions w a d s --speeds 0.2 0.2 0.2 0.2
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import psutil
import torch

# ── Default paths ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VBENCH_ROOT = os.path.join(_ROOT, "..", "VBench", "vbench2_beta_i2v")
_DEFAULT_INFO_JSON = os.path.join(_VBENCH_ROOT, "vbench2_beta_i2v", "data", "i2v-bench-info.json")
_DEFAULT_CROP_DIR  = os.path.join(_VBENCH_ROOT, "vbench2_beta_i2v", "data", "crop")
_DEFAULT_NEG = (
    "overexposed, low quality, deformation, a poor composition, "
    "bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, "
    "text, subtitles, static, picture, black border."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="GameCraft VBench batch inference — N samples per VBench prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── VBench / loop settings ─────────────────────────────────────────────────
    p.add_argument("--vbench_output_dir", default="results_low_mem/videos",
                   help="Flat output directory for VBench-named videos (default: results_low_mem/videos)")
    p.add_argument("--vbench_info_json", default=_DEFAULT_INFO_JSON,
                   help="Path to i2v-bench-info.json")
    p.add_argument("--crop_dir", default=_DEFAULT_CROP_DIR,
                   help="Path to VBench crop image root (contains resolution subfolders)")
    p.add_argument("--resolution", default="1-1",
                   help="Resolution subfolder name inside crop_dir (default: 1-1)")
    p.add_argument("--image_types", default="scenery,indoor",
                   help="Comma-separated type filter, e.g. 'scenery,indoor' (default: scenery,indoor)")
    p.add_argument("--num_samples", type=int, default=5,
                   help="Number of samples to generate per prompt (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed; sample i uses seed+i (default: 42)")

    # ── GameCraft inference settings ───────────────────────────────────────────
    p.add_argument("--ckpt",
                   default=os.path.join(_ROOT, "weights", "gamecraft_models",
                                        "mp_rank_00_model_states_distill.pt"),
                   help="Path to GameCraft checkpoint")
    p.add_argument("--neg_prompt", default=_DEFAULT_NEG,
                   help="Negative prompt text")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width",  type=int, default=960)
    p.add_argument("--steps",  type=int, default=5,   help="Inference steps (default: 5)")
    p.add_argument("--frames", type=int, default=161,  help="Frames per video (default: 161)")
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--actions", nargs="+", default=["w", "a", "d", "s"])
    p.add_argument("--speeds",  nargs="+", type=float, default=[0.2, 0.2, 0.2, 0.2])
    p.add_argument("--flow_shift", type=float, default=5.0)

    return p.parse_args()


def load_vbench_entries(info_json, image_types_str):
    with open(info_json, encoding="utf-8") as f:
        entries = json.load(f)

    allowed = {t.strip() for t in image_types_str.split(",") if t.strip()} if image_types_str else None

    seen = set()
    result = []
    for e in entries:
        name = e["file_name"]
        if name in seen:
            continue
        if allowed and e.get("type") not in allowed:
            continue
        seen.add(name)
        result.append((name, e["caption"], e.get("type", "")))
    return result


def build_sample_cmd(args, image_path, prompt, seed, tmp_dir):
    return [
        sys.executable,
        os.path.join(_ROOT, "hymm_sp", "sample_batch.py"),
        "--image-path",      image_path,
        "--prompt",          prompt,
        "--add-neg-prompt",  args.neg_prompt,
        "--ckpt",            args.ckpt,
        "--video-size",      str(args.height), str(args.width),
        "--cfg-scale",       str(args.cfg_scale),
        "--image-start",
        "--action-list",     *args.actions,
        "--action-speed-list", *[str(s) for s in args.speeds],
        "--seed",            str(seed),
        "--sample-n-frames", str(args.frames),
        "--infer-steps",     str(args.steps),
        "--flow-shift-eval-video", str(args.flow_shift),
        "--use-fp8",
        "--save-path",       tmp_dir,
    ]


def _poll_vram(stop_event, readings):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, text=True
            ).strip().splitlines()[0].strip()
            if out.isdigit():
                readings.append(int(out))
        except Exception:
            pass
        time.sleep(5)


def _vram_peak_gb(readings):
    return round(max(readings) / 1024.0, 2) if readings else ''


def _ram_gb():
    try:
        return round(psutil.virtual_memory().used / 1024**3, 2)
    except Exception:
        return ''


def main():
    args = parse_args()

    print("[GC-VBench] Flags:")
    print(f"  --vbench_output_dir  {args.vbench_output_dir}")
    print(f"  --num_samples        {args.num_samples}")
    print(f"  --seed               {args.seed}")
    print(f"  --resolution         {args.resolution}")
    print(f"  --image_types        {args.image_types or '(all)'}")
    print(f"  --ckpt               {args.ckpt}")
    print(f"  --height/width       {args.height}x{args.width}")
    print(f"  --steps/frames       {args.steps}/{args.frames}")
    print(f"  --actions            {args.actions}  speeds={args.speeds}")

    # Resolve paths
    info_json = os.path.abspath(args.vbench_info_json)
    crop_dir  = os.path.abspath(args.crop_dir)
    out_dir   = os.path.abspath(args.vbench_output_dir)
    image_dir = os.path.join(crop_dir, args.resolution)
    os.makedirs(out_dir, exist_ok=True)

    stats_path   = os.path.join(os.path.dirname(out_dir), 'vbench_stats.csv')
    stats_is_new = not os.path.isfile(stats_path)
    stats_f      = open(stats_path, 'a', newline='', encoding='utf-8')
    stats_w      = csv.writer(stats_f)
    if stats_is_new:
        stats_w.writerow(['task_idx', 'prompt', 'type', 'sample_idx', 'seed',
                          'duration_s', 'gen_fps', 'vram_gb', 'ram_gb', 'out_path', 'status'])
        stats_f.flush()

    fps_path = os.path.join(os.path.dirname(out_dir), 'vbench_fps.txt')
    fps_f    = open(fps_path, 'w', encoding='utf-8')
    fps_f.write(f'GameCraft VBench Per-Video Timing\n')
    fps_f.write(f'frames={args.frames}  steps={args.steps}  size={args.height}x{args.width}\n')
    fps_f.write('=' * 78 + '\n')
    fps_f.write(f'{"#":>4}  {"prompt":<50}  {"s":>1}  {"dur_s":>6}  {"fps":>5}  status\n')
    fps_f.write('-' * 78 + '\n')

    if not os.path.isfile(info_json):
        print(f"[GC-VBench] ERROR: VBench info JSON not found: {info_json}")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"[GC-VBench] ERROR: Crop image directory not found: {image_dir}")
        sys.exit(1)

    entries = load_vbench_entries(info_json, args.image_types)
    print(f"\n[GC-VBench] Found {len(entries)} prompts | {args.num_samples} samples each "
          f"= {len(entries) * args.num_samples} total videos\n")

    skipped   = 0
    generated = 0
    errors    = 0
    total     = len(entries) * args.num_samples
    done      = 0
    t_start   = time.time()

    for task_idx, (image_name, prompt, img_type) in enumerate(entries):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"[GC-VBench] Skipping task {task_idx}: image not found — {image_path}")
            continue

        image_stem = os.path.splitext(image_name)[0]

        for sample_idx in range(args.num_samples):
            seed = args.seed + sample_idx
            out_path = os.path.join(out_dir, f"{prompt}-{sample_idx}-{seed}.mp4")

            if glob.glob(os.path.join(out_dir, f"{prompt}-{sample_idx}-*.mp4")):
                skipped += 1
                done += 1
                existing = glob.glob(os.path.join(out_dir, f"{prompt}-{sample_idx}-*.mp4"))[0]
                stats_w.writerow([task_idx, prompt, img_type, sample_idx, seed, '', '', '', '', existing, 'skipped'])
                stats_f.flush()
                fps_f.write(f'{task_idx+1:>4}  {prompt[:50]:<50}  {sample_idx}  {"":>6}  {"":>5}  skipped\n')
                fps_f.flush()
                continue

            pct = 100 * done / total if total else 0
            eta = ''
            if done > 0:
                secs_left = (time.time() - t_start) / done * (total - done)
                eta = f'  ETA {int(secs_left//3600):02d}h{int(secs_left%3600//60):02d}m{int(secs_left%60):02d}s'
            print(f"[GC-VBench] [{done+1}/{total}  {pct:.0f}%{eta}]  prompt {task_idx+1}/{len(entries)}  sample {sample_idx+1}/{args.num_samples}  seed {seed}: {prompt[:50]}")

            tmp_dir = os.path.join(out_dir, f"_tmp_{task_idx}_{sample_idx}")
            os.makedirs(tmp_dir, exist_ok=True)

            cmd = build_sample_cmd(args, image_path, prompt, seed, tmp_dir)

            env = os.environ.copy()
            env.update({"PYTHONPATH": _ROOT, "DISABLE_SP": "1",
                        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"})

            vram_readings = []
            stop_evt = threading.Event()
            vram_thread = threading.Thread(target=_poll_vram, args=(stop_evt, vram_readings), daemon=True)
            vram_thread.start()

            t0 = time.time()
            result = subprocess.run(cmd, cwd=_ROOT, env=env)
            elapsed = time.time() - t0

            stop_evt.set()
            vram_thread.join(timeout=10)

            gen_fps = args.frames / elapsed if elapsed > 0 else 0.0
            vram = _vram_peak_gb(vram_readings)
            ram  = _ram_gb()

            tmp_mp4 = os.path.join(tmp_dir, f"{image_stem}.mp4")
            if result.returncode == 0 and os.path.isfile(tmp_mp4):
                shutil.move(tmp_mp4, out_path)
                print(f"[GC-VBench] Saved: {out_path}  ({gen_fps:.1f} gen-fps)  VRAM {vram}GB  RAM {ram}GB")
                stats_w.writerow([task_idx, prompt, img_type, sample_idx, seed,
                                  f'{elapsed:.2f}', f'{gen_fps:.2f}', vram, ram, out_path, 'ok'])
                stats_f.flush()
                fps_f.write(f'{task_idx+1:>4}  {prompt[:50]:<50}  {sample_idx}  {elapsed:>6.1f}  {gen_fps:>5.2f}  ok\n')
                fps_f.flush()
                generated += 1
            else:
                print(f"[GC-VBench] ERROR: task {task_idx} sample {sample_idx} "
                      f"(exit={result.returncode}, mp4_found={os.path.isfile(tmp_mp4)})")
                stats_w.writerow([task_idx, prompt, img_type, sample_idx, seed,
                                  f'{elapsed:.2f}', f'{gen_fps:.2f}', vram, ram, out_path, 'error'])
                stats_f.flush()
                fps_f.write(f'{task_idx+1:>4}  {prompt[:50]:<50}  {sample_idx}  {elapsed:>6.1f}  {gen_fps:>5.2f}  ERROR\n')
                fps_f.flush()
                errors += 1

            done += 1
            shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed_total = time.time() - t_start
    stats_f.close()
    fps_f.write('=' * 78 + '\n')
    fps_f.write(f'generated={generated}  skipped={skipped}  errors={errors}\n')
    fps_f.close()
    print(f"\n[GC-VBench] Done — generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed_total/60:.1f}m")
    print(f"[GC-VBench] Stats → {stats_path}")
    print(f"[GC-VBench] FPS   → {fps_path}")


if __name__ == "__main__":
    main()
