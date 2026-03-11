"""
GameCraft VBench In-Process Batch Inference
===========================================
Like gc_vbench_batch.py but loads HunyuanVideoSampler ONCE and runs all
inference in-process — no subprocess overhead per video.

Usage (from Hunyuan-GameCraft-1.0 root):
    python scripts/gc_vbench_inprocess.py --help

Example:
    python scripts/gc_vbench_inprocess.py \\
        --vbench_output_dir results_low_mem\\videos \\
        --num_samples 5 \\
        --image_types scenery,indoor \\
        --ckpt weights\\gamecraft_models\\mp_rank_00_model_states_distill.pt \\
        --neg_prompt "overexposed, low quality" \\
        --height 720 --width 960 \\
        --steps 5 --frames 161 --cfg_scale 1.0 \\
        --actions w a d s --speeds 0.2 0.2 0.2 0.2
"""

import argparse
import csv
import glob
import json
import os
import random
import shutil
import sys
import threading
import time

# Ensure repo root is on sys.path so hymm_sp is importable when called via absolute path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
import torch
import torchvision.transforms as transforms
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VBENCH_ROOT = os.path.join(_ROOT, "..", "VBench", "vbench2_beta_i2v")
_DEFAULT_INFO_JSON = os.path.join(_VBENCH_ROOT, "vbench2_beta_i2v", "data", "i2v-bench-info.json")
_DEFAULT_CROP_DIR  = os.path.join(_VBENCH_ROOT, "vbench2_beta_i2v", "data", "crop")
_DEFAULT_NEG = (
    "overexposed, low quality, deformation, a poor composition, "
    "bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, "
    "text, subtitles, static, picture, black border."
)

_CLOSEST_SIZE = (704, 1216)


class _Tee:
    def __init__(self, stream, log_path):
        self._stream = stream
        self._log = open(log_path, 'w', encoding='utf-8', buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def fileno(self):
        return self._stream.fileno()

    def close(self):
        self._log.close()


class _CropResize:
    """Resize + center-crop to target (H, W) preserving aspect ratio."""
    def __init__(self, size=_CLOSEST_SIZE):
        self.target_h, self.target_w = size

    def __call__(self, img):
        w, h = img.size
        scale = max(self.target_w / w, self.target_h / h)
        new_size = (int(h * scale), int(w * scale))
        img = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR)(img)
        return transforms.CenterCrop((self.target_h, self.target_w))(img)


def parse_args():
    p = argparse.ArgumentParser(
        description="GameCraft VBench in-process inference — loads model once",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--vbench_output_dir", default="results_distill/videos")
    p.add_argument("--vbench_info_json", default=_DEFAULT_INFO_JSON)
    p.add_argument("--crop_dir", default=_DEFAULT_CROP_DIR)
    p.add_argument("--resolution", default="1-1")
    p.add_argument("--image_types", default="scenery,indoor")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt",
                   default=os.path.join(_ROOT, "weights", "gamecraft_models",
                                        "mp_rank_00_model_states_distill.pt"))
    p.add_argument("--neg_prompt", default=_DEFAULT_NEG)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width",  type=int, default=960)
    p.add_argument("--steps",  type=int, default=5)
    p.add_argument("--frames", type=int, default=161)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--actions", nargs="+", default=["w", "a", "d", "s"])
    p.add_argument("--speeds",  nargs="+", type=float, default=[0.2, 0.2, 0.2, 0.2])
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--log_file", default=None)
    return p.parse_args()


def load_vbench_entries(info_json, image_types_str):
    with open(info_json, encoding="utf-8") as f:
        entries = json.load(f)
    allowed = {t.strip() for t in image_types_str.split(",") if t.strip()} if image_types_str else None
    seen, result = set(), []
    for e in entries:
        name = e["file_name"]
        if name in seen:
            continue
        if allowed and e.get("type") not in allowed:
            continue
        seen.add(name)
        result.append((name, e["caption"], e.get("type", "")))
    return result


def _build_hymm_args(args):
    """Build hymm_sp args namespace by temporarily overriding sys.argv."""
    argv = [
        "gc_vbench_inprocess.py",
        "--ckpt",            args.ckpt,
        "--video-size",      str(args.height), str(args.width),
        "--sample-n-frames", str(args.frames),
        "--infer-steps",     str(args.steps),
        "--cfg-scale",       str(args.cfg_scale),
        "--add-neg-prompt",  args.neg_prompt,
        "--action-list",     *args.actions,
        "--action-speed-list", *[str(s) for s in args.speeds],
        "--flow-shift-eval-video", str(args.flow_shift),
        "--use-fp8",
        "--image-start",
        "--image-path",      "",
        "--save-path",       "",
    ]
    if args.cpu_offload:
        argv.append("--cpu-offload")

    saved = sys.argv[:]
    sys.argv = argv
    try:
        # Import here so sys.argv is already set when parse_args reads it
        from hymm_sp.config import parse_args as hymm_parse
        hymm_args = hymm_parse()
    finally:
        sys.argv = saved
    return hymm_args


def _load_sampler(hymm_args):
    """Load HunyuanVideoSampler and return (sampler, updated_args)."""
    from hymm_sp.sample_inference import HunyuanVideoSampler
    from hymm_sp.modules.parallel_states import initialize_distributed

    # Set env vars expected by the sampler
    os.environ.setdefault("DISABLE_SP", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MODEL_BASE", os.path.join(_ROOT, "weights", "stdmodels"))
    os.environ.setdefault("PYTHONPATH", _ROOT)

    initialize_distributed(hymm_args.seed)
    device = torch.device("cuda") if not hymm_args.cpu_offload else torch.device("cpu")
    sampler = HunyuanVideoSampler.from_pretrained(hymm_args.ckpt, args=hymm_args, device=device)
    hymm_args = sampler.args

    if hymm_args.cpu_offload:
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(
            sampler.pipeline.transformer,
            onload_device=torch.device("cuda"),
            offload_type="block_level",
            num_blocks_per_group=1,
        )
    return sampler, hymm_args


def _run_sample(sampler, hymm_args, image_path, prompt, seed, tmp_dir):
    """Run inference for one sample. Returns path of generated mp4."""
    device = torch.device("cuda")
    ref_tf = transforms.Compose([
        _CropResize(_CLOSEST_SIZE),
        transforms.CenterCrop(_CLOSEST_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    raw_ref_images = [Image.open(image_path).convert("RGB")]
    pv = torch.cat([ref_tf(img) for img in raw_ref_images]).unsqueeze(0).unsqueeze(2).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        if hymm_args.cpu_offload:
            sampler.vae.quant_conv.to("cuda")
            sampler.vae.encoder.to("cuda")
        sampler.pipeline.vae.enable_tiling()
        raw_last = sampler.vae.encode(pv).latent_dist.sample().to(dtype=torch.float16)
        raw_last.mul_(sampler.vae.config.scaling_factor)
        raw_ref = raw_last.clone()
        sampler.pipeline.vae.disable_tiling()
        if hymm_args.cpu_offload:
            sampler.vae.quant_conv.to("cpu")
            sampler.vae.encoder.to("cpu")

    last_latents, ref_latents = raw_last, raw_ref
    out_cat = None

    for idx, action_id in enumerate(hymm_args.action_list):
        outputs = sampler.predict(
            prompt=prompt,
            action_id=action_id,
            action_speed=hymm_args.action_speed_list[idx],
            is_image=(idx == 0),
            size=_CLOSEST_SIZE,
            seed=seed,
            last_latents=last_latents,
            ref_latents=ref_latents,
            video_length=hymm_args.sample_n_frames,
            guidance_scale=hymm_args.cfg_scale,
            num_images_per_prompt=hymm_args.num_images,
            negative_prompt=hymm_args.add_neg_prompt,
            infer_steps=hymm_args.infer_steps,
            flow_shift=hymm_args.flow_shift_eval_video,
            use_linear_quadratic_schedule=hymm_args.use_linear_quadratic_schedule,
            linear_schedule_end=hymm_args.linear_schedule_end,
            use_deepcache=hymm_args.use_deepcache,
            cpu_offload=hymm_args.cpu_offload,
            ref_images=raw_ref_images,
            output_dir=tmp_dir,
            return_latents=True,
            use_sage=hymm_args.use_sage,
        )
        ref_latents  = outputs["ref_latents"]
        last_latents = outputs["last_latents"]
        sub = outputs["samples"][0]
        out_cat = sub if out_cat is None else torch.cat([out_cat, sub], dim=2)

    from hymm_sp.data_kits.data_tools import save_videos_grid
    stem = os.path.splitext(os.path.basename(image_path))[0]
    mp4 = os.path.join(tmp_dir, f"{stem}.mp4")
    save_videos_grid(out_cat, mp4, n_rows=1, fps=24)
    return mp4


def _poll_vram(stop_event, readings):
    import subprocess as _sp
    while not stop_event.is_set():
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stderr=_sp.DEVNULL, text=True,
            ).strip().splitlines()[0].strip()
            if out.isdigit():
                readings.append(int(out))
        except Exception:
            pass
        time.sleep(5)


def main():
    args = parse_args()

    if args.log_file:
        sys.stdout = _Tee(sys.stdout, args.log_file)

    print("[GC-VBench-IP] Loading model (once)...")
    hymm_args = _build_hymm_args(args)
    sampler, hymm_args = _load_sampler(hymm_args)
    print("[GC-VBench-IP] Model ready.\n")

    info_json = os.path.abspath(args.vbench_info_json)
    crop_dir  = os.path.abspath(args.crop_dir)
    out_dir   = os.path.abspath(args.vbench_output_dir)
    image_dir = os.path.join(crop_dir, args.resolution)
    os.makedirs(out_dir, exist_ok=True)

    stats_path   = os.path.join(os.path.dirname(out_dir), "vbench_stats.csv")
    stats_is_new = not os.path.isfile(stats_path)
    stats_f      = open(stats_path, "a", newline="", encoding="utf-8")
    stats_w      = csv.writer(stats_f)
    if stats_is_new:
        stats_w.writerow(["task_idx", "prompt", "type", "sample_idx", "seed",
                          "duration_s", "gen_fps", "vram_gb", "ram_gb", "out_path", "status"])
        stats_f.flush()

    fps_path = os.path.join(os.path.dirname(out_dir), "vbench_fps.txt")
    fps_f    = open(fps_path, "w", encoding="utf-8")
    fps_f.write("GameCraft VBench Per-Video Timing (in-process)\n")
    fps_f.write(f"frames={args.frames}  steps={args.steps}  size={args.height}x{args.width}\n")
    fps_f.write("=" * 78 + "\n")
    fps_f.write(f'{"#":>4}  {"prompt":<50}  {"s":>1}  {"dur_s":>6}  {"fps":>5}  status\n')
    fps_f.write("-" * 78 + "\n")

    if not os.path.isfile(info_json):
        print(f"[GC-VBench-IP] ERROR: VBench info JSON not found: {info_json}")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"[GC-VBench-IP] ERROR: Crop image directory not found: {image_dir}")
        sys.exit(1)

    entries = load_vbench_entries(info_json, args.image_types)
    total   = len(entries) * args.num_samples
    print(f"[GC-VBench-IP] {len(entries)} prompts × {args.num_samples} samples = {total} videos\n")

    skipped = generated = errors = done = 0
    t_start = time.time()

    for task_idx, (image_name, prompt, img_type) in enumerate(entries):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"[GC-VBench-IP] SKIP: image not found — {image_path}")
            continue

        image_stem = os.path.splitext(image_name)[0]

        for sample_idx in range(args.num_samples):
            seed     = random.randint(0, 2**31 - 1)
            out_path = os.path.join(out_dir, f"{prompt}-{sample_idx}.mp4")

            if glob.glob(os.path.join(out_dir, f"{prompt}-{sample_idx}.mp4")):
                skipped += 1
                done    += 1
                stats_w.writerow([task_idx, prompt, img_type, sample_idx, seed,
                                  "", "", "", "", out_path, "skipped"])
                stats_f.flush()
                fps_f.write(f"{task_idx+1:>4}  {prompt[:50]:<50}  {sample_idx}  {'':>6}  {'':>5}  skipped\n")
                fps_f.flush()
                continue

            pct = 100 * done / total if total else 0
            eta = ""
            if done > 0:
                secs_left = (time.time() - t_start) / done * (total - done)
                eta = f"  ETA {int(secs_left//3600):02d}h{int(secs_left%3600//60):02d}m{int(secs_left%60):02d}s"
            print(f"[GC-VBench-IP] [{done+1}/{total}  {pct:.0f}%{eta}]  "
                  f"prompt {task_idx+1}/{len(entries)}  sample {sample_idx+1}/{args.num_samples}  "
                  f"seed {seed}: {prompt[:50]}")

            tmp_dir = os.path.join(out_dir, f"_tmp_{task_idx}_{sample_idx}")
            os.makedirs(tmp_dir, exist_ok=True)

            vram_readings = []
            stop_evt      = threading.Event()
            vram_thread   = threading.Thread(target=_poll_vram, args=(stop_evt, vram_readings), daemon=True)
            vram_thread.start()

            t0 = time.time()
            try:
                tmp_mp4 = _run_sample(sampler, hymm_args, image_path, prompt, seed, tmp_dir)
                elapsed = time.time() - t0
                stop_evt.set()
                vram_thread.join(timeout=10)

                gen_fps  = args.frames / elapsed if elapsed > 0 else 0.0
                vram     = round(max(vram_readings) / 1024.0, 2) if vram_readings else ""
                ram      = round(psutil.virtual_memory().used / 1024**3, 2)

                if os.path.isfile(tmp_mp4):
                    shutil.move(tmp_mp4, out_path)
                    print(f"[GC-VBench-IP] Saved: {out_path}  ({gen_fps:.1f} gen-fps)  VRAM {vram}GB  RAM {ram}GB")
                    stats_w.writerow([task_idx, prompt, img_type, sample_idx, seed,
                                      f"{elapsed:.2f}", f"{gen_fps:.2f}", vram, ram, out_path, "ok"])
                    fps_f.write(f"{task_idx+1:>4}  {prompt[:50]:<50}  {sample_idx}  {elapsed:>6.1f}  {gen_fps:>5.2f}  ok\n")
                    generated += 1
                else:
                    print(f"[GC-VBench-IP] ERROR: mp4 not found after inference: {tmp_mp4}")
                    stats_w.writerow([task_idx, prompt, img_type, sample_idx, seed,
                                      f"{elapsed:.2f}", f"{gen_fps:.2f}", vram, ram, out_path, "error"])
                    fps_f.write(f"{task_idx+1:>4}  {prompt[:50]:<50}  {sample_idx}  {elapsed:>6.1f}  {gen_fps:>5.2f}  ERROR\n")
                    errors += 1

            except Exception as exc:
                elapsed = time.time() - t0
                stop_evt.set()
                vram_thread.join(timeout=10)
                print(f"[GC-VBench-IP] ERROR: {exc}")
                ram = round(psutil.virtual_memory().used / 1024**3, 2)
                stats_w.writerow([task_idx, prompt, img_type, sample_idx, seed,
                                  f"{elapsed:.2f}", "", "", ram, out_path, "error"])
                fps_f.write(f"{task_idx+1:>4}  {prompt[:50]:<50}  {sample_idx}  {elapsed:>6.1f}  {'':>5}  ERROR\n")
                errors += 1

            stats_f.flush()
            fps_f.flush()
            done += 1
            shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed_total = time.time() - t_start
    stats_f.close()
    fps_f.write("=" * 78 + "\n")
    fps_f.write(f"generated={generated}  skipped={skipped}  errors={errors}\n")
    fps_f.close()
    print(f"\n[GC-VBench-IP] Done — generated={generated}  skipped={skipped}  errors={errors}  "
          f"elapsed={elapsed_total/60:.1f}m")
    print(f"[GC-VBench-IP] Stats → {stats_path}")
    print(f"[GC-VBench-IP] FPS   → {fps_path}")


if __name__ == "__main__":
    main()
