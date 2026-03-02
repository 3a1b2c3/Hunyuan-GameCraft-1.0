@echo off
setlocal

:: ── configurable defaults ──────────────────────────────────────────────────
set IMAGE=asset\village.png
set PROMPT=A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky.
set NEG_PROMPT=overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.
set CKPT=weights\gamecraft_models\mp_rank_00_model_states_distill.pt
set SAVE=results_low_mem
set HEIGHT=704
set WIDTH=1216
set STEPS=8
set FRAMES=33
set CFG_SCALE=1.0
set SEED=42
set ACTIONS=w a d s
set SPEEDS=0.2 0.2 0.2 0.2
:: ──────────────────────────────────────────────────────────────────────────

:: Script directory as ROOT (strip trailing backslash)
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

:: Validate inputs
if not exist "%ROOT%\%CKPT%" (
    echo ERROR: Checkpoint not found: %ROOT%\%CKPT%
    echo Download from: https://huggingface.co/tencent/Hunyuan-GameCraft-1.0
    exit /b 1
)
if not exist "%ROOT%\%IMAGE%" (
    echo ERROR: Image not found: %ROOT%\%IMAGE%
    exit /b 1
)

mkdir "%ROOT%\%SAVE%" 2>nul

:: Single-GPU environment
set PYTHONPATH=%ROOT%
set MODEL_BASE=%ROOT%\weights\stdmodels
set DISABLE_SP=1
set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1

echo ============================================================
echo Hunyuan-GameCraft  ^|  low-memory single-GPU  ^|  Windows
echo ============================================================
echo   image   : %IMAGE%
echo   actions : %ACTIONS%  speeds: %SPEEDS%
echo   size    : %HEIGHT%x%WIDTH%   frames: %FRAMES%   steps: %STEPS%
echo   output  : %SAVE%
echo ============================================================

python "%ROOT%\hymm_sp\sample_batch.py" ^
    --image-path "%ROOT%\%IMAGE%" ^
    --prompt "%PROMPT%" ^
    --add-neg-prompt "%NEG_PROMPT%" ^
    --ckpt "%ROOT%\%CKPT%" ^
    --video-size %HEIGHT% %WIDTH% ^
    --cfg-scale %CFG_SCALE% ^
    --image-start ^
    --action-list %ACTIONS% ^
    --action-speed-list %SPEEDS% ^
    --seed %SEED% ^
    --sample-n-frames %FRAMES% ^
    --infer-steps %STEPS% ^
    --flow-shift-eval-video 5.0 ^
    --use-fp8 ^
    --save-path "%ROOT%\%SAVE%"

exit /b %ERRORLEVEL%
