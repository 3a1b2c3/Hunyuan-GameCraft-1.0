@echo off
cd /d "%~dp0.."
:: Hunyuan-GameCraft-1.0 - VBench Batch Inference (distill checkpoint)
:: Usage: run_distill_vbench.bat [output_base] [num_samples] [image_types]

setlocal enabledelayedexpansion

:: ── configurable defaults ──────────────────────────────────────────────────
set CKPT=weights\gamecraft_models\mp_rank_00_model_states_distill.pt
set NEG_PROMPT=overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.
set HEIGHT=720
set WIDTH=960
set STEPS=5
set FRAMES=161
set CFG_SCALE=1.0
set ACTIONS=w a d s
set SPEEDS=0.2 0.2 0.2 0.2
set RESOLUTION=1-1
:: ──────────────────────────────────────────────────────────────────────────

set OUTPUT_BASE=%~1
if "%OUTPUT_BASE%"=="" set OUTPUT_BASE=results_distill
set NUM_SAMPLES=%~2
if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=5
set IMAGE_TYPES=%~3
if "%IMAGE_TYPES%"=="" set IMAGE_TYPES=scenery,indoor

set VBENCH_OUTPUT_DIR=%OUTPUT_BASE%\videos
set STATS_FILE=%OUTPUT_BASE%\vbench_stats.txt

set ROOT=%~dp0..
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value 2^>nul') do set _DT=%%a
set LOG_FILE=%ROOT%\%OUTPUT_BASE%\vbench_run_%_DT:~0,8%_%_DT:~8,6%.log

if not exist "%ROOT%\%CKPT%" (
    echo ERROR: Checkpoint not found: %ROOT%\%CKPT%
    exit /b 1
)

echo ============================================================
echo Hunyuan-GameCraft  ^|  VBench batch  ^|  distill
echo ============================================================
echo   output    : %VBENCH_OUTPUT_DIR%
echo   samples   : %NUM_SAMPLES%
echo   types     : %IMAGE_TYPES%
echo   size      : %HEIGHT%x%WIDTH%   frames: %FRAMES%   steps: %STEPS%
echo   actions   : %ACTIONS%  speeds: %SPEEDS%
echo ============================================================

set START_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a START_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)

set OPTIONAL_ARGS=--num_samples %NUM_SAMPLES% --resolution %RESOLUTION% --cpu_offload
if not "%IMAGE_TYPES%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --image_types "%IMAGE_TYPES%"

echo.
echo [GC-VBench] Generating %NUM_SAMPLES% samples per prompt...
python "%ROOT%\scripts\gc_vbench_batch.py" ^
    --vbench_output_dir "%ROOT%\%VBENCH_OUTPUT_DIR%" ^
    --ckpt "%ROOT%\%CKPT%" ^
    --neg_prompt "%NEG_PROMPT%" ^
    --height %HEIGHT% --width %WIDTH% ^
    --steps %STEPS% --frames %FRAMES% --cfg_scale %CFG_SCALE% ^
    --actions %ACTIONS% --speeds %SPEEDS% ^
    --log_file "%LOG_FILE%" ^
    %OPTIONAL_ARGS%
set EXIT_CODE=%ERRORLEVEL%
echo [GC-VBench] Done. Exit: %EXIT_CODE%

set END_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a END_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)
set /a ELAPSED=END_S-START_S
if %ELAPSED% lss 0 set /a ELAPSED+=86400
set /a ELAPSED_H=ELAPSED/3600
set /a ELAPSED_M=(ELAPSED%%3600)/60
set /a ELAPSED_SS=ELAPSED%%60

echo ============================================================
echo Done. Elapsed: %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s  Exit: %EXIT_CODE%
echo Stats: %STATS_FILE%
echo ============================================================

(
    echo GameCraft VBench Batch Stats ^(distill^)
    echo =======================================
    echo Date:           %DATE%
    echo Start:          %START_TIME%
    echo End:            %END_TIME%
    echo Elapsed:        %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s ^(%ELAPSED%s^)
    echo Exit code:      %EXIT_CODE%
    echo.
    echo === Settings ===
    echo Checkpoint:     %CKPT%
    echo Num samples:    %NUM_SAMPLES%
    echo Image types:    %IMAGE_TYPES%
    echo Resolution:     %RESOLUTION%
    echo Size:           %HEIGHT%x%WIDTH%
    echo Steps:          %STEPS%
    echo Frames:         %FRAMES%
    echo CPU offload:    yes
    echo.
    echo === Output ===
    echo Output base:    %OUTPUT_BASE%
    echo VBench videos:  %VBENCH_OUTPUT_DIR%
    echo Log:            %LOG_FILE%
    echo Stats CSV:      %OUTPUT_BASE%\vbench_stats.csv
) > "%STATS_FILE%"

exit /b %EXIT_CODE%
