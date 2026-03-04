@echo off
:: Hunyuan-GameCraft-1.0 - VBench Batch Inference
:: Loops over VBench crop images, generates num_samples videos per prompt,
:: skips already-generated outputs.
:: Usage: run_low_mem_vbench.bat [output_base] [num_samples] [image_types] [num_chunks]
:: Example: run_low_mem_vbench.bat results_low_mem 5 "abstract,background"

setlocal enabledelayedexpansion

:: --help
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h"     goto :help
if /i "%~1"=="/?"     goto :help
goto :run

:help
echo.
echo Hunyuan-GameCraft-1.0 - VBench Batch Inference
echo.
echo Usage:
echo   run_low_mem_vbench.bat [output_base] [num_samples] [image_types]
echo.
echo Arguments (positional, all optional):
echo   1  output_base    Base output directory            (default: results_low_mem)
echo   2  num_samples    Videos to generate per prompt   (default: 5)
echo   3  image_types    Comma-separated type filter      (default: all)
echo                       e.g. "abstract,background,scenery"
echo.
echo Hardcoded settings (edit top of this file to change):
echo   CKPT             Checkpoint path
echo   HEIGHT / WIDTH   Video resolution
echo   STEPS            Inference steps
echo   FRAMES           Frames per video
echo   CFG_SCALE        Classifier-free guidance scale
echo   SEED             Base random seed (sample i uses SEED+i)
echo   ACTIONS          Action sequence  (default: w a d s)
echo   SPEEDS           Speed per action (default: 0.2 0.2 0.2 0.2)
echo.
echo Notes:
echo   - VBench requires 5 samples per prompt (num_samples=5)
echo   - Already-generated videos are skipped automatically
echo   - Outputs: {output_base}\videos\{prompt}-{0..N-1}.mp4
echo   - Log:     {output_base}\vbench_run.log
echo   - Stats:   {output_base}\vbench_stats.txt
echo.
echo Example:
echo   run_low_mem_vbench.bat results_low_mem 5 "abstract,background"
echo.
exit /b 0

:run
:: ── configurable defaults ──────────────────────────────────────────────────
set CKPT=weights\gamecraft_models\mp_rank_00_model_states_distill.pt
set NEG_PROMPT=overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.
set HEIGHT=720
set WIDTH=960
set STEPS=5
set FRAMES=161
set CFG_SCALE=1.0
set /a SEED=%RANDOM% * 32768 + %RANDOM%
set ACTIONS=w a d s
set SPEEDS=0.2 0.2 0.2 0.2
set RESOLUTION=1-1
:: ──────────────────────────────────────────────────────────────────────────

:: Parameters
set OUTPUT_BASE=%~1
if "%OUTPUT_BASE%"=="" set OUTPUT_BASE=results_low_mem
set NUM_SAMPLES=%~2
if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=5
set IMAGE_TYPES=%~3
if "%IMAGE_TYPES%"=="" set IMAGE_TYPES=scenery,indoor

set VBENCH_OUTPUT_DIR=%OUTPUT_BASE%\videos
set STATS_FILE=%OUTPUT_BASE%\vbench_stats.txt

set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value 2^>nul') do set _DT=%%a
set LOG_FILE=%OUTPUT_BASE%\vbench_run_%_DT:~0,8%_%_DT:~8,6%.log

:: Validate checkpoint
if not exist "%ROOT%\%CKPT%" (
    echo ERROR: Checkpoint not found: %ROOT%\%CKPT%
    echo Download from: https://huggingface.co/tencent/Hunyuan-GameCraft-1.0
    exit /b 1
)

echo ============================================================
echo Hunyuan-GameCraft  ^|  VBench batch  ^|  Windows
echo ============================================================
echo   output    : %VBENCH_OUTPUT_DIR%
echo   samples   : %NUM_SAMPLES%
if not "%IMAGE_TYPES%"=="" echo   types     : %IMAGE_TYPES%
echo   size      : %HEIGHT%x%WIDTH%   frames: %FRAMES%   steps: %STEPS%
echo   actions   : %ACTIONS%  speeds: %SPEEDS%
echo ============================================================

:: Snapshot GPU before
set GPU_INFO_BEFORE=N/A
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "skip=1 tokens=1,2,3 delims=, " %%a in ('nvidia-smi --query-gpu^=name^,memory.used^,memory.total --format^=csv^,noheader') do (
        set GPU_INFO_BEFORE=%%a  used=%%b  total=%%c
    )
)

:: Record start time
set START_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a START_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)

:: Build optional args
set OPTIONAL_ARGS=--num_samples %NUM_SAMPLES% --seed %SEED% --resolution %RESOLUTION%
if not "%IMAGE_TYPES%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --image_types "%IMAGE_TYPES%"

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

echo.
echo [GC-VBench] Generating %NUM_SAMPLES% samples per prompt...
python "%ROOT%\scripts\gc_vbench_batch.py" ^
    --vbench_output_dir "%ROOT%\%VBENCH_OUTPUT_DIR%" ^
    --ckpt "%ROOT%\%CKPT%" ^
    --neg_prompt "%NEG_PROMPT%" ^
    --height %HEIGHT% --width %WIDTH% ^
    --steps %STEPS% --frames %FRAMES% --cfg_scale %CFG_SCALE% ^
    --actions %ACTIONS% --speeds %SPEEDS% ^
    %OPTIONAL_ARGS% ^
    > "%ROOT%\%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
type "%ROOT%\%LOG_FILE%"
echo [GC-VBench] Done. Exit: %EXIT_CODE%

:: Record end time
set END_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a END_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)
set /a ELAPSED=END_S-START_S
if %ELAPSED% lss 0 set /a ELAPSED+=86400
set /a ELAPSED_H=ELAPSED/3600
set /a ELAPSED_M=(ELAPSED%%3600)/60
set /a ELAPSED_SS=ELAPSED%%60

:: Snapshot GPU after
set GPU_INFO_AFTER=N/A
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "skip=1 tokens=1,2,3 delims=, " %%a in ('nvidia-smi --query-gpu^=name^,memory.used^,memory.total --format^=csv^,noheader') do (
        set GPU_INFO_AFTER=%%a  used=%%b  total=%%c
    )
)

echo ============================================================
echo Done. Elapsed: %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s  Exit: %EXIT_CODE%
echo Stats: %STATS_FILE%
echo ============================================================

(
    echo GameCraft VBench Batch Stats
    echo ============================
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
    echo Seed:           %SEED%
    echo.
    echo === GPU ===
    echo GPU before:     %GPU_INFO_BEFORE%
    echo GPU after:      %GPU_INFO_AFTER%
    echo.
    echo === Output ===
    echo Output base:    %OUTPUT_BASE%
    echo VBench videos:  %VBENCH_OUTPUT_DIR%
    echo Log:            %LOG_FILE%
    echo FPS log:        %OUTPUT_BASE%\vbench_fps.txt
    echo Stats CSV:      %OUTPUT_BASE%\vbench_stats.csv
) > "%STATS_FILE%"

exit /b %EXIT_CODE%
