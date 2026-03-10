@echo off
cd /d "%~dp0"
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
echo   - Per-video CSV: {output_base}\vbench_stats.csv  (written by gc_vbench_batch.py)
   - Run summary:   {output_base}\vbench_run_summary.csv
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
set SEED=42
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
set STATS_FILE=%OUTPUT_BASE%\vbench_run_summary.csv

set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value 2^>nul') do set _DT=%%a
set LOG_FILE=%ROOT%\%OUTPUT_BASE%\vbench_run_%_DT:~0,8%_%_DT:~8,6%.log

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
set VRAM_USED_BEFORE=N/A
set VRAM_TOTAL=N/A
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "skip=1 tokens=1,2 delims=, " %%a in ('nvidia-smi --query-gpu^=memory.used^,memory.total --format^=csv^,noheader^,nounits') do (
        set VRAM_USED_BEFORE=%%a
        set VRAM_TOTAL=%%b
    )
)

:: Snapshot RAM before
set RAM_FREE_BEFORE_MB=N/A
set RAM_TOTAL_MB=N/A
for /f "tokens=2 delims==" %%a in ('wmic OS get FreePhysicalMemory /value 2^>nul') do if not "%%a"=="" set /a RAM_FREE_BEFORE_MB=%%a/1024
for /f "tokens=2 delims==" %%a in ('wmic OS get TotalVisibleMemorySize /value 2^>nul') do if not "%%a"=="" set /a RAM_TOTAL_MB=%%a/1024

:: Record start time
set START_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a START_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)

:: Build optional args
set OPTIONAL_ARGS=--num_samples %NUM_SAMPLES% --seed %SEED% --resolution %RESOLUTION%
if not "%IMAGE_TYPES%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --image_types "%IMAGE_TYPES%"

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

echo.
echo [GC-VBench] Generating %NUM_SAMPLES% samples per prompt...
python "%ROOT%\scripts\gc_vbench_inprocess.py" ^
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

:: Record end time
set END_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a END_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)
set /a ELAPSED=END_S-START_S
if %ELAPSED% lss 0 set /a ELAPSED+=86400
set /a ELAPSED_H=ELAPSED/3600
set /a ELAPSED_M=(ELAPSED%%3600)/60
set /a ELAPSED_SS=ELAPSED%%60

:: Snapshot GPU after
set VRAM_USED_AFTER=N/A
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "skip=1 tokens=1 delims=, " %%a in ('nvidia-smi --query-gpu^=memory.used --format^=csv^,noheader^,nounits') do (
        set VRAM_USED_AFTER=%%a
    )
)

:: Snapshot RAM after
set RAM_FREE_AFTER_MB=N/A
for /f "tokens=2 delims==" %%a in ('wmic OS get FreePhysicalMemory /value 2^>nul') do if not "%%a"=="" set /a RAM_FREE_AFTER_MB=%%a/1024

:: Count generated videos and estimate FPS
set VIDEO_COUNT=0
for /f %%a in ('dir /b /s "%VBENCH_OUTPUT_DIR%\*.mp4" 2^>nul ^| find /c /v ""') do set VIDEO_COUNT=%%a
set /a TOTAL_FRAMES_GEN=VIDEO_COUNT*FRAMES
set EST_FPS=0
if %ELAPSED% gtr 0 set /a EST_FPS=TOTAL_FRAMES_GEN/ELAPSED

echo ============================================================
echo Done. Elapsed: %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s  Exit: %EXIT_CODE%
echo Stats: %STATS_FILE%
echo ============================================================

:: Write CSV header if file does not exist yet
if not exist "%STATS_FILE%" (
    echo date,start,end,elapsed_s,exit_code,ckpt,num_samples,image_types,height,width,steps,frames,seed,vram_used_before_mib,vram_total_mib,vram_used_after_mib,ram_free_before_mb,ram_total_mb,ram_free_after_mb,videos_generated,est_fps,output_base,log_file >> "%STATS_FILE%"
)

:: Append one data row
echo %DATE%,%START_TIME%,%END_TIME%,%ELAPSED%,%EXIT_CODE%,%CKPT%,%NUM_SAMPLES%,%IMAGE_TYPES%,%HEIGHT%,%WIDTH%,%STEPS%,%FRAMES%,%SEED%,%VRAM_USED_BEFORE%,%VRAM_TOTAL%,%VRAM_USED_AFTER%,%RAM_FREE_BEFORE_MB%,%RAM_TOTAL_MB%,%RAM_FREE_AFTER_MB%,%VIDEO_COUNT%,%EST_FPS%,%OUTPUT_BASE%,%LOG_FILE% >> "%STATS_FILE%"

exit /b %EXIT_CODE%
