@echo off
setlocal enabledelayedexpansion

:: ========== Configuration ==========
set NEG_PROMPT=overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.
set ACTION_LIST=w s d a
set ACTION_SPEED_LIST=0.2 0.2 0.2 0.2
set EXAMPLE_DIR=C:\workspace\world\Infinite-World\assets\example_case
set DEFAULT_PROMPT=A first-person view exploring an interactive game scene.
:: ====================================

set JOBS_DIR=%~dp0..
set PYTHONPATH=%JOBS_DIR%;%PYTHONPATH%
set MODEL_BASE=%JOBS_DIR%\weights\stdmodels
set checkpoint_path=%JOBS_DIR%\weights\gamecraft_models\mp_rank_00_model_states_distill.pt

set DISABLE_SP=1
set CPU_OFFLOAD=1
set LOCAL_RANK=
set RANK=
set WORLD_SIZE=
set MASTER_ADDR=
set MASTER_PORT=

cd /d "%JOBS_DIR%"

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format ''yyyyMMdd_HHmmss''"') do set RUN_TS=%%T
set SAVE_PATH_BASE=./results_distill/%RUN_TS%

set ACTION_COUNT=0
for %%a in (%ACTION_LIST%) do set /a ACTION_COUNT+=1

for /f %%t in ('powershell -NoProfile -Command "[int64](Get-Date).Ticks"') do set OVERALL_START=%%t
set CASE_NUM=0

goto :main

:: -------------------------------------------------------
:: Subroutine: reads RC_IMG, RC_NAME, RC_PROMPT from env
:: (avoids passing long strings as call arguments, which
::  breaks on commas/special chars; also moves ^ continuations
::  outside for-loop bodies so CMD doesn't eat them)
:: -------------------------------------------------------
:run_case
    set /a CASE_NUM+=1
    echo.
    echo === !CASE_NUM!: !RC_NAME! ===
    for /f %%t in ('powershell -NoProfile -Command "[int64](Get-Date).Ticks"') do set CASE_START=%%t

    set RC_PROMPT=!RC_PROMPT:"=!

    python hymm_sp/sample_batch.py ^
        --image-path "!RC_IMG!" ^
        --prompt "!RC_PROMPT!" ^
        --add-neg-prompt "%NEG_PROMPT%" ^
        --ckpt "%checkpoint_path%" ^
        --video-size 64 112 ^
        --cfg-scale 1.0 ^
        --image-start ^
        --action-list %ACTION_LIST% ^
        --action-speed-list %ACTION_SPEED_LIST% ^
        --seed 250160 ^
        --infer-steps 8 ^
        --use-fp8 ^
        --cpu-offload ^
        --flow-shift-eval-video 5.0 ^
        --save-path "%SAVE_PATH_BASE%/!RC_NAME!/"

    powershell -NoProfile -Command "$e=([int64](Get-Date).Ticks-!CASE_START!)/1e7; $f=%ACTION_COUNT%*33; $spf=[math]::Round($e/$f,2); $fps=[math]::Round($f/$e,3); Write-Host ('  time='+[math]::Round($e,1)+'s  frames='+$f+'  '+$spf+'s/frame  '+$fps+' fps')"
    exit /b 0

:main

:: --- priority cases first (streetview, racer) ---
for %%D in ("%EXAMPLE_DIR%\gc" "%EXAMPLE_DIR%\racer") do (
    set RC_NAME=%%~nxD
    set RC_IMG=
    for %%F in (%%~D\*.png %%~D\*.jpg) do if not defined RC_IMG set RC_IMG=%%F
    if not defined RC_IMG (
        echo Skipping !RC_NAME! - no image found
    ) else (
        set RC_PROMPT=%DEFAULT_PROMPT%
        if exist "%%~D\prompt.txt" set /p RC_PROMPT=<"%%~D\prompt.txt"
        call :run_case
    )
)

:: --- remaining subdirectory cases ---
for /d %%D in (%EXAMPLE_DIR%\*) do (
    set RC_NAME=%%~nxD
    if /i not "!RC_NAME!"=="gc" if /i not "!RC_NAME!"=="racer" (
        set RC_IMG=
        for %%F in (%%D\*.png %%D\*.jpg) do if not defined RC_IMG set RC_IMG=%%F
        if not defined RC_IMG (
            echo Skipping !RC_NAME! - no image found
        ) else (
            set RC_PROMPT=%DEFAULT_PROMPT%
            if exist "%%D\prompt.txt" set /p RC_PROMPT=<"%%D\prompt.txt"
            call :run_case
        )
    )
)

:: --- root-level images ---
for %%F in (%EXAMPLE_DIR%\*.png %EXAMPLE_DIR%\*.jpg) do (
    set RC_IMG=%%F
    set RC_NAME=%%~nF
    set RC_PROMPT=%DEFAULT_PROMPT%
    call :run_case
)

echo.
powershell -NoProfile -Command "$e=([int64](Get-Date).Ticks-%OVERALL_START%)/1e7; Write-Host ('Total: !CASE_NUM! cases  '+[math]::Round($e,1)+'s')"

exit /b 0
