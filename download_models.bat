@echo off
setlocal

set REPO=tencent/Hunyuan-GameCraft-1.0
set LOCAL_DIR=%~dp0weights

echo Repo:      %REPO%
echo Local dir: %LOCAL_DIR%
echo.

huggingface-cli download %REPO% ^
    --local-dir "%LOCAL_DIR%" ^
    --exclude "*.metadata" ".cache/*"

echo.
echo Done. Weights in: %LOCAL_DIR%
endlocal
