@echo off
setlocal

set REPO=tencent/Hunyuan-GameCraft-1.0
set LOCAL_DIR=%~dp0weights

echo Repo:      %REPO%
echo Local dir: %LOCAL_DIR%
echo.

python download_model.py

echo.
echo Done. Weights in: %LOCAL_DIR%
echo.
echo To run inference, set MODEL_BASE before launching:
echo   set MODEL_BASE=%LOCAL_DIR%
endlocal
