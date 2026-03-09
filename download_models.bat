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
endlocal
