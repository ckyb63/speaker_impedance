@echo off
echo Setting up environment variables for CUDA 12.8 and cuDNN 9.7.1...

REM Set CUDA_PATH environment variable
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
echo CUDA_PATH set to %CUDA_PATH%

REM Add CUDA paths to PATH environment variable
set CUDA_BIN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
set CUDA_LIBNVVP_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
set CUDA_CUPTI_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\CUPTI\lib64
set CUDA_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include

REM Add to PATH for current session
set PATH=%CUDA_BIN_PATH%;%CUDA_LIBNVVP_PATH%;%CUDA_CUPTI_PATH%;%CUDA_INCLUDE_PATH%;%PATH%

REM Add to PATH permanently
setx PATH "%CUDA_BIN_PATH%;%CUDA_LIBNVVP_PATH%;%CUDA_CUPTI_PATH%;%CUDA_INCLUDE_PATH%;%PATH%"

echo Environment variables set for current session and future sessions.
echo.
echo Please restart your command prompt or IDE for changes to take effect.
echo.
echo Verifying CUDA installation...
where nvcc
if %ERRORLEVEL% EQU 0 (
    echo CUDA compiler (nvcc) found in PATH.
    nvcc --version
) else (
    echo CUDA compiler (nvcc) not found in PATH. Please check your CUDA installation.
)

echo.
echo Checking for NVIDIA GPU...
nvidia-smi
if %ERRORLEVEL% NEQ 0 (
    echo Could not run nvidia-smi. Please check your NVIDIA driver installation.
)

echo.
echo Next steps:
echo 1. Restart your command prompt or IDE
echo 2. Run: python setup_cuda_compatibility.py
echo 3. Run: python verify_cuda12_setup.py
echo 4. If everything looks good, run: python main.py

pause 