@echo off
echo ===== cuDNN Installation Helper for CUDA 12.8 =====
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrator privileges.
    echo Please right-click and select "Run as administrator".
    pause
    exit /b 1
)

REM Check CUDA_PATH environment variable
if "%CUDA_PATH%"=="" (
    echo CUDA_PATH environment variable is not set.
    echo Setting it to the default location for CUDA 12.8...
    setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
)

echo Using CUDA installation at: %CUDA_PATH%
if not exist "%CUDA_PATH%" (
    echo ERROR: CUDA installation not found at %CUDA_PATH%
    echo Please install CUDA 12.8 first.
    pause
    exit /b 1
)

echo.
echo This script will help you install cuDNN files for CUDA 12.8.
echo.
echo You need to download cuDNN v9.x for CUDA 12.x from NVIDIA:
echo https://developer.nvidia.com/cudnn-downloads
echo.
echo After downloading, extract the ZIP file and place this script in the
echo extracted folder (the folder containing 'bin', 'include', and 'lib' directories).
echo.

REM Check if cuDNN files exist in current directory
set CUDNN_FOUND=0
if exist "include\cudnn.h" (
    set CUDNN_FOUND=1
) else (
    echo ERROR: cuDNN files not found in current directory.
    echo Please run this script from the directory where you extracted cuDNN.
    echo The directory should contain 'bin', 'include', and 'lib' folders.
    pause
    exit /b 1
)

echo Found cuDNN files in current directory.
echo.
echo Installing cuDNN files to CUDA directory...

REM Create directories if they don't exist
if not exist "%CUDA_PATH%\include" mkdir "%CUDA_PATH%\include"
if not exist "%CUDA_PATH%\lib\x64" mkdir "%CUDA_PATH%\lib\x64"
if not exist "%CUDA_PATH%\bin" mkdir "%CUDA_PATH%\bin"

REM Copy files
echo Copying include files...
copy /Y "include\cudnn*.h" "%CUDA_PATH%\include" >nul
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy include files.
    pause
    exit /b 1
)

echo Copying library files...
if exist "lib\x64\cudnn*.lib" (
    copy /Y "lib\x64\cudnn*.lib" "%CUDA_PATH%\lib\x64" >nul
) else if exist "lib\cudnn*.lib" (
    copy /Y "lib\cudnn*.lib" "%CUDA_PATH%\lib\x64" >nul
)
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy library files.
    pause
    exit /b 1
)

echo Copying DLL files...
if exist "bin\cudnn*.dll" (
    copy /Y "bin\cudnn*.dll" "%CUDA_PATH%\bin" >nul
)
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy DLL files.
    pause
    exit /b 1
)

echo.
echo cuDNN files have been successfully installed!
echo.
echo Next steps:
echo 1. Restart your command prompt or IDE
echo 2. Run 'python fix_gpu_detection.py' to verify the installation
echo 3. Run 'python test_gpu.py' to test GPU functionality
echo.

pause 