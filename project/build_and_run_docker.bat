@echo off
echo Building and running TensorFlow GPU Docker container for speaker impedance project...

REM Check if Docker is installed
docker --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Docker Compose is not installed. It should be included with Docker Desktop.
    echo If not, please install it separately.
    exit /b 1
)

REM Check if NVIDIA Container Toolkit is installed
docker info | findstr "Runtimes" | findstr "nvidia" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo NVIDIA Container Toolkit not detected. Please install it from https://github.com/NVIDIA/nvidia-docker
    echo You can also try running without GPU support by modifying docker-compose.yml.
)

echo.
echo Building Docker image...
docker-compose build

echo.
echo Starting Docker container...
docker-compose up -d

echo.
echo Attaching to container...
docker exec -it speaker-impedance-tensorflow bash

echo.
echo Container session ended. Container is still running in the background.
echo To stop the container, run: docker-compose down
echo. 