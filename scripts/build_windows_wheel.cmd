@echo off
setlocal enabledelayedexpansion

:: get arguments
set PYTHON_VERSION=%1
set TORCH_VERSION=%2
set CUDA_VERSION=%3

set BUILD_WITH_ROCM=0
if defined ROCM_HOME set BUILD_WITH_ROCM=1
if %BUILD_WITH_ROCM%==0 if defined HIPSDK_PATH set BUILD_WITH_ROCM=1
if %BUILD_WITH_ROCM%==0 if defined HIP_SDK_PATH set BUILD_WITH_ROCM=1

if %BUILD_WITH_ROCM%==0 (
    set CUDA_SHORT_VERSION=%CUDA_VERSION:.=%
    echo %CUDA_SHORT_VERSION%
) else (
    if not defined ROCM_HOME (
        if defined HIPSDK_PATH set ROCM_HOME=%HIPSDK_PATH%
        if defined HIP_SDK_PATH set ROCM_HOME=%HIP_SDK_PATH%
    )
    if not defined ROCM_HOME (
        echo ROCM_HOME or HIPSDK_PATH must be defined for ROCm builds.
        exit /b 1
    )
    echo Building with ROCm from %ROCM_HOME%
    set HIPSDK_PATH=%ROCM_HOME%
    set HIP_SDK_PATH=%ROCM_HOME%
    set HIP_PATH=%ROCM_HOME%
    set ROCM_PATH=%ROCM_HOME%
    set PATH=%ROCM_HOME%\bin;%PATH%
)

:: setup some variables
if "%TORCH_VERSION%"=="2.5" (
    set TORCHVISION_VERSION=0.20
    set TORCHAUDIO_VERSION=2.5
) else if "%TORCH_VERSION%"=="2.6" (
    set TORCHVISION_VERSION=0.21
    set TORCHAUDIO_VERSION=2.6
) else if "%TORCH_VERSION%"=="2.7" (
    set TORCHVISION_VERSION=0.22
    set TORCHAUDIO_VERSION=2.7
) else if "%TORCH_VERSION%"=="2.8" (
    set TORCHVISION_VERSION=0.23
    set TORCHAUDIO_VERSION=2.8
) else (
    echo TORCH_VERSION is not 2.5, 2.6, 2.7 or 2.8, no changes to versions.
)
echo setting TORCHVISION_VERSION to %TORCHVISION_VERSION% and TORCHAUDIO_VERSION to %TORCHAUDIO_VERSION%

:: conda environment name
set ENV_NAME=build_env_%PYTHON_VERSION%_%TORCH_VERSION%
echo Using conda environment: %ENV_NAME%

:: create conda environment
call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
call conda activate %ENV_NAME%

:: install dependencies
call pip install ninja setuptools wheel
if %BUILD_WITH_ROCM%==0 (
    call pip install --no-cache-dir torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url "https://download.pytorch.org/whl/cu%CUDA_SHORT_VERSION%/"
) else (
    echo Expecting ROCm PyTorch wheels to be preinstalled or available on PATH.
)

:: set environment variables
set NUNCHAKU_INSTALL_MODE=ALL
set NUNCHAKU_BUILD_WHEELS=1

:: cd to the parent directory
cd /d "%~dp0.."
if exist build rd /s /q build
if exist dist rd /s /q dist

:: set up Visual Studio compilation environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
set DISTUTILS_USE_SDK=1

:: build wheels
pip wheel . --no-deps --no-build-isolation -w dist

:: exit conda
call conda deactivate
call conda remove -y -n %ENV_NAME% --all

echo Build complete!
