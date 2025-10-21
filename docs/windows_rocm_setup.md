# Windows ROCm 7 Environment Setup for AMD AI Pro R9700

This guide documents the steps required to bring up a Windows 11 ROCm 7 environment on a workstation equipped with an AMD AI Pro R9700 accelerator. It focuses on enabling PyTorch ROCm binaries (or source builds) and making HIP runtime libraries available to Python workloads.

## 1. Hardware and OS prerequisites

- **GPU**: AMD AI Pro R9700 (Navi 31 based) with at least 48 GB VRAM.
- **CPU**: A modern x86_64 CPU with support for AVX2 and virtualization extensions.
- **Operating System**: Windows 11 Pro/Enterprise 22H2 or later with the latest updates.
- **Storage**: ≥ 200 GB free SSD space for ROCm SDK, build caches, and sample data.
- **Memory**: ≥ 64 GB system RAM.
- **BIOS Settings**: Enable Above 4G decoding and Resizable BAR if available.

## 2. Install AMD Software: PRO Edition with ROCm 7

1. Download the **AMD Software: PRO Edition** package that includes ROCm 7 support for Windows from [the AMD professional drivers portal](https://www.amd.com/en/support/professional-graphics).
2. Install the driver package with **Factory Reset** unchecked so existing device settings are preserved.
3. During installation, enable the optional **ROCm** components. This deploys HIP runtimes, ROCm math libraries (rocBLAS, hipSPARSE, MIOpen, etc.), and developer tools under `C:\Program Files\AMD\ROCm`.
4. Reboot after installation and verify driver functionality using **AMD Software → Performance → Metrics** or `C:\Program Files\AMD\CNext\CNext\RadeonSoftware.exe`.

## 3. Configure the developer environment

Install development prerequisites:

- **Python** ≥ 3.10 (recommend the official installer with "Add Python to PATH" enabled).
- **Microsoft Visual Studio Build Tools 2022** with the following workloads:
  - Desktop development with C++
  - C++ CMake tools for Windows
  - MSVC v143 build tools
- **Git for Windows**
- **CMake** ≥ 3.26 (if not included with Visual Studio).
- **Ninja** build system (optional but recommended for faster builds).

### 3.1. Environment variables

Add the following to the system-level environment (Control Panel → System → Advanced system settings → Environment Variables):

- `ROCM_PATH=C:\Program Files\AMD\ROCm`
- `HIP_PATH=%ROCM_PATH%\hip`
- `PYTORCH_ROCM_ARCH=gfx1102` (for the R9700’s RDNA3 compute unit).
- Append `%ROCM_PATH%\bin` and `%HIP_PATH%\bin` to the `PATH` variable.

Open a new **x64 Native Tools Command Prompt for VS 2022** to pick up the changes. Validate the HIP runtime via:

```bat
where hipcc
hipcc --version
rocminfo
```

## 4. Install PyTorch ROCm binaries

AMD publishes ROCm-enabled PyTorch wheels for Windows under the `rocm` extra index. Install them in a fresh virtual environment (recommended: `python -m venv %USERPROFILE%\venvs\nunchaku-rocm`):

```bat
python -m venv %USERPROFILE%\venvs\nunchaku-rocm
%USERPROFILE%\venvs\nunchaku-rocm\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0
```

> **Note:** The `--pre` flag is required until ROCm 7 wheels are promoted to stable.

Validate the installation and HIP runtime visibility to Python:

```python
>>> import torch
>>> torch.version.hip
'7.0.0'
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'AMD AI Pro R9700'
```

## 5. Building PyTorch from source (optional)

If you need custom PyTorch builds—for example to incorporate experimental patches—use the ROCm toolchain:

```bat
set CMAKE_GENERATOR=Ninja
set PYTORCH_BUILD_VERSION=2.3.0
set PYTORCH_BUILD_NUMBER=1
set USE_ROCM=1
set HIP_PATH=%ROCM_PATH%\hip
set USE_MIOPEN=1
set USE_MAGMA=0
set PYTORCH_ROCM_ARCH=gfx1102

:: Clone the PyTorch repo and initialize submodules
set SRC_DIR=%USERPROFILE%\src\pytorch
if not exist %SRC_DIR% git clone --recursive https://github.com/pytorch/pytorch %SRC_DIR%
cd /d %SRC_DIR%
python tools\build_libtorch.py
python setup.py bdist_wheel
```

The generated wheel(s) will be located under `dist\`. Install them into a virtual environment and verify as in the previous section.

## 6. Ensuring HIP libraries are importable from Python

When running Python applications, ensure the ROCm DLLs are discoverable:

- Keep the ROCm directories at the front of the `PATH` environment variable in shells or automation scripts.
- For isolated environments (e.g., GitHub Actions self-hosted runners), prepend the ROCm paths:

  ```powershell
  $env:PATH = "C:\\Program Files\\AMD\\ROCm\\bin;C:\\Program Files\\AMD\\ROCm\\hip\\bin;" + $env:PATH
  ```

- If packaging Python applications, bundle a PowerShell activation script that exports the same `PATH` adjustments before launching workloads.

## 7. Validating with nunchaku examples

After installing dependencies (see the [Profiling](#8-profiling-nunchaku-examples-on-rocm) section for additional tools), clone this repository and install it editable:

```bat
cd %SRC_DIR%\nunchaku_rocm_poc
python -m pip install -e .[dev]
```

Run a quick smoke test to verify HIP runtime visibility:

```bat
python -m torch.utils.collect_env | findstr HIP
python -c "import torch; print(torch.cuda.get_device_name())"
```

## 8. Profiling nunchaku examples on ROCm

1. Install optional profiling utilities:

   ```bat
   python -m pip install psutil pandas
   winget install --id AMD.RadeonSoftware
   ```

2. Use the `scripts\profile_rocm_examples.py` helper (added in this change set) to run selected examples and capture execution metrics:

   ```bat
   python scripts\profile_rocm_examples.py \
       --examples examples\v1\qwen-image-edit-2509.py examples\flux.1-dev.py \
       --output logs\rocm_profile.json
   ```

   The script reports runtime, peak GPU memory usage, and exposes the HIP runtime version to help compare against NVIDIA or CPU runs.

3. Compare results against a reference JSON file created on other hardware:

   ```bat
   python scripts\profile_rocm_examples.py --examples @ci\nvidia_baseline.txt \
       --output logs\rocm_profile.json --baseline logs\nvidia_profile.json
   ```

## 9. Known limitations

- ROCm for Windows currently supports a limited set of professional GPUs (AI Pro R9700/Pro W7900). Consumer RDNA3 parts are unsupported.
- Some CUDA-specific kernels may not yet have optimized HIP equivalents; expect longer compile times for custom kernels on first run.
- Mixed-precision (FP8/BF16) performance is still maturing. Monitor release notes for updates.
- PyTorch extension builds require MSVC toolchain components. Ensure `cl.exe` is on PATH when compiling custom ops.
- ROCm debug and profiling tools (such as rocprofiler) have limited Windows support compared to Linux.

## 10. Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `hipErrorNoBinaryForGpu` | Confirm `PYTORCH_ROCM_ARCH=gfx1102` and reinstall PyTorch ROCm wheels with matching version. |
| `ImportError: HIP runtime not found` | Ensure `%ROCM_PATH%\bin` and `%HIP_PATH%\bin` precede Python directories in `PATH`. |
| Example scripts fail to find DirectML | Remove any conflicting DirectML packages; ROCm kernels should be preferred. |
| Slow first-run performance | Warm up the kernels by running example once; the HIP runtime caches compiled binaries under `%LOCALAPPDATA%\AMD\HIP`. |

For additional details, refer to AMD’s official [ROCm documentation](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html) and PyTorch’s [ROCm notes](https://pytorch.org/docs/stable/notes/rocm.html).
