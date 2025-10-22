# ROCm 7 Windows Setup for Nunchaku INT4 Inference

This guide walks through preparing a Windows 11 workstation with an AMD AI Pro R9700 (gfx1201) GPU to build and run the ROCm 7 port of Nunchaku. Follow the steps in order; many of them require administrator privileges and a reboot.

## 1. Install Windows ROCm 7 driver stack

1. Download the latest **ROCm 7 for Windows** package from AMD's official support site. Ensure the release notes list support for `gfx1201` (RDNA4).
2. Run the installer and select the HIP SDK, ROCm runtime, rocBLAS/hipBLASLt components, and the Visual Studio integration.
3. After installation, verify that the environment variables point to the ROCm SDK location (the installer defaults to `C:\Program Files\AMD\ROCm`). Note the following paths—we will reference them later:
   - `%ROCM_HOME%` (or `%HIPSDK_PATH%`)
   - `%ROCM_HOME%\bin` (contains `hipcc.exe`, `amdhip64.dll`, etc.)
   - `%ROCM_HOME%\include`
   - `%ROCM_HOME%\lib` / `%ROCM_HOME%\lib64`
4. Reboot to finalize the driver installation.

## 2. Prepare the Python environment

1. Install **Python 3.11** (64-bit) from python.org and add it to `PATH`.
2. Install **Git** and **CMake** (3.26 or newer).
3. Install the Microsoft Visual Studio 2022 Build Tools with the "Desktop development with C++" workload. This provides `cl.exe`, the Windows SDK, and MSVC runtime libraries required by hipcc.
4. Install **pipx** (optional) and update `pip`:
   ```powershell
   python -m pip install --upgrade pip virtualenv
   ```
5. (Recommended) Create an isolated virtual environment for Nunchaku:
   ```powershell
   python -m venv %USERPROFILE%\venvs\nunchaku-rocm
   %USERPROFILE%\venvs\nunchaku-rocm\Scripts\activate
   ```

## 3. Install PyTorch with ROCm support

1. Download or build a ROCm-enabled PyTorch wheel that targets Windows ROCm 7. AMD publishes preview builds in the **AMD ROCm for Windows** GitHub releases.
2. Install the ROCm PyTorch package:
   ```powershell
   pip install torch torchvision --index-url https://download.amd.com/rocm/manylinux/rocm-rel-7.0
   ```
   Replace the index URL with the actual release location if AMD moves the packages.
3. Validate that PyTorch detects the GPU:
   ```powershell
   python - <<'PY'
   import torch
   assert torch.version.hip is not None, "PyTorch build is not HIP-enabled"
   print(torch.cuda.get_device_name(0))
   PY
   ```

## 4. Fetch repository dependencies

1. Clone the repository:
   ```powershell
   git clone https://github.com/<org>/nunchaku.git
   cd nunchaku
   ```
2. Initialize submodules (for CUTLASS headers and other bundled third-party code):
   ```powershell
   git submodule update --init --recursive
   ```
3. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   If a consolidated requirements file is not available, install the packages referenced in `pyproject.toml` and `setup.py` (e.g., `packaging`, `numpy`, `einops`).

## 5. Configure environment variables

Set the HIP toolchain paths before building the extension. Replace `C:\Program Files\AMD\ROCm` if you installed to a different directory.

```powershell
setx ROCM_HOME "C:\Program Files\AMD\ROCm"
setx HIPSDK_PATH "C:\Program Files\AMD\ROCm"
setx HIP_SDK_PATH "C:\Program Files\AMD\ROCm"
setx PATH "%ROCM_HOME%\bin;%PATH%"
```

After updating `PATH`, restart the terminal to pick up the changes.

If you keep additional ROCm builds or custom libraries outside of `%ROCM_HOME%`, append them via:

```
setx ROCM_EXTRA_LIB_DIRS "D:\rocm\lib64"
setx ROCM_EXTRA_BIN_DIRS "D:\rocm\bin"
```

These variables are optional but help `setup.py` locate preview builds of `rocblaslt`/`hipblaslt` without copying DLLs into the
repository.

## 6. Build the HIP extension

1. From the repository root, build the wheel in release mode:
   ```powershell
   python -m build
   ```
   The HIP branch of `setup.py` detects `ROCM_HOME`, defines `NUNCHAKU_USE_HIP`, and enumerates the installed GPUs to emit a
   `--offload-arch=<gfx>` flag per device (multi-GPU hosts receive one entry per architecture). Override the detected list by
   exporting `NUNCHAKU_HIP_ARCHES` before invoking the build.
2. On Windows, the custom `build_ext` step bundles `amdhip64.dll`, `hiprtc.dll`, and `hiprtc-builtins.dll` into the wheel directory. Verify these files are present in `build\lib\nunchaku`.
3. Install the generated wheel into your virtual environment:
   ```powershell
   pip install dist\nunchaku-<version>-cp311-cp311-win_amd64.whl
   ```

## 7. Runtime validation checklist

1. Run the unit tests that exercise HIP runtime primitives:
   ```powershell
   pytest tests\hip --maxfail=1 --disable-warnings
   ```
   (If HIP-specific tests are not available yet, start with basic smoke tests that import `nunchaku._C`.)
2. Execute the INT4 GEMM smoke test to ensure kernel launches succeed:
   ```powershell
   python -m nunchaku.sanity.int4_gemm
   ```
3. Load a representative model (e.g., Qwen image edit) with an INT4 checkpoint:
   ```powershell
   python examples\v1\qwen-image-edit-2509.py --quant-int4 --device hip:0
   ```
4. Monitor GPU utilization with `rocminfo.exe` or the AMD GPU Performance HUD to confirm kernels target `gfx1201`.

## 8. Troubleshooting tips

- **hipcc not found**: Re-run the ROCm installer and ensure the HIP SDK component is installed. Verify `%ROCM_HOME%\bin` appears before other `clang` or `LLVM` distributions in `PATH`.
- **Missing ROCm DLLs at runtime**: Confirm that the build step copied the HIP runtime DLLs into the `nunchaku` package directory. Manually copy them from `%ROCM_HOME%\bin` if necessary.
- **Undefined references to rocBLAS/hipBLAS**: Make sure the ROCm libraries reside in `%ROCM_HOME%\lib64` and that `hipcc` can discover them. You can pass additional library directories through the `LIB` environment variable or edit `setup.py` to append custom paths.
- **PyTorch reports no HIP devices**: Check that the AMD Windows GPU driver is active (Device Manager) and that Secure Boot is either enabled with Microsoft keys or disabled, per AMD's documentation.
- **Kernel launch failures**: Set `NUNCHAKU_DEBUG_GPU=1` before running tests to surface verbose logging from the unified GPU runtime helpers.

## 9. Linux ROCm 7 quickstart

While the primary focus of this guide is Windows, the HIP toolchain in this repository also supports Linux hosts running ROCm 7. The steps below were validated on Ubuntu 22.04 with an Instinct MI300 workstation and the ROCm 7.0.2 packages.

### 9.1 Install ROCm runtime and developer packages

1. Add AMD's ROCm APT repository and install the developer stack:

   ```bash
   sudo wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
   echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.0 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dev rocblas rocblas-dev rocblaslt hipblaslt hipblaslt-dev hipfft-dev python3-venv build-essential ninja-build cmake pkg-config
   ```

2. Add your user to the `video` group and reboot to ensure the GPU devices are accessible:

   ```bash
   sudo usermod -aG video $USER
   sudo reboot
   ```

### 9.2 Configure environment variables

Add the following lines to `~/.bashrc` (adjust `/opt/rocm` if ROCm is installed elsewhere):

```bash
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm
export HIPSDK_PATH=/opt/rocm
export HIP_SDK_PATH=/opt/rocm
export PATH="$ROCM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_HOME/lib:$ROCM_HOME/lib64:$LD_LIBRARY_PATH"
# Optional: point the build system at out-of-tree libraries
export ROCM_EXTRA_LIB_DIRS="$HOME/rocm/lib"
export ROCM_EXTRA_BIN_DIRS="$HOME/rocm/bin"
```

Reload your shell (`source ~/.bashrc`) after editing the file.

### 9.3 Validate ROCm installation

Run the following commands to confirm that HIP can see the devices and toolchain:

```bash
rocminfo | grep -m1 "gfx"
hipconfig --version
python - <<'PY'
import torch
assert torch.version.hip, "PyTorch was not compiled with HIP support"
print("HIP devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
import importlib.util
import pathlib
spec = importlib.util.spec_from_file_location("nunchaku_setup", pathlib.Path("setup.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print("Detected gfx targets:", module.detect_hip_arches())
PY
```

### 9.4 Build and test the extension

1. Clone the repository and initialize submodules:

   ```bash
   git clone https://github.com/<org>/nunchaku.git
   cd nunchaku
   git submodule update --init --recursive
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip wheel build
   pip install -r requirements.txt
   ```

3. Build the HIP extension. `setup.py` will automatically detect the available `gfx` targets (via `rocminfo` and PyTorch) and enable the correct `--offload-arch` flags:

   ```bash
   python -m build
   ```

4. Install the wheel and run a smoke test:

   ```bash
   pip install dist/nunchaku-*-cp$(python -c 'import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")')-linux_x86_64.whl
   python - <<'PY'
import nunchaku
import torch
print("has_block_sparse_attention:", nunchaku.has_block_sparse_attention)
print("nunchaku._C available:", hasattr(nunchaku, "ops"))
PY
   ```

5. Optional: run the HIP smoke tests and a sample workflow.

   ```bash
   pytest tests/hip --maxfail=1 --disable-warnings
   python examples/v1/qwen-image-edit-2509.py --quant-int4 --device hip:0
   ```

### 9.5 Troubleshooting

- If `rocminfo` fails with permission errors, verify your user is in the `video` group and that Secure Boot is configured according to AMD's ROCm documentation.
- To force a specific architecture list, set `NUNCHAKU_HIP_ARCHES="gfx942,gfx940"` before invoking `python -m build`.
- Missing `rocblaslt`/`hipblaslt` libraries can be resolved by installing the `rocblaslt` and `hipblaslt` packages or by pointing `ROCM_EXTRA_LIB_DIRS` at custom build locations.
- Block-sparse attention is disabled on ROCm unless HIP kernels and the `rocblaslt`/`hipblaslt` libraries are detected. Review the build log for the `[nunchaku]` messages describing the capability check before enabling it manually via `NUNCHAKU_BLOCK_SPARSE=1`.

## 10. Future automation ideas

- Integrate the steps above into a PowerShell script or `Invoke-Build` recipe that configures the environment, runs `python -m build`, and packages the resulting wheel automatically.
- Add GitHub Actions workflows that target a self-hosted Windows ROCm runner to continuously build and smoke-test the HIP wheel.

Following this guide will leave you with a HIP-enabled Python wheel capable of running INT4 quantized models on an AMD AI Pro R9700 GPU.
