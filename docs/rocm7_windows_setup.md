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

## 6. Build the HIP extension

1. From the repository root, build the wheel in release mode:
   ```powershell
   python -m build
   ```
   The HIP branch of `setup.py` detects `ROCM_HOME`, defines `NUNCHAKU_USE_HIP`, and invokes `hipcc` with the `--offload-arch=gfx1201` flag.
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

## 9. Future automation ideas

- Integrate the steps above into a PowerShell script or `Invoke-Build` recipe that configures the environment, runs `python -m build`, and packages the resulting wheel automatically.
- Add GitHub Actions workflows that target a self-hosted Windows ROCm runner to continuously build and smoke-test the HIP wheel.

Following this guide will leave you with a HIP-enabled Python wheel capable of running INT4 quantized models on an AMD AI Pro R9700 GPU.
