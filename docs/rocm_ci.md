# Continuous Integration for ROCm builds

This document explains how to operate the ROCm GitHub Actions workflows added in `.github/workflows/windows-rocm.yml` and the companion Linux job. It covers the expected runner configuration, secrets, and artifact handling for both platforms.

## Self-hosted runner requirements

| Component | Requirement |
| --- | --- |
| Hardware | AMD AI Pro R9700 GPU, ≥ 32 GB RAM, ≥ 200 GB SSD |
| OS | Windows 11 Pro/Enterprise 22H2+ with AMD Software: PRO Edition (ROCm 7) |
| Toolchain | Python 3.10+, Visual Studio Build Tools 2022 (Desktop C++ workload), Git, CMake 3.26+, Ninja |
| Networking | Outbound HTTPS (github.com, pypi.org, download.pytorch.org, download.microsoft.com) |

### Runner labels

Register the runner with the labels `self-hosted`, `Windows`, and `ROCm` so the workflow can target it:

```powershell
.\config.cmd --url https://github.com/<org>/<repo> --token <runner-token> --labels "Windows,ROCm"
```

Ensure the runner service account has permission to install device drivers and access the AMD ROCm directories under `C:\Program Files\AMD\ROCm`.

### Environment hardening

- Keep GPU firmware and AMD Software: PRO Edition updated to the latest ROCm-compatible version.
- Pin Python and toolchain versions using [winget](https://learn.microsoft.com/windows/package-manager/) or Chocolatey so the workflow receives deterministic builds.
- Enable disk cleanup (e.g., Storage Sense) to prevent caches from filling the SSD.

### Linux ROCm runner

| Component | Requirement |
| --- | --- |
| Hardware | AMD Instinct MI300 or MI210 GPU, ≥ 64 GB RAM, ≥ 200 GB SSD |
| OS | Ubuntu 22.04 LTS with ROCm 7.0.2 repositories enabled |
| Toolchain | Python 3.10+, `rocm-dev`, `rocblas{,lt}`, `hipblaslt`, `hipfft`, CMake 3.26+, Ninja, `python3-venv` |
| Networking | Outbound HTTPS (github.com, pypi.org, repo.radeon.com) |

Register the runner with labels `self-hosted`, `Linux`, and `ROCm`. The runner service account must belong to the `video` group and source `/opt/rocm` in its shell profile (export `ROCM_HOME`, `HIP_PATH`, `ROCM_EXTRA_LIB_DIRS`, `ROCM_EXTRA_BIN_DIRS`, and update `LD_LIBRARY_PATH`).

## Workflow overview

The `windows-rocm.yml` workflow executes the following steps for pushes, pull requests, and version tags (prefixed with `v`). A second job, `linux-rocm`, mirrors the build/test stages using the Ubuntu runner described above.

1. **Prepare ROCm paths** – verifies that ROCm is installed and prepends the HIP and ROCm binary folders to the `PATH` so Python can import HIP runtime DLLs.
2. **Set up Python** – installs Python 3.10 on the runner.
3. **Install project dependencies** – upgrades packaging tools, installs the project in editable mode (with `dev` extras), and ensures build tooling is present.
4. **Validate HIP availability** – runs a short Python snippet to check `torch.cuda.is_available()`, printing the HIP runtime, GPU name, and the `gfx` list returned by `setup.detect_hip_arches()`.
5. **Run unit tests** – executes `pytest` with failure-fast semantics.
6. **Profile ROCm examples** – runs `scripts/profile_rocm_examples.py` against `examples/v1/qwen-image-edit-2509.py` and `examples/flux.1-dev.py`, recording runtime and memory metrics to `logs/rocm_profile.json`.
7. **Build ROCm wheel** – builds a binary wheel with `python -m build`, placing the artifact in `dist\rocm`.
8. **Upload artifacts** – publishes wheels and profiling JSON as workflow artifacts.
9. **Publish release artifacts** – when a `v*` tag is pushed, attaches the wheel(s) and profile report to the GitHub Release via `softprops/action-gh-release`.
10. **Linux verification** – builds the HIP wheel on Ubuntu, asserts `nunchaku.has_block_sparse_attention` matches the expected backend, and runs the HIP pytest subset.

## Device coverage matrix

- **Windows ROCm** – Targets RDNA4 (e.g., AMD AI Pro R9700, `gfx1201`). The workflow consumes the GPU exposed by the Windows runner and bundles HIP runtime DLLs into the wheel artifact.
- **Linux ROCm** – Targets Instinct accelerators (`gfx90a`, `gfx942`). Populate the runner with one or more GPUs and let `setup.detect_hip_arches()` enumerate each `gfx` identifier. If multiple devices are present, export `NUNCHAKU_HIP_ARCHES="gfx90a,gfx942"` so the build emits one `--offload-arch` flag per architecture.

Document the GPUs attached to each runner inside the repository wiki or README so release managers can verify that new hardware remains represented in CI.

## Secrets and permissions

No additional secrets are required beyond the default `GITHUB_TOKEN`. If you plan to push wheels to an external package index, add the appropriate API token as a repository secret and extend the workflow to authenticate during the upload step.

The workflow expects the runner service to have access to `hipcc` and other ROCm binaries. If they are installed in a non-default location, override the `ROCM_PATH` environment variable via repository or organization-level variables.

## Baseline comparisons

To compare ROCm performance with NVIDIA or CPU runs, generate baseline JSON metrics (using the same script) and store them as workflow artifacts in another pipeline. During analysis, download both artifacts and compare the `duration_delta_*` and `peak_memory_delta_mb` fields recorded in the JSON output.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| Workflow fails on `hipcc --version` | Confirm ROCm tooling is installed and accessible to the runner’s service account. Update `ROCM_PATH` if using a custom install location. |
| `torch.cuda.is_available()` returns `False` | Reboot the runner, verify GPU is detected in Device Manager, and ensure AMD Software: PRO Edition installed ROCm components. |
| Example profiling step exits early | Check `logs/rocm_profile.json` for the `status` and `error` fields. Missing assets (e.g., model weights) may require caching or stub downloads. |
| Wheels not attached to release | Confirm the workflow ran on a tag starting with `v` and that the `GITHUB_TOKEN` has `contents: write` permission (default for public repos). |
| Linux job cannot locate rocBLASLt | Install the `rocblaslt` and `hipblaslt` packages or set `ROCM_EXTRA_LIB_DIRS` to include the custom library build directory. |
| Linux job reports empty HIP target list | Ensure `rocminfo` is installed and on the runner `PATH`. The validation step imports `setup.detect_hip_arches()`; inspect its output to confirm the expected `gfx` identifiers or set `NUNCHAKU_HIP_ARCHES`. |
| Block-sparse attention disabled unexpectedly | Check that HIP-specific block-sparse sources are present and that `rocblaslt`/`hipblaslt` resolved from the directories listed in `ROCM_EXTRA_LIB_DIRS` / `ROCM_EXTRA_BIN_DIRS`. |

## Updating dependencies

- When PyTorch publishes new ROCm wheels, update the index URL in documentation and ensure the runner installs matching drivers.
- Regenerate baselines when significant framework changes occur to maintain meaningful comparisons.
- Keep the `scripts/profile_rocm_examples.py` helper in sync with new examples by updating the default list or maintaining example lists (see `@file` syntax in the script).
