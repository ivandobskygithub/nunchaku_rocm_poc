# Continuous Integration for Windows ROCm builds

This document explains how to operate the Windows ROCm GitHub Actions workflow added in `.github/workflows/windows-rocm.yml`. It covers the expected runner configuration, secrets, and artifact handling.

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

## Workflow overview

The `windows-rocm.yml` workflow executes the following steps for pushes, pull requests, and version tags (prefixed with `v`):

1. **Prepare ROCm paths** – verifies that ROCm is installed and prepends the HIP and ROCm binary folders to the `PATH` so Python can import HIP runtime DLLs.
2. **Set up Python** – installs Python 3.10 on the runner.
3. **Install project dependencies** – upgrades packaging tools, installs the project in editable mode (with `dev` extras), and ensures build tooling is present.
4. **Validate HIP availability** – runs a short Python snippet to check `torch.cuda.is_available()`, printing the HIP runtime and GPU name.
5. **Run unit tests** – executes `pytest` with failure-fast semantics.
6. **Profile ROCm examples** – runs `scripts/profile_rocm_examples.py` against `examples/v1/qwen-image-edit-2509.py` and `examples/flux.1-dev.py`, recording runtime and memory metrics to `logs/rocm_profile.json`.
7. **Build ROCm wheel** – builds a binary wheel with `python -m build`, placing the artifact in `dist\rocm`.
8. **Upload artifacts** – publishes wheels and profiling JSON as workflow artifacts.
9. **Publish release artifacts** – when a `v*` tag is pushed, attaches the wheel(s) and profile report to the GitHub Release via `softprops/action-gh-release`.

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

## Updating dependencies

- When PyTorch publishes new ROCm wheels, update the index URL in documentation and ensure the runner installs matching drivers.
- Regenerate baselines when significant framework changes occur to maintain meaningful comparisons.
- Keep the `scripts/profile_rocm_examples.py` helper in sync with new examples by updating the default list or maintaining example lists (see `@file` syntax in the script).
