import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import setuptools
import torch
from packaging import version as packaging_version
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
from torch.utils.cpp_extension import ROCM_HOME as TORCH_ROCM_HOME
import torch.utils.cpp_extension as torch_cpp_extension


ROOT_DIR = Path(__file__).resolve().parent


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            if isinstance(ext.extra_compile_args, dict):
                ext.extra_compile_args.setdefault("cxx", [])
                if USING_ROCM:
                    ext.extra_compile_args.setdefault("hipcc", [])
                ext.extra_compile_args.setdefault("nvcc", [])
        super().build_extensions()
        if USING_ROCM and IS_WINDOWS:
            self._bundle_rocm_runtime()

    def _bundle_rocm_runtime(self) -> None:
        runtime_dirs = [Path(p) for p in HIP_BIN_DIRS if Path(p).is_dir()]
        if not runtime_dirs:
            print(
                "[nunchaku] Unable to locate ROCm runtime bin directory; skipping DLL bundling.",
                file=sys.stderr,
            )
            return

        target_dir = Path(self.build_lib) / "nunchaku"
        target_dir.mkdir(parents=True, exist_ok=True)

        for dll in HIP_RUNTIME_DLLS:
            for directory in runtime_dirs:
                candidate = directory / dll
                if candidate.exists():
                    shutil.copy2(candidate, target_dir / dll)
                    break
            else:
                print(f"[nunchaku] Warning: {dll} not found under {runtime_dirs}", file=sys.stderr)


def HIPExtension(*args, **kwargs):
    extra_compile_args = kwargs.get("extra_compile_args", {})
    hipcc_args = extra_compile_args.get("hipcc")
    if hipcc_args is not None:
        hipcc_args = list(hipcc_args)
        extra_compile_args["hipcc"] = hipcc_args
        extra_compile_args.setdefault("nvcc", [])
        extra_compile_args["nvcc"].extend(hipcc_args)
        kwargs["extra_compile_args"] = extra_compile_args
    return CUDAExtension(*args, **kwargs)


def get_sm_targets() -> list[str]:
    nvcc_path = os.path.join(CUDA_HOME, "bin/nvcc") if CUDA_HOME else "nvcc"
    try:
        nvcc_output = subprocess.check_output([nvcc_path, "--version"]).decode()
        match = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", nvcc_output)
        if match:
            nvcc_version = match.group(2)
        else:
            raise Exception("nvcc version not found")
        print(f"Found nvcc version: {nvcc_version}")
    except:
        raise Exception("nvcc not found")

    support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")

    install_mode = os.getenv("NUNCHAKU_INSTALL_MODE", "FAST")
    if install_mode == "FAST":
        ret = []
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            sm = f"{capability[0]}{capability[1]}"
            if sm == "120" and support_sm120:
                sm = "120a"
            assert sm in ["75", "80", "86", "89", "120a"], f"Unsupported SM {sm}"
            if sm not in ret:
                ret.append(sm)
    else:
        assert install_mode == "ALL"
        ret = ["75", "80", "86", "89"]
        if support_sm120:
            ret.append("120a")
    return ret


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _parse_hip_arch_list(raw: str) -> list[str]:
    tokens = re.split(r"[\s,;]+", raw)
    return [token for token in tokens if token.startswith("gfx")]


def detect_hip_arches() -> list[str]:
    override = os.environ.get("NUNCHAKU_HIP_ARCHES")
    if override:
        arches = _parse_hip_arch_list(override)
        if arches:
            return _unique_preserving_order(arches)

    detected: list[str] = []

    try:
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                arch = getattr(props, "gcnArchName", None)
                if arch and arch.startswith("gfx"):
                    detected.append(arch)
    except Exception:
        pass

    def _extract_gfx_from_output(output: str) -> list[str]:
        matches = re.findall(r"gfx[0-9a-zA-Z]+", output)
        return matches

    for command in (["rocminfo"], ["rocminfo", "--hip"]):
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        detected.extend(_extract_gfx_from_output(output))

    try:
        hipconfig_output = subprocess.check_output(["hipconfig", "--amdgpu-target"], stderr=subprocess.STDOUT, text=True)
        detected.extend(_parse_hip_arch_list(hipconfig_output))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    detected = _unique_preserving_order(detected)
    if detected:
        return detected

    fallback_arch = os.environ.get("NUNCHAKU_DEFAULT_HIP_ARCH", "gfx1201")
    print(
        f"[nunchaku] Unable to detect HIP offload targets; defaulting to {fallback_arch}",
        file=sys.stderr,
    )
    return [fallback_arch]


def _hip_block_sparse_backend_available() -> bool:
    override = os.environ.get("NUNCHAKU_HIP_BLOCK_SPARSE_DIR")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))

    candidates.extend(
        [
            ROOT_DIR / "third_party" / "Block-Sparse-Attention" / "csrc" / "block_sparse_attn" / "hip",
            ROOT_DIR
            / "third_party"
            / "Block-Sparse-Attention"
            / "csrc"
            / "block_sparse_attn"
            / "src"
            / "hip",
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = candidate.expanduser().resolve()
        except (FileNotFoundError, OSError):
            continue
        if path.is_file():
            return True
        if path.is_dir():
            try:
                next(path.rglob("*"))
            except (StopIteration, OSError):
                continue
            else:
                return True
    return False


def detect_block_sparse_support(sm_targets: list[str], hip_arches: list[str], is_rocm: bool) -> bool:
    override = os.environ.get("NUNCHAKU_BLOCK_SPARSE")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}

    if is_rocm:
        hip_arch_bases = {arch.split(":", 1)[0] for arch in hip_arches}
        supported_hip_arches = hip_arch_bases & HIP_BLOCK_SPARSE_SUPPORTED_ARCHES
        backend_available = _hip_block_sparse_backend_available()
        if supported_hip_arches and backend_available:
            return True

        reasons: list[str] = []
        if not hip_arch_bases:
            reasons.append("no HIP devices detected")
        elif not supported_hip_arches:
            reasons.append(
                "supported gfx architecture not detected (requires one of "
                + ", ".join(sorted(HIP_BLOCK_SPARSE_SUPPORTED_ARCHES))
                + ")"
            )
        if not backend_available:
            reasons.append("HIP block-sparse backend sources not found")
        if reasons:
            print(
                "[nunchaku] Disabling block-sparse attention on ROCm: "
                + "; ".join(reasons),
                file=sys.stderr,
            )
        return False

    for target in sm_targets:
        digits = "".join(ch for ch in target if ch.isdigit())
        if not digits:
            continue
        try:
            sm_int = int(digits)
        except ValueError:
            continue
        if sm_int >= 80:
            return True

    return False


ROCM_HOME = (
    os.environ.get("ROCM_HOME")
    or os.environ.get("HIPSDK_PATH")
    or os.environ.get("HIP_SDK_PATH")
    or TORCH_ROCM_HOME
)

USING_ROCM = bool(ROCM_HOME)
IS_WINDOWS = os.name == "nt"

HIP_INCLUDE_DIRS: list[str] = []
HIP_LIBRARY_DIRS: list[str] = []
HIP_BIN_DIRS: list[str] = []
HIP_RUNTIME_DLLS: list[str] = []

HIP_LIBRARIES = [
    "amdhip64",
    "hiprtc",
    "hipfft",
    "rocblas",
    "hipblas",
]

HIP_OPTIONAL_LIBRARIES: dict[str, list[str]] = {
    "rocblaslt": ["librocblaslt.so*", "rocblaslt.dll", "librocblaslt.a"],
    "hipblaslt": ["libhipblaslt.so*", "hipblaslt.dll", "libhipblaslt.a"],
}

HIP_OPTIONAL_LIBRARY_DIRS = [
    ("rocblas", ["lib", "lib64"]),
    ("rocblaslt", ["lib", "lib64"]),
    ("hipblaslt", ["lib", "lib64"]),
]

HIP_BLOCK_SPARSE_SUPPORTED_ARCHES = {
    "gfx90a",
    "gfx940",
    "gfx941",
    "gfx942",
    "gfx1200",
    "gfx1201",
}

if USING_ROCM:
    torch_cpp_extension.IS_HIP_EXTENSION = True
    os.environ.setdefault("ROCM_HOME", ROCM_HOME)
    os.environ.setdefault("ROCM_PATH", ROCM_HOME)
    os.environ.setdefault("HIP_PATH", ROCM_HOME)
    os.environ.setdefault("HIPSDK_PATH", ROCM_HOME)
    os.environ.setdefault("HIP_SDK_PATH", ROCM_HOME)

    HIP_INCLUDE_DIRS = [
        os.path.join(ROCM_HOME, "include"),
        os.path.join(ROCM_HOME, "hip", "include"),
        os.path.join(ROCM_HOME, "llvm", "include"),
    ]

    HIP_LIBRARY_DIRS = [
        os.path.join(ROCM_HOME, "lib"),
        os.path.join(ROCM_HOME, "lib64"),
        os.path.join(ROCM_HOME, "hip", "lib"),
        os.path.join(ROCM_HOME, "hip", "lib64"),
    ]

    for subdir, candidates in HIP_OPTIONAL_LIBRARY_DIRS:
        for candidate in candidates:
            HIP_LIBRARY_DIRS.append(os.path.join(ROCM_HOME, subdir, candidate))

    for extra_dir in os.environ.get("ROCM_EXTRA_LIB_DIRS", "").split(os.pathsep):
        if extra_dir:
            HIP_LIBRARY_DIRS.append(extra_dir)

    HIP_BIN_DIRS = [
        os.path.join(ROCM_HOME, "bin"),
    ]

    for extra_dir in os.environ.get("ROCM_EXTRA_BIN_DIRS", "").split(os.pathsep):
        if extra_dir:
            HIP_BIN_DIRS.append(extra_dir)

    HIP_RUNTIME_DLLS = [
        "amdhip64.dll",
        "hiprtc.dll",
        "hiprtc-builtins.dll",
    ]


def hip_library_exists(search_dirs: list[str], patterns: list[str]) -> bool:
    for directory in search_dirs:
        path = Path(directory)
        if not path.is_dir():
            continue
        for pattern in patterns:
            if any(path.glob(pattern)):
                return True
    return False


def format_define(symbol: str) -> str:
    return f"/D{symbol}" if IS_WINDOWS else f"-D{symbol}"


def format_undef(symbol: str) -> str:
    return f"/U{symbol}" if IS_WINDOWS else f"-U{symbol}"


def host_compile_flags(debug_enabled: bool) -> list[str]:
    defines = ["ENABLE_BF16=1", "BUILD_NUNCHAKU=1"]
    flags: list[str] = []
    if IS_WINDOWS:
        flags.extend(["/std:c++20", "/Zc:__cplusplus", "/FS", "/DNOMINMAX", "/EHsc"])
        flags.append("/MDd" if debug_enabled else "/MD")
        flags.extend(["/Zi", "/Od"] if debug_enabled else ["/O2"])
    else:
        flags.extend(["-std=c++20", "-fPIC", "-fvisibility=hidden"])
        flags.extend(["-O0", "-g"] if debug_enabled else ["-O3"])
    flags.append(format_undef("NDEBUG"))
    flags.extend(format_define(d) for d in defines)
    if USING_ROCM:
        flags.append(format_define("NUNCHAKU_USE_HIP"))
    return flags


def cuda_device_flags(debug_enabled: bool, sm_targets: list[str], block_sparse_enabled: bool) -> list[str]:
    flags = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-std=c++20",
        "-Xcudafe",
        "--diag_suppress=20208",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=--allow-expensive-optimizations=true",
    ]
    flags.extend(["-G", "-O0"] if debug_enabled else ["-O3"])
    if os.getenv("NUNCHAKU_BUILD_WHEELS", "0") == "0":
        flags.append("--generate-line-info")
    for target in sm_targets:
        flags += ["-gencode", f"arch=compute_{target},code=sm_{target}"]
    if IS_WINDOWS:
        flags += ["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/FS", "-Xcompiler", "/bigobj"]
    flags.append(f"-DNUNCHAKU_WITH_BLOCK_SPARSE={'1' if block_sparse_enabled else '0'}")
    return flags


def hip_device_flags(debug_enabled: bool, hip_arches: list[str], block_sparse_enabled: bool) -> list[str]:
    if not hip_arches:
        hip_arches = detect_hip_arches()

    flags = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-DNUNCHAKU_USE_HIP=1",
        "-std=c++20",
        f"--rocm-path={ROCM_HOME}",
    ]
    for arch in hip_arches:
        flags.append(f"--offload-arch={arch}")
    flags.extend(["-O0", "-g"] if debug_enabled else ["-O3"])
    flags.append(f"-DNUNCHAKU_WITH_BLOCK_SPARSE={'1' if block_sparse_enabled else '0'}")
    return flags


if __name__ == "__main__":
    fp = open("nunchaku/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    torch_version = torch.__version__.split("+")[0]
    torch_major_minor_version = ".".join(torch_version.split(".")[:2])
    if "dev" in version:
        version = version + date.today().strftime("%Y%m%d")  # data
    version = version + "+torch" + torch_major_minor_version

    ROOT_DIR = Path(__file__).resolve().parent

    INCLUDE_DIRS = [
        "src",
        "third_party/cutlass/include",
        "third_party/json/include",
        "third_party/mio/include",
        "third_party/spdlog/include",
        "third_party/Block-Sparse-Attention/csrc/block_sparse_attn",
    ]

    INCLUDE_DIRS = [ROOT_DIR / dir for dir in INCLUDE_DIRS]
    INCLUDE_DIRS = [str(path) for path in INCLUDE_DIRS]

    DEBUG = False

    def ncond(s) -> list:
        if DEBUG:
            return []
        else:
            return [s]

    def cond(s) -> list:
        if DEBUG:
            return [s]
        else:
            return []

    debug_mode = DEBUG

    extra_include_dirs = INCLUDE_DIRS.copy()
    library_dirs: list[str] = []
    libraries: list[str] = []

    hip_arches: list[str] = []
    sm_targets: list[str] = []
    block_sparse_available = False
    hip_optional_libraries_found: dict[str, bool] = {}

    if USING_ROCM:
        hip_includes = [path for path in HIP_INCLUDE_DIRS if os.path.isdir(path)]
        hip_libs = [path for path in HIP_LIBRARY_DIRS if os.path.isdir(path)]
        extra_include_dirs.extend(hip_includes)
        library_dirs.extend(hip_libs)
        libraries.extend(HIP_LIBRARIES)
        for lib_name, patterns in HIP_OPTIONAL_LIBRARIES.items():
            found = hip_library_exists(hip_libs, patterns)
            hip_optional_libraries_found[lib_name] = found
            if found:
                libraries.append(lib_name)

    ExtensionImpl = HIPExtension if USING_ROCM else CUDAExtension

    if USING_ROCM:
        hip_arches = detect_hip_arches()
        print(f"[nunchaku] Detected HIP targets: {hip_arches}", file=sys.stderr)
        block_sparse_available = detect_block_sparse_support([], hip_arches, True)
        if block_sparse_available:
            required_optional_libs = {"rocblaslt", "hipblaslt"}
            missing_optional = [lib for lib in required_optional_libs if not hip_optional_libraries_found.get(lib, False)]
            if missing_optional:
                block_sparse_available = False
                print(
                    "[nunchaku] Disabling block-sparse attention on ROCm: missing optional libraries "
                    + ", ".join(sorted(missing_optional)),
                    file=sys.stderr,
                )
        device_flags = hip_device_flags(debug_mode, hip_arches, block_sparse_available)
        sm_targets = []
    else:
        sm_targets = get_sm_targets()
        print(f"Detected SM targets: {sm_targets}", file=sys.stderr)
        assert len(sm_targets) > 0, "No SM targets found"
        block_sparse_available = detect_block_sparse_support(sm_targets, [], False)
        device_flags = cuda_device_flags(debug_mode, sm_targets, block_sparse_available)

    host_flags = host_compile_flags(debug_mode)
    block_sparse_define = "NUNCHAKU_WITH_BLOCK_SPARSE=1" if block_sparse_available else "NUNCHAKU_WITH_BLOCK_SPARSE=0"
    host_flags.append(format_define(block_sparse_define))
    print(
        f"[nunchaku] Block-sparse attention backend {'enabled' if block_sparse_available else 'disabled'}",
        file=sys.stderr,
    )

    extra_compile_args = {"cxx": host_flags}
    if USING_ROCM:
        extra_compile_args["hipcc"] = device_flags
        extra_compile_args["nvcc"] = []
    else:
        extra_compile_args["nvcc"] = device_flags

    sources = [
        "nunchaku/csrc/pybind.cpp",
        "src/interop/torch.cpp",
        "src/activation.cpp",
        "src/layernorm.cpp",
        "src/Linear.cpp",
        *ncond("src/FluxModel.cpp"),
        *ncond("src/SanaModel.cpp"),
        "src/Serialization.cpp",
        "src/Module.cpp",
        "src/kernels/activation_kernels.cu",
        "src/kernels/layernorm_kernels.cu",
        "src/kernels/misc_kernels.cu",
        "src/kernels/zgemm/gemm_w4a4.cu",
        "src/kernels/zgemm/gemm_w4a4_test.cu",
        "src/kernels/zgemm/gemm_w4a4_launch_fp16_int4.cu",
        "src/kernels/zgemm/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
        "src/kernels/zgemm/gemm_w4a4_launch_fp16_fp4.cu",
        "src/kernels/zgemm/gemm_w4a4_launch_bf16_int4.cu",
        "src/kernels/zgemm/gemm_w4a4_launch_bf16_fp4.cu",
        "src/kernels/zgemm/gemm_w8a8.cu",
        "src/kernels/zgemm/attention.cu",
        "src/kernels/dwconv.cu",
        "src/kernels/gemm_batched.cu",
        "src/kernels/gemm_f16.cu",
        "src/kernels/awq/gemm_awq.cu",
        "src/kernels/awq/gemv_awq.cu",
    ]

    if block_sparse_available:
        sources.extend(
            [
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim64_fp16_sm80.cu",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim64_bf16_sm80.cu",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim128_fp16_sm80.cu",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim128_bf16_sm80.cu",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/flash_api.cpp",
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/flash_api_adapter.cpp",
            ]
        )

    extension_kwargs = dict(
        name="nunchaku._C",
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=extra_include_dirs,
    )

    if library_dirs:
        extension_kwargs["library_dirs"] = library_dirs
    if libraries:
        extension_kwargs["libraries"] = libraries

    nunchaku_extension = ExtensionImpl(**extension_kwargs)

    setuptools.setup(
        name="nunchaku",
        version=version,
        packages=setuptools.find_packages(),
        ext_modules=[nunchaku_extension],
        cmdclass={"build_ext": CustomBuildExtension},
        include_package_data=True,
        package_data={"nunchaku": ["*.dll"]},
    )
