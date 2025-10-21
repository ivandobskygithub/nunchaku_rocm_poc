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


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            if isinstance(ext.extra_compile_args, dict):
                ext.extra_compile_args.setdefault("cxx", [])
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


class HIPExtension(CUDAExtension):
    def __init__(self, *args, **kwargs):
        extra_compile_args = kwargs.get("extra_compile_args", {})
        if "hipcc" in extra_compile_args:
            hipcc_args = extra_compile_args.pop("hipcc")
            extra_compile_args.setdefault("nvcc", []).extend(hipcc_args)
            kwargs["extra_compile_args"] = extra_compile_args
        super().__init__(*args, **kwargs)


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

HIP_BIN_DIRS = [
    os.path.join(ROCM_HOME, "bin"),
]

HIP_RUNTIME_DLLS = [
    "amdhip64.dll",
    "hiprtc.dll",
    "hiprtc-builtins.dll",
]


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
    return flags


def cuda_device_flags(debug_enabled: bool, sm_targets: list[str]) -> list[str]:
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
    return flags


def hip_device_flags(debug_enabled: bool) -> list[str]:
    flags = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-std=c++20",
        "--offload-arch=gfx1201",
        f"--rocm-path={ROCM_HOME}",
    ]
    flags.extend(["-O0", "-g"] if debug_enabled else ["-O3"])
    return flags


if __name__ == "__main__":
    fp = open("nunchaku/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    torch_version = torch.__version__.split("+")[0]
    torch_major_minor_version = ".".join(torch_version.split(".")[:2])
    if "dev" in version:
        version = version + date.today().strftime("%Y%m%d")  # data
    version = version + "+torch" + torch_major_minor_version

    ROOT_DIR = os.path.dirname(__file__)

    INCLUDE_DIRS = [
        "src",
        "third_party/cutlass/include",
        "third_party/json/include",
        "third_party/mio/include",
        "third_party/spdlog/include",
        "third_party/Block-Sparse-Attention/csrc/block_sparse_attn",
    ]

    INCLUDE_DIRS = [os.path.join(ROOT_DIR, dir) for dir in INCLUDE_DIRS]

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

    host_flags = host_compile_flags(debug_mode)

    extra_include_dirs = INCLUDE_DIRS.copy()
    library_dirs: list[str] = []

    if USING_ROCM:
        hip_includes = [path for path in HIP_INCLUDE_DIRS if os.path.isdir(path)]
        hip_libs = [path for path in HIP_LIBRARY_DIRS if os.path.isdir(path)]
        extra_include_dirs.extend(hip_includes)
        library_dirs.extend(hip_libs)

    ExtensionImpl = HIPExtension if USING_ROCM else CUDAExtension

    if USING_ROCM:
        device_flags = hip_device_flags(debug_mode)
        sm_targets: list[str] = []
    else:
        sm_targets = get_sm_targets()
        print(f"Detected SM targets: {sm_targets}", file=sys.stderr)
        assert len(sm_targets) > 0, "No SM targets found"
        device_flags = cuda_device_flags(debug_mode, sm_targets)

    extra_compile_args = {
        "cxx": host_flags,
        "nvcc": device_flags,
    }

    extension_kwargs = dict(
        name="nunchaku._C",
        sources=[
            "nunchaku/csrc/pybind.cpp",
            "src/interop/torch.cpp",
            "src/activation.cpp",
            "src/layernorm.cpp",
            "src/Linear.cpp",
            *ncond("src/FluxModel.cpp"),
            *ncond("src/SanaModel.cpp"),
            "src/Serialization.cpp",
            "src/Module.cpp",
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim64_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim64_bf16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim128_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim128_bf16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim64_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim64_bf16_sm80.cu"),
            *ncond(
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim128_fp16_sm80.cu"
            ),
            *ncond(
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim128_bf16_sm80.cu"
            ),
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
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/flash_api.cpp"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/flash_api_adapter.cpp"),
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=extra_include_dirs,
    )

    if library_dirs:
        extension_kwargs["library_dirs"] = library_dirs

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
