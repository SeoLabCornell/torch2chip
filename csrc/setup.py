import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10
print(cuda_arch)

setup(
    name="t2c_gemm",
    ext_modules=[
        CppExtension(
            name="t2c_gemm",
            sources=[
                "kernel/bmm.cu",
                "kernel/bmw.cu",
                "kernel/qbmw.cu",
                "kernel/bcmm.cu",
                "kernel/bindings.cpp",
            ],
            include_dirs=["t2c_gemm/kernel/include"],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    f"-DCUDA_ARCH={cuda_arch}"
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)