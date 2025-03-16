export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd submodules
git clone https://github.com/NVIDIA/cutlass.git

cd cutlass
rm -rf build
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j 16