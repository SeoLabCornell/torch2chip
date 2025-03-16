#include "include/qbmw.h"
#include "include/common.h"

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/array.h>
#include <torch/extension.h>

#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/device_memory.h>
#include <cuda_runtime.h>


__global__ void quantize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    int8_t* __restrict__ output,
    int M, 
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        int index = row * K + col;
        float scaled_value = input[index] * scale[index];

        int8_t quantized_value = static_cast<int8_t>(roundf(scaled_value));

        // clamp the range
        output[index] = max(-128, min(127, quantized_value));
    }
}

torch::Tensor quantize_tensor(const torch::Tensor& input, const torch::Tensor& scale) {

    // shape of input
    int M = input.size(0);
    int K = input.size(1);

    auto output = torch::empty({M, K}, torch::dtype(torch::kInt8).device(input.device()));
    
    // get pointers
    const float* input_ptr = input.data_ptr<float>();
    const float* scale_ptr = scale.data_ptr<float>();
    int8_t* output_ptr = output.data_ptr<int8_t>();

    // define CUDA block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // launch the kernel
    quantize_kernel<<<gridDim, blockDim>>>(
        input_ptr, scale_ptr, output_ptr, M, K
    );

    return output;
}

torch::Tensor qbmw(torch::Tensor A, torch::Tensor W) {
    auto M = A.size(0);
    auto N = W.size(0);
    auto K = A.size(1);

    // automatically define the data type of output tensor C
    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementComputeEpilogue = int32_t;
    using ElementInputA = int8_t;
    using ElementInputW = int8_t;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputW = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA, ElementInputW, LayoutInputW,
        ElementOutput, LayoutOutput, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

    auto input_size = cutlass::MatrixCoord(M, K);
    auto weight_size = cutlass::MatrixCoord(K, N);
    auto output_size = cutlass::MatrixCoord(M, N);

    auto device = A.device();
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
        A.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
    
    cutlass::TensorRef<ElementInputW, LayoutInputW> weight_ref(
        W.data_ptr<int8_t>(), LayoutInputW::packed(weight_size));
    
    cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
        C.data_ptr<int32_t>(), LayoutOutput::packed(output_size));
    
    typename Gemm::Arguments arguments{
        problem_size,
        input_ref,
        weight_ref,
        out_ref,
        out_ref,
        {1, 0}, 1};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot implement");
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot initialize");
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot run");
    }

    return C;

}

torch::Tensor qbmw_scaled(torch::Tensor A, torch::Tensor W, torch::Tensor scale) {

    torch::Tensor out = qbmw(A, W);
    torch::Tensor Y = (out * scale);

    Y = Y.to(torch::kFloat16);

    return Y;
}

torch::Tensor qbmw_quantized(torch::Tensor A, torch::Tensor W, torch::Tensor scale) {
    torch::Tensor out = qbmw(A, W);
    out = out.to(torch::kFloat32);

    torch::Tensor Y = quantize_tensor(out, scale);
    return Y;
}