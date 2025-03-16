#include "include/bmw.h"
#include "include/common.h"
#include "cutlass/core_io.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/host_tensor.h"

torch::Tensor bmw_int8(torch::Tensor A, torch::Tensor W, torch::Tensor alpha) {
    int batch_size = A.size(0);
    int M = A.size(1);
    int N = W.size(0);
    int K = A.size(2);

    // automatically define the data type of output tensor C
    auto C = torch::empty({batch_size, M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    // define the leading dimension of each matrix
    int lda = A.size(2);
    int ldb = W.size(1);
    int ldc = C.size(2);

    // define the memory layout
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutW = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    // define data type and accumulation precision
    using FormatInputA = int8_t;
    using FormatInputW = int8_t;
    using FormatOutputC = float;
    using FormatAccumulator = int32_t;
    using ElementComputeEpilogue = float;

    // device-dependent architecture
    #if CUDA_ARCH >= 800
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<FormatOutputC, 128/cutlass::sizeof_bits<FormatOutputC>::value, FormatAccumulator, ElementComputeEpilogue>;

        using Gemm = cutlass::gemm::device::GemmBatched<
        FormatInputA, LayoutA, FormatInputW, LayoutW, FormatOutputC, LayoutC, FormatAccumulator, cutlass::arch::OpClassTensorOp, 
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>, EpilogueOp>;

    #elif CUDA_ARCH >= 750
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<FormatOutputC, 128/cutlass::sizeof_bits<FormatOutputC>::value, FormatAccumulator, ElementComputeEpilogue>;
        
        using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        FormatInputA, FormatInputW, FormatOutputC, FormatAccumulator>;
        
        using Gemm = cutlass::gemm::device::GemmBatched<
        FormatInputA, LayoutA, FormatInputW, LayoutW, FormatOutputC, LayoutC, FormatAccumulator, cutlass::arch::OpClassTensorOp, 
        cutlass::arch::Sm75, DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape, DefaultGemmCfg::InstructionShape, EpilogueOp>;

    #elif CUDA_ARCH >= 700
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<FormatOutputC, 1, FormatAccumulator, ElementComputeEpilogue>;

        using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm70
        FormatInputA, FormatInputW, FormatOutputC, FormatAccumulator>;

        using Gemm = cutlass::gemm::device::GemmBatched<
        FormatInputA, LayoutA, FormatInputW, LayoutW, FormatOutputC, LayoutC, FormatAccumulator, cutlass::arch::OpClassSimt,
        cutlass::arch::Sm70, DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape, DefaultGemmCfg::InstructionShape, EpilogueOp>;

    #else
        #error "Unsupported GPU type"
    #endif

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // stride between two matrices within each batch
    long long int batch_stride_A = M * K;
    long long int batch_stride_W = 0;
    long long int batch_stride_C = M * N;

    // Define the operation of Gemm
    Gemm gemm_op;

    // Argument of Gemm Op
    typename Gemm::Arguments arguments{
        {M, N, K},
        {A.data_ptr<FormatInputA>(), lda}, batch_stride_A,
        {W.data_ptr<FormatInputW>(), ldb}, batch_stride_W,
        {C.data_ptr<FormatOutputC>(), ldc}, batch_stride_C,
        {C.data_ptr<FormatOutputC>(), ldc}, batch_stride_C,
        {1.0f, 0},
        batch_size
    };

    // request  extra space for GEMM operation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot implement");
    }

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot initialize");
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot run");
    }

    torch::Tensor Y = (C * alpha);
    return Y;

}
