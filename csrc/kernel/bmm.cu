#include "include/bmm.h"
#include "include/common.h"
#include "cutlass/core_io.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/host_tensor.h"

// the return type of the function is torch Tensor
torch::Tensor bmm_int8(torch::Tensor A, torch::Tensor B, float alpha) {
    int batch_size = A.size(0);
    int M = A.size(1);
    int N = B.size(1);
    int K = A.size(2);

    // pad the height and width of the input matrices
    int int_pad_A = (M + 7) / 8 * 8;
    int int_pad_B = (N + 7) / 8 * 8;

    torch::Tensor A_pad;
    torch::Tensor B_pad;

    if (int_pad_A > M) {
        A_pad = torch::cat({A, torch::zeros({batch_size, int_pad_A - M, K}, A.options())}, 1);
    } else {
        A_pad = A;
    }

    if (int_pad_B > N){
        B_pad = torch::cat({B, torch::zeros({batch_size, int_pad_B - N, K}, B.options())}, 1);
    } else {
        B_pad = B;
    }

    // automatically define the data type of output tensor C
    auto C = torch::empty({batch_size, int_pad_A, int_pad_B}, torch::dtype(torch::kFloat32).device(A_pad.device()));

    // define the leading dimension of each matrix
    int lda = A_pad.size(2);
    int ldb = B_pad.size(2);
    int ldc = C.size(2);

    // define the layout of memory storage for each matrix
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    // define data type and accumulation precision
    using FormatInputA = int8_t;
    using FormatInputB = int8_t;
    using FormatOutputC = float;
    using FormatAccumulator = int32_t;
    using ElementComputeEpilogue = float;

    // deivce-dependent definition
    #if CUDA_ARCH >= 800
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<FormatOutputC, 128/cutlass::sizeof_bits<FormatOutputC>::value, FormatAccumulator, ElementComputeEpilogue>;
        
        using Gemm = cutlass::gemm::device::GemmBatched<
        FormatInputA, LayoutA, FormatInputB, LayoutB, FormatOutputC, LayoutC, FormatAccumulator, cutlass::arch::OpClassTensorOp, 
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>, EpilogueOp>;
    
    #elif CUDA_ARCH >= 750
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<FormatOutputC, 128/cutlass::sizeof_bits<FormatOutputC>::value, FormatAccumulator, ElementComputeEpilogue>;
        
        using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        FormatInputA, FormatInputB, FormatOutputC, FormatAccumulator>;
        
        using Gemm = cutlass::gemm::device::GemmBatched<
        FormatInputA, LayoutA, FormatInputB, LayoutB, FormatOutputC, LayoutC, FormatAccumulator, cutlass::arch::OpClassTensorOp, 
        cutlass::arch::Sm75, DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape, DefaultGemmCfg::InstructionShape, EpilogueOp>;

    #elif CUDA_ARCH >= 700
        using EpilogueOp = cutlass::epilogue::thread::LinearCombination<FormatOutputC, 1, FormatAccumulator, ElementComputeEpilogue>;
        
        using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm70
        FormatInputA, FormatInputB, FormatOutputC, FormatAccumulator>;

        using Gemm = cutlass::gemm::device::GemmBatched<
        FormatInputA, LayoutA, FormatInputB, LayoutB, FormatOutputC, LayoutC, FormatAccumulator, cutlass::arch::OpClassSimt,
        cutlass::arch::Sm70, DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape, DefaultGemmCfg::InstructionShape, EpilogueOp>;
    #else
        #error "Unsupported GPU type"
    #endif

    // stride between two matrices within each batch
    long long int batch_stride_A = int_pad_A * K;
    long long int batch_stride_B = int_pad_B * K;
    long long int batch_stride_C = int_pad_A * int_pad_B;

    // Define the operation of Gemm
    Gemm gemm_op;
    // Argument of Gemm Op
    typename Gemm::Arguments arguments{
        {int_pad_A, int_pad_B, K},      {A_pad.data_ptr<FormatInputA>(), lda},
        batch_stride_A, {B_pad.data_ptr<FormatInputB>(), ldb},
        batch_stride_B, {C.data_ptr<FormatOutputC>(), ldc},
        batch_stride_C, {C.data_ptr<FormatOutputC>(), ldc},
        batch_stride_C, {alpha, 0},
        batch_size};

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

    if (int_pad_A > M){
        return C.slice(1, 0, M).slice(2, 0, N);
    } else {
        return C;
    }
}

