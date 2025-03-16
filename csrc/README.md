# Customized MatMul Kernel of Torch2Chip

The customized cuda kernel is built based on CUTLASS, with the support of multi-dimensional matrix multiplication.

### Installation

Make sure you have NVCC 12.1 or 12.4 installed. Start from the home of `torch2chip`.

```
cd csrc
source environment.sh
bash build_cutlass.sh
pip install .
```
The dedicated kernels will be installed as the `t2c_gemm` package to your anaconda environment. 

### Usage

> Version = 0.1.0

Support INT8 matrix multiplication between 3-D and 2-D tensors (`_QBaseLinear`), 4-D x 4-D tensors (`BatchHeadIntMatMul`). 


**[Warning]**: Current MatMul kernel doesn't support the MXINT quantizer due to the fine-grained group-wise shifting. We will update the dedicated CUDA kernel implementation soon, stay tunned. 