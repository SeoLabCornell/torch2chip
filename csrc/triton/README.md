# Triton Implementation of MX Quantization

This document introduces the mx scaling and matrix multiplication kernel implemented by Triton. 

## MX implementation of scaling

By default, the shared exponent of microscaling is computed along the reduction dimension. Given the input matrix with shape of $(M, K)$, the shape of the scale is $(M, K // G)$, where $G$ is the group size of MX. 


### Basics of Triton Kernel

#### **Question #1**: What is `tl.program_id`? 
> Triton `program_id` is similar to the concept of `blockIdx` in CUDA. Each `program_id` tells you the which program (in the specific axis) is currently executing. 

The `program_id` returns a single integer value that indicates which program is currently running on the given axis/dimension. Everytime your kernel is invoked for one "program" along that dimension. Each program will process a set of elements with size of `BLOCK_SIZE`. 


**Understand the computation hierarchy of GPU**
- **Grid:** On the top-level, we define a grid of thread blocks (CUDA) or programs (Triton).
- **Block:** The block is defined as `blockIdx` (CUDA) or `program` (Triton). Each block or prgram is responsible for a portion of data. 
- **Thread:** Inside each block or program, we further split down into smaller lanes or parallelisms. In CUDA, threads are defined as `threadIdx.x`, `threadIdx.y`, and `threadIdx.z`. Different from CUDA, Triton follows the SPMD (single program, multiple data) fashion, which often start with `tl.arange`. 

#### Memory Hierarchy of GPU

##### Global memory
- Global memory is the large region of device memory accessible by all blocks. The memory can be accessed by passing pointers into the kernel. 

- For Triton, we receive pointers (e.g., x_ptr, y_ptr) that refer to device memory. The values are fetched from the global memory via `tl.load()` or `tl.store()`

##### L2 / L1 Cache
- L2 Cache located at each Streaming Multiprocess (SM), shared across all SMs on the GPU. 
    - Properly loading items on L2 cache helps reduce global memory access. 
- L1 Cache located within each SM, faster but smaller than L2 Cache. 

A detailed configuration can be found in this [flowchart](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_levels) and [datasheet](https://stevengong.co/notes/Streaming-Multiprocessor). 

In Triton, we do not explicitly manage L1 or L2 Caches. Instead, we use correct memory-coalescing to encourage efficient usage of caches. 

##### Shared memory
Shared memory is a programmable on-chip memory, unique to each thread block. In CUDA, a kernel can be declared as: `__shared__ float tile[256];`. 

By saying "Shared Memory", it means all threads in the same block share data here. Shared memory is much faster than global memory but with very limited size. 

**Triton** offers "scratchpad" or "shared" allocations, you can use certain intrinsics or advanced techniques to store data in fast memory location to a "program" or block of programs. 

##### Triton memory model vs CUDA memory model
- Global Memory in Triton → same concept as CUDA global memory.
- Shared Memory in Triton → specialized scratchpad allocations for a Triton program (similar to a block in CUDA). 

### Example 01: Microscaling

We perform the dequantization process of micro-scaling based on the quantized integer matrix `A` with a shape of `[M, K]`. 

Overall, we will have two pointers for matrix `A` and the shift-based scaling `As_ptr`. 

First, we initiate a set of programs along the row axis. E.g., each row block will be processed individually and the size of each row group is `BLOCK_SIZE_M`. 

Apparently, the total number of programs is defined by the ceiling division between `M` and group size: 

```
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
```

**What is `GROUP_SIZE_M`?**

We define a specific group size `GROUP_SIZE_M` to fetching multiple groups of row blocks for better load balancing: Instead of fetching one row at a time, loading multiple row blocks to L2 cache leads to better memory efficiency and reduce the latency. 

```
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
```

`group_id`: Which “group” of row-blocks we’re in. Or it tells us which group the current `pid` belongs to.
If pid is 7 and each group has size 4, then group_id = 1 (the second group). 
`first_pid_m`: The row-block ID of the first block in that group.

> **Small Example** 
- M = 68
- `BLOCK_SIZE_M = 16`
    - Each row block contains 16 rows. 
- Then `num_pid_m` = `ceiling(68 / 16)` = 5 => Indecies of [0,1,2,3,4]
- Suppose the `GROUP_SIZE_M = 2`, it means we group the row blocks with group size = 2. 
- It is easy to tell that the indexes of row blocks: 
**Row Blocks**
- Block 0 → rows [0..15]
- Block 1 → rows [16..31]
- Block 2 → rows [32..47]
- Block 3 → rows [48..63]
- Block 4 → rows [64..67] (partial block)

To make sure the group size can also properly cover the non-full group, the actual group size is dynamically computed based on the comparison between the selected group size and the remained row  blocks. 

1. Group 0: row-blocks 0 and 1
    - group_id=0 → first_pid_m = 0 * 2 = 0
    - group_size_m = min(num_pid_m - 0, 2) = min(5, 2) = 2
    - So it handles row-blocks `[0,1]`

2. Group 1: row-blocks 2 and 3
    - `group_id=1` $\rightarrow$ `first_pid_m` = `1 * 2` = 2
    - `group_size_m` = `min(num_pid_m - 2, 2)` = `min(3, 2)` = 2
    - So it handles row-blocks [2,3]

3. Group 2: only row-block 4 left
    - `group_id=2` $\rightarrow$ `first_pid_m = 2 * 2` = 4
    - `group_size_m = min(num_pid_m - 4, 2) = min(1, 2)` = 1
    - So it only handle the very last row block.

In other words, `group_size_m` characterizes how many row-blocks (not rows) does the current group handle, starting from `first_pid_m`. 

The final program_id (`pid`) along the row dimension should poroperly divide the program_id into multiple groups, where the starting id of each group is `first_pid_m`. 

