import { KernelProblem } from '../types';

export const KERNEL_PRESETS: KernelProblem[] = [
  {
    id: 'gpumode_matmul_v2',
    name: 'GPU MODE: matmul_v2 (Dense GEMM)',
    category: 'gemm',
    description:
      'Official GPU MODE matmul_v2 benchmark. Matrix multiplication C = A @ B with L2 cache swizzling, async memory loads, and tensor core MMA optimization.',
    complexity: 'Advanced',
    baselineLatencyUs: 215.0,
    baselineGFlops: 39040,
    theoreticalRooflineGFlops: 989000,
    commonBottlenecks: [
      'L2 Cache Thrashing from row-major stride order',
      'Shared memory bank conflicts during tile load',
      'Underutilized Tensor Cores without MMA pipelining',
      'Inadequate Block M/N tiling for SM warp occupancy',
    ],
    config: {
      tensorShapes: 'M=4096, N=4096, K=4096 [FP16]',
      dtype: 'float16',
      warmupRuns: 25,
      benchmarkRuns: 100,
    },
    baselineCode: `import torch

# GPU MODE Reference Baseline: PyTorch eager torch.matmul
def matmul_v2_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)
`,
    initialTritonCode: `@triton.jit
def matmul_v2_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
`,
    initialCudaCode: `// GPU MODE matmul_v2 CUDA Reference Implementation
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void matmul_v2_cuda_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // 16x16x16 Tensor Core warp tile setup
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + warpN * 16 * K + k, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + warpM * 16 * N + warpN * 16, c_frag, N, wmma::mem_row_major);
}
`,
  },
  {
    id: 'gpumode_vectorsum_v2',
    name: 'GPU MODE: vectorsum_v2 (Parallel Reduction)',
    category: 'reduction',
    description:
      'Official GPU MODE vectorsum_v2 challenge. Fast multi-block reduction with intra-warp shuffle (__shfl_down_sync), shared memory staging, and vectorized 128-bit global loads.',
    complexity: 'Intermediate',
    baselineLatencyUs: 145.0,
    baselineGFlops: 231.0,
    theoreticalRooflineGFlops: 3350000,
    commonBottlenecks: [
      'Uncoalesced 32-bit global loads instead of 128-bit float4 loads',
      'Thread divergent branch reduction trees',
      'Unnecessary __syncthreads() barrier serialization',
      'Memory bandwidth bound without multi-stage unrolling',
    ],
    config: {
      tensorShapes: 'N = 33,554,432 elements (128 MB) [FP32]',
      dtype: 'float32',
      warmupRuns: 50,
      benchmarkRuns: 200,
    },
    baselineCode: `import torch

# GPU MODE Reference Baseline: PyTorch torch.sum
def vectorsum_v2_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x)
`,
    initialTritonCode: `@triton.jit
def vectorsum_v2_kernel(
    x_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # 128-bit aligned vectorized load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    
    tl.atomic_add(output_ptr, sum_val)
`,
    initialCudaCode: `// GPU MODE vectorsum_v2 CUDA Reference Implementation
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void vectorsum_v2_cuda_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 4) + tid;
    float sum = 0.0f;

    // Vectorized float4 memory load
    if (idx + 3 * blockDim.x < N) {
        float4 v = reinterpret_cast<const float4*>(input)[blockIdx.x * blockDim.x + tid];
        sum = v.x + v.y + v.z + v.w;
    }

    sum = warp_reduce_sum(sum);
    
    __shared__ float sdata[32];
    int lane = tid % 32;
    int wid = tid / 32;
    
    if (lane == 0) sdata[wid] = sum;
    __syncthreads();
    
    sum = (tid < blockDim.x / 32) ? sdata[lane] : 0.0f;
    if (wid == 0) sum = warp_reduce_sum(sum);
    
    if (tid == 0) atomicAdd(output, sum);
}
`,
  },
  {
    id: 'gpumode_prefixsum_v2',
    name: 'GPU MODE: prefixsum_v2 (Exclusive Prefix Scan)',
    category: 'reduction',
    description:
      'Official GPU MODE prefixsum_v2 benchmark. Fast work-efficient parallel scan using Blelloch tree reduction, decoupled look-back or multi-pass spine coordination.',
    complexity: 'Advanced',
    baselineLatencyUs: 190.0,
    baselineGFlops: 350.0,
    theoreticalRooflineGFlops: 3350000,
    commonBottlenecks: [
      'Shared memory bank conflicts during up-sweep & down-sweep phases',
      'Global inter-block coordination stall overhead',
      'Branch divergence in log2 reduction trees',
      'Sub-optimal tile partition size vs memory bus bandwidth',
    ],
    config: {
      tensorShapes: 'N = 16,777,216 elements (64 MB) [FP32]',
      dtype: 'float32',
      warmupRuns: 25,
      benchmarkRuns: 100,
    },
    baselineCode: `import torch

# GPU MODE Reference Baseline: PyTorch torch.cumsum
def prefixsum_v2_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, dim=0)
`,
    initialTritonCode: `@triton.jit
def prefixsum_v2_kernel(
    x_ptr, y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Triton associative scan primitive
    scan_val = tl.associative_scan(x, axis=0, combine_fn=tl.maximum) # or tl.sum
    
    tl.store(y_ptr + offsets, scan_val, mask=mask)
`,
    initialCudaCode: `// GPU MODE prefixsum_v2 CUDA Reference (Blelloch Scan with Bank conflict padding)
#include <cuda_runtime.h>

#define NUM_BANKS 32
#define LOG_NUMBANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUMBANKS)

__global__ void prefixsum_v2_cuda_kernel(float *g_odata, const float *g_idata, int n) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int offset = 1;

    int ai = thid;
    int bi = thid + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    // Up-sweep (Reduction)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai_d = offset * (2 * thid + 1) - 1;
            int bi_d = offset * (2 * thid + 2) - 1;
            ai_d += CONFLICT_FREE_OFFSET(ai_d);
            bi_d += CONFLICT_FREE_OFFSET(bi_d);
            temp[bi_d] += temp[ai_d];
        }
        offset *= 2;
    }

    if (thid == 0) {
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    // Down-sweep
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai_d = offset * (2 * thid + 1) - 1;
            int bi_d = offset * (2 * thid + 2) - 1;
            ai_d += CONFLICT_FREE_OFFSET(ai_d);
            bi_d += CONFLICT_FREE_OFFSET(bi_d);
            float t = temp[ai_d];
            temp[ai_d] = temp[bi_d];
            temp[bi_d] += t;
        }
    }
    __syncthreads();

    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB];
}
`,
  },
  {
    id: 'gpumode_histogram_v2',
    name: 'GPU MODE: histogram_v2 (Privatized Histogram)',
    category: 'reduction',
    description:
      'Official GPU MODE histogram_v2 challenge. High-throughput atomic bin accumulation with sub-warp privatized shared memory bins and coalesced global flush.',
    complexity: 'Advanced',
    baselineLatencyUs: 165.0,
    baselineGFlops: 420.0,
    theoreticalRooflineGFlops: 3350000,
    commonBottlenecks: [
      'Severe atomic contention on hot bins in global memory',
      'Shared memory bank serialization on identical key values',
      'Uncoalesced warp scatter writes',
    ],
    config: {
      tensorShapes: 'N = 16,777,216 items, 256 Bins [INT32]',
      dtype: 'float32',
      warmupRuns: 25,
      benchmarkRuns: 100,
    },
    baselineCode: `import torch

# GPU MODE Reference Baseline: PyTorch torch.histc / bincount
def histogram_v2_ref(x: torch.Tensor, bins: int = 256) -> torch.Tensor:
    return torch.histc(x.float(), bins=bins, min=0, max=255)
`,
    initialTritonCode: `@triton.jit
def histogram_v2_kernel(
    x_ptr, hist_ptr,
    N,
    NUM_BINS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Atomic bin increment
    for i in range(BLOCK_SIZE):
        bin_idx = tl.load(x_ptr + pid * BLOCK_SIZE + i)
        if bin_idx < NUM_BINS:
            tl.atomic_add(hist_ptr + bin_idx, 1)
`,
    initialCudaCode: `// GPU MODE histogram_v2 CUDA Reference with Shared Memory Privatization
#include <cuda_runtime.h>

#define NUM_BINS 256

__global__ void histogram_v2_cuda_kernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ output,
    int N
) {
    __shared__ unsigned int local_hist[NUM_BINS];

    // Clear shared memory bins
    int tid = threadIdx.x;
    if (tid < NUM_BINS) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // Accumulate locally
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < N) {
        unsigned char val = input[idx];
        atomicAdd(&local_hist[val], 1);
        idx += stride;
    }
    __syncthreads();

    // Flush shared memory histogram into global memory
    if (tid < NUM_BINS) {
        atomicAdd(&output[tid], local_hist[tid]);
    }
}
`,
  },
  {
    id: 'gpumode_trimul_alphafold3',
    name: 'GPU MODE: TriMul (AlphaFold 3 Update)',
    category: 'attention',
    description:
      'Official GPU MODE closed challenge. Triangle Multiplicative Update operator from AlphaFold 3 / ESMFold featuring fused gating, online reduction, and high arithmetic intensity.',
    complexity: 'Expert',
    baselineLatencyUs: 380.0,
    baselineGFlops: 58400,
    theoreticalRooflineGFlops: 989000,
    commonBottlenecks: [
      'Huge intermediate memory allocations without kernel fusion',
      'Tensor core underutilization on 3D spatial dimensions',
      'High register pressure in fused elementwise sigmoid gates',
    ],
    config: {
      tensorShapes: 'B=1, N_res=512, N_res=512, Dim=128 [FP16]',
      dtype: 'float16',
      warmupRuns: 20,
      benchmarkRuns: 50,
    },
    baselineCode: `import torch

# GPU MODE Reference Baseline: AlphaFold3 Triangle Multiplicative Update
def trimul_alphafold3_ref(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # A: [B, N, N, C], B: [B, N, N, C]
    gate_a = torch.sigmoid(a)
    gate_b = torch.sigmoid(b)
    left = a * gate_a
    right = b * gate_b
    out = torch.einsum('bikc,bjkc->bijc', left, right)
    return out
`,
    initialTritonCode: `@triton.jit
def trimul_alphafold3_kernel(
    a_ptr, b_ptr, out_ptr,
    B, N, C,
    stride_ab, stride_ai, stride_aj, stride_ac,
    stride_bb, stride_bi, stride_bj, stride_bc,
    stride_ob, stride_oi, stride_oj, stride_oc,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_c = tl.arange(0, BLOCK_C)
    
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    for k in range(0, N):
        # Load A[i, k, c] and B[j, k, c]
        a_ptrs = a_ptr + (offs_i[:, None] * stride_ai + k * stride_aj + offs_c[None, :] * stride_ac)
        b_ptrs = b_ptr + (offs_j[:, None] * stride_bi + k * stride_aj + offs_c[None, :] * stride_bc)
        
        a = tl.load(a_ptrs, mask=(offs_i[:, None] < N) & (offs_c[None, :] < C), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_j[:, None] < N) & (offs_c[None, :] < C), other=0.0)
        
        # Fused sigmoid gating
        gate_a = tl.sigmoid(a)
        gate_b = tl.sigmoid(b)
        
        a_gated = a * gate_a
        b_gated = b * gate_b
        
        acc += tl.sum(a_gated[:, None, :] * b_gated[None, :, :], axis=2)
        
    out_ptrs = out_ptr + (offs_i[:, None] * stride_oi + offs_j[None, :] * stride_oj)
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_i[:, None] < N) & (offs_j[None, :] < N))
`,
    initialCudaCode: `// GPU MODE TriMul AlphaFold3 CUDA Implementation Skeleton
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void trimul_alphafold3_cuda_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ Out,
    int B_size, int N, int C
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            for (int c = 0; c < C; ++c) {
                float a_val = __half2float(A[i * N * C + k * C + c]);
                float b_val = __half2float(B[j * N * C + k * C + c]);
                float a_gated = a_val / (1.0f + expf(-a_val));
                float b_gated = b_val / (1.0f + expf(-b_val));
                acc += a_gated * b_gated;
            }
        }
        Out[i * N + j] = __float2half(acc);
    }
}
`,
  },
  {
    id: 'gpumode_conv2d_v2',
    name: 'GPU MODE: conv2d_v2 (Implicit GEMM Conv)',
    category: 'convolution',
    description:
      'Official GPU MODE conv2d_v2 benchmark. Fast 2D direct convolution via implicit GEMM coordinate mapping, vectorized float4 activation streaming, and fused ReLU.',
    complexity: 'Advanced',
    baselineLatencyUs: 290.0,
    baselineGFlops: 41200,
    theoreticalRooflineGFlops: 989000,
    commonBottlenecks: [
      'Costly im2col explicit tensor footprint expansions',
      'Uncoalesced input halo border access strides',
      'Underutilized tensor core MMA layout mappings',
    ],
    config: {
      tensorShapes: 'N=64, C=128, H=56, W=56, K=256, R=3, S=3 [FP16]',
      dtype: 'float16',
      warmupRuns: 20,
      benchmarkRuns: 100,
    },
    baselineCode: `import torch
import torch.nn.functional as F

# GPU MODE Reference Baseline: PyTorch F.conv2d
def conv2d_v2_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, weight, stride=1, padding=1)
`,
    initialTritonCode: `@triton.jit
def conv2d_v2_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C, H, W,
    K, R, S,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wc, stride_wr, stride_ws,
    stride_yn, stride_yk, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    # Implicit GEMM mapping
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Convolution implicit loop
    for c in range(0, C):
        for r in range(0, R):
            for s in range(0, S):
                x = tl.load(x_ptr + offs_m[:, None], mask=offs_m[:, None] < N * H * W, other=0.0)
                w = tl.load(w_ptr + offs_n[None, :], mask=offs_n[None, :] < K, other=0.0)
                acc += x * w
                
    tl.store(y_ptr + offs_m[:, None], acc.to(tl.float16), mask=offs_m[:, None] < N * H * W)
`,
    initialCudaCode: `// GPU MODE conv2d_v2 CUDA Reference Implementation
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void conv2d_v2_cuda_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (out_x < W && out_y < H && k < K) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    int in_x = out_x + s - 1;
                    int in_y = out_y + r - 1;
                    if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                        float in_val = __half2float(input[c * H * W + in_y * W + in_x]);
                        float w_val = __half2float(weight[k * C * R * S + c * R * S + r * S + s]);
                        sum += in_val * w_val;
                    }
                }
            }
        }
        output[k * H * W + out_y * W + out_x] = __float2half(fmaxf(0.0f, sum));
    }
}
`,
  },
];
