import { HardwareProfile, HardwareId } from '../types';

export const HARDWARE_PROFILES: Record<HardwareId, HardwareProfile> = {
  'b200-sxm': {
    id: 'b200-sxm',
    name: 'NVIDIA Blackwell B200 SXM 192GB',
    vendor: 'NVIDIA',
    architecture: 'Blackwell (GB200/GB208)',
    computeCapability: 'sm_100a',
    peakTFlopsFp16: 2250,
    peakTFlopsFp32: 90,
    memoryBandwidthGBs: 8000,
    sharedMemPerSMKB: 256,
    warpSize: 32,
    maxThreadsPerBlock: 1024,
    numComputeUnits: 148,
    description: 'GPU MODE Next-Gen Blackwell Target. 2nd Gen Transformer Engine, 8 TB/s HBM3e bandwidth, FP4/FP8 micro-scaling Tensor Cores, and 5th Gen NVLink.'
  },
  'h100-sxm': {
    id: 'h100-sxm',
    name: 'NVIDIA H100 SXM5 80GB',
    vendor: 'NVIDIA',
    architecture: 'Hopper (GH100)',
    computeCapability: 'sm_90a',
    peakTFlopsFp16: 989,
    peakTFlopsFp32: 67,
    memoryBandwidthGBs: 3350,
    sharedMemPerSMKB: 228,
    warpSize: 32,
    maxThreadsPerBlock: 1024,
    numComputeUnits: 132,
    description: 'GPU MODE Flagship Target. 4th Gen Tensor Cores with DPX, TMA async copy pipelines, and Hopper Distributed Shared Memory (DSMEM).'
  },
  'a100-80g': {
    id: 'a100-80g',
    name: 'NVIDIA A100 SXM4 80GB',
    vendor: 'NVIDIA',
    architecture: 'Ampere (GA100)',
    computeCapability: 'sm_80',
    peakTFlopsFp16: 312,
    peakTFlopsFp32: 19.5,
    memoryBandwidthGBs: 2039,
    sharedMemPerSMKB: 164,
    warpSize: 32,
    maxThreadsPerBlock: 1024,
    numComputeUnits: 108,
    description: 'GPU MODE Standard Cluster Target. 3rd Gen Tensor Cores with cp.async copy pipeline and L2 cache residency control.'
  },
  'l4': {
    id: 'l4',
    name: 'NVIDIA L4 Tensor Core 24GB',
    vendor: 'NVIDIA',
    architecture: 'Ada Lovelace (AD104)',
    computeCapability: 'sm_89',
    peakTFlopsFp16: 120,
    peakTFlopsFp32: 30.3,
    memoryBandwidthGBs: 300,
    sharedMemPerSMKB: 100,
    warpSize: 32,
    maxThreadsPerBlock: 1024,
    numComputeUnits: 58,
    description: 'Google Cloud & GPU MODE Cost-Efficient Inference Target. 4th Gen Tensor Cores with FP8 acceleration and 48MB L2 cache.'
  },
  'rtx4090': {
    id: 'rtx4090',
    name: 'NVIDIA GeForce RTX 4090 24GB',
    vendor: 'NVIDIA',
    architecture: 'Ada Lovelace (AD102)',
    computeCapability: 'sm_89',
    peakTFlopsFp16: 165,
    peakTFlopsFp32: 82.6,
    memoryBandwidthGBs: 1008,
    sharedMemPerSMKB: 100,
    warpSize: 32,
    maxThreadsPerBlock: 1024,
    numComputeUnits: 128,
    description: 'GPU MODE Workstation Target. High core clocks with 72MB L2 cache and FP8 Tensor Cores for local kernel testing.'
  }
};
