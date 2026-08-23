export type HardwareId = 'b200-sxm' | 'h100-sxm' | 'a100-80g' | 'l4' | 'rtx4090';

export interface HardwareProfile {
  id: HardwareId;
  name: string;
  vendor: 'NVIDIA';
  architecture: string;
  computeCapability: string;
  peakTFlopsFp16: number;
  peakTFlopsFp32: number;
  memoryBandwidthGBs: number;
  sharedMemPerSMKB: number;
  warpSize: number;
  maxThreadsPerBlock: number;
  numComputeUnits: number;
  description: string;
}

export type KernelCategory = 'gemm' | 'attention' | 'normalization' | 'recommendation' | 'reduction' | 'convolution';

export interface KernelBenchmarkConfig {
  tensorShapes: string;
  dtype: 'float16' | 'bfloat16' | 'float32' | 'float8';
  warmupRuns: number;
  benchmarkRuns: number;
  batchSize?: number;
}

export interface KernelProblem {
  id: string;
  name: string;
  category: KernelCategory;
  description: string;
  complexity: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  baselineLatencyUs: number;
  baselineGFlops: number;
  baselineCode: string;
  initialTritonCode: string;
  initialCudaCode: string;
  config: KernelBenchmarkConfig;
  theoreticalRooflineGFlops: number;
  commonBottlenecks: string[];
}

export type VariantStatus = 'pending' | 'evaluating' | 'success' | 'failed' | 'elite';

export interface MutationDetail {
  strategy: 'tile_tuning' | 'vectorization' | 'shared_memory' | 'loop_unroll' | 'warp_shuffle' | 'split_k' | 'swizzle' | 'custom_ai';
  title: string;
  description: string;
  tileParams?: Record<string, number | string>;
}

export interface BenchmarkMetrics {
  latencyUs: number;
  gflops: number;
  tflopsRooflinePercent: number;
  memoryBandwidthGBs: number;
  bandwidthUtilPercent: number;
  sharedMemBytes: number;
  registersPerThread: number;
  numericalMaxError: number;
  numericalPassed: boolean;
  speedupVsBaseline: number;
}

export interface KernelVariant {
  id: string;
  name: string;
  generation: number;
  parentId: string | null;
  code: string;
  language: 'triton' | 'cuda' | 'cpp';
  status: VariantStatus;
  mutation: MutationDetail;
  metrics?: BenchmarkMetrics;
  aiExplanation?: string;
  compilerLogs?: string[];
  createdAt: number;
}

export interface EvolutionRun {
  id: string;
  problemId: string;
  targetHardware: HardwareId;
  currentGeneration: number;
  maxGenerations: number;
  populationSize: number;
  mutationTemperature: number;
  status: 'idle' | 'running' | 'paused' | 'completed';
  variants: KernelVariant[];
  bestVariantId: string | null;
  startTime: number;
  updatedAt: number;
}
