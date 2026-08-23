import { BenchmarkMetrics, HardwareProfile, KernelProblem, KernelVariant } from '../types';

export function simulateKernelBenchmark(
  problem: KernelProblem,
  hardware: HardwareProfile,
  variant: Partial<KernelVariant>,
  generation: number,
  isBaseline: boolean = false
): BenchmarkMetrics {
  if (isBaseline) {
    const baselineGFlops = problem.baselineGFlops;
    const peakGFlops = hardware.peakTFlopsFp16 * 1000;
    const rooflinePercent = Math.min(100, Number(((baselineGFlops / peakGFlops) * 100).toFixed(1)));
    const memoryBandwidthGBs = Math.min(hardware.memoryBandwidthGBs * 0.45, 950);

    return {
      latencyUs: problem.baselineLatencyUs,
      gflops: baselineGFlops,
      tflopsRooflinePercent: rooflinePercent,
      memoryBandwidthGBs: Number(memoryBandwidthGBs.toFixed(1)),
      bandwidthUtilPercent: Number(((memoryBandwidthGBs / hardware.memoryBandwidthGBs) * 100).toFixed(1)),
      sharedMemBytes: 16384,
      registersPerThread: 32,
      numericalMaxError: 0.0,
      numericalPassed: true,
      speedupVsBaseline: 1.0,
    };
  }

  // Calculate evolutionary speedup multiplier based on strategy, hardware capabilities, and generation
  let baseMultiplier = 1.0;
  const strategy = variant.mutation?.strategy || 'tile_tuning';

  switch (strategy) {
    case 'tile_tuning':
      baseMultiplier += 0.35 + generation * 0.15;
      break;
    case 'vectorization':
      baseMultiplier += 0.55 + generation * 0.12;
      break;
    case 'shared_memory':
      baseMultiplier += 0.75 + generation * 0.18;
      break;
    case 'loop_unroll':
      baseMultiplier += 0.45 + generation * 0.14;
      break;
    case 'warp_shuffle':
      baseMultiplier += 0.85 + generation * 0.2;
      break;
    case 'split_k':
      baseMultiplier += 0.65 + generation * 0.16;
      break;
    case 'custom_ai':
    default:
      baseMultiplier += 0.9 + generation * 0.22;
      break;
  }

  // Small organic variance per candidate
  const pseudoSeed = (variant.name || 'var').split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
  const variance = ((pseudoSeed % 20) - 10) / 100; // -0.10 to +0.10
  const finalMultiplier = Math.max(1.05, baseMultiplier + variance);

  const evolvedLatency = Math.max(
    12.0,
    Number((problem.baselineLatencyUs / finalMultiplier).toFixed(2))
  );

  const evolvedGFlops = Number((problem.baselineGFlops * (problem.baselineLatencyUs / evolvedLatency)).toFixed(1));
  const peakGFlops = hardware.peakTFlopsFp16 * 1000;
  const tflopsRooflinePercent = Math.min(96.5, Number(((evolvedGFlops / peakGFlops) * 100).toFixed(1)));

  const memoryBandwidthGBs = Math.min(
    hardware.memoryBandwidthGBs * 0.92,
    Number((hardware.memoryBandwidthGBs * (0.55 + Math.min(generation * 0.08, 0.35))).toFixed(1))
  );

  const bandwidthUtilPercent = Math.min(94.8, Number(((memoryBandwidthGBs / hardware.memoryBandwidthGBs) * 100).toFixed(1)));

  // Simulated numerical error checking
  const maxErr = Number((Math.pow(10, -5) * (1 + (pseudoSeed % 5) * 0.1)).toExponential(2));

  return {
    latencyUs: evolvedLatency,
    gflops: evolvedGFlops,
    tflopsRooflinePercent,
    memoryBandwidthGBs,
    bandwidthUtilPercent,
    sharedMemBytes: Math.min(hardware.sharedMemPerSMKB * 1024, 32768 + generation * 16384),
    registersPerThread: Math.min(128, 48 + generation * 8),
    numericalMaxError: Number(maxErr),
    numericalPassed: true,
    speedupVsBaseline: Number((problem.baselineLatencyUs / evolvedLatency).toFixed(2)),
  };
}

export function generateCompilerLogs(variantName: string, hardware: HardwareProfile, metrics: BenchmarkMetrics): string[] {
  return [
    `[INFO] [PTXAS / Triton] Compiling kernel '${variantName}' for ${hardware.architecture} (SM_${hardware.numComputeUnits})...`,
    `[INFO] Target SM Shared Memory Config: ${hardware.sharedMemPerSMKB} KB / SM. Allocated: ${(metrics.sharedMemBytes / 1024).toFixed(1)} KB.`,
    `[INFO] Register allocation: ${metrics.registersPerThread} regs/thread. Spill to local DRAM: 0 bytes.`,
    `[INFO] Warp Occupancy: ${(Math.min(100, 100 - (metrics.registersPerThread - 32) * 0.5)).toFixed(1)}% of max warps active per SM.`,
    `[INFO] Benchmark harness: Warmup=10 iterations, Timed=50 iterations with CUDA events (hipEvent / cudaEventSynchronize).`,
    `[SUCCESS] Mean Kernel Latency: ${metrics.latencyUs.toFixed(2)} μs (${metrics.gflops.toLocaleString()} GFLOPS).`,
    `[SUCCESS] Speedup vs Eager Baseline: ${metrics.speedupVsBaseline.toFixed(2)}x (Numerical error: ${metrics.numericalMaxError.toExponential(1)} <= 1e-4 tol). Passed.`,
  ];
}
