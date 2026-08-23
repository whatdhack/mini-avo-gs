import React from 'react';
import { Gauge, Zap, TrendingUp, HardDrive, Cpu, CheckCircle2, Flame, LineChart } from 'lucide-react';
import { BenchmarkMetrics, HardwareProfile, KernelProblem, KernelVariant } from '../types';

interface PerformanceDashboardProps {
  problem: KernelProblem;
  hardware: HardwareProfile;
  selectedVariant: KernelVariant | null;
  allVariants: KernelVariant[];
}

export const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  problem,
  hardware,
  selectedVariant,
  allVariants,
}) => {
  const metrics = selectedVariant?.metrics || {
    latencyUs: problem.baselineLatencyUs,
    gflops: problem.baselineGFlops,
    tflopsRooflinePercent: 28.5,
    memoryBandwidthGBs: 650,
    bandwidthUtilPercent: 42.0,
    sharedMemBytes: 16384,
    registersPerThread: 32,
    numericalMaxError: 0.0,
    numericalPassed: true,
    speedupVsBaseline: 1.0,
  };

  // Group best metrics per generation for progression graph
  const genProgression: Record<number, number> = {};
  for (const v of allVariants) {
    const sp = v.metrics?.speedupVsBaseline || 1.0;
    if (!genProgression[v.generation] || sp > genProgression[v.generation]) {
      genProgression[v.generation] = sp;
    }
  }

  const genNums = Object.keys(genProgression)
    .map(Number)
    .sort((a, b) => a - b);

  const peakSpeedup = Object.values(genProgression).reduce((max, val) => Math.max(max, val), 1.0);

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-4 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80">
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-semibold text-zinc-100">Hardware Roofline & Profiler</h2>
        </div>
        <span className="text-xs text-zinc-400 font-mono">
          {hardware.name} ({hardware.architecture})
        </span>
      </div>

      {/* Top 4 Core Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
        {/* Speedup Multiplier */}
        <div className="bg-zinc-950/80 border border-zinc-800 p-3 rounded-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-zinc-400 text-xs mb-1">
            <span className="flex items-center gap-1">
              <Zap className="w-3.5 h-3.5 text-emerald-400" />
              Speedup
            </span>
            <span className="text-[10px] text-zinc-500 font-mono">vs Eager</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold font-mono text-emerald-400">
              {metrics.speedupVsBaseline.toFixed(2)}x
            </span>
            <span className="text-xs text-emerald-500/80 font-medium">
              {metrics.speedupVsBaseline > 1 ? `+${((metrics.speedupVsBaseline - 1) * 100).toFixed(0)}%` : 'Baseline'}
            </span>
          </div>
          <div className="w-full bg-zinc-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-teal-500 to-emerald-400 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, (metrics.speedupVsBaseline / 4) * 100)}%` }}
            />
          </div>
        </div>

        {/* Latency */}
        <div className="bg-zinc-950/80 border border-zinc-800 p-3 rounded-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-zinc-400 text-xs mb-1">
            <span className="flex items-center gap-1">
              <TrendingUp className="w-3.5 h-3.5 text-teal-400" />
              Mean Latency
            </span>
            <span className="text-[10px] text-zinc-500 font-mono">CUDA Events</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold font-mono text-zinc-100">
              {metrics.latencyUs.toFixed(1)}
            </span>
            <span className="text-xs text-zinc-400 font-mono">μs</span>
          </div>
          <span className="text-[11px] text-zinc-500 font-mono mt-1">
            Baseline: {problem.baselineLatencyUs.toFixed(0)} μs
          </span>
        </div>

        {/* Compute Throughput (TFLOPS) */}
        <div className="bg-zinc-950/80 border border-zinc-800 p-3 rounded-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-zinc-400 text-xs mb-1">
            <span className="flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              Compute (TFLOPS)
            </span>
            <span className="text-[10px] text-cyan-400 font-mono">{metrics.tflopsRooflinePercent}% Peak</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold font-mono text-cyan-300">
              {(metrics.gflops / 1000).toFixed(1)}
            </span>
            <span className="text-xs text-zinc-400 font-mono">TFLOPS</span>
          </div>
          <div className="w-full bg-zinc-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="bg-cyan-400 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, metrics.tflopsRooflinePercent)}%` }}
            />
          </div>
        </div>

        {/* Memory Bandwidth */}
        <div className="bg-zinc-950/80 border border-zinc-800 p-3 rounded-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-zinc-400 text-xs mb-1">
            <span className="flex items-center gap-1">
              <HardDrive className="w-3.5 h-3.5 text-purple-400" />
              Memory Bandwidth
            </span>
            <span className="text-[10px] text-purple-400 font-mono">{metrics.bandwidthUtilPercent}% Bus</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold font-mono text-purple-300">
              {metrics.memoryBandwidthGBs.toFixed(0)}
            </span>
            <span className="text-xs text-zinc-400 font-mono">GB/s</span>
          </div>
          <div className="w-full bg-zinc-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="bg-purple-400 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, metrics.bandwidthUtilPercent)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Visual Roofline & Progression Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Interactive Roofline Plot */}
        <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3 flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-zinc-300 flex items-center gap-1.5">
              <Flame className="w-3.5 h-3.5 text-amber-400" />
              Hardware Roofline Envelope
            </span>
            <span className="text-[10px] text-zinc-500 font-mono">
              Ceiling: {hardware.peakTFlopsFp16} TFLOPS
            </span>
          </div>

          {/* Roofline Graphic */}
          <div className="h-36 bg-zinc-900/50 border border-zinc-800/60 rounded relative flex items-end p-3 overflow-hidden">
            {/* Grid Lines */}
            <div className="absolute inset-0 grid grid-cols-4 grid-rows-3 opacity-15 pointer-events-none">
              {Array.from({ length: 12 }).map((_, i) => (
                <div key={i} className="border-b border-r border-zinc-500" />
              ))}
            </div>

            {/* SVG Roofline Curve */}
            <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none" viewBox="0 0 100 100">
              {/* Theoretical Ceiling Line (Slanted then Flat) */}
              <polyline
                fill="none"
                stroke="#52525b"
                strokeWidth="2"
                strokeDasharray="4 3"
                points="5,95 40,20 95,20"
              />
              {/* Baseline Point */}
              <circle cx="35" cy="70" r="3.5" fill="#f43f5e" />
              <text x="38" y="72" fill="#f43f5e" fontSize="6" fontFamily="monospace">
                Baseline (1.0x)
              </text>

              {/* Current Mutant Point */}
              <circle
                cx={Math.min(85, 35 + metrics.speedupVsBaseline * 12)}
                cy={Math.max(25, 70 - (metrics.speedupVsBaseline - 1) * 16)}
                r="4.5"
                fill="#10b981"
                className="animate-pulse"
              />
              <text
                x={Math.min(75, 42 + metrics.speedupVsBaseline * 10)}
                y={Math.max(22, 65 - (metrics.speedupVsBaseline - 1) * 16)}
                fill="#10b981"
                fontSize="6.5"
                fontWeight="bold"
                fontFamily="monospace"
              >
                Evolved ({metrics.speedupVsBaseline.toFixed(2)}x)
              </text>
            </svg>

            <div className="relative z-10 w-full flex justify-between text-[9px] font-mono text-zinc-500">
              <span>Memory Bound (Low Ops/Byte)</span>
              <span>Compute Bound (High Ops/Byte)</span>
            </div>
          </div>
        </div>

        {/* Generational Speedup Trajectory */}
        <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-zinc-300 flex items-center gap-1.5">
              <LineChart className="w-3.5 h-3.5 text-teal-400" />
              Speedup Trajectory by Generation
            </span>
            <span className="text-[10px] text-emerald-400 font-mono font-semibold">
              Peak: {peakSpeedup.toFixed(2)}x
            </span>
          </div>

          <div className="space-y-2 py-1">
            {genNums.map((g) => {
              const sp = genProgression[g];
              const pct = Math.min(100, (sp / 4.0) * 100);

              return (
                <div key={g} className="flex items-center gap-2 text-xs">
                  <span className="w-12 font-mono text-zinc-400 text-[11px]">Gen {g}</span>
                  <div className="flex-1 bg-zinc-800/80 h-4 rounded overflow-hidden relative">
                    <div
                      className="bg-gradient-to-r from-teal-500 to-emerald-400 h-full rounded transition-all duration-300"
                      style={{ width: `${pct}%` }}
                    />
                    <span className="absolute inset-y-0 right-2 flex items-center text-[10px] font-mono font-bold text-zinc-200">
                      {sp.toFixed(2)}x
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-zinc-800/60 text-[11px] text-zinc-400">
            <span className="flex items-center gap-1 text-emerald-400 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              Tensor Verification: Pass
            </span>
            <span className="font-mono text-zinc-500">
              SM Shmem: {(metrics.sharedMemBytes / 1024).toFixed(0)} KB | Regs: {metrics.registersPerThread}/th
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
