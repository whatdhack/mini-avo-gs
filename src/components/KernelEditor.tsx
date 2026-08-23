import React, { useState } from 'react';
import { Code2, Terminal, Play, Copy, Check, FileCode, Cpu } from 'lucide-react';
import { BenchmarkMetrics, HardwareProfile, KernelProblem, KernelVariant } from '../types';
import { generateCompilerLogs } from '../utils/benchmarkSimulator';

interface KernelEditorProps {
  problem: KernelProblem;
  hardware: HardwareProfile;
  selectedVariant: KernelVariant | null;
  onRunCustomBenchmark: (code: string) => void;
}

export const KernelEditor: React.FC<KernelEditorProps> = ({
  problem,
  hardware,
  selectedVariant,
  onRunCustomBenchmark,
}) => {
  const [activeTab, setActiveTab] = useState<'triton' | 'cuda' | 'pytorch' | 'logs'>('triton');
  const [copied, setCopied] = useState(false);
  const [customCode, setCustomCode] = useState(selectedVariant?.code || problem.initialTritonCode);

  React.useEffect(() => {
    if (selectedVariant) {
      setCustomCode(selectedVariant.code);
    }
  }, [selectedVariant]);

  const handleCopy = () => {
    const textToCopy =
      activeTab === 'triton'
        ? customCode
        : activeTab === 'cuda'
        ? problem.initialCudaCode
        : activeTab === 'pytorch'
        ? problem.baselineCode
        : logs.join('\n');

    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const metrics: BenchmarkMetrics = selectedVariant?.metrics || {
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

  const logs = selectedVariant?.compilerLogs || generateCompilerLogs(selectedVariant?.name || 'BaselineKernel', hardware, metrics);

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-4 flex flex-col h-full overflow-hidden">
      {/* Header Tabs */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80 mb-3 shrink-0 flex-wrap gap-2">
        <div className="flex items-center gap-1 bg-zinc-950 border border-zinc-800 rounded-lg p-0.5 text-xs">
          <button
            onClick={() => setActiveTab('triton')}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded font-medium transition-colors ${
              activeTab === 'triton'
                ? 'bg-zinc-800 text-emerald-400 font-semibold'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            <span>Triton ({selectedVariant?.name || 'Active'})</span>
          </button>

          <button
            onClick={() => setActiveTab('cuda')}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded font-medium transition-colors ${
              activeTab === 'cuda'
                ? 'bg-zinc-800 text-cyan-400 font-semibold'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>CUDA C++</span>
          </button>

          <button
            onClick={() => setActiveTab('pytorch')}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded font-medium transition-colors ${
              activeTab === 'pytorch'
                ? 'bg-zinc-800 text-teal-400 font-semibold'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span>PyTorch Reference</span>
          </button>

          <button
            onClick={() => setActiveTab('logs')}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded font-medium transition-colors ${
              activeTab === 'logs'
                ? 'bg-zinc-800 text-amber-400 font-semibold'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>Compiler Logs</span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          {activeTab === 'triton' && (
            <button
              onClick={() => onRunCustomBenchmark(customCode)}
              className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-2.5 py-1 rounded-lg transition-colors shadow-sm"
              title="Benchmark updated code"
            >
              <Play className="w-3 h-3 fill-current" />
              <span>Benchmark Code</span>
            </button>
          )}

          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 bg-zinc-800/90 hover:bg-zinc-700 text-zinc-200 text-xs px-2.5 py-1 rounded-lg border border-zinc-700 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>

      {/* Editor & Content Body */}
      <div className="flex-1 overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950 font-mono text-xs text-zinc-300 flex flex-col">
        {activeTab === 'triton' && (
          <textarea
            value={customCode}
            onChange={(e) => setCustomCode(e.target.value)}
            className="w-full h-full p-3 bg-transparent text-emerald-300 font-mono text-xs resize-none focus:outline-none custom-scrollbar leading-relaxed"
            spellCheck={false}
          />
        )}

        {activeTab === 'cuda' && (
          <div className="w-full h-full p-3 overflow-auto custom-scrollbar leading-relaxed">
            <pre className="text-cyan-300 whitespace-pre">{problem.initialCudaCode}</pre>
          </div>
        )}

        {activeTab === 'pytorch' && (
          <div className="w-full h-full p-3 overflow-auto custom-scrollbar leading-relaxed">
            <pre className="text-teal-300 whitespace-pre">{problem.baselineCode}</pre>
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="w-full h-full p-3 overflow-auto custom-scrollbar leading-relaxed bg-black/80 space-y-1 text-[11px]">
            {logs.map((line, idx) => (
              <div
                key={idx}
                className={`${
                  line.includes('[SUCCESS]')
                    ? 'text-emerald-400 font-bold'
                    : line.includes('[INFO]')
                    ? 'text-zinc-400'
                    : 'text-amber-400'
                }`}
              >
                {line}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
