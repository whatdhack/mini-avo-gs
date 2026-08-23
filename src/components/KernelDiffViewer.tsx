import React, { useState } from 'react';
import { GitCompare, Copy, Check, Sparkles, Sliders, Code2, ArrowRight } from 'lucide-react';
import { KernelProblem, KernelVariant } from '../types';

interface KernelDiffViewerProps {
  problem: KernelProblem;
  selectedVariant: KernelVariant | null;
  parentVariant: KernelVariant | null;
}

export const KernelDiffViewer: React.FC<KernelDiffViewerProps> = ({
  problem,
  selectedVariant,
  parentVariant,
}) => {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<'side_by_side' | 'single'>('side_by_side');

  const baselineCode = problem.initialTritonCode;
  const targetCode = selectedVariant?.code || baselineCode;
  const parentCode = parentVariant?.code || baselineCode;

  const handleCopy = () => {
    navigator.clipboard.writeText(targetCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const tileParams = selectedVariant?.mutation?.tileParams || {
    BLOCK_M: 128,
    BLOCK_N: 128,
    BLOCK_K: 32,
    num_warps: 4,
    num_stages: 3,
    GROUP_SIZE_M: 8,
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-4 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80 mb-3 shrink-0 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <GitCompare className="w-4 h-4 text-teal-400" />
          <h2 className="text-sm font-semibold text-zinc-100">
            {selectedVariant ? `Diff: ${selectedVariant.name} vs Baseline` : 'Kernel Code Inspector'}
          </h2>
        </div>

        <div className="flex items-center gap-2">
          {/* Mode Switcher */}
          <div className="flex items-center bg-zinc-950 border border-zinc-800 rounded-lg p-0.5 text-xs">
            <button
              onClick={() => setViewMode('side_by_side')}
              className={`px-2 py-1 rounded ${
                viewMode === 'side_by_side'
                  ? 'bg-zinc-800 text-zinc-100 font-medium'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Side-by-Side Diff
            </button>
            <button
              onClick={() => setViewMode('single')}
              className={`px-2 py-1 rounded ${
                viewMode === 'single'
                  ? 'bg-zinc-800 text-zinc-100 font-medium'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Mutant Only
            </button>
          </div>

          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 bg-zinc-800/90 hover:bg-zinc-700 text-zinc-200 text-xs px-2.5 py-1 rounded-lg border border-zinc-700 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>

      {/* AI Mutation Reasoning & Tile Parameters Banner */}
      {selectedVariant && (
        <div className="mb-3 space-y-2 shrink-0">
          {/* AI Explanation */}
          <div className="bg-emerald-950/20 border border-emerald-500/30 rounded-lg p-3 text-xs">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold mb-1">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Architectural Mutation: {selectedVariant.mutation.title}</span>
            </div>
            <p className="text-zinc-300 leading-relaxed">
              {selectedVariant.aiExplanation || selectedVariant.mutation.description}
            </p>
          </div>

          {/* Tile Tuner Table */}
          <div className="bg-zinc-950/80 border border-zinc-800/80 rounded-lg px-3 py-2 flex items-center justify-between text-xs flex-wrap gap-2">
            <div className="flex items-center gap-1.5 text-zinc-400 font-medium">
              <Sliders className="w-3.5 h-3.5 text-teal-400" />
              <span>Triton Compiler Tiling Config:</span>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {Object.entries(tileParams).map(([key, val]) => (
                <span
                  key={key}
                  className="bg-zinc-900 border border-zinc-700/80 px-2 py-0.5 rounded font-mono text-[11px] text-zinc-300"
                >
                  <strong className="text-zinc-500">{key}:</strong> {String(val)}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Code Viewer */}
      <div className="flex-1 overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950 font-mono text-xs text-zinc-300 flex">
        {viewMode === 'side_by_side' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 w-full h-full divide-y md:divide-y-0 md:divide-x divide-zinc-800">
            {/* Left: Baseline / Parent */}
            <div className="flex flex-col h-full overflow-hidden">
              <div className="bg-zinc-900/90 px-3 py-1.5 border-b border-zinc-800 text-[11px] font-semibold text-zinc-400 flex items-center justify-between">
                <span>Baseline (Seed Gen 0)</span>
                <span className="text-[10px] text-zinc-500 font-normal">Reference implementation</span>
              </div>
              <div className="flex-1 overflow-auto p-3 custom-scrollbar leading-relaxed">
                <pre className="text-zinc-400 whitespace-pre font-mono select-text">{baselineCode}</pre>
              </div>
            </div>

            {/* Right: Evolved Mutant */}
            <div className="flex flex-col h-full overflow-hidden bg-emerald-950/5">
              <div className="bg-zinc-900/90 px-3 py-1.5 border-b border-zinc-800 text-[11px] font-semibold text-emerald-400 flex items-center justify-between">
                <span>{selectedVariant ? selectedVariant.name : 'Evolved Mutant'}</span>
                <span className="text-[10px] text-emerald-400/80 font-mono">
                  {selectedVariant?.metrics ? `${selectedVariant.metrics.speedupVsBaseline}x speedup` : 'Ready'}
                </span>
              </div>
              <div className="flex-1 overflow-auto p-3 custom-scrollbar leading-relaxed">
                <pre className="text-emerald-300/90 whitespace-pre font-mono select-text">{targetCode}</pre>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col w-full h-full overflow-hidden">
            <div className="bg-zinc-900/90 px-3 py-1.5 border-b border-zinc-800 text-[11px] font-semibold text-zinc-300 flex items-center justify-between">
              <span>{selectedVariant?.name || 'Active Kernel'}</span>
              <span className="text-[10px] text-zinc-500">Triton GPU Kernel</span>
            </div>
            <div className="flex-1 overflow-auto p-3 custom-scrollbar leading-relaxed">
              <pre className="text-zinc-200 whitespace-pre font-mono select-text">{targetCode}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
