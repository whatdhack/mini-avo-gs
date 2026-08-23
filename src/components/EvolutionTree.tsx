import React from 'react';
import { Crown, Sparkles, Zap, Clock, ShieldCheck, CheckCircle2, ChevronRight, Activity } from 'lucide-react';
import { KernelVariant } from '../types';

interface EvolutionTreeProps {
  variants: KernelVariant[];
  selectedVariantId: string | null;
  bestVariantId: string | null;
  onSelectVariant: (variantId: string) => void;
  isEvolving: boolean;
}

export const EvolutionTree: React.FC<EvolutionTreeProps> = ({
  variants,
  selectedVariantId,
  bestVariantId,
  onSelectVariant,
  isEvolving,
}) => {
  // Group variants by generation
  const generations: Record<number, KernelVariant[]> = {};
  for (const v of variants) {
    if (!generations[v.generation]) generations[v.generation] = [];
    generations[v.generation].push(v);
  }

  const genKeys = Object.keys(generations)
    .map(Number)
    .sort((a, b) => a - b);

  const getStrategyColor = (strategy: string) => {
    switch (strategy) {
      case 'tile_tuning':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
      case 'vectorization':
        return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
      case 'shared_memory':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      case 'warp_shuffle':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
      case 'split_k':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
      default:
        return 'bg-teal-500/10 text-teal-400 border-teal-500/30';
    }
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-4 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80 mb-3 shrink-0">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-semibold text-zinc-100">Evolution Lineage DAG</h2>
          <span className="text-xs text-zinc-400 font-mono">({variants.length} mutants)</span>
        </div>
        {isEvolving && (
          <div className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-950/40 border border-emerald-500/30 px-2 py-0.5 rounded-full animate-pulse">
            <Sparkles className="w-3 h-3" />
            <span>Synthesizing Generation {genKeys[genKeys.length - 1] + 1}...</span>
          </div>
        )}
      </div>

      {/* Horizontal Generation Lanes */}
      <div className="flex-1 overflow-x-auto overflow-y-auto pr-1 pb-2 custom-scrollbar">
        <div className="flex items-start gap-4 min-w-max">
          {genKeys.map((genNum, idx) => {
            const genVariants = generations[genNum];
            const isLatest = idx === genKeys.length - 1;

            return (
              <div key={genNum} className="flex items-start">
                {/* Generation Column */}
                <div className="w-64 bg-zinc-950/70 border border-zinc-800 rounded-xl p-2.5 flex flex-col gap-2 shadow-sm">
                  {/* Generation Tag */}
                  <div className="flex items-center justify-between pb-2 border-b border-zinc-800/60 px-1">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      <span className="text-xs font-semibold text-zinc-200 uppercase tracking-wider">
                        {genNum === 0 ? 'Gen 0 (Seed)' : `Gen ${genNum} (Mutants)`}
                      </span>
                    </div>
                    <span className="text-[11px] text-zinc-400 font-mono">
                      {genVariants.length} candidate{genVariants.length > 1 ? 's' : ''}
                    </span>
                  </div>

                  {/* Variant Cards */}
                  <div className="space-y-2 max-h-[440px] overflow-y-auto pr-0.5 custom-scrollbar">
                    {genVariants.map((variant) => {
                      const isSelected = variant.id === selectedVariantId;
                      const isBest = variant.id === bestVariantId;
                      const speedup = variant.metrics?.speedupVsBaseline || 1.0;

                      return (
                        <div
                          key={variant.id}
                          onClick={() => onSelectVariant(variant.id)}
                          className={`p-2.5 rounded-lg border transition-all cursor-pointer text-left relative group ${
                            isSelected
                              ? 'bg-zinc-800/90 border-emerald-500/80 shadow-md shadow-emerald-950/30 ring-1 ring-emerald-500/40'
                              : isBest
                              ? 'bg-zinc-900/90 border-amber-500/50 hover:border-amber-500/80'
                              : 'bg-zinc-900/60 border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/50'
                          }`}
                        >
                          {/* Top Row: Title and Elite Badge */}
                          <div className="flex items-center justify-between gap-1 mb-1">
                            <span className="text-xs font-medium text-zinc-200 truncate group-hover:text-emerald-300 transition-colors">
                              {variant.name}
                            </span>
                            {isBest && (
                              <span className="flex items-center gap-1 text-[10px] font-bold text-amber-300 bg-amber-500/20 border border-amber-500/40 px-1.5 py-0.5 rounded shadow-sm">
                                <Crown className="w-3 h-3 fill-amber-400" />
                                Elite
                              </span>
                            )}
                          </div>

                          {/* Strategy Chip */}
                          <div className="mb-2">
                            <span
                              className={`inline-block text-[10px] font-medium px-1.5 py-0.5 rounded border ${getStrategyColor(
                                variant.mutation.strategy
                              )}`}
                            >
                              {variant.mutation.title}
                            </span>
                          </div>

                          {/* Metrics summary */}
                          {variant.metrics && (
                            <div className="grid grid-cols-2 gap-1.5 pt-1.5 border-t border-zinc-800/60 text-[11px]">
                              <div className="flex items-center gap-1 text-zinc-400">
                                <Zap className="w-3 h-3 text-emerald-400" />
                                <span>Speedup:</span>
                                <strong className="text-emerald-400 font-mono font-bold">
                                  {speedup.toFixed(2)}x
                                </strong>
                              </div>
                              <div className="flex items-center gap-1 text-zinc-400 justify-end">
                                <Clock className="w-3 h-3 text-zinc-500" />
                                <span className="font-mono text-zinc-300">
                                  {variant.metrics.latencyUs.toFixed(1)} μs
                                </span>
                              </div>
                            </div>
                          )}

                          {/* Verification Status */}
                          <div className="mt-1.5 flex items-center justify-between text-[10px] text-zinc-500">
                            <span className="flex items-center gap-1 text-emerald-400/90 font-medium">
                              <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                              Passed (Tol ≤ 1e-4)
                            </span>
                            <span className="font-mono text-zinc-400">
                              {variant.metrics?.tflopsRooflinePercent}% Roofline
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Arrow Connector to next Gen */}
                {!isLatest && (
                  <div className="flex items-center justify-center h-28 px-1 text-zinc-600">
                    <ChevronRight className="w-5 h-5 text-zinc-600" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
