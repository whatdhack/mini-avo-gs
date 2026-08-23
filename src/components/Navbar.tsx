import React from 'react';
import { Cpu, Dna, Sparkles, Download, Layers, Play, Pause, RotateCcw, ShieldCheck, Zap } from 'lucide-react';
import { HARDWARE_PROFILES } from '../data/hardwareProfiles';
import { KERNEL_PRESETS } from '../data/kernelPresets';
import { HardwareId } from '../types';

interface NavbarProps {
  selectedProblemId: string;
  onSelectProblem: (id: string) => void;
  selectedHardwareId: HardwareId;
  onSelectHardware: (id: HardwareId) => void;
  isEvolving: boolean;
  onToggleEvolve: () => void;
  onResetEvolution: () => void;
  onOpenExport: () => void;
  currentGen: number;
  bestSpeedup: number;
  hasApiKey: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  selectedProblemId,
  onSelectProblem,
  selectedHardwareId,
  onSelectHardware,
  isEvolving,
  onToggleEvolve,
  onResetEvolution,
  onOpenExport,
  currentGen,
  bestSpeedup,
  hasApiKey,
}) => {
  const currentHardware = HARDWARE_PROFILES[selectedHardwareId] || HARDWARE_PROFILES['h100-sxm'];

  return (
    <header className="border-b border-zinc-800 bg-zinc-950/90 backdrop-blur-md sticky top-0 z-40 px-4 py-2.5">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
        {/* Brand & Identity */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500/20 via-teal-500/10 to-cyan-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-sm shadow-emerald-950/50">
            <Dna className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold tracking-tight text-zinc-100 text-base">MiniAVO</h1>
              <span className="px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                GPU MODE Edition
              </span>
            </div>
            <p className="text-xs text-zinc-400 hidden sm:block">
              AI Evolutionary Kernel Studio for GPU MODE Leaderboards
            </p>
          </div>
        </div>

        {/* Problem & Hardware Selectors */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Workload Picker */}
          <div className="flex items-center gap-1.5 bg-zinc-900/90 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300">
            <Layers className="w-3.5 h-3.5 text-teal-400 shrink-0" />
            <span className="text-zinc-500 hidden md:inline">Kernel:</span>
            <select
              value={selectedProblemId}
              onChange={(e) => onSelectProblem(e.target.value)}
              className="bg-transparent text-zinc-200 font-medium focus:outline-none cursor-pointer pr-1"
            >
              {KERNEL_PRESETS.map((p) => (
                <option key={p.id} value={p.id} className="bg-zinc-900 text-zinc-200">
                  {p.name} ({p.complexity})
                </option>
              ))}
            </select>
          </div>

          {/* Hardware Picker */}
          <div className="flex items-center gap-1.5 bg-zinc-900/90 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300">
            <Cpu className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
            <span className="text-zinc-500 hidden md:inline">Target:</span>
            <select
              value={selectedHardwareId}
              onChange={(e) => onSelectHardware(e.target.value as HardwareId)}
              className="bg-transparent text-zinc-200 font-medium focus:outline-none cursor-pointer pr-1"
            >
              {Object.values(HARDWARE_PROFILES).map((h) => (
                <option key={h.id} value={h.id} className="bg-zinc-900 text-zinc-200">
                  {h.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Status Indicators & Main Actions */}
        <div className="flex items-center gap-2.5">
          {/* Generation & Speedup Badge */}
          <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 px-2.5 py-1 rounded-md text-xs">
            <span className="text-zinc-400">
              Gen <strong className="text-zinc-200 font-mono">{currentGen}</strong>
            </span>
            <span className="text-zinc-700">|</span>
            <span className="text-zinc-400 flex items-center gap-1">
              Top:{' '}
              <strong className="text-emerald-400 font-mono flex items-center gap-0.5">
                <Zap className="w-3 h-3 fill-emerald-400" />
                {bestSpeedup.toFixed(2)}x
              </strong>
            </span>
          </div>

          {/* Evolution Start/Pause Button */}
          <button
            onClick={onToggleEvolve}
            id="btn-evolve-toggle"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold shadow-sm transition-all ${
              isEvolving
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-950/40'
            }`}
          >
            {isEvolving ? (
              <>
                <Pause className="w-3.5 h-3.5 fill-current animate-pulse" />
                <span>Pause Loop</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Evolve Step</span>
              </>
            )}
          </button>

          {/* Reset Loop */}
          <button
            onClick={onResetEvolution}
            title="Reset Evolution Population"
            className="p-1.5 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80 rounded-md transition-colors border border-transparent hover:border-zinc-700"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Export Code Modal */}
          <button
            onClick={onOpenExport}
            className="flex items-center gap-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 px-2.5 py-1.5 rounded-lg text-xs font-medium border border-zinc-700 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Export</span>
          </button>
        </div>
      </div>
    </header>
  );
};
