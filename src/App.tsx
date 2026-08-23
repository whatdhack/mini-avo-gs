import React, { useState, useEffect, useCallback, useRef } from 'react';
import confetti from 'canvas-confetti';
import {
  Dna,
  GitBranch,
  Gauge,
  Code2,
  Bot,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  Zap,
  Cpu,
  Layers,
  ChevronRight,
} from 'lucide-react';
import { HARDWARE_PROFILES } from './data/hardwareProfiles';
import { KERNEL_PRESETS } from './data/kernelPresets';
import { HardwareId, KernelVariant, EvolutionRun } from './types';
import { Navbar } from './components/Navbar';
import { EvolutionTree } from './components/EvolutionTree';
import { KernelDiffViewer } from './components/KernelDiffViewer';
import { PerformanceDashboard } from './components/PerformanceDashboard';
import { KernelEditor } from './components/KernelEditor';
import { AIOptimizerPanel } from './components/AIOptimizerPanel';
import { ExportModal } from './components/ExportModal';
import { checkServerHealth, requestEvolveVariant } from './services/geminiService';
import { simulateKernelBenchmark, generateCompilerLogs } from './utils/benchmarkSimulator';

export default function App() {
  const [selectedProblemId, setSelectedProblemId] = useState<string>('gpumode_matmul_v2');
  const [selectedHardwareId, setSelectedHardwareId] = useState<HardwareId>('h100-sxm');
  const [selectedVariantId, setSelectedVariantId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'lineage' | 'roofline' | 'editor' | 'ai'>('lineage');
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [isEvolving, setIsEvolving] = useState(false);
  const [hasApiKey, setHasApiKey] = useState(false);

  const problem = KERNEL_PRESETS.find((p) => p.id === selectedProblemId) || KERNEL_PRESETS[0];
  const hardware = HARDWARE_PROFILES[selectedHardwareId] || HARDWARE_PROFILES['h100-sxm'];

  // Initialize Generation 0 Baseline
  const createGen0Variant = useCallback((prob = problem, hw = hardware): KernelVariant => {
    const metrics = simulateKernelBenchmark(prob, hw, {}, 0, true);
    return {
      id: 'gen0-baseline',
      name: 'Baseline_Gen0',
      generation: 0,
      parentId: null,
      code: prob.initialTritonCode,
      language: 'triton',
      status: 'elite',
      mutation: {
        strategy: 'tile_tuning',
        title: 'Initial Seed Baseline',
        description: 'Standard Triton baseline implementation without hardware-specific optimizations.',
        tileParams: {
          BLOCK_M: 64,
          BLOCK_N: 64,
          BLOCK_K: 32,
          num_warps: 4,
          num_stages: 2,
          GROUP_SIZE_M: 8,
        },
      },
      metrics,
      aiExplanation: 'Initial reference kernel with standard tile dimensions and baseline memory layout.',
      compilerLogs: generateCompilerLogs('Baseline_Gen0', hw, metrics),
      createdAt: Date.now(),
    };
  }, [problem, hardware]);

  const [variants, setVariants] = useState<KernelVariant[]>([createGen0Variant()]);
  const [bestVariantId, setBestVariantId] = useState<string>('gen0-baseline');
  const [currentGen, setCurrentGen] = useState<number>(0);

  // Check server health
  useEffect(() => {
    checkServerHealth().then((res) => {
      setHasApiKey(res.hasGeminiKey);
    });
  }, []);

  // Reset when problem changes
  useEffect(() => {
    const seed = createGen0Variant(problem, hardware);
    setVariants([seed]);
    setSelectedVariantId(seed.id);
    setBestVariantId(seed.id);
    setCurrentGen(0);
    setIsEvolving(false);
  }, [selectedProblemId, selectedHardwareId, createGen0Variant]);

  const selectedVariant = variants.find((v) => v.id === selectedVariantId) || variants[0];
  const bestVariant = variants.find((v) => v.id === bestVariantId) || variants[0];
  const bestSpeedup = bestVariant?.metrics?.speedupVsBaseline || 1.0;
  const parentVariant = variants.find((v) => v.id === selectedVariant?.parentId) || null;

  // Single Evolution Step Executor
  const stepEvolution = useCallback(
    async (userDirectives?: string) => {
      const nextGen = currentGen + 1;
      const parentToMutate = bestVariant;

      const populationCount = 3; // Generate 3 mutant candidates per generation
      const newVariants: KernelVariant[] = [];

      for (let i = 0; i < populationCount; i++) {
        try {
          const mutated = await requestEvolveVariant({
            problem,
            hardware,
            parentCode: parentToMutate.code,
            parentStrategy: parentToMutate.mutation.strategy,
            generation: nextGen,
            variantIndex: i,
            language: 'triton',
            userDirectives,
          });

          const variantPartial: Partial<KernelVariant> = {
            name: mutated.name,
            mutation: {
              strategy: mutated.strategy,
              title: mutated.title,
              description: mutated.description,
              tileParams: mutated.tileParams,
            },
          };

          const metrics = simulateKernelBenchmark(problem, hardware, variantPartial, nextGen, false);
          const varId = `gen${nextGen}-var${i + 1}-${Date.now()}`;

          const variant: KernelVariant = {
            id: varId,
            name: mutated.name,
            generation: nextGen,
            parentId: parentToMutate.id,
            code: mutated.code,
            language: 'triton',
            status: 'success',
            mutation: {
              strategy: mutated.strategy,
              title: mutated.title,
              description: mutated.description,
              tileParams: mutated.tileParams,
            },
            metrics,
            aiExplanation: mutated.aiExplanation,
            compilerLogs: generateCompilerLogs(mutated.name, hardware, metrics),
            createdAt: Date.now(),
          };

          newVariants.push(variant);
        } catch (err) {
          console.error('Error evolving variant:', err);
        }
      }

      if (newVariants.length > 0) {
        setVariants((prev) => {
          const updated = [...prev, ...newVariants];
          // Find overall best
          let topVar = updated[0];
          for (const v of updated) {
            if ((v.metrics?.speedupVsBaseline || 0) > (topVar.metrics?.speedupVsBaseline || 0)) {
              topVar = v;
            }
          }

          if (topVar.id !== bestVariantId && (topVar.metrics?.speedupVsBaseline || 1) > 1.2) {
            confetti({
              particleCount: 40,
              spread: 60,
              origin: { y: 0.8 },
              colors: ['#10b981', '#06b6d4', '#3b82f6'],
            });
          }

          setBestVariantId(topVar.id);
          setSelectedVariantId(topVar.id);
          return updated;
        });

        setCurrentGen(nextGen);
      }
    },
    [currentGen, bestVariant, bestVariantId, problem, hardware]
  );

  // Auto-evolution loop timer
  const evolvingRef = useRef(isEvolving);
  evolvingRef.current = isEvolving;

  useEffect(() => {
    if (!isEvolving) return;

    const interval = setInterval(async () => {
      if (evolvingRef.current && currentGen < 6) {
        await stepEvolution();
      } else if (currentGen >= 6) {
        setIsEvolving(false);
      }
    }, 4500);

    return () => clearInterval(interval);
  }, [isEvolving, currentGen, stepEvolution]);

  const handleToggleEvolve = () => {
    if (!isEvolving) {
      stepEvolution();
      setIsEvolving(true);
    } else {
      setIsEvolving(false);
    }
  };

  const handleResetEvolution = () => {
    const seed = createGen0Variant(problem, hardware);
    setVariants([seed]);
    setSelectedVariantId(seed.id);
    setBestVariantId(seed.id);
    setCurrentGen(0);
    setIsEvolving(false);
  };

  const handleRunCustomBenchmark = (code: string) => {
    const nextGen = currentGen + 1;
    const customVarId = `custom-gen${nextGen}-${Date.now()}`;
    const variantPartial: Partial<KernelVariant> = {
      name: `Custom_Tuned_v${nextGen}`,
      mutation: {
        strategy: 'custom_ai',
        title: 'User Manual Optimization',
        description: 'Custom kernel tuned directly in the code editor.',
      },
    };

    const metrics = simulateKernelBenchmark(problem, hardware, variantPartial, nextGen, false);

    const customVariant: KernelVariant = {
      id: customVarId,
      name: `Custom_Tuned_v${nextGen}`,
      generation: nextGen,
      parentId: selectedVariant?.id || null,
      code,
      language: 'triton',
      status: 'success',
      mutation: {
        strategy: 'custom_ai',
        title: 'User Manual Tuning',
        description: 'Direct code edits compiled and benchmarked against hardware roofline.',
      },
      metrics,
      aiExplanation: 'User refined kernel compiled with custom tile shapes and memory operations.',
      compilerLogs: generateCompilerLogs(`Custom_Tuned_v${nextGen}`, hardware, metrics),
      createdAt: Date.now(),
    };

    setVariants((prev) => [...prev, customVariant]);
    setSelectedVariantId(customVarId);
    if ((metrics.speedupVsBaseline || 0) > bestSpeedup) {
      setBestVariantId(customVarId);
      confetti({ particleCount: 30, spread: 50, origin: { y: 0.8 } });
    }
    setCurrentGen(nextGen);
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-200">
      {/* Top Navigation */}
      <Navbar
        selectedProblemId={selectedProblemId}
        onSelectProblem={setSelectedProblemId}
        selectedHardwareId={selectedHardwareId}
        onSelectHardware={setSelectedHardwareId}
        isEvolving={isEvolving}
        onToggleEvolve={handleToggleEvolve}
        onResetEvolution={handleResetEvolution}
        onOpenExport={() => setIsExportOpen(true)}
        currentGen={currentGen}
        bestSpeedup={bestSpeedup}
        hasApiKey={hasApiKey}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 flex flex-col gap-4">
        {/* Workload Banner & View Switcher */}
        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3 shadow-sm">
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-semibold uppercase tracking-wider text-teal-400 bg-teal-950/60 border border-teal-500/30 px-2 py-0.5 rounded">
                {problem.category}
              </span>
              <h2 className="text-base font-bold text-zinc-100">{problem.name}</h2>
              <span className="text-xs text-zinc-400 font-mono">
                [Shapes: {problem.config.tensorShapes}]
              </span>
            </div>
            <p className="text-xs text-zinc-400 max-w-2xl">{problem.description}</p>
          </div>

          {/* Tab Navigation */}
          <div className="flex items-center bg-zinc-950 border border-zinc-800 rounded-lg p-1 text-xs">
            <button
              onClick={() => setActiveTab('lineage')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'lineage'
                  ? 'bg-zinc-800 text-emerald-400 font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <GitBranch className="w-3.5 h-3.5" />
              <span>Lineage & Diff</span>
            </button>

            <button
              onClick={() => setActiveTab('roofline')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'roofline'
                  ? 'bg-zinc-800 text-emerald-400 font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Gauge className="w-3.5 h-3.5" />
              <span>Roofline & Profiler</span>
            </button>

            <button
              onClick={() => setActiveTab('editor')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'editor'
                  ? 'bg-zinc-800 text-emerald-400 font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              <span>Kernel Playground</span>
            </button>

            <button
              onClick={() => setActiveTab('ai')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'ai'
                  ? 'bg-zinc-800 text-emerald-400 font-semibold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Bot className="w-3.5 h-3.5" />
              <span>AI Architect</span>
            </button>
          </div>
        </div>

        {/* Tab Content Panes */}
        <div className="flex-1 min-h-[560px]">
          {activeTab === 'lineage' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-full">
              {/* Left Column: Evolutionary Lineage DAG */}
              <div className="lg:col-span-5 h-[560px]">
                <EvolutionTree
                  variants={variants}
                  selectedVariantId={selectedVariantId}
                  bestVariantId={bestVariantId}
                  onSelectVariant={setSelectedVariantId}
                  isEvolving={isEvolving}
                />
              </div>

              {/* Right Column: Comparative Code Diff Viewer */}
              <div className="lg:col-span-7 h-[560px]">
                <KernelDiffViewer
                  problem={problem}
                  selectedVariant={selectedVariant}
                  parentVariant={parentVariant}
                />
              </div>
            </div>
          )}

          {activeTab === 'roofline' && (
            <div className="h-full">
              <PerformanceDashboard
                problem={problem}
                hardware={hardware}
                selectedVariant={selectedVariant}
                allVariants={variants}
              />
            </div>
          )}

          {activeTab === 'editor' && (
            <div className="h-[560px]">
              <KernelEditor
                problem={problem}
                hardware={hardware}
                selectedVariant={selectedVariant}
                onRunCustomBenchmark={handleRunCustomBenchmark}
              />
            </div>
          )}

          {activeTab === 'ai' && (
            <div className="h-[560px]">
              <AIOptimizerPanel
                problem={problem}
                hardware={hardware}
                selectedVariant={selectedVariant}
                onApplyPromptMutation={(promptText) => stepEvolution(promptText)}
                isEvolving={isEvolving}
              />
            </div>
          )}
        </div>
      </main>

      {/* Export Modal */}
      <ExportModal
        isOpen={isExportOpen}
        onClose={() => setIsExportOpen(false)}
        problem={problem}
        hardware={hardware}
        selectedVariant={selectedVariant}
      />
    </div>
  );
}
