import React, { useState } from 'react';
import { Bot, Send, Sparkles, Terminal, SlidersHorizontal, Zap, Loader2 } from 'lucide-react';
import { HardwareProfile, KernelProblem, KernelVariant } from '../types';
import { requestAIKernelConsult } from '../services/geminiService';

interface AIOptimizerPanelProps {
  problem: KernelProblem;
  hardware: HardwareProfile;
  selectedVariant: KernelVariant | null;
  onApplyPromptMutation: (customPrompt: string) => void;
  isEvolving: boolean;
}

export const AIOptimizerPanel: React.FC<AIOptimizerPanelProps> = ({
  problem,
  hardware,
  selectedVariant,
  onApplyPromptMutation,
  isEvolving,
}) => {
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([
    {
      role: 'assistant',
      content: `Hello! I am your AI Kernel Optimization Assistant for **${hardware.name}** (${hardware.architecture}).
I can help eliminate shared memory bank conflicts, vectorize global memory loads with TMA / cp.async, tune tile block sizes, or restructure kernel logic for maximal speedup. 

Type an optimization goal below or click a quick prompt!`,
    },
  ]);

  const [inputPrompt, setInputPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const quickPrompts = [
    'Eliminate shared memory bank conflicts and align tile strides',
    'Vectorize memory loads with 128-bit float4 and TMA hints',
    'Enable Split-K reduction for small batch latency',
    'Optimize register allocation to hit 100% warp occupancy',
  ];

  const handleSendMessage = async (promptToSend?: string) => {
    const text = promptToSend || inputPrompt;
    if (!text.trim() || isLoading) return;

    const newMessages = [...messages, { role: 'user' as const, content: text }];
    setMessages(newMessages);
    setInputPrompt('');
    setIsLoading(true);

    try {
      const reply = await requestAIKernelConsult({
        problem,
        hardware,
        currentCode: selectedVariant?.code || problem.initialTritonCode,
        prompt: text,
        messageHistory: messages,
      });

      setMessages([...newMessages, { role: 'assistant' as const, content: reply }]);
    } catch (err: any) {
      setMessages([
        ...newMessages,
        {
          role: 'assistant' as const,
          content: `⚠️ Error during AI consultation: ${err.message || 'Failed to reach backend'}. Using local compiler heuristics.`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-4 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80 mb-3 shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-semibold text-zinc-100">AI Kernel Architect & Consultant</h2>
        </div>
        <span className="text-[11px] font-mono text-zinc-400 bg-zinc-800/80 px-2 py-0.5 rounded border border-zinc-700">
          Gemini 3.7 Flash Engine
        </span>
      </div>

      {/* Message Stream */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-1 mb-3 custom-scrollbar text-xs">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg leading-relaxed ${
              m.role === 'assistant'
                ? 'bg-zinc-950/80 border border-zinc-800 text-zinc-200'
                : 'bg-emerald-950/30 border border-emerald-500/30 text-emerald-200 ml-4'
            }`}
          >
            <div className="flex items-center gap-1.5 font-semibold mb-1 text-[11px]">
              {m.role === 'assistant' ? (
                <>
                  <Sparkles className="w-3 h-3 text-emerald-400" />
                  <span className="text-emerald-400">EvolveKernel AI</span>
                </>
              ) : (
                <span className="text-emerald-300">You (Kernel Engineer)</span>
              )}
            </div>
            <div className="whitespace-pre-wrap font-sans text-zinc-300">{m.content}</div>
          </div>
        ))}
        {isLoading && (
          <div className="p-3 bg-zinc-950/80 border border-zinc-800 rounded-lg flex items-center gap-2 text-xs text-zinc-400">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-400" />
            <span>Analyzing kernel architecture for {hardware.name}...</span>
          </div>
        )}
      </div>

      {/* Quick Optimization Suggestions */}
      <div className="flex flex-wrap gap-1.5 mb-2.5 shrink-0">
        {quickPrompts.map((qp, i) => (
          <button
            key={i}
            onClick={() => {
              setInputPrompt(qp);
              handleSendMessage(qp);
            }}
            className="text-[10px] bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200 px-2 py-1 rounded transition-colors"
          >
            + {qp}
          </button>
        ))}
      </div>

      {/* Input Box & Action Buttons */}
      <div className="flex items-center gap-2 pt-2 border-t border-zinc-800/80 shrink-0">
        <input
          type="text"
          value={inputPrompt}
          onChange={(e) => setInputPrompt(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder={`Ask AI to optimize kernel (e.g. "Use TMA async copy on ${hardware.name}")`}
          className="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-emerald-500/80 focus:ring-1 focus:ring-emerald-500/30"
        />

        <button
          onClick={() => handleSendMessage()}
          disabled={isLoading || !inputPrompt.trim()}
          className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 p-2 rounded-lg border border-zinc-700 disabled:opacity-40 transition-colors"
          title="Send message to AI"
        >
          <Send className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={() => {
            if (inputPrompt.trim()) {
              onApplyPromptMutation(inputPrompt);
              setInputPrompt('');
            }
          }}
          disabled={isEvolving || !inputPrompt.trim()}
          className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-3 py-2 rounded-lg shadow-sm shadow-emerald-950/50 flex items-center gap-1.5 disabled:opacity-40 transition-all shrink-0"
          title="Directly trigger an evolutionary mutant with this guidance"
        >
          <Zap className="w-3.5 h-3.5 fill-current" />
          <span>Evolve Mutant</span>
        </button>
      </div>
    </div>
  );
};
