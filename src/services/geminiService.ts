import { HardwareProfile, KernelProblem, KernelVariant } from '../types';

export interface EvolveStepParams {
  problem: KernelProblem;
  hardware: HardwareProfile;
  parentCode: string;
  parentStrategy?: string;
  generation: number;
  variantIndex: number;
  language?: 'triton' | 'cuda' | 'cpp';
  userDirectives?: string;
}

export async function checkServerHealth(): Promise<{ status: string; hasGeminiKey: boolean }> {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (err) {
    return { status: 'offline', hasGeminiKey: false };
  }
}

export async function requestEvolveVariant(params: EvolveStepParams): Promise<{
  name: string;
  strategy: any;
  title: string;
  description: string;
  code: string;
  tileParams?: Record<string, any>;
  aiExplanation?: string;
}> {
  const res = await fetch('/api/evolve-step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `Server returned ${res.status}`);
  }

  const data = await res.json();
  return data.variant;
}

export async function requestAIKernelConsult(params: {
  problem: KernelProblem;
  hardware: HardwareProfile;
  currentCode: string;
  prompt: string;
  messageHistory?: Array<{ role: 'user' | 'assistant'; content: string }>;
}): Promise<string> {
  const res = await fetch('/api/ai-kernel-consult', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || 'Failed to consult AI');
  }

  const data = await res.json();
  return data.reply;
}
