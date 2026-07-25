import type * as t from '@/types';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

/**
 * The overflow-recovery bookkeeping on AgentContext: the budget correction,
 * its restoration between runs, and the stall detector that stops a recovery
 * loop when a correction demonstrably changed nothing.
 */
describe('AgentContext overflow recovery state', () => {
  const createContext = (maxContextTokens?: number): AgentContext =>
    AgentContext.fromConfig({
      agentId: 'overflow-agent',
      provider: Providers.ANTHROPIC,
      instructions: 'Test instructions',
      maxContextTokens,
    } as Partial<t.AgentInputs> as t.AgentInputs);

  it('records the correction and counts the attempt', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 274_468);

    expect(context.maxContextTokens).toBe(190_000);
    expect(context.overflowRecoveryAttempts).toBe(1);
    /** Forces the pruner to be rebuilt against the corrected budget. */
    expect(context.pruneMessages).toBeUndefined();
  });

  it('restores the pre-correction budget on reset', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 274_468);
    context.applyContextBudgetCorrection(133_000, 180_000);
    context.reset();

    expect(context.maxContextTokens).toBe(1_000_000);
    expect(context.overflowRecoveryAttempts).toBe(0);
  });

  it('leaves an untouched budget alone on reset', () => {
    const context = createContext(1_000_000);
    context.maxContextTokens = 500_000;
    context.reset();

    expect(context.maxContextTokens).toBe(500_000);
  });

  it('reports a stall when the prompt did not shrink', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 250_000);

    expect(context.overflowRecoveryStalled(250_000)).toBe(true);
    expect(context.overflowRecoveryStalled(260_000)).toBe(true);
  });

  it('reports no stall while the prompt is still shrinking', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 250_000);

    expect(context.overflowRecoveryStalled(180_000)).toBe(false);
  });

  it('reports no stall before any correction, or without a measurement', () => {
    const context = createContext(1_000_000);
    expect(context.overflowRecoveryStalled(250_000)).toBe(false);

    context.applyContextBudgetCorrection(190_000, 250_000);
    expect(context.overflowRecoveryStalled(undefined)).toBe(false);
  });

  it('clears the stall measurement on reset', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 250_000);
    context.reset();

    expect(context.overflowRecoveryStalled(250_000)).toBe(false);
  });
});
