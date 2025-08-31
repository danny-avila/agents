// src/graphs/SupervisedGraph.ts
import { START, END, StateGraph } from '@langchain/langgraph';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import { GraphNodeKeys, Providers } from '@/common';
import { toolsCondition } from '@/tools/ToolNode';
import type * as l from '@/types/llm';
import { StandardGraph } from '@/graphs/Graph';

const { ROUTER, AGENT, TOOLS } = GraphNodeKeys;

export type SupervisedGraphInput = t.StandardGraphInput & {
  routerEnabled?: boolean;
  routingPolicies?: Array<{
    stage: string;
    agents?: string[];
    model?: Providers;
    parallel?: boolean;
  }>;
  featureFlags?: { multi_model_routing?: boolean; fan_out?: boolean };
  models?: Record<string, l.LLMConfig>;
};

/**
 * SupervisedGraph (opt-in)
 * Inserts a ROUTER node ahead of AGENT/TOOLS and optionally applies
 * stage-based provider overrides using existing extension points.
 */
export class SupervisedGraph extends StandardGraph {
  routerEnabled?: boolean;
  routingPolicies?: SupervisedGraphInput['routingPolicies'];
  featureFlags?: SupervisedGraphInput['featureFlags'];
  models?: SupervisedGraphInput['models'];
  /** Tracks which routing policy/stage is active. Starts at -1 so first router pass selects index 0. */
  private currentStageIndex: number = -1;

  constructor({
    routerEnabled,
    routingPolicies,
    featureFlags,
    models,
    ...standardInput
  }: SupervisedGraphInput) {
    super(standardInput);
    this.routerEnabled = routerEnabled;
    this.routingPolicies = routingPolicies;
    this.featureFlags = featureFlags;
    this.models = models;
  }

  /** Decide the next node after ROUTER and apply per-stage provider overrides if configured. */
  private routerFn = (
    state: t.BaseGraphState,
    config?: RunnableConfig
  ): string => {
    this.config = config;
    if (this.routerEnabled !== true) {
      return AGENT;
    }

    const totalPolicies = this.routingPolicies?.length ?? 0;
    const lastMsg = Array.isArray(state.messages) && state.messages.length > 0 ? state.messages[state.messages.length - 1] : undefined;

    const pickIndex = (): number => {
      if (!this.routingPolicies || this.routingPolicies.length === 0) return -1;
      // simple content-aware matching (only used when at least one policy has `when`)
      for (let i = 0; i < this.routingPolicies.length; i++) {
        const pol = this.routingPolicies[i];
        const cond = (pol as unknown as { when?: unknown }).when as
          | 'always'
          | 'has_tools'
          | 'no_tools'
          | { includes?: string[]; excludes?: string[] }
          | undefined;
        if (cond === 'always') return i;
        if (cond != null) {
          const contentVal = (lastMsg as { content?: unknown })?.content as unknown;
          const contentStr = typeof contentVal === 'string' ? contentVal : '';
          const hasTools = 'tool_calls' in (lastMsg ?? {}) && ((lastMsg as unknown as { tool_calls?: unknown[] }).tool_calls?.length ?? 0) > 0;
          if (cond === 'has_tools' && hasTools) return i;
          if (cond === 'no_tools' && !hasTools) return i;
          if (typeof cond === 'object') {
            const incOk = !cond.includes || cond.includes.some((s) => contentStr && contentStr.includes(s));
            const excOk = !cond.excludes || !cond.excludes.some((s) => contentStr && contentStr.includes(s));
            if (incOk && excOk) return i;
          }
        }
      }
      return -1;
    };

    if (totalPolicies > 0) {
      const hasConditional = (this.routingPolicies ?? []).some((p) => (p as unknown as { when?: unknown }).when != null);
      if (hasConditional) {
        const nextIndex = pickIndex();
        this.currentStageIndex = nextIndex >= 0 ? nextIndex : Math.min(this.currentStageIndex + 1, totalPolicies - 1);
      } else {
        this.currentStageIndex = Math.min(this.currentStageIndex + 1, totalPolicies - 1);
      }
      const policy = this.routingPolicies?.[this.currentStageIndex];
      if (policy && policy.model != null) {
        // If a per-stage model config is provided, use its provider & client options; otherwise fall back to provider only
        const stageConfig = this.models?.[policy.stage];
        const enableFanOut = this.featureFlags?.fan_out === true && policy.parallel === true && (policy.agents?.length ?? 0) > 1;
        if (enableFanOut) {
          const agentNames = policy.agents as string[];
          const self = this;
          const buildModel = (agentName: string) => {
            const cfg = self.models?.[agentName];
            const provider = cfg?.provider ?? policy.model!;
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const { provider: _p, ...clientOptions } = (cfg ?? {}) as l.LLMConfig & t.ClientOptions;
            const instance = self.getNewModel({ provider, clientOptions: clientOptions as unknown as t.ClientOptions });
            return (!self.tools || self.tools.length === 0)
              ? (instance as unknown as Runnable)
              : (instance as t.ModelWithTools).bindTools(self.tools);
          };
          const aggregator = {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            invoke: async (messages: any, cfg?: RunnableConfig) => {
              const TIMEOUT_MS = 12000;
              const MAX_RETRIES = Math.max(0, ((self.featureFlags as unknown as { fan_out_retries?: number })?.fan_out_retries ?? 1));
              const BASE_BACKOFF = Math.max(100, ((self.featureFlags as unknown as { fan_out_backoff_ms?: number })?.fan_out_backoff_ms ?? 400));
              const MAX_CONCURRENCY = Math.max(1, ((self.featureFlags as unknown as { fan_out_concurrency?: number })?.fan_out_concurrency ?? 2));

              // Simple in-memory circuit breaker per provider
              // provider -> { failures: number, openedUntil: number }
              const breaker = new Map<string, { failures: number; openedUntil: number }>();

              const jitter = (n: number): number => n + Math.floor(Math.random() * 100);
              const sleepMs = (n: number): Promise<void> => new Promise((r) => setTimeout(r, n));

              const withBreaker = async (
                providerKey: string,
                fn: () => Promise<unknown>
              ): Promise<unknown> => {
                const now = Date.now();
                const state = breaker.get(providerKey);
                if (state && state.openedUntil > now) {
                  return undefined; // short-circuit while open
                }
                try {
                  const res = await fn();
                  breaker.set(providerKey, { failures: 0, openedUntil: 0 });
                  return res;
                } catch (e) {
                  const current = breaker.get(providerKey) ?? { failures: 0, openedUntil: 0 };
                  const failures = current.failures + 1;
                  // open for 10s after 3 consecutive failures
                  const openedUntil = failures >= 3 ? now + 10_000 : 0;
                  breaker.set(providerKey, { failures, openedUntil });
                  throw e;
                }
              };
              const runBranch = async (name: string): Promise<unknown> => {
                const model = buildModel(name) as unknown as { invoke?: Function };
                if (!model.invoke) { return undefined; }
                const modelCfg = self.models?.[name];
                const providerKey = modelCfg?.provider ?? policy.model!;

                const attemptOnce = async (): Promise<unknown> => {
                  const parentSignal = (cfg as unknown as { signal?: AbortSignal })?.signal;
                  const controller = new AbortController();
                  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
                  const onAbort = (): void => controller.abort();
                  if (parentSignal) {
                    try { parentSignal.addEventListener('abort', onAbort, { once: true }); } catch {}
                  }
                  const started = Date.now();
                  try {
                    const out = await (model.invoke as Function)(messages, { ...(cfg as object), signal: controller.signal } as RunnableConfig);
                    const duration = Date.now() - started;
                    try { dispatchCustomEvent('fanout_branch_end', { provider: providerKey, duration, status: 'ok' }, self.config); } catch {}
                    return out;
                  } catch (e) {
                    const duration = Date.now() - started;
                    try { dispatchCustomEvent('fanout_branch_end', { provider: providerKey, duration, status: 'error' }, self.config); } catch {}
                    throw e;
                  } finally {
                    clearTimeout(timer);
                    if (parentSignal) {
                      try { parentSignal.removeEventListener('abort', onAbort as EventListener); } catch {}
                    }
                  }
                };

                for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
                  try {
                    return await withBreaker(String(providerKey), attemptOnce);
                  } catch (e) {
                    if (attempt === MAX_RETRIES) return undefined;
                    const delay = jitter(BASE_BACKOFF * Math.pow(2, attempt));
                    await sleepMs(delay);
                  }
                }
                return undefined;
              };

              // Concurrency limiter
              const queue = agentNames.slice();
              const running: Promise<unknown>[] = [];
              const settledResults: PromiseSettledResult<unknown>[] = [] as unknown as PromiseSettledResult<unknown>[];
              const launchNext = (): void => {
                if (queue.length === 0) return;
                const name = queue.shift() as string;
                const p = runBranch(name)
                  .then((v) => ({ status: 'fulfilled', value: v } as PromiseSettledResult<unknown>))
                  .catch((err) => ({ status: 'rejected', reason: err } as PromiseSettledResult<unknown>))
                  .finally(() => {
                    settledResults.push(running.splice(running.indexOf(p as unknown as Promise<unknown>), 1)[0] as unknown as PromiseSettledResult<unknown>);
                    launchNext();
                  });
                running.push(p as unknown as Promise<unknown>);
                if (running.length < MAX_CONCURRENCY) launchNext();
              };
              launchNext();
              await Promise.all(running);
              const settled = settledResults.length > 0 ? settledResults : (await Promise.allSettled(agentNames.map((n) => runBranch(n))));
              const outputs = settled.map((s) => (s.status === 'fulfilled' ? s.value : undefined)).filter((v) => v != null);
              // Combine text content strings
              const textParts = outputs.map((o) => (typeof (o as { content?: unknown })?.content === 'string' ? (o as { content: string }).content : '')).filter(Boolean);
              const combinedText = textParts.join('\n\n');

              // Synthesize a single final answer using a chosen synthesizer model
              const synthCfg = self.models?.['synth'] as l.LLMConfig | undefined;
              const synthProvider = synthCfg?.provider ?? policy.model!;
              // eslint-disable-next-line @typescript-eslint/no-unused-vars
              const { provider: _p2, ...synthClientOptions } = (synthCfg ?? {}) as l.LLMConfig & t.ClientOptions;
              const synthInstance = self.getNewModel({ provider: synthProvider, clientOptions: synthClientOptions as unknown as t.ClientOptions });
              const synthModel = (!self.tools || self.tools.length === 0)
                ? (synthInstance as unknown as Runnable)
                : (synthInstance as t.ModelWithTools).bindTools(self.tools);

              const prompt = new HumanMessage(`Synthesize the following model outputs into one concise, high-quality answer. Do not repeat, deduplicate overlap, and keep it short.\n\n${combinedText}`);
              const baseMessages = Array.isArray(messages) ? (messages as unknown as unknown[]) : [];
              const finalMsg = await (synthModel as unknown as { invoke: Function }).invoke([...baseMessages, prompt], cfg);
              return finalMsg as unknown;
            },
            // Provide a minimal stream API yielding the final aggregated chunk
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            stream: async function* (messages: any, cfg?: RunnableConfig) {
              const selfAny = this as unknown as { invoke?: Function };
              const finalMsg = selfAny.invoke ? await selfAny.invoke(messages, cfg) : { content: '' };
              // yield a single chunk-like object
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              yield finalMsg as any;
            },
          } as unknown as Runnable;
          this.boundModel = aggregator;
        } else if (stageConfig) {
          const stageProvider = stageConfig.provider ?? policy.model;
          // Extract provider-specific client options from stage config
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { provider: _p, ...clientOptions } = stageConfig as l.LLMConfig & t.ClientOptions;
          const modelInstance = this.getNewModel({
            provider: stageProvider,
            clientOptions: clientOptions as unknown as t.ClientOptions,
          });
          this.boundModel =
            !this.tools || this.tools.length === 0
              ? (modelInstance as unknown as Runnable)
              : (modelInstance as t.ModelWithTools).bindTools(this.tools);
        } else {
          const modelInstance = this.getNewModel({ provider: policy.model });
          this.boundModel =
            !this.tools || this.tools.length === 0
              ? (modelInstance as unknown as Runnable)
              : (modelInstance as t.ModelWithTools).bindTools(this.tools);
        }
      }
    }
    // Route to tools immediately if tool calls are present and not yet invoked; else agent
    return toolsCondition(state.messages as unknown as t.BaseGraphState['messages'], this.invokedToolIds) === TOOLS ? TOOLS : AGENT;
  };

  /** After AGENT, route to TOOLS or END using existing logic. */
  private routeAfterAgent = (
    state: t.BaseGraphState,
    config?: RunnableConfig
  ): string => {
    this.config = config;
    return toolsCondition(state, this.invokedToolIds);
  };

  /**
   * ROUTER -> (AGENT | TOOLS | END)
   * AGENT  --(conditional)--> (TOOLS | END)
   * TOOLS  --(edge)--> (END | ROUTER)
   */
  createWorkflow(): t.CompiledWorkflow<t.BaseGraphState> {
    const workflow = new StateGraph<t.BaseGraphState>({
      channels: (this as unknown as { graphState: t.GraphStateChannels<t.BaseGraphState> }).graphState,
    })
      .addNode(AGENT, this.createCallModel())
      .addNode(TOOLS, this.initializeTools())
      .addNode(ROUTER, async () => ({}))
      .addEdge(START, ROUTER)
      .addConditionalEdges(ROUTER, this.routerFn)
      .addConditionalEdges(AGENT, this.routeAfterAgent)
      .addEdge(TOOLS, this.toolEnd ? END : ROUTER);

    // Cast to unknown to avoid tight coupling to external types; options are opt-in
    return workflow.compile(this.compileOptions as unknown as never);
  }
}
