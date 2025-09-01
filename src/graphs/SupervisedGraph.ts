// src/graphs/SupervisedGraph.ts
import { START, END, StateGraph } from '@langchain/langgraph';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessageLike } from '@langchain/core/messages';
import { GraphNodeKeys, Providers } from '@/common';
import { toolsCondition } from '@/tools/ToolNode';
import type * as l from '@/types/llm';
import { StandardGraph } from '@/graphs/Graph';

const { ROUTER, AGENT, TOOLS } = GraphNodeKeys;

/** Circuit breaker state for a provider */
interface CircuitBreakerState {
  failures: number;
  openedUntil: number;
}

/** Configuration for fan-out execution */
interface FanOutConfig {
  timeoutMs: number;
  maxRetries: number;
  baseBackoffMs: number;
  maxConcurrency: number;
}

/** Aggregator for fan-out model execution */
interface ModelAggregator {
  invoke: (
    messages: BaseMessageLike[],
    config?: RunnableConfig
  ) => Promise<unknown>;
  stream: (
    messages: BaseMessageLike[],
    config?: RunnableConfig
  ) => AsyncGenerator<unknown, void, unknown>;
}

export type RoutingPolicy = {
  stage: string;
  agents?: string[];
  model?: Providers;
  parallel?: boolean;
};

export type SupervisedGraphInput = t.StandardGraphInput & {
  routerEnabled?: boolean;
  routingPolicies?: Array<RoutingPolicy>;
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
  /** Circuit breaker for providers */
  private circuitBreaker = new Map<string, CircuitBreakerState>();

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

  buildModel(agentName: string, policy: RoutingPolicy): t.ChatModelInstance {
    const cfg = this.models?.[agentName];
    const provider = cfg?.provider ?? policy.model!;
    const { provider: _p, ...clientOptions } = (cfg ?? {}) as l.LLMConfig &
      t.ClientOptions;
    const instance = this.getNewModel({
      provider,
      clientOptions: clientOptions as unknown as t.ClientOptions,
    });
    return !this.tools || this.tools.length === 0
      ? instance
      : (instance.bindTools(this.tools) as t.ChatModelInstance);
  }

  /** Execute a function with circuit breaker protection */
  private async withCircuitBreaker<T>(
    providerKey: string,
    fn: () => Promise<T>
  ): Promise<T | undefined> {
    const now = Date.now();
    const state = this.circuitBreaker.get(providerKey);

    if (state && state.openedUntil > now) {
      return undefined; // Circuit is open
    }

    try {
      const result = await fn();
      this.circuitBreaker.set(providerKey, { failures: 0, openedUntil: 0 });
      return result;
    } catch (error) {
      const current = this.circuitBreaker.get(providerKey) ?? {
        failures: 0,
        openedUntil: 0,
      };
      const failures = current.failures + 1;
      // Open circuit for 10s after 3 consecutive failures
      const openedUntil = failures >= 3 ? now + 10_000 : 0;
      this.circuitBreaker.set(providerKey, { failures, openedUntil });
      throw error;
    }
  }

  /** Get fan-out configuration from feature flags */
  private getFanOutConfig(): FanOutConfig {
    const flags = this.featureFlags as Record<string, unknown> | undefined;
    return {
      timeoutMs: 12000,
      maxRetries: Math.max(
        0,
        flags?.fan_out_retries ? (flags.fan_out_retries as number) : 1
      ),
      baseBackoffMs: Math.max(
        100,
        flags?.fan_out_backoff_ms ? (flags.fan_out_backoff_ms as number) : 400
      ),
      maxConcurrency: Math.max(
        1,
        flags?.fan_out_concurrency ? (flags.fan_out_concurrency as number) : 2
      ),
    };
  }

  /** Add jitter to a delay value */
  private jitter(ms: number): number {
    return ms + Math.floor(Math.random() * 100);
  }

  /** Sleep for specified milliseconds */
  private sleepMs(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /** Execute a single branch with timeout and abort handling */
  private async executeBranchWithTimeout(
    model: t.ChatModelInstance,
    messages: BaseMessageLike[],
    config: RunnableConfig | undefined,
    timeoutMs: number,
    providerKey: string
  ): Promise<unknown> {
    const parentSignal = (config as { signal?: AbortSignal } | undefined)
      ?.signal;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    const onAbort = (): void => controller.abort();
    if (parentSignal) {
      try {
        parentSignal.addEventListener('abort', onAbort, { once: true });
      } catch {
        // Ignore error adding abort event listener
      }
    }

    const started = Date.now();
    try {
      const result = await model.invoke(messages, {
        ...(config ?? {}),
        signal: controller.signal,
      } as RunnableConfig);

      const duration = Date.now() - started;
      try {
        this.handlerRegistry?.getHandler('fanout_branch_end')?.handle(
          'fanout_branch_end',
          {
            provider: providerKey,
            duration,
            status: 'ok',
          } as never,
          undefined,
          this
        );
      } catch {
        // Ignore error dispatching fanout branch end event
      }

      return result;
    } catch (error) {
      const duration = Date.now() - started;
      try {
        this.handlerRegistry?.getHandler('fanout_branch_end')?.handle(
          'fanout_branch_end',
          {
            provider: providerKey,
            duration,
            status: 'error',
          } as never,
          undefined,
          this
        );
      } catch {
        // Ignore error dispatching fanout branch end event
      }
      throw error;
    } finally {
      clearTimeout(timer);
      if (parentSignal) {
        try {
          parentSignal.removeEventListener('abort', onAbort as EventListener);
        } catch {
          console.error('Error removing abort event listener');
        }
      }
    }
  }

  /** Execute a branch with retries and circuit breaker */
  private async executeBranchWithRetries(
    name: string,
    policy: RoutingPolicy,
    messages: BaseMessageLike[],
    config: RunnableConfig | undefined,
    fanOutConfig: FanOutConfig
  ): Promise<unknown> {
    const model = this.buildModel(name, policy);
    if (!model.invoke) {
      return undefined;
    }

    const modelCfg = this.models?.[name];
    const providerKey = modelCfg?.provider ?? policy.model!;

    for (let attempt = 0; attempt <= fanOutConfig.maxRetries; attempt++) {
      try {
        return await this.withCircuitBreaker(String(providerKey), () =>
          this.executeBranchWithTimeout(
            model,
            messages,
            config,
            fanOutConfig.timeoutMs,
            String(providerKey)
          )
        );
      } catch (error) {
        console.error('Error with circuit breaker:', error);
        if (attempt === fanOutConfig.maxRetries) {
          return undefined;
        }
        const delay = this.jitter(
          fanOutConfig.baseBackoffMs * Math.pow(2, attempt)
        );
        await this.sleepMs(delay);
      }
    }

    return undefined;
  }

  /** Create an aggregator for fan-out model execution */
  private createFanOutAggregator(
    agentNames: string[],
    policy: RoutingPolicy
  ): ModelAggregator {
    return {
      invoke: async (
        messages: BaseMessageLike[],
        config?: RunnableConfig
      ): Promise<unknown> => {
        const fanOutConfig = this.getFanOutConfig();

        // Concurrent execution with controlled concurrency
        const queue = [...agentNames];
        const running: Promise<unknown>[] = [];
        const results: PromiseSettledResult<unknown>[] = [];

        const launchNext = (): void => {
          if (queue.length === 0) return;

          const name = queue.shift()!;
          const promise = this.executeBranchWithRetries(
            name,
            policy,
            messages,
            config,
            fanOutConfig
          )
            .then(
              (value) =>
                ({
                  status: 'fulfilled',
                  value,
                }) as PromiseSettledResult<unknown>
            )
            .catch(
              (reason) =>
                ({
                  status: 'rejected',
                  reason,
                }) as PromiseSettledResult<unknown>
            )
            .finally(() => {
              const index = running.indexOf(promise);
              if (index !== -1) {
                results.push(
                  running.splice(
                    index,
                    1
                  )[0] as unknown as PromiseSettledResult<unknown>
                );
              }
              launchNext();
            });

          running.push(promise);

          if (running.length < fanOutConfig.maxConcurrency) {
            launchNext();
          }
        };

        // Start execution
        launchNext();
        await Promise.all(running);

        // Use results if available, otherwise fallback to allSettled
        const settled =
          results.length > 0
            ? results
            : await Promise.allSettled(
              agentNames.map((name) =>
                this.executeBranchWithRetries(
                  name,
                  policy,
                  messages,
                  config,
                  fanOutConfig
                )
              )
            );

        // Extract successful outputs
        const outputs = settled
          .map((result) =>
            result.status === 'fulfilled' ? result.value : undefined
          )
          .filter((value) => value != null);

        // Combine text content
        const textParts = outputs
          .map((output) => {
            const content = (output as { content?: unknown }).content;
            return typeof content === 'string' ? content : '';
          })
          .filter(Boolean);

        const combinedText = textParts.join('\n\n');

        // Synthesize final answer
        return this.synthesizeResults(combinedText, messages, config, policy);
      },

      stream: async function* (
        messages: BaseMessageLike[],
        config?: RunnableConfig
      ): AsyncGenerator<unknown, void, unknown> {
        // For streaming, we invoke and yield the final result
        const result = await this.invoke(messages, config);
        yield result;
      },
    } as ModelAggregator;
  }

  /** Synthesize multiple model outputs into a single result */
  private async synthesizeResults(
    combinedText: string,
    originalMessages: BaseMessageLike[],
    config: RunnableConfig | undefined,
    policy: RoutingPolicy
  ): Promise<unknown> {
    const synthCfg = this.models?.['synth'] as l.LLMConfig | undefined;
    const synthProvider = synthCfg?.provider ?? policy.model!;

    const { provider: _p, ...synthClientOptions } = (synthCfg ??
      {}) as l.LLMConfig & t.ClientOptions;

    const synthInstance = this.getNewModel({
      provider: synthProvider,
      clientOptions: synthClientOptions as unknown as t.ClientOptions,
    });

    const synthModel =
      !this.tools || this.tools.length === 0
        ? synthInstance
        : (synthInstance as t.ModelWithTools).bindTools(this.tools);

    const prompt = new HumanMessage(
      `Synthesize the following model outputs into one concise, high-quality answer. Do not repeat, deduplicate overlap, and keep it short.\n\n${combinedText}`
    );

    return synthModel.invoke([...originalMessages, prompt], config);
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
    const lastMsg =
      Array.isArray(state.messages) && state.messages.length > 0
        ? state.messages[state.messages.length - 1]
        : undefined;

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
          const contentVal = (lastMsg as { content?: unknown })
            .content as unknown;
          const contentStr = typeof contentVal === 'string' ? contentVal : '';
          const hasTools =
            'tool_calls' in (lastMsg ?? {}) &&
            ((lastMsg as unknown as { tool_calls?: unknown[] }).tool_calls
              ?.length ?? 0) > 0;
          if (cond === 'has_tools' && hasTools) return i;
          if (cond === 'no_tools' && !hasTools) return i;
          if (typeof cond === 'object') {
            const incOk =
              !cond.includes ||
              cond.includes.some((s) => contentStr && contentStr.includes(s));
            const excOk =
              !cond.excludes ||
              !cond.excludes.some((s) => contentStr && contentStr.includes(s));
            if (incOk && excOk) return i;
          }
        }
      }
      return -1;
    };

    if (totalPolicies > 0) {
      const hasConditional = (this.routingPolicies ?? []).some(
        (p) => (p as unknown as { when?: unknown }).when != null
      );
      if (hasConditional) {
        const nextIndex = pickIndex();
        this.currentStageIndex =
          nextIndex >= 0
            ? nextIndex
            : Math.min(this.currentStageIndex + 1, totalPolicies - 1);
      } else {
        this.currentStageIndex = Math.min(
          this.currentStageIndex + 1,
          totalPolicies - 1
        );
      }
      const policy = this.routingPolicies?.[this.currentStageIndex];
      if (policy && policy.model != null) {
        // If a per-stage model config is provided, use its provider & client options; otherwise fall back to provider only
        const stageConfig = this.models?.[policy.stage];
        const enableFanOut =
          this.featureFlags?.fan_out === true &&
          policy.parallel === true &&
          (policy.agents?.length ?? 0) > 1;
        if (enableFanOut) {
          const agentNames = policy.agents as string[];
          this.boundModel = this.createFanOutAggregator(
            agentNames,
            policy
          ) as unknown as Runnable;
        } else if (stageConfig) {
          const stageProvider = stageConfig.provider ?? policy.model;
          // Extract provider-specific client options from stage config
          const { provider: _p, ...clientOptions } =
            stageConfig as l.LLMConfig & t.ClientOptions;
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
    /** Route to tools immediately if tool calls are present and not yet invoked; else agent */
    return toolsCondition(state.messages, this.invokedToolIds) === TOOLS
      ? TOOLS
      : AGENT;
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
      channels: (
        this as unknown as {
          graphState: t.GraphStateChannels<t.BaseGraphState>;
        }
      ).graphState,
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
