import {
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import {
  createCompactionCacheNamespace,
  createCompactionReplayRecipe,
  createCompactionToolProjectionFingerprint,
  EMPTY_COMPACTION_SYSTEM_PROJECTION_FINGERPRINT,
  inspectCompactionReplayEligibility,
  type CompactionReplayRecipe,
  type CompactionReplayState,
} from '@/llm/compactionReplay';
import type * as t from '@/types';
import { setProviderMessageProvenance } from '@/messages/provenance';
import { Providers } from '@/common';

function sourceMessage(id: string): HumanMessage {
  const message = new HumanMessage({ content: id, id });
  setProviderMessageProvenance(message, [
    { attribution: 'user', sourceMessageId: id },
  ]);
  return message;
}

function createEnvelope(
  overrides: Partial<{
    provider: t.ProviderName;
    modelId: string;
    projectionMode: 'chat-messages' | 'openai-responses';
    cacheNamespace: ReturnType<typeof createCompactionCacheNamespace>;
    promptCacheEnabled: boolean;
    systemProjectionFingerprint: string | null;
    toolProjectionFingerprint: string;
    systemRevision: number;
    toolRevision: number;
    messages: BaseMessage[];
    sourceMessages: BaseMessage[];
  }> = {}
): CompactionReplayRecipe {
  const sourceMessages = overrides.sourceMessages ?? [
    sourceMessage('a'),
    sourceMessage('b'),
    sourceMessage('c'),
  ];
  const envelope = createCompactionReplayRecipe({
    provider: overrides.provider ?? Providers.ANTHROPIC,
    modelId: overrides.modelId ?? 'claude-sonnet',
    projectionMode: overrides.projectionMode ?? 'chat-messages',
    cacheNamespace:
      overrides.cacheNamespace ??
      createCompactionCacheNamespace(Providers.ANTHROPIC, {
        apiKey: 'test-key',
        baseURL: 'https://provider.test',
      }),
    promptCacheEnabled: overrides.promptCacheEnabled ?? true,
    systemProjectionFingerprint:
      overrides.systemProjectionFingerprint === null
        ? undefined
        : (overrides.systemProjectionFingerprint ??
          EMPTY_COMPACTION_SYSTEM_PROJECTION_FINGERPRINT),
    toolProjectionFingerprint:
      overrides.toolProjectionFingerprint ??
      createCompactionToolProjectionFingerprint(undefined),
    systemRevision: overrides.systemRevision ?? 2,
    toolRevision: overrides.toolRevision ?? 3,
    messages: overrides.messages ?? [
      new SystemMessage('stable'),
      ...sourceMessages,
    ],
    sourceMessages,
  });
  return envelope;
}

function inspect(
  state: CompactionReplayState | undefined,
  overrides: Partial<{
    provider: t.ProviderName;
    modelId: string;
    projectionMode: 'chat-messages' | 'openai-responses';
    cacheNamespace: ReturnType<typeof createCompactionCacheNamespace>;
    promptCacheEnabled: boolean;
    systemProjectionFingerprint: string;
    toolProjectionFingerprint: string;
    systemRevision: number;
    toolRevision: number;
    messages: BaseMessage[];
    restoredToolSubstitution: boolean;
    summarizerFallbackServed: boolean;
  }> = {}
) {
  return inspectCompactionReplayEligibility(state, {
    provider: overrides.provider ?? Providers.ANTHROPIC,
    modelId: overrides.modelId,
    projectionMode: overrides.projectionMode,
    cacheNamespace:
      overrides.cacheNamespace ??
      createCompactionCacheNamespace(Providers.ANTHROPIC, {
        apiKey: 'test-key',
        baseURL: 'https://provider.test',
      }),
    promptCacheEnabled: overrides.promptCacheEnabled ?? true,
    systemProjectionFingerprint:
      overrides.systemProjectionFingerprint ??
      EMPTY_COMPACTION_SYSTEM_PROJECTION_FINGERPRINT,
    toolProjectionFingerprint:
      overrides.toolProjectionFingerprint ??
      createCompactionToolProjectionFingerprint(undefined),
    systemRevision: overrides.systemRevision ?? 2,
    toolRevision: overrides.toolRevision ?? 3,
    messages: overrides.messages ?? [sourceMessage('a'), sourceMessage('b')],
    restoredToolSubstitution: overrides.restoredToolSubstitution ?? false,
    summarizerFallbackServed: overrides.summarizerFallbackServed,
  });
}

describe('compaction replay eligibility', () => {
  it.each([Providers.ANTHROPIC, Providers.BEDROCK, Providers.OPENROUTER])(
    'finds the exact source prefix for %s',
    (provider) => {
      const envelope = createEnvelope({ provider });
      const result = inspect(envelope, { provider });

      expect(result).toEqual({
        eligible: true,
        replayMessageCount: 3,
        replaySourceCount: 2,
        requestSourceCount: 3,
      });
    }
  );

  it('inherits the captured model when no dedicated summary model is set', () => {
    const envelope = createEnvelope();
    expect(inspect(envelope)).toMatchObject({
      eligible: true,
    });
  });

  it('treats a changed prompt-cache marker policy as a namespace mismatch', () => {
    const envelope = createEnvelope({
      cacheNamespace: createCompactionCacheNamespace(Providers.ANTHROPIC, {
        apiKey: 'test-key',
        promptCache: true,
        promptCacheTtl: '1h',
      }),
    });

    expect(
      inspect(envelope, {
        cacheNamespace: createCompactionCacheNamespace(Providers.ANTHROPIC, {
          apiKey: 'test-key',
          promptCache: true,
          promptCacheTtl: '5m',
        }),
      })
    ).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_mismatch',
    });
  });

  it.each(['anthropicApiKey', 'anthropicApiUrl'] as const)(
    'includes the Anthropic %s routing alias in cache identity',
    (key) => {
      const recipe = createEnvelope({
        cacheNamespace: createCompactionCacheNamespace(Providers.ANTHROPIC, {
          apiKey: 'shared-key',
          [key]: 'primary',
        }),
      });

      expect(
        inspect(recipe, {
          cacheNamespace: createCompactionCacheNamespace(Providers.ANTHROPIC, {
            apiKey: 'shared-key',
            [key]: 'summary',
          }),
        })
      ).toMatchObject({
        eligible: false,
        reason: 'cache_namespace_mismatch',
      });
    }
  );

  it.each(['openAIApiKey', 'openAIBasePath', 'openAIApiVersion'] as const)(
    'includes the Azure %s routing alias in cache identity',
    (key) => {
      const recipe = createEnvelope({
        provider: Providers.AZURE,
        cacheNamespace: createCompactionCacheNamespace(Providers.AZURE, {
          azureOpenAIApiKey: 'shared-key',
          [key]: 'primary',
        }),
      });

      expect(
        inspect(recipe, {
          provider: Providers.AZURE,
          cacheNamespace: createCompactionCacheNamespace(Providers.AZURE, {
            azureOpenAIApiKey: 'shared-key',
            [key]: 'summary',
          }),
        })
      ).toMatchObject({
        eligible: false,
        reason: 'cache_namespace_mismatch',
      });
    }
  );

  it('snapshots nested cache identity before host mutation', () => {
    const options = {
      configuration: { baseURL: 'https://primary.test' },
      apiKey: 'test-key',
    };
    const recipe = createEnvelope({
      cacheNamespace: createCompactionCacheNamespace(
        Providers.ANTHROPIC,
        options
      ),
    });
    options.configuration.baseURL = 'https://summary.test';

    expect(
      inspect(recipe, {
        cacheNamespace: createCompactionCacheNamespace(
          Providers.ANTHROPIC,
          options
        ),
      })
    ).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_mismatch',
    });
  });

  it('includes the Vertex project ID in cache identity', () => {
    const recipe = createEnvelope({
      provider: Providers.VERTEXAI,
      cacheNamespace: createCompactionCacheNamespace(Providers.VERTEXAI, {
        credentials: { client_email: 'test@example.test' },
        projectId: 'primary-project',
      }),
    });

    expect(
      inspect(recipe, {
        provider: Providers.VERTEXAI,
        cacheNamespace: createCompactionCacheNamespace(Providers.VERTEXAI, {
          credentials: { client_email: 'test@example.test' },
          projectId: 'summary-project',
        }),
      })
    ).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_mismatch',
    });
  });

  it('fails closed instead of throwing on an opaque route object', () => {
    const revocable = Proxy.revocable({}, {});
    revocable.revoke();

    expect(() =>
      createCompactionCacheNamespace(Providers.ANTHROPIC, {
        configuration: revocable.proxy,
      })
    ).not.toThrow();
    expect(
      createCompactionCacheNamespace(Providers.ANTHROPIC, {
        configuration: revocable.proxy,
      })
    ).toMatchObject({ complete: false });
  });

  it('fingerprints the effective thinking mode', () => {
    const recipe = createEnvelope({
      cacheNamespace: createCompactionCacheNamespace(Providers.ANTHROPIC, {
        apiKey: 'test-key',
        thinking: { type: 'enabled', budget_tokens: 1_024 },
      }),
    });

    expect(
      inspect(recipe, {
        cacheNamespace: createCompactionCacheNamespace(Providers.ANTHROPIC, {
          apiKey: 'test-key',
        }),
      })
    ).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_mismatch',
    });
  });

  it('fails closed when the normal system projection is not replayed', () => {
    const recipe = createEnvelope({
      systemProjectionFingerprint: null,
    });

    expect(inspect(recipe)).toMatchObject({
      eligible: false,
      reason: 'system_projection_unknown',
    });
  });

  it('includes Google API version in cache identity', () => {
    const credentials = { apiKey: 'test-key' };
    const recipe = createEnvelope({
      provider: Providers.GOOGLE,
      cacheNamespace: createCompactionCacheNamespace(Providers.GOOGLE, {
        ...credentials,
        apiVersion: 'v1',
      }),
    });

    expect(
      inspect(recipe, {
        provider: Providers.GOOGLE,
        cacheNamespace: createCompactionCacheNamespace(Providers.GOOGLE, {
          ...credentials,
          apiVersion: 'v1beta',
        }),
      })
    ).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_mismatch',
    });
  });

  it('includes OpenRouter upstream routing in cache identity', () => {
    const recipe = createEnvelope({
      provider: Providers.OPENROUTER,
      cacheNamespace: createCompactionCacheNamespace(Providers.OPENROUTER, {
        apiKey: 'test-key',
        modelKwargs: { provider: { order: ['anthropic'] } },
        promptCache: true,
      }),
    });

    expect(
      inspect(recipe, {
        provider: Providers.OPENROUTER,
        cacheNamespace: createCompactionCacheNamespace(Providers.OPENROUTER, {
          apiKey: 'test-key',
          modelKwargs: { provider: { order: ['google'] } },
          promptCache: true,
        }),
      })
    ).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_mismatch',
    });
  });

  it('includes the Bedrock application inference profile in cache identity', () => {
    const recipe = createEnvelope({
      provider: Providers.BEDROCK,
      cacheNamespace: createCompactionCacheNamespace(Providers.BEDROCK, {
        credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
        applicationInferenceProfile: 'arn:primary',
        promptCache: true,
      }),
    });

    expect(
      inspect(recipe, {
        provider: Providers.BEDROCK,
        cacheNamespace: createCompactionCacheNamespace(Providers.BEDROCK, {
          credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
          applicationInferenceProfile: 'arn:summary',
          promptCache: true,
        }),
      })
    ).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_mismatch',
    });
  });

  it('includes OpenAI explicit cache projection mode in cache identity', () => {
    const recipe = createEnvelope({
      provider: Providers.OPENAI,
      cacheNamespace: createCompactionCacheNamespace(Providers.OPENAI, {
        apiKey: 'test-key',
        promptCacheExplicit: true,
      }),
    });

    expect(
      inspect(recipe, {
        provider: Providers.OPENAI,
        cacheNamespace: createCompactionCacheNamespace(Providers.OPENAI, {
          apiKey: 'test-key',
          promptCacheExplicit: false,
        }),
      })
    ).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_mismatch',
    });
  });

  it('fails closed when credentials come from an opaque external source', () => {
    const cacheNamespace = createCompactionCacheNamespace(Providers.ANTHROPIC, {
      baseURL: 'https://provider.test',
    });
    const recipe = createEnvelope({ cacheNamespace });

    expect(inspect(recipe, { cacheNamespace })).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_unknown',
    });
  });

  it('snapshots environment-derived endpoint routing', () => {
    const originalBaseURL = process.env.OPENAI_BASE_URL;
    try {
      process.env.OPENAI_BASE_URL = 'https://primary.test';
      const recipe = createEnvelope({
        provider: Providers.OPENAI,
        cacheNamespace: createCompactionCacheNamespace(Providers.OPENAI, {
          apiKey: 'test-key',
        }),
      });
      process.env.OPENAI_BASE_URL = 'https://summary.test';

      expect(
        inspect(recipe, {
          provider: Providers.OPENAI,
          cacheNamespace: createCompactionCacheNamespace(Providers.OPENAI, {
            apiKey: 'test-key',
          }),
        })
      ).toMatchObject({
        eligible: false,
        reason: 'cache_namespace_mismatch',
      });
    } finally {
      if (originalBaseURL == null) {
        delete process.env.OPENAI_BASE_URL;
      } else {
        process.env.OPENAI_BASE_URL = originalBaseURL;
      }
    }
  });

  it('ignores an environment endpoint overridden by explicit options', () => {
    const originalBaseURL = process.env.OPENAI_BASE_URL;
    try {
      process.env.OPENAI_BASE_URL = 'https://ignored-primary.test';
      const recipe = createEnvelope({
        provider: Providers.OPENAI,
        cacheNamespace: createCompactionCacheNamespace(Providers.OPENAI, {
          apiKey: 'test-key',
          baseURL: 'https://explicit.test',
        }),
      });
      process.env.OPENAI_BASE_URL = 'https://ignored-summary.test';

      expect(
        inspect(recipe, {
          provider: Providers.OPENAI,
          cacheNamespace: createCompactionCacheNamespace(Providers.OPENAI, {
            apiKey: 'test-key',
            baseURL: 'https://explicit.test',
          }),
        })
      ).toMatchObject({ eligible: true });
    } finally {
      if (originalBaseURL == null) {
        delete process.env.OPENAI_BASE_URL;
      } else {
        process.env.OPENAI_BASE_URL = originalBaseURL;
      }
    }
  });

  it('fails closed when the serving model owns an unknown route', () => {
    const cacheNamespace = createCompactionCacheNamespace(
      Providers.ANTHROPIC,
      { baseURL: 'https://configured.test' },
      false
    );
    const recipe = createEnvelope({ cacheNamespace });

    expect(inspect(recipe, { cacheNamespace })).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_unknown',
    });
  });

  it('requires explicit prompt caching for gated providers', () => {
    const recipe = createEnvelope({ promptCacheEnabled: false });

    expect(inspect(recipe)).toMatchObject({
      eligible: false,
      reason: 'prompt_cache_disabled',
    });
  });

  it('compares the actual ordered tool projection', () => {
    const primaryTools = createCompactionToolProjectionFingerprint([
      { name: 'stable', extras: { cache_control: { type: 'ephemeral' } } },
      { name: 'deferred', defer_loading: true },
    ]);
    const summaryTools = createCompactionToolProjectionFingerprint([
      { name: 'stable' },
      { name: 'deferred', defer_loading: true },
    ]);
    const recipe = createEnvelope({
      toolProjectionFingerprint: primaryTools,
    });

    expect(
      inspect(recipe, { toolProjectionFingerprint: summaryTools })
    ).toMatchObject({
      eligible: false,
      reason: 'tool_projection_changed',
    });
  });

  it('fails closed when a runtime provider cannot prove its cache namespace', () => {
    const provider = 'runtime-provider' as t.ProviderName;
    const cacheNamespace = createCompactionCacheNamespace(provider, {
      baseURL: 'https://runtime-provider.test',
    });
    const recipe = createEnvelope({ provider, cacheNamespace });

    expect(inspect(recipe, { provider, cacheNamespace })).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_unknown',
    });
  });

  it('fails closed when cache identity contains an unsupported value', () => {
    const cacheNamespace = createCompactionCacheNamespace(Providers.ANTHROPIC, {
      apiKey: () => 'dynamic-key',
    });
    const recipe = createEnvelope({ cacheNamespace });

    expect(inspect(recipe, { cacheNamespace })).toMatchObject({
      eligible: false,
      reason: 'cache_namespace_unknown',
    });
  });

  it('rejects changed content even when source lineage is unchanged', () => {
    const recipe = createEnvelope();

    expect(
      inspect(recipe, {
        messages: [
          sourceMessage('a'),
          new HumanMessage({ content: 'changed', id: 'b' }),
        ],
      })
    ).toMatchObject({
      eligible: false,
      reason: 'source_content_mismatch',
    });
  });

  it.each([
    ['no_request_snapshot', undefined, {}],
    ['fallback_served_request', 'fallback', {}],
    [
      'summarizer_fallback_served_request',
      createEnvelope(),
      { summarizerFallbackServed: true },
    ],
    ['provider_mismatch', createEnvelope(), { provider: Providers.OPENAI }],
    ['model_mismatch', createEnvelope(), { modelId: 'different-model' }],
    [
      'cache_namespace_mismatch',
      createEnvelope(),
      {
        cacheNamespace: createCompactionCacheNamespace(Providers.ANTHROPIC, {
          apiKey: 'test-key',
          baseURL: 'https://other.test',
        }),
      },
    ],
    ['system_projection_changed', createEnvelope(), { systemRevision: 4 }],
    ['tool_projection_changed', createEnvelope(), { toolRevision: 4 }],
    [
      'projection_mode_mismatch',
      createEnvelope(),
      { projectionMode: 'openai-responses' },
    ],
    [
      'restored_tool_substitution',
      createEnvelope(),
      { restoredToolSubstitution: true },
    ],
    ['source_not_prefix', createEnvelope(), { messages: [sourceMessage('b')] }],
    [
      'ambiguous_lineage',
      createEnvelope(),
      { messages: [new HumanMessage('missing id')] },
    ],
  ] as const)('fails closed with %s', (reason, state, overrides) => {
    expect(
      inspect(
        state as CompactionReplayState | undefined,
        overrides as Parameters<typeof inspect>[1]
      )
    ).toMatchObject({ eligible: false, reason });
  });

  it('rejects a cut through one coalesced provider message', () => {
    const coalesced = new HumanMessage({
      content: 'a then b',
      additional_kwargs: { sourceMessageIds: ['a', 'b'] },
    });
    const envelope = createEnvelope({
      messages: [new SystemMessage('stable'), coalesced, sourceMessage('c')],
      sourceMessages: [
        sourceMessage('a'),
        sourceMessage('b'),
        sourceMessage('c'),
      ],
    });

    expect(inspect(envelope, { messages: [sourceMessage('a')] })).toMatchObject(
      { eligible: false, reason: 'ambiguous_lineage' }
    );
  });

  it('accepts a complete strict-alternation coalesced prefix', () => {
    const coalesced = new HumanMessage({
      content: 'a then b',
      additional_kwargs: { sourceMessageIds: ['a', 'b'] },
    });
    const envelope = createEnvelope({
      messages: [new SystemMessage('stable'), coalesced, sourceMessage('c')],
      sourceMessages: [
        sourceMessage('a'),
        sourceMessage('b'),
        sourceMessage('c'),
      ],
    });

    expect(
      inspect(envelope, {
        messages: [sourceMessage('a'), sourceMessage('b')],
      })
    ).toEqual({
      eligible: true,
      replayMessageCount: 2,
      replaySourceCount: 2,
      requestSourceCount: 3,
    });
  });

  it('includes every repeated artifact projection at the replay boundary', () => {
    const firstArtifact = new ToolMessage({
      content: 'first placeholder',
      tool_call_id: 'call_1',
      additional_kwargs: { sourceMessageIds: ['a'] },
    });
    const secondArtifact = new ToolMessage({
      content: 'second placeholder',
      tool_call_id: 'call_2',
      additional_kwargs: { sourceMessageIds: ['b'] },
    });
    const aggregate = new HumanMessage({
      content: 'expanded artifacts',
      additional_kwargs: { sourceMessageIds: ['a', 'b'] },
    });
    const sourceMessages = [sourceMessage('a'), sourceMessage('b')];
    const envelope = createEnvelope({
      messages: [
        new SystemMessage('stable'),
        firstArtifact,
        secondArtifact,
        aggregate,
      ],
      sourceMessages,
    });

    expect(inspect(envelope, { messages: sourceMessages })).toEqual({
      eligible: true,
      replayMessageCount: 4,
      replaySourceCount: 2,
      requestSourceCount: 2,
    });
  });

  it('ignores a synthetic prior checkpoint when proving source order', () => {
    const priorCheckpoint = new HumanMessage({
      content: '<summary>prior checkpoint</summary>',
      additional_kwargs: { injected: true, source: 'summary' },
    });
    const envelope = createEnvelope({
      messages: [
        new SystemMessage('stable'),
        priorCheckpoint,
        sourceMessage('a'),
        sourceMessage('b'),
        sourceMessage('c'),
      ],
      sourceMessages: [
        priorCheckpoint,
        sourceMessage('a'),
        sourceMessage('b'),
        sourceMessage('c'),
      ],
    });

    expect(
      inspect(envelope, {
        messages: [priorCheckpoint, sourceMessage('a'), sourceMessage('b')],
      })
    ).toEqual({
      eligible: true,
      replayMessageCount: 4,
      replaySourceCount: 2,
      requestSourceCount: 3,
    });
  });

  it('retains only a frozen recipe container, not the live model', () => {
    const envelope = createEnvelope();

    expect(Object.isFrozen(envelope)).toBe(true);
    expect('model' in envelope).toBe(false);
  });

  it('retains the prepared-message reference without mutating its messages', () => {
    const messages = [sourceMessage('a'), sourceMessage('b')];
    const before = [...messages];
    const recipe = createEnvelope({ messages, sourceMessages: messages });

    expect(recipe.messages).toBe(messages);
    inspect(recipe, { messages: [sourceMessage('a')] });
    expect(messages).toEqual(before);
  });

  it('treats a restored tool result as ineligible even with exact lineage', () => {
    const tool = new ToolMessage({
      content: 'result',
      tool_call_id: 'call_1',
      id: 'tool-source',
    });
    setProviderMessageProvenance(tool, [
      { attribution: 'tool', sourceMessageId: 'tool-source' },
    ]);
    const envelope = createEnvelope({
      messages: [new SystemMessage('stable'), sourceMessage('a'), tool],
      sourceMessages: [sourceMessage('a'), tool],
    });

    expect(
      inspect(envelope, {
        messages: [sourceMessage('a'), tool],
        restoredToolSubstitution: true,
      })
    ).toMatchObject({
      eligible: false,
      reason: 'restored_tool_substitution',
    });
  });
});
