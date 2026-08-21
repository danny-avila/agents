import type {
  ProviderToolCallIndex,
  ProviderToolCallPartDescriptor,
} from './toolResultTypes';
import {
  PROVIDER_TOOL_PAIRING_MAX_IDENTIFIER_CHARS,
  PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS,
  PROVIDER_TOOL_RESULT_MAX_ARRAY_ENTRIES,
  appendProviderToolCallDescriptor,
  consumeProviderToolResultPair,
  getBoundedProviderPairingArray,
  getBoundedProviderPairingArrayProperty,
  getProviderAIMessageToolCallDescriptor,
  getProviderToolCallPartDescriptor,
  getProviderToolResultPartDescriptor,
  hasStructurallyValidAnthropicWebSearchResultContent,
} from './toolResultTypes';

describe('provider tool-result shape validation', () => {
  it('rejects oversized nested arrays before reading an element', () => {
    let reads = 0;
    const content = new Array(PROVIDER_TOOL_RESULT_MAX_ARRAY_ENTRIES + 1);
    Object.defineProperty(content, '0', {
      enumerable: true,
      get: () => {
        reads++;
        return { type: 'text', text: 'attacker' };
      },
    });

    expect(
      getProviderToolResultPartDescriptor({
        type: 'mcp_tool_result',
        tool_use_id: 'mcp-call',
        is_error: false,
        content,
      })
    ).toBeUndefined();
    expect(reads).toBe(0);
  });

  it('treats nested output blocks as opaque without invoking accessors', () => {
    let reads = 0;
    const content: unknown[] = [];
    Object.defineProperty(content, '0', {
      configurable: true,
      enumerable: true,
      get: () => {
        reads++;
        return { type: 'text', text: 'attacker' };
      },
    });

    expect(
      getProviderToolResultPartDescriptor({
        type: 'tool_result',
        tool_use_id: 'tool-call',
        content,
      })
    ).toBeDefined();
    expect(reads).toBe(0);
  });

  it('accepts a dense structurally valid nested array at the cap', () => {
    const content = Array.from(
      { length: PROVIDER_TOOL_RESULT_MAX_ARRAY_ENTRIES },
      () => ({ type: 'text', text: 'ok' })
    );

    expect(
      getProviderToolResultPartDescriptor({
        type: 'mcp_tool_result',
        tool_use_id: 'mcp-call',
        is_error: false,
        content,
      })
    ).toMatchObject({ type: 'mcp_tool_result', toolCallId: 'mcp-call' });
  });

  it('accepts Anthropic MCP response text blocks with citations', () => {
    expect(
      getProviderToolResultPartDescriptor({
        type: 'mcp_tool_result',
        tool_use_id: 'mcp-call',
        is_error: false,
        content: [{ type: 'text', text: 'found', citations: null }],
      })
    ).toMatchObject({ type: 'mcp_tool_result', toolCallId: 'mcp-call' });
  });

  it('requires the Anthropic MCP error flag to be present and boolean', () => {
    expect(
      getProviderToolResultPartDescriptor({
        type: 'mcp_tool_result',
        tool_use_id: 'mcp-call',
        content: 'safe',
      })
    ).toBeUndefined();
    expect(
      getProviderToolResultPartDescriptor({
        type: 'mcp_tool_result',
        tool_use_id: 'mcp-call',
        is_error: 'false',
        content: 'safe',
      })
    ).toBeUndefined();
    for (const isError of [false, true]) {
      expect(
        getProviderToolResultPartDescriptor({
          type: 'mcp_tool_result',
          tool_use_id: 'mcp-call',
          is_error: isError,
          content: 'safe',
        })
      ).toMatchObject({ type: 'mcp_tool_result', toolCallId: 'mcp-call' });
    }
  });

  it('requires the declared payload field on optional-output protocols', () => {
    expect(
      getProviderToolResultPartDescriptor({
        type: 'tool_result',
        tool_use_id: 'tool-call',
        text: 'attacker bytes',
      })
    ).toBeUndefined();
    expect(
      getProviderToolResultPartDescriptor({
        type: 'mcp_tool_result',
        tool_use_id: 'mcp-call',
        is_error: false,
        text: 'attacker bytes',
      })
    ).toBeUndefined();
    expect(
      getProviderToolResultPartDescriptor({
        type: 'server_tool_result',
        tool_call_id: 'server-call',
        status: 'success',
        text: 'attacker bytes',
      })
    ).toBeUndefined();
  });

  it.each([
    {
      type: 'tool_result',
      tool_use_id: 'tool-call',
      content: 'safe',
      text: 'attacker bytes',
    },
    {
      type: 'mcp_tool_result',
      tool_use_id: 'mcp-call',
      is_error: false,
      content: 'safe',
      text: 'attacker bytes',
    },
    {
      type: 'server_tool_result',
      tool_call_id: 'server-call',
      status: 'success',
      output: 'safe',
      text: 'attacker bytes',
    },
    {
      type: 'codeExecutionResult',
      codeExecutionResult: {
        outcome: 'OUTCOME_OK',
        output: 'safe',
        text: 'attacker bytes',
      },
    },
    {
      type: 'toolResult',
      toolResult: {
        toolUseId: 'bedrock-call',
        content: [{ text: 'safe' }],
        text: 'attacker bytes',
      },
    },
  ])('rejects extra content-bearing fields in $type shapes', (part) => {
    expect(getProviderToolResultPartDescriptor(part)).toBeUndefined();
  });

  it('never classifies top-level result leaves as tool-result envelopes', () => {
    expect(
      getProviderToolResultPartDescriptor({
        type: 'search_result',
        source: 'https://example.com',
        title: 'Result',
        content: [{ type: 'text', text: 'bytes' }],
      })
    ).toBeUndefined();
    expect(
      getProviderToolResultPartDescriptor({
        type: 'web_search_result',
        url: 'https://example.com',
        title: 'Result',
        encrypted_content: 'ciphertext',
      })
    ).toBeUndefined();
  });

  it('rejects over-cap call ids, result ids, and tool names', () => {
    const overCap = 'x'.repeat(PROVIDER_TOOL_PAIRING_MAX_IDENTIFIER_CHARS + 1);

    expect(
      getProviderToolCallPartDescriptor({
        type: 'tool_use',
        id: overCap,
        name: 'lookup',
      })
    ).toBeUndefined();
    expect(
      getProviderToolCallPartDescriptor({
        type: 'tool_use',
        id: 'tool-call',
        name: overCap,
      })
    ).toBeUndefined();
    expect(
      getProviderToolResultPartDescriptor({
        type: 'tool_result',
        tool_use_id: overCap,
        content: 'bytes',
      })
    ).toBeUndefined();
  });

  it.each([
    {
      label: 'nested LibreChat call',
      call: {
        type: 'tool_call',
        tool_call: { id: 'call-id', name: 'lookup', args: {} },
      },
    },
    {
      label: 'Anthropic tool use',
      call: {
        type: 'tool_use',
        id: 'call-id',
        name: 'lookup',
        input: {},
      },
    },
    {
      label: 'Anthropic server tool use',
      call: {
        type: 'server_tool_use',
        id: 'call-id',
        name: 'web_search',
        input: {},
      },
    },
    {
      label: 'Anthropic MCP tool use',
      call: {
        type: 'mcp_tool_use',
        id: 'call-id',
        name: 'lookup',
        input: {},
        server_name: 'docs',
      },
    },
    {
      label: 'LangChain server call',
      call: {
        type: 'server_tool_call',
        id: 'call-id',
        name: 'lookup',
        args: {},
      },
    },
    {
      label: 'Gemini call',
      call: {
        type: 'toolCall',
        toolCall: { id: 'call-id', name: 'lookup', args: {} },
      },
    },
    {
      label: 'Bedrock call',
      call: {
        type: 'toolUse',
        toolUse: {
          toolUseId: 'call-id',
          name: 'lookup',
          input: {},
        },
      },
    },
  ])('requires the declared payload slot on $label', ({ call }) => {
    expect(getProviderToolCallPartDescriptor(call)).toBeDefined();

    const copy = structuredClone(call) as Record<string, unknown>;
    let payload = copy;
    if ('tool_call' in copy) {
      payload = copy.tool_call as Record<string, unknown>;
    } else if ('toolCall' in copy) {
      payload = copy.toolCall as Record<string, unknown>;
    } else if ('toolUse' in copy) {
      payload = copy.toolUse as Record<string, unknown>;
    }
    let payloadKey = 'server_name';
    if ('input' in payload) {
      payloadKey = 'input';
    } else if ('args' in payload) {
      payloadKey = 'args';
    }
    delete payload[payloadKey];

    expect(getProviderToolCallPartDescriptor(copy)).toBeUndefined();
  });

  it('rejects extra and accessor-backed call fields without invoking accessors', () => {
    const extra = {
      type: 'tool_use',
      id: 'call-id',
      name: 'lookup',
      input: {},
      text: 'attacker',
    };
    expect(getProviderToolCallPartDescriptor(extra)).toBeUndefined();

    let reads = 0;
    const accessor = {
      type: 'tool_use',
      id: 'call-id',
      name: 'lookup',
    };
    Object.defineProperty(accessor, 'input', {
      enumerable: true,
      get: () => {
        reads++;
        return {};
      },
    });
    expect(getProviderToolCallPartDescriptor(accessor)).toBeUndefined();
    expect(reads).toBe(0);
  });

  it('requires exact AI tool-call keys and arguments', () => {
    expect(
      getProviderAIMessageToolCallDescriptor({
        id: 'call-id',
        name: 'lookup',
        args: {},
        type: 'tool_call',
      })
    ).toBeDefined();
    expect(
      getProviderAIMessageToolCallDescriptor({
        id: 'call-id',
        name: 'lookup',
        type: 'tool_call',
      })
    ).toBeUndefined();
    expect(
      getProviderAIMessageToolCallDescriptor({
        id: 'call-id',
        name: 'lookup',
        args: {},
        type: 'tool_call',
        text: 'attacker',
      })
    ).toBeUndefined();
  });

  it('fails closed without throwing for revoked proxies', () => {
    const record = Proxy.revocable({}, {});
    const array = Proxy.revocable([], {});
    record.revoke();
    array.revoke();

    expect(() => getProviderToolResultPartDescriptor(record.proxy)).not.toThrow();
    expect(getProviderToolResultPartDescriptor(record.proxy)).toBeUndefined();
    expect(() => getBoundedProviderPairingArray(array.proxy)).not.toThrow();
    expect(getBoundedProviderPairingArray(array.proxy)).toBeUndefined();
  });

  it.each(['symbol', 'non-enumerable', 'inherited'] as const)(
    'rejects %s fields hidden outside the exact result wrapper',
    (kind) => {
      const base = {
        type: 'tool_result',
        tool_use_id: 'tool-call',
        content: 'safe',
      };
      let part: Record<PropertyKey, unknown>;
      if (kind === 'inherited') {
        part = Object.assign(Object.create({ text: 'attacker' }), base);
      } else {
        part = { ...base };
        Object.defineProperty(
          part,
          kind === 'symbol' ? Symbol('attacker') : 'text',
          { value: 'attacker', enumerable: kind === 'symbol' }
        );
      }

      expect(getProviderToolResultPartDescriptor(part)).toBeUndefined();
    }
  );

  it('accepts only installed Anthropic result callers', () => {
    const result = (caller: Record<string, unknown>) => ({
      type: 'web_search_tool_result',
      tool_use_id: 'server-call',
      caller,
      content: [
        {
          type: 'web_search_result',
          encrypted_content: 'ciphertext',
          title: 'Result',
          url: 'https://example.com',
        },
      ],
    });

    expect(
      getProviderToolResultPartDescriptor(result({ type: 'direct' }))
    ).toBeDefined();
    expect(
      getProviderToolResultPartDescriptor(
        result({ type: 'code_execution_20250825', tool_id: 'code-call' })
      )
    ).toBeDefined();
    expect(
      getProviderToolResultPartDescriptor(
        result({ type: 'code_execution_20260120', tool_id: 'code-call' })
      )
    ).toBeDefined();
    expect(
      getProviderToolResultPartDescriptor(
        result({ type: 'code_execution_20260521', tool_id: 'code-call' })
      )
    ).toBeUndefined();
  });

  it('rejects hidden fields in discriminated provider payload wrappers', () => {
    const response = {
      id: 'google-call',
      name: 'lookup',
      response: { output: 'safe' },
    };
    Object.defineProperty(response, 'text', {
      value: 'attacker',
      enumerable: false,
    });

    expect(
      getProviderToolResultPartDescriptor({
        type: 'toolResponse',
        toolResponse: response,
      })
    ).toBeUndefined();
  });

  it('requires one exact Gemini tool response variant', () => {
    expect(
      getProviderToolResultPartDescriptor({
        type: 'toolResponse',
        toolResponse: {
          id: 'google-call',
          name: 'lookup',
          response: { output: 'safe' },
        },
      })
    ).toBeDefined();
    expect(
      getProviderToolResultPartDescriptor({
        type: 'toolResponse',
        toolResponse: {
          id: 'google-call',
          toolType: 'google_search',
          result: { output: 'safe' },
        },
      })
    ).toBeDefined();
    expect(
      getProviderToolResultPartDescriptor({
        type: 'toolResponse',
        toolResponse: {
          id: 'google-call',
          name: 'lookup',
          response: { output: 'safe' },
          result: { output: 'attacker' },
        },
      })
    ).toBeUndefined();
  });

  it('keeps Bedrock designated result blocks opaque', () => {
    expect(
      getProviderToolResultPartDescriptor({
        type: 'toolResult',
        toolResult: {
          toolUseId: 'bedrock-call',
          content: [{ $unknown: ['attacker', 'bytes'] }],
        },
      })
    ).toBeDefined();
  });

  it('keeps deep web-search checks isolated to wire repair', () => {
    const valid = {
      type: 'web_search_tool_result',
      tool_use_id: 'server-call',
      content: [
        {
          type: 'web_search_result',
          encrypted_content: 'ciphertext',
          title: 'Result',
          url: 'https://example.com',
        },
      ],
    };
    expect(
      hasStructurallyValidAnthropicWebSearchResultContent(valid)
    ).toBe(true);
    expect(
      hasStructurallyValidAnthropicWebSearchResultContent({
        ...valid,
        content: [{ ...valid.content[0], text: 'unexpected wire field' }],
      })
    ).toBe(false);
    expect(
      getProviderToolResultPartDescriptor({
        ...valid,
        content: [{ ...valid.content[0], text: 'opaque tool output' }],
      })
    ).toBeDefined();
  });

  it('validates 4096 maximum nested outputs without walking their blocks', () => {
    let reads = 0;
    const content = new Array(PROVIDER_TOOL_RESULT_MAX_ARRAY_ENTRIES);
    Object.defineProperty(content, '0', {
      enumerable: true,
      get: () => {
        reads++;
        return { type: 'text', text: 'opaque' };
      },
    });
    const startedAt = performance.now();
    let validated = 0;
    for (
      let index = 0;
      index < PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS;
      index++
    ) {
      if (
        getProviderToolResultPartDescriptor({
          type: 'tool_result',
          tool_use_id: `call-${index}`,
          content,
        }) != null
      ) {
        validated++;
      }
    }

    expect(validated).toBe(PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS);
    expect(reads).toBe(0);
    expect(performance.now() - startedAt).toBeLessThan(2_000);
  });
});

describe('provider tool-result pairing index', () => {
  const call = (
    sourceType: string,
    name = 'lookup'
  ): ProviderToolCallPartDescriptor => ({
    callId: 'call-id',
    kind: 'tool',
    name,
    sourceType,
  });

  const result = () => {
    const descriptor = getProviderToolResultPartDescriptor({
      type: 'tool_result',
      tool_use_id: 'call-id',
      content: 'bytes',
    });
    expect(descriptor).toBeDefined();
    return descriptor!;
  };

  it('deduplicates identical dual representations and consumes once', () => {
    const calls: ProviderToolCallIndex = new Map();
    appendProviderToolCallDescriptor(calls, call('tool_call'));
    appendProviderToolCallDescriptor(calls, call('ai_tool_calls'));

    expect(calls.get('call-id')).not.toBeNull();
    expect(consumeProviderToolResultPair(result(), calls)).toBe(true);
    expect(consumeProviderToolResultPair(result(), calls)).toBe(false);
  });

  it('marks same-representation duplicates and conflicts ambiguous', () => {
    const duplicateCalls: ProviderToolCallIndex = new Map();
    appendProviderToolCallDescriptor(duplicateCalls, call('tool_call'));
    for (let index = 0; index < 20_000; index++) {
      appendProviderToolCallDescriptor(duplicateCalls, call('tool_call'));
    }
    expect(duplicateCalls.get('call-id')).toBeNull();
    expect(consumeProviderToolResultPair(result(), duplicateCalls)).toBe(false);

    const conflictingCalls: ProviderToolCallIndex = new Map();
    appendProviderToolCallDescriptor(conflictingCalls, call('tool_call'));
    appendProviderToolCallDescriptor(
      conflictingCalls,
      call('ai_tool_calls', 'different')
    );
    expect(conflictingCalls.get('call-id')).toBeNull();
  });

  it('pairs Anthropic server results only with the server protocol', () => {
    const resultDescriptor = getProviderToolResultPartDescriptor({
      type: 'web_search_tool_result',
      tool_use_id: 'server-call',
      content: [
        {
          type: 'web_search_result',
          encrypted_content: 'ciphertext',
          title: 'Result',
          url: 'https://example.com',
        },
      ],
    });
    expect(resultDescriptor).toBeDefined();

    const anthropicCalls: ProviderToolCallIndex = new Map();
    const anthropicCall = getProviderToolCallPartDescriptor({
      type: 'server_tool_use',
      id: 'server-call',
      name: 'web_search',
      input: {},
      index: 0,
    });
    expect(anthropicCall?.kind).toBe('anthropic-server');
    appendProviderToolCallDescriptor(anthropicCalls, anthropicCall!);
    expect(
      consumeProviderToolResultPair(resultDescriptor!, anthropicCalls)
    ).toBe(true);

    const langChainCalls: ProviderToolCallIndex = new Map();
    const langChainCall = getProviderToolCallPartDescriptor({
      type: 'server_tool_call',
      id: 'server-call',
      name: 'web_search',
      args: {},
    });
    appendProviderToolCallDescriptor(langChainCalls, langChainCall!);
    expect(
      consumeProviderToolResultPair(resultDescriptor!, langChainCalls)
    ).toBe(false);
  });

  it('canonicalizes indexed and dual Anthropic server-call representations', () => {
    const callId = 'srvtoolu_server-call';
    const resultDescriptor = getProviderToolResultPartDescriptor({
      type: 'web_search_tool_result',
      tool_use_id: callId,
      content: [
        {
          type: 'web_search_result',
          encrypted_content: 'ciphertext',
          title: 'Result',
          url: 'https://example.com',
        },
      ],
    });
    expect(resultDescriptor).toBeDefined();

    const calls: ProviderToolCallIndex = new Map();
    const nativeCall = getProviderToolCallPartDescriptor({
      type: 'server_tool_use',
      id: callId,
      name: 'web_search',
      input: {},
      index: 3,
    });
    const genericCall = getProviderAIMessageToolCallDescriptor({
      type: 'tool_call',
      id: callId,
      name: 'web_search',
      args: {},
    });
    expect(nativeCall?.kind).toBe('anthropic-server');
    expect(genericCall?.kind).toBe('anthropic-server');
    appendProviderToolCallDescriptor(calls, nativeCall!);
    appendProviderToolCallDescriptor(calls, genericCall!);
    expect(calls.get(callId)).not.toBeNull();
    expect(consumeProviderToolResultPair(resultDescriptor!, calls)).toBe(true);

    const streamedCalls: ProviderToolCallIndex = new Map();
    const streamedCall = getProviderToolCallPartDescriptor({
      type: 'server_tool_call',
      id: callId,
      name: 'web_search',
      args: {},
      index: 3,
    });
    expect(streamedCall?.kind).toBe('anthropic-server');
    appendProviderToolCallDescriptor(streamedCalls, streamedCall!);
    expect(
      consumeProviderToolResultPair(resultDescriptor!, streamedCalls)
    ).toBe(true);
  });

  it('does not pair MCP results with an ordinary generic call', () => {
    const resultDescriptor = getProviderToolResultPartDescriptor({
      type: 'mcp_tool_result',
      tool_use_id: 'mcp-call',
      is_error: false,
      content: 'found',
    });
    const calls: ProviderToolCallIndex = new Map();
    const genericCall = getProviderAIMessageToolCallDescriptor({
      type: 'tool_call',
      id: 'mcp-call',
      name: 'lookup',
      args: {},
    });
    appendProviderToolCallDescriptor(calls, genericCall!);

    expect(consumeProviderToolResultPair(resultDescriptor!, calls)).toBe(
      false
    );
  });

  it('stores the common call index without allocating per-call sets', () => {
    const calls: ProviderToolCallIndex = new Map();
    for (
      let index = 0;
      index < PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS;
      index++
    ) {
      appendProviderToolCallDescriptor(calls, {
        callId: `call-${index}`,
        kind: 'tool',
        name: 'lookup',
        sourceType: 'tool_call',
      });
    }

    expect(calls.size).toBe(PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS);
    calls.forEach((entry) => {
      expect(entry).not.toBeNull();
      expect(entry).not.toHaveProperty('sourceTypes');
      expect(entry).not.toHaveProperty('secondarySourceType');
    });
  });

  it('handles a maximum mixed-provider pairing wave in bounded time', () => {
    const cases = [
      (id: string) => ({
        call: { type: 'tool_use', id, name: 'lookup', input: {} },
        result: { type: 'tool_result', tool_use_id: id, content: 'ok' },
      }),
      (id: string) => ({
        call: {
          type: 'server_tool_call',
          id,
          name: 'lookup',
          args: {},
        },
        result: {
          type: 'server_tool_call_result',
          toolCallId: id,
          status: 'success',
          output: 'ok',
        },
      }),
      (id: string) => ({
        call: {
          type: 'server_tool_use',
          id,
          name: 'web_search',
          input: {},
        },
        result: {
          type: 'web_search_tool_result',
          tool_use_id: id,
          content: [
            {
              type: 'web_search_result',
              encrypted_content: 'ciphertext',
              title: 'Result',
              url: 'https://example.com',
            },
          ],
        },
      }),
      (id: string) => ({
        call: {
          type: 'mcp_tool_use',
          id,
          name: 'lookup',
          input: {},
          server_name: 'docs',
        },
        result: {
          type: 'mcp_tool_result',
          tool_use_id: id,
          is_error: false,
          content: 'ok',
        },
      }),
      (id: string) => ({
        call: {
          type: 'toolCall',
          toolCall: { id, name: 'google_search', args: {} },
        },
        result: {
          type: 'toolResponse',
          toolResponse: {
            id,
            name: 'google_search',
            response: { output: 'ok' },
          },
        },
      }),
      (id: string) => ({
        call: {
          type: 'toolUse',
          toolUse: { toolUseId: id, name: 'lookup', input: {} },
        },
        result: {
          type: 'toolResult',
          toolResult: { toolUseId: id, content: [{ text: 'ok' }] },
        },
      }),
    ];
    const calls: ProviderToolCallIndex = new Map();
    const iterations = Math.floor(
      PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS / cases.length
    );
    let consumed = 0;
    const startedAt = performance.now();
    for (let iteration = 0; iteration < iterations; iteration++) {
      for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
        const id =
          caseIndex === 2
            ? `srvtoolu_${iteration}-${caseIndex}`
            : `call-${iteration}-${caseIndex}`;
        const { call, result } = cases[caseIndex](id);
        const callDescriptor = getProviderToolCallPartDescriptor(call);
        const resultDescriptor = getProviderToolResultPartDescriptor(result);
        if (callDescriptor != null && resultDescriptor != null) {
          appendProviderToolCallDescriptor(calls, callDescriptor);
          if (consumeProviderToolResultPair(resultDescriptor, calls)) {
            consumed++;
          }
        }
      }
    }
    const elapsedMs = performance.now() - startedAt;

    expect(consumed).toBe(iterations * cases.length);
    expect(calls.size).toBe(0);
    expect(elapsedMs).toBeLessThan(5_000);
  });

  it('bounds outer pairing scans without allocating a copy', () => {
    const atCap = Array.from(
      { length: PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS },
      () => ({ type: 'text', text: 'ok' })
    );
    expect(getBoundedProviderPairingArray(atCap)).toBe(atCap);

    let reads = 0;
    const oversized = new Array(PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS + 1);
    Object.defineProperty(oversized, '0', {
      enumerable: true,
      get: () => {
        reads++;
        return {};
      },
    });
    expect(getBoundedProviderPairingArray(oversized)).toBeUndefined();
    expect(reads).toBe(0);
  });

  it('never invokes a custom iterator on a validated zero-copy array', () => {
    let iteratorReads = 0;
    const content = [{ type: 'text', text: 'safe' }];
    Object.defineProperty(content, Symbol.iterator, {
      configurable: true,
      get: () => {
        iteratorReads++;
        return Array.prototype[Symbol.iterator];
      },
    });

    expect(getBoundedProviderPairingArray(content)).toBe(content);
    expect(iteratorReads).toBe(0);
  });

  it('rejects unsafe outer properties without invoking accessors', () => {
    let reads = 0;
    const message = {};
    Object.defineProperty(message, 'content', {
      enumerable: true,
      get: () => {
        reads++;
        return [];
      },
    });

    expect(
      getBoundedProviderPairingArrayProperty(message, 'content')
    ).toBeUndefined();
    expect(reads).toBe(0);
  });
});
