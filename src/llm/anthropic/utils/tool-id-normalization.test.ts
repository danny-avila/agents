import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  _convertLangChainToolCallToAnthropic,
  _convertMessagesToAnthropicPayload,
  normalizeAnthropicToolCallId,
} from './message_inputs';

describe('normalizeAnthropicToolCallId', () => {
  it('returns valid IDs unchanged', () => {
    expect(normalizeAnthropicToolCallId('toolu_01ABcdEFgh')).toBe(
      'toolu_01ABcdEFgh'
    );
    expect(normalizeAnthropicToolCallId('call_abc123XYZ')).toBe(
      'call_abc123XYZ'
    );
    expect(normalizeAnthropicToolCallId('a-b_c-d')).toBe('a-b_c-d');
  });

  it('replaces invalid characters with underscores', () => {
    expect(
      normalizeAnthropicToolCallId(
        'fc_67abc1234def567|call_abc123def456ghi789jkl0mnopqrs'
      )
    ).toBe('fc_67abc1234def567_call_abc123def456ghi789jkl0mnopqrs');
    expect(normalizeAnthropicToolCallId('a.b@c#d')).toBe('a_b_c_d');
  });

  it('truncates IDs longer than 64 characters', () => {
    const long = 'fc_' + 'a'.repeat(80);
    const out = normalizeAnthropicToolCallId(long);
    expect(out).toHaveLength(64);
    expect(out.startsWith('fc_aaa')).toBe(true);
  });

  it('handles combined length and character violations', () => {
    const id = 'fc_' + 'x|'.repeat(100);
    const out = normalizeAnthropicToolCallId(id);
    expect(out).toHaveLength(64);
    expect(/^[a-zA-Z0-9_-]+$/.test(out)).toBe(true);
  });

  it('is deterministic — same input always yields same output', () => {
    const id = 'fc_a|b|c';
    expect(normalizeAnthropicToolCallId(id)).toBe(
      normalizeAnthropicToolCallId(id)
    );
  });

  it('passes through undefined for the optional overload', () => {
    expect(normalizeAnthropicToolCallId(undefined)).toBeUndefined();
  });
});

describe('_convertMessagesToAnthropicPayload — cross-provider ID normalization', () => {
  it('normalizes Responses-style IDs on tool_use AND matching tool_result', () => {
    const responsesId = 'fc_67abc1234def567|call_abc123def456ghi789jkl0mnopqrs';

    const payload = _convertMessagesToAnthropicPayload([
      new HumanMessage('weather?'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: responsesId,
            name: 'get_weather',
            args: { location: 'Tokyo' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        tool_call_id: responsesId,
        content: '{"temp": 21}',
      }),
    ]);

    const assistantMsg = payload.messages.find((m) => m.role === 'assistant')!;
    const userToolResultMsg = payload.messages.find(
      (m) =>
        m.role === 'user' &&
        Array.isArray(m.content) &&
        (m.content as Array<{ type: string }>)[0]?.type === 'tool_result'
    )!;

    const toolUseBlock = (
      assistantMsg.content as Array<{ type: string; id?: string }>
    ).find((b) => b.type === 'tool_use')!;
    const toolResultBlock = (
      userToolResultMsg.content as Array<{
        type: string;
        tool_use_id?: string;
      }>
    ).find((b) => b.type === 'tool_result')!;

    const expected = normalizeAnthropicToolCallId(responsesId);
    expect(toolUseBlock.id).toBe(expected);
    expect(toolResultBlock.tool_use_id).toBe(expected);
    expect(toolUseBlock.id).toBe(toolResultBlock.tool_use_id);
    expect(/^[a-zA-Z0-9_-]+$/.test(toolUseBlock.id!)).toBe(true);
    expect(toolUseBlock.id!.length).toBeLessThanOrEqual(64);
  });

  it('passes through Anthropic-native IDs unchanged', () => {
    const nativeId = 'toolu_01ABcdEFgh23ijKL';

    const payload = _convertMessagesToAnthropicPayload([
      new HumanMessage('hi'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: nativeId,
            name: 'noop',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        tool_call_id: nativeId,
        content: 'ok',
      }),
    ]);

    const assistantMsg = payload.messages.find((m) => m.role === 'assistant')!;
    const toolUseBlock = (
      assistantMsg.content as Array<{ type: string; id?: string }>
    ).find((b) => b.type === 'tool_use')!;

    expect(toolUseBlock.id).toBe(nativeId);
  });

  it('does not normalize server tool IDs (srvtoolu_ prefix)', () => {
    const serverId = 'srvtoolu_01abcXYZ';

    const block = _convertLangChainToolCallToAnthropic({
      id: serverId,
      name: 'web_search',
      args: { query: 'x' },
      type: 'tool_call',
    });

    expect(block.type).toBe('server_tool_use');
    expect(block.id).toBe(serverId);
  });
});
