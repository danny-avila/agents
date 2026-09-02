import test from 'node:test';
import assert from 'node:assert/strict';

import {
  classifyReport,
  collectFailures,
  isTransientProviderFailure,
} from './provider-flake.mjs';

const overloadedFailure = `
    {"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"},"request_id":"req_011CeeC3ZvHan2b9KRF5xNCV"}

      at Stream.iterator (node_modules/@anthropic-ai/sdk/src/core/streaming.ts:142:19)
      at src/llm/anthropic/index.ts:692:24
`;

const contextOverflowFailure = `
    ContextOverflowError: {"type":"final_context_overflow","info":"Provider message formatting exceeded the context budget and no safe synthetic-context compaction could make it fit.","provider":"anthropic","projectedMessageTokens":15,"availableMessageTokens":0}

      at createProviderPayloadOverflowError (src/graphs/Graph.ts:3339:23)
`;

const assertionFailure = `
    expect(received).toBeGreaterThan(expected)

    Expected: > 0
    Received:   0

      at Object.<anonymous> (src/specs/summarization.test.ts:213:52)
`;

const suite = (name, status, extra) => ({
  name,
  status,
  assertionResults: [],
  ...extra,
});

const failedAssertion = (fullName, message) => ({
  fullName,
  status: 'failed',
  failureMessages: [message],
});

test('provider overload and throttling responses are transient', () => {
  assert.equal(isTransientProviderFailure(overloadedFailure), true);
  assert.equal(
    isTransientProviderFailure(
      'ThrottlingException: Too many requests, please wait'
    ),
    true
  );
  assert.equal(
    isTransientProviderFailure('APIConnectionError: fetch failed'),
    true
  );
  assert.equal(isTransientProviderFailure('Error: socket hang up'), true);
  assert.equal(
    isTransientProviderFailure('AI_APICallError: status code 529'),
    true
  );
});

test('library errors and assertion diffs are not transient', () => {
  assert.equal(isTransientProviderFailure(contextOverflowFailure), false);
  assert.equal(isTransientProviderFailure(assertionFailure), false);
  assert.equal(isTransientProviderFailure(''), false);
  assert.equal(
    isTransientProviderFailure(
      'TypeError: cannot read properties of undefined'
    ),
    false
  );
});

test('an assertion diff wins over transient wording quoted inside it', () => {
  const quoting = `
    expect(received).toEqual(expected)

    Expected: {"error": "overloaded_error"}
    Received: {"error": "invalid_request_error"}
  `;

  assert.equal(isTransientProviderFailure(quoting), false);
});

test('test titles are ignored when classifying a suite-level failure', () => {
  const titled = [
    '  ● Rate limit handling › retries an overloaded stream',
    '',
    '    TypeError: retry is not a function',
  ].join('\n');

  assert.equal(isTransientProviderFailure(titled), false);
});

test('collectFailures reads assertions, then falls back to suite output', () => {
  const failures = collectFailures({
    testResults: [
      suite('src/specs/cache.simple.test.ts', 'failed', {
        assertionResults: [
          failedAssertion(
            'Anthropic Prompt Caching › multi-turn',
            overloadedFailure
          ),
          {
            fullName: 'Anthropic Prompt Caching › tool calls',
            status: 'passed',
          },
        ],
      }),
      suite('src/specs/broken.test.ts', 'failed', {
        message: 'SyntaxError: Unexpected token',
      }),
      suite('src/specs/green.test.ts', 'passed'),
    ],
  });

  assert.deepEqual(
    failures.map((failure) => failure.name),
    ['Anthropic Prompt Caching › multi-turn', 'src/specs/broken.test.ts']
  );
});

test('a run is tolerable only when every failure is provider-side', () => {
  const providerOnly = {
    testResults: [
      suite('src/specs/cache.simple.test.ts', 'failed', {
        assertionResults: [
          failedAssertion('caching › multi-turn', overloadedFailure),
        ],
      }),
    ],
  };
  const mixed = {
    testResults: [
      suite('src/specs/cache.simple.test.ts', 'failed', {
        assertionResults: [
          failedAssertion('caching › multi-turn', overloadedFailure),
        ],
      }),
      suite('src/specs/summarization.test.ts', 'failed', {
        assertionResults: [
          failedAssertion(
            'summarization › cross-provider',
            contextOverflowFailure
          ),
        ],
      }),
    ],
  };

  assert.equal(classifyReport(providerOnly).tolerable, true);
  assert.equal(classifyReport(mixed).tolerable, false);
});

test('a failure jest never attributed to a test stays red', () => {
  assert.equal(
    classifyReport({ success: false, testResults: [] }).tolerable,
    false
  );
  assert.equal(classifyReport(null).tolerable, false);
});
