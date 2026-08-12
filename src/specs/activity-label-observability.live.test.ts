/**
 * Live provider + Langfuse export verification for activity-label traces.
 *
 * Run with:
 * RUN_ACTIVITY_LABEL_LANGFUSE_LIVE=1 OPENAI_API_KEY=... \
 * LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_BASE_URL=... \
 * LANGFUSE_FORCE_FLUSH_ON_DISPOSE=true \
 * npm test -- activity-label-observability.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import { Providers } from '@/common';
import { Run } from '@/run';

type LangfuseMetadata = {
  sourceRunId?: string;
  responseId?: string;
  activityIndex?: string | number;
  phaseIndex?: string | number;
  activityCount?: string | number;
};

type LangfuseObservation = {
  id: string;
  parentObservationId?: string | null;
  type: string;
  name: string;
  model?: string | null;
  usage?: { input?: number; output?: number; total?: number };
};

type LangfuseTrace = {
  id: string;
  name: string;
  tags?: string[];
  metadata?: LangfuseMetadata;
  observations?: LangfuseObservation[];
};

type LangfuseTraceList = {
  data: LangfuseTrace[];
};

const shouldRunLive =
  process.env.RUN_ACTIVITY_LABEL_LANGFUSE_LIVE === '1' &&
  (process.env.OPENAI_API_KEY ?? '') !== '' &&
  (process.env.LANGFUSE_PUBLIC_KEY ?? '') !== '' &&
  (process.env.LANGFUSE_SECRET_KEY ?? '') !== '' &&
  (process.env.LANGFUSE_BASE_URL ?? '') !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;

function langfuseHeaders(): HeadersInit {
  const credentials = Buffer.from(
    `${process.env.LANGFUSE_PUBLIC_KEY}:${process.env.LANGFUSE_SECRET_KEY}`
  ).toString('base64');
  return { Authorization: `Basic ${credentials}` };
}

async function requestLangfuse(path: string): Promise<Response> {
  const baseUrl = process.env.LANGFUSE_BASE_URL?.replace(/\/$/, '');
  return fetch(`${baseUrl}${path}`, {
    headers: langfuseHeaders(),
  });
}

async function getJson<T>(path: string): Promise<T> {
  const response = await requestLangfuse(path);
  if (!response.ok) {
    throw new Error(`Langfuse request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

async function getJsonIfPresent<T>(path: string): Promise<T | undefined> {
  const response = await requestLangfuse(path);
  if (response.status === 404) {
    return undefined;
  }
  if (!response.ok) {
    throw new Error(`Langfuse request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

async function findExportedTraces(
  sourceRunId: string,
  fromTimestamp: string
): Promise<{ label: LangfuseTrace; phase: LangfuseTrace }> {
  const query = new URLSearchParams({
    limit: '100',
    fromTimestamp,
  });
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const list = await getJson<LangfuseTraceList>(
      `/api/public/traces?${query.toString()}`
    );
    const candidates = list.data.filter(
      (trace) => trace.metadata?.sourceRunId === sourceRunId
    );
    const labelSummary = candidates.find(
      (trace) => trace.tags?.includes('activity-label') === true
    );
    const phaseSummary = candidates.find(
      (trace) => trace.tags?.includes('activity-phase') === true
    );
    if (labelSummary != null && phaseSummary != null) {
      const [label, phase] = await Promise.all([
        getJsonIfPresent<LangfuseTrace>(
          `/api/public/traces/${labelSummary.id}`
        ),
        getJsonIfPresent<LangfuseTrace>(
          `/api/public/traces/${phaseSummary.id}`
        ),
      ]);
      if (label != null && phase != null) {
        const labelReady =
          label.observations?.some(
            (observation) => observation.type === 'GENERATION'
          ) === true;
        const phaseRootReady =
          phase.observations?.some(
            (observation) =>
              observation.parentObservationId == null &&
              observation.type === 'CHAIN'
          ) === true;
        const phaseGenerationReady =
          phase.observations?.some(
            (observation) => observation.type === 'GENERATION'
          ) === true;
        if (labelReady && phaseRootReady && phaseGenerationReady) {
          return { label, phase };
        }
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
  throw new Error(`Timed out waiting for activity traces from ${sourceRunId}`);
}

describeIfLive('activity label Langfuse export (live)', () => {
  jest.setTimeout(120_000);

  it('exports stable, correlated traces with correct observation types', async () => {
    const startedAt = new Date(Date.now() - 5000).toISOString();
    const sourceRunId = `activity-label-live-${Date.now()}`;
    const sessionId = `${sourceRunId}-session`;
    const model = process.env.ACTIVITY_LABEL_LIVE_MODEL ?? 'gpt-4.1-mini';
    const run = await Run.create({
      runId: sourceRunId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'activity-label-live-agent',
            name: 'Volatile Agent Display Name',
            provider: Providers.OPENAI,
            clientOptions: {
              apiKey: process.env.OPENAI_API_KEY,
              model,
            },
            tools: [],
          },
        ],
      },
    });
    if (run.Graph != null) {
      run.Graph.messages = [
        new HumanMessage('Verify activity-label Langfuse trace semantics'),
      ];
    }
    const chainOptions = {
      configurable: {
        thread_id: sessionId,
        user_id: 'activity-label-live-user',
        requestBody: { parentMessageId: 'activity-label-live-parent' },
      },
    };

    await run.generateActivityLabel({
      provider: Providers.OPENAI,
      agentId: 'activity-label-live-agent',
      clientOptions: {
        apiKey: process.env.OPENAI_API_KEY,
        model,
      },
      entries: [
        {
          toolName: 'inspect_runtime',
          toolInput: { target: 'trace-shape' },
          toolOutput: { status: 'verified' },
          status: 'success',
        },
      ],
      chainOptions,
    });
    await run.generateActivityPhaseLabel({
      provider: Providers.OPENAI,
      clientOptions: {
        apiKey: process.env.OPENAI_API_KEY,
        model,
      },
      activities: [
        { label: 'Inspected the activity-label trace shape' },
        { label: 'Verified stable correlation metadata' },
      ],
      sourceRunId,
      responseId: sourceRunId,
      phaseIndex: 0,
      chainOptions,
    });

    const { label, phase } = await findExportedTraces(sourceRunId, startedAt);
    expect(label).toMatchObject({
      name: 'LibreChat Activity Label',
      metadata: {
        sourceRunId,
        responseId: sourceRunId,
        activityIndex: 0,
      },
    });
    expect(phase).toMatchObject({
      name: 'LibreChat Activity Phase',
      metadata: {
        sourceRunId,
        responseId: sourceRunId,
        phaseIndex: 0,
        activityCount: 2,
      },
    });

    const labelGeneration = label.observations?.find(
      (observation) => observation.type === 'GENERATION'
    );
    expect(labelGeneration).toMatchObject({
      parentObservationId: null,
      name: 'llm',
      model: expect.stringContaining(model),
    });
    expect(labelGeneration?.usage?.total).toBeGreaterThan(0);

    const phaseRoot = phase.observations?.find(
      (observation) => observation.parentObservationId == null
    );
    expect(phaseRoot).toMatchObject({
      type: 'CHAIN',
      name: 'summarize-activity-phase',
    });
    const phaseGeneration = phase.observations?.find(
      (observation) => observation.type === 'GENERATION'
    );
    expect(phaseGeneration).toMatchObject({
      parentObservationId: phaseRoot?.id,
      name: 'llm',
      model: expect.stringContaining(model),
    });
    expect(phaseGeneration?.usage?.total).toBeGreaterThan(0);
  });
});
