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
  reasoningStepId?: string;
  revision?: string | number;
};

type LangfuseObservation = {
  id: string;
  traceId: string | null;
  parentObservationId?: string | null;
  type: string;
  name?: string | null;
  traceName?: string | null;
  tags?: string[];
  metadata?: LangfuseMetadata;
  input?: unknown;
  output?: unknown;
  providedModelName?: string | null;
  usageDetails?: Record<string, number>;
};

type LangfuseTrace = {
  id: string;
  name?: string | null;
  tags?: string[];
  metadata?: LangfuseMetadata;
  observations: LangfuseObservation[];
};

type LangfuseObservationList = {
  data: LangfuseObservation[];
  meta?: { cursor?: string | null };
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

async function listObservations(
  query: URLSearchParams
): Promise<LangfuseObservation[]> {
  const observations: LangfuseObservation[] = [];
  let cursor: string | undefined;
  do {
    if (cursor != null) {
      query.set('cursor', cursor);
    }
    const page = await getJson<LangfuseObservationList>(
      `/api/public/v2/observations?${query.toString()}`
    );
    observations.push(...page.data);
    cursor = page.meta?.cursor ?? undefined;
  } while (cursor != null);
  return observations;
}

function groupObservationsByTrace(
  observations: LangfuseObservation[]
): LangfuseTrace[] {
  const traces = new Map<string, LangfuseTrace>();
  for (const observation of observations) {
    if (observation.traceId == null) {
      continue;
    }
    const existing = traces.get(observation.traceId);
    if (existing != null) {
      existing.observations.push(observation);
      continue;
    }
    traces.set(observation.traceId, {
      id: observation.traceId,
      name: observation.traceName,
      tags: observation.tags,
      metadata: observation.metadata,
      observations: [observation],
    });
  }
  return [...traces.values()];
}

async function findExportedTraces(
  sourceRunId: string,
  fromTimestamp: string
): Promise<{
  label: LangfuseTrace;
  phase: LangfuseTrace;
  reasoning: LangfuseTrace;
}> {
  const query = new URLSearchParams({
    limit: '1000',
    fields: 'core,basic,io,metadata,model,usage,trace_context',
    fromStartTime: fromTimestamp,
  });
  for (let attempt = 0; attempt < 20; attempt += 1) {
    query.set('toStartTime', new Date(Date.now() + 60_000).toISOString());
    query.delete('cursor');
    const candidates = groupObservationsByTrace(
      await listObservations(query)
    ).filter((trace) => trace.metadata?.sourceRunId === sourceRunId);
    const labelSummary = candidates.find(
      (trace) => trace.tags?.includes('activity-label') === true
    );
    const phaseSummary = candidates.find(
      (trace) => trace.tags?.includes('activity-phase') === true
    );
    const reasoningSummary = candidates.find(
      (trace) => trace.tags?.includes('reasoning-label') === true
    );
    if (
      labelSummary != null &&
      phaseSummary != null &&
      reasoningSummary != null
    ) {
      const labelReady = labelSummary.observations.some(
        (observation) => observation.type === 'GENERATION'
      );
      const phaseRootReady = phaseSummary.observations.some(
        (observation) =>
          observation.parentObservationId == null && observation.type === 'SPAN'
      );
      const phaseGenerationReady = phaseSummary.observations.some(
        (observation) => observation.type === 'GENERATION'
      );
      const reasoningReady = reasoningSummary.observations.some(
        (observation) => observation.type === 'GENERATION'
      );
      if (
        labelReady &&
        phaseRootReady &&
        phaseGenerationReady &&
        reasoningReady
      ) {
        return {
          label: labelSummary,
          phase: phaseSummary,
          reasoning: reasoningSummary,
        };
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
  throw new Error(`Timed out waiting for activity traces from ${sourceRunId}`);
}

describeIfLive('activity label Langfuse export (live)', () => {
  jest.setTimeout(120_000);

  it('exports stable, correlated traces with the expected observation hierarchy', async () => {
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
    await run.generateReasoningLabel({
      provider: Providers.OPENAI,
      agentId: 'activity-label-live-agent',
      clientOptions: {
        apiKey: process.env.OPENAI_API_KEY,
        model,
      },
      visibleReasoning:
        'I am verifying that reasoning revisions remain correlated in Langfuse.',
      reasoningStepId: 'reasoning-step-live-1',
      revision: 0,
      sourceRunId,
      responseId: sourceRunId,
      chainOptions,
    });

    const { label, phase, reasoning } = await findExportedTraces(
      sourceRunId,
      startedAt
    );
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
    expect(reasoning).toMatchObject({
      name: 'LibreChat Reasoning Label',
      metadata: {
        sourceRunId,
        responseId: sourceRunId,
        reasoningStepId: 'reasoning-step-live-1',
        revision: 0,
      },
    });

    for (const [trace, tag] of [
      [label, 'activity-label'],
      [phase, 'activity-phase'],
      [reasoning, 'reasoning-label'],
    ] as const) {
      for (const observation of trace.observations) {
        expect(observation.metadata).toMatchObject({ sourceRunId });
        expect(observation.tags).toContain(tag);
      }
      const root = trace.observations.find(
        (observation) => observation.parentObservationId == null
      );
      expect(root?.input).toBeDefined();
      expect(root?.output).toBeDefined();
    }

    const labelGeneration = label.observations.find(
      (observation) => observation.type === 'GENERATION'
    );
    expect(labelGeneration).toMatchObject({
      parentObservationId: null,
      name: 'StepLabel',
      providedModelName: expect.stringContaining(model),
    });
    expect(labelGeneration?.usageDetails?.total).toBeGreaterThan(0);

    const phaseRoot = phase.observations.find(
      (observation) => observation.parentObservationId == null
    );
    expect(phaseRoot).toMatchObject({
      type: 'SPAN',
      name: 'MultiStepLabel',
    });
    const phaseGeneration = phase.observations.find(
      (observation) => observation.type === 'GENERATION'
    );
    expect(phaseGeneration).toMatchObject({
      parentObservationId: phaseRoot?.id,
      name: 'MultiStepLabelGeneration',
      providedModelName: expect.stringContaining(model),
    });
    expect(phaseGeneration?.usageDetails?.total).toBeGreaterThan(0);

    const reasoningGeneration = reasoning.observations.find(
      (observation) => observation.type === 'GENERATION'
    );
    expect(reasoningGeneration).toMatchObject({
      parentObservationId: null,
      name: 'ReasoningLabel',
      providedModelName: expect.stringContaining(model),
    });
    expect(reasoningGeneration?.usageDetails?.total).toBeGreaterThan(0);
  });
});
