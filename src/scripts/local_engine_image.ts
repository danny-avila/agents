/**
 * src/scripts/local_engine_image.ts
 *
 * Live demo of the local read_file tool returning an image as an
 * inline `MessageContentComplex[]` attachment when
 * `local.attachReadAttachments: 'images-only'` is set. We seed a tiny
 * red square PNG into the workspace, ask the agent to read it, and
 * verify the model receives the image bytes (not just a refusal stub).
 *
 * Best run with a vision-capable provider, e.g.:
 *   npm run local:image -- --provider anthropic
 */
import { config } from 'dotenv';
config();
import { tmpdir } from 'os';
import { join } from 'path';
import { copyFile, mkdtemp, rm, writeFile } from 'fs/promises';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { GraphEvents } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

// Canonical 1x1 transparent PNG fallback if the system PNG isn't readable.
const TINY_PNG_HEX =
  '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15' +
  'c4890000000a49444154789c63000100000005000165be7e6e0000000049' +
  '454e44ae426082';

const SAMPLE_PNG_PATHS = [
  '/System/Library/CoreServices/Certificate Assistant.app/Contents/Resources/droppedImage.png',
  '/System/Library/CoreServices/Certificate Assistant.app/Contents/Resources/shapeimage_1.png',
  '/System/Library/CoreServices/BluetoothUIServer.app/Contents/Resources/handoff.png',
];

async function main(): Promise<void> {
  const { userName, provider } = await getArgs();
  const cwd = await mkdtemp(join(tmpdir(), 'lc-local-image-'));
  console.log(`[local_engine_image] workspace: ${cwd}`);

  const pngPath = join(cwd, 'sample.png');
  let copied = false;
  for (const sample of SAMPLE_PNG_PATHS) {
    try {
      await copyFile(sample, pngPath);
      copied = true;
      break;
    } catch {
      // try next
    }
  }
  if (!copied) {
    await writeFile(pngPath, Buffer.from(TINY_PNG_HEX, 'hex'));
  }
  console.log(`[local_engine_image] wrote ${pngPath} (real-png=${copied})`);

  const { aggregateContent } = createContentAggregator();
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        const obj = data as unknown as { name?: string; input?: unknown };
        console.log(
          `====== TOOL_START tool=${obj.name ?? '?'} input=${JSON.stringify(
            obj.input
          )} ======`
        );
      },
    },
  };

  const llmConfig = getLLMConfig(provider);

  const runConfig: t.RunConfig = {
    runId: 'local-engine-image-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      instructions:
        `You are ${userName}'s assistant. When asked to inspect a file, ` +
        'use `read_file`. The host has enabled inline image attachments, so ' +
        'reading an image will deliver the actual image bytes to your vision ' +
        'context. Describe the image you see.',
    },
    toolExecution: {
      engine: 'local',
      local: {
        cwd,
        attachReadAttachments: 'images-only',
        timeoutMs: 30_000,
      },
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers,
  };
  const run = await Run.create<t.IState>(runConfig);

  const streamConfig = {
    configurable: { provider, thread_id: 'local-engine-image-thread-1' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage = new HumanMessage(
    'Please call `read_file` on `sample.png` (in the working directory) and ' +
      'then briefly describe what the image shows.'
  );
  conversationHistory.push(userMessage);
  console.log('====== USER ======\n' + userMessage.content + '\n');

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
  }

  console.log('\n====== TOOL MESSAGES IN HISTORY ======');
  let imageBlockSeen = false;
  for (const msg of conversationHistory) {
    if (msg instanceof ToolMessage) {
      const isArray = Array.isArray(msg.content);
      const blocks = isArray
        ? (msg.content as Array<{ type?: string; image_url?: { url?: string } }>)
        : [];
      const types = blocks.map((b) => b.type ?? '?').join(',');
      const url = blocks.find((b) => b.type === 'image_url')?.image_url?.url;
      console.log(
        `- ToolMessage(name=${msg.name}, content=${
          isArray ? `[${types}]` : typeof msg.content
        }${url ? ` url=${url.slice(0, 40)}…` : ''})`
      );
      if (isArray && blocks.some((b) => b.type === 'image_url')) {
        imageBlockSeen = true;
      }
    }
  }
  console.log(
    `\nImage attachment landed in tool result: ${imageBlockSeen} ${
      imageBlockSeen ? '✔' : '✖'
    }`
  );

  console.log('\n====== ASSISTANT FINAL TEXT ======');
  const lastAssistant = [...conversationHistory]
    .reverse()
    .find((m) => m._getType() === 'ai');
  if (lastAssistant) {
    const c = lastAssistant.content;
    console.log(
      typeof c === 'string'
        ? c
        : Array.isArray(c)
          ? c
            .map((b) => ('text' in b ? b.text : `<${b.type}>`))
            .join(' ')
          : JSON.stringify(c)
    );
  }

  if (!imageBlockSeen) {
    process.exitCode = 1;
  }
  await rm(cwd, { recursive: true, force: true });
}

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
