export const DEFAULT_STREAM_DELAY = 25;
export const SMOOTH_TARGET_LATENCY_MS = 250;
export const MAX_STREAM_QUEUE_CHUNKS = 256;
export const MAX_STREAM_QUEUE_TEXT_CHARS = 8192;
export const STREAM_CHUNK_MIN_SIZE = 4;
export const STREAM_BOUNDARIES: ReadonlySet<string> = new Set([
  ' ',
  '.',
  ',',
  '!',
  '?',
  ';',
  ':',
]);

export const STREAM_ABORT_MESSAGE = 'AbortError: User aborted the request.';

/** Resolves a configured stream delay to its effective value (default 25ms; 0 disables smoothing). */
export function resolveStreamDelay(delay?: number): number {
  return Math.max(0, delay ?? DEFAULT_STREAM_DELAY);
}

export function isSignalAborted(signal?: AbortSignal): boolean {
  return signal?.aborted === true;
}

export function findStreamChunkBoundary(
  text: string,
  minSize: number
): number {
  if (minSize >= text.length) {
    return text.length;
  }

  for (let position = minSize; position < text.length; position++) {
    if (STREAM_BOUNDARIES.has(text[position])) {
      return position + 1;
    }
  }

  return text.length;
}

/**
 * Backlog-proportional piece sizing: emit enough per tick that the current
 * backlog drains in ~`targetLatencyMs`, so render lag stays pinned near the
 * target regardless of how fast the provider streams. Token-sized arrivals
 * never exceed the minimum piece, matching the legacy fixed-size splitter.
 */
export function computeAdaptivePieceSize(
  bufferedTextLength: number,
  tickMs: number,
  targetLatencyMs: number = SMOOTH_TARGET_LATENCY_MS
): number {
  if (bufferedTextLength <= 0) {
    return STREAM_CHUNK_MIN_SIZE;
  }
  if (tickMs <= 0 || targetLatencyMs <= 0) {
    return bufferedTextLength;
  }
  return Math.max(
    STREAM_CHUNK_MIN_SIZE,
    Math.ceil((bufferedTextLength * tickMs) / targetLatencyMs)
  );
}

/**
 * A cadence, not an additive sleep: time the consumer already spent since the
 * last visible emission counts against the target delay, so slow downstream
 * handlers never compound latency.
 */
export function getCadencedStreamDelay({
  targetDelay,
  lastVisibleTextAt,
  now,
}: {
  targetDelay: number;
  lastVisibleTextAt?: number;
  now: number;
}): number {
  if (targetDelay <= 0 || lastVisibleTextAt == null) {
    return 0;
  }
  return Math.max(0, targetDelay - (now - lastVisibleTextAt));
}

/** Abort-aware sleep that resolves (never rejects) on abort; callers re-check the signal. */
export async function waitForStreamDelay(
  delay: number,
  signal?: AbortSignal
): Promise<void> {
  if (delay <= 0 || isSignalAborted(signal)) {
    return;
  }
  await new Promise<void>((resolve) => {
    const timeoutRef: { current?: ReturnType<typeof setTimeout> } = {};
    const onAbort = (): void => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      signal?.removeEventListener('abort', onAbort);
      resolve();
    };
    timeoutRef.current = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, delay);
    signal?.addEventListener('abort', onAbort, { once: true });
    if (isSignalAborted(signal)) {
      onAbort();
    }
  });
}

export type SmoothPiece = {
  text: string;
  isFirst: boolean;
  isLast: boolean;
};

/**
 * One classified unit of provider stream output.
 *
 * - `smooth: true` — visible text, paced at the configured cadence and (unless
 *   `atomic`) sliced adaptively at dequeue time.
 * - `atomic: true` — paced as a single piece, never split (text-bearing chunks
 *   whose metadata cannot survive slicing, e.g. logprobs / finish_reason).
 * - `smooth: false` — passthrough: tool-call deltas, usage-only, id-only and
 *   seal chunks. Zero delay, strict FIFO with the text around them.
 *
 * `emit` builds the provider-specific output for one piece; `isFirst` lets
 * providers keep usage_metadata on only the first piece of a split.
 */
export type SmoothItem<TEmit> = {
  text: string;
  smooth: boolean;
  atomic?: boolean;
  emit: (piece: SmoothPiece) => TEmit;
};

type ProducerState = {
  done: boolean;
  error?: unknown;
};

type QueuedSmoothItem<TEmit> = {
  item: SmoothItem<TEmit>;
  textLength: number;
};

/**
 * Bounded producer/consumer smoothing engine.
 *
 * The producer drains `source` eagerly into a bounded queue (the buffer is the
 * backlog measurement adaptive sizing needs); at capacity it parks, applying
 * backpressure to the underlying stream. The consumer emits paced pieces,
 * decrementing the text budget and waking the producer *before* each cadenced
 * sleep so the provider stream keeps being read during pacing.
 *
 * `delayMs <= 0` disables smoothing entirely: every item passes through FIFO,
 * unsplit and undelayed.
 */
export async function* smoothStream<TEmit>({
  source,
  delayMs,
  signal,
  abortUpstream,
}: {
  source: AsyncIterable<SmoothItem<TEmit>>;
  delayMs: number;
  signal?: AbortSignal;
  abortUpstream?: () => void;
}): AsyncGenerator<TEmit> {
  const smoothingEnabled = delayMs > 0;
  const queuedItems: QueuedSmoothItem<TEmit>[] = [];
  const producerState: ProducerState = { done: false };
  let queuedItemIndex = 0;
  let bufferedTextLength = 0;
  let consumerClosed = false;
  let notifyConsumer: (() => void) | undefined;
  let notifyProducer: (() => void) | undefined;

  const notifyConsumerForItem = (): void => {
    notifyConsumer?.();
    notifyConsumer = undefined;
  };

  const notifyProducerForSpace = (): void => {
    notifyProducer?.();
    notifyProducer = undefined;
  };

  const hasQueuedItems = (): boolean => queuedItemIndex < queuedItems.length;

  const getQueuedItemCount = (): number =>
    queuedItems.length - queuedItemIndex;

  const isQueueAtCapacity = (): boolean =>
    getQueuedItemCount() >= MAX_STREAM_QUEUE_CHUNKS ||
    bufferedTextLength >= MAX_STREAM_QUEUE_TEXT_CHARS;

  const waitForNextItem = async (): Promise<void> => {
    if (hasQueuedItems() || producerState.done || producerState.error != null) {
      return;
    }
    await new Promise<void>((resolve) => {
      notifyConsumer = resolve;
    });
  };

  const waitForQueueSpace = async (): Promise<void> => {
    while (
      isQueueAtCapacity() &&
      !consumerClosed &&
      !isSignalAborted(signal)
    ) {
      await new Promise<void>((resolve) => {
        const onAbort = (): void => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        };
        const onSpace = (): void => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        };
        notifyProducer = onSpace;
        signal?.addEventListener('abort', onAbort, { once: true });
        if (isSignalAborted(signal)) {
          onAbort();
        }
      });
    }
  };

  const dequeue = (): QueuedSmoothItem<TEmit> | undefined => {
    if (!hasQueuedItems()) {
      return undefined;
    }
    const queuedItem = queuedItems[queuedItemIndex];
    queuedItemIndex++;
    if (queuedItemIndex > 128 && queuedItemIndex * 2 >= queuedItems.length) {
      queuedItems.splice(0, queuedItemIndex);
      queuedItemIndex = 0;
    }
    return queuedItem;
  };

  const throwAborted = (): never => {
    abortUpstream?.();
    throw new Error(STREAM_ABORT_MESSAGE);
  };

  const enqueue = async (item: SmoothItem<TEmit>): Promise<void> => {
    await waitForQueueSpace();
    if (consumerClosed || isSignalAborted(signal)) {
      throwAborted();
    }
    const isSmooth = smoothingEnabled && item.smooth;
    const textLength = isSmooth ? item.text.length : 0;
    queuedItems.push({ item, textLength });
    bufferedTextLength += textLength;
    notifyConsumerForItem();
  };

  const producer = (async (): Promise<void> => {
    try {
      for await (const item of source) {
        if (isSignalAborted(signal)) {
          throwAborted();
        }
        await enqueue(item);
      }
    } catch (error) {
      producerState.error = error;
    } finally {
      producerState.done = true;
      notifyConsumerForItem();
    }
  })();

  let hasEmittedText = false;
  let lastVisibleTextAt: number | undefined;
  let keepStreaming = true;
  try {
    while (keepStreaming) {
      if (isSignalAborted(signal)) {
        throwAborted();
      }

      await waitForNextItem();
      const queuedItem = dequeue();

      if (!queuedItem) {
        if (producerState.error != null) {
          throw producerState.error;
        }
        if (producerState.done) {
          keepStreaming = false;
        }
        continue;
      }

      const { item } = queuedItem;
      const isSmooth = smoothingEnabled && item.smooth;

      if (!isSmooth) {
        notifyProducerForSpace();
        yield item.emit({ text: item.text, isFirst: true, isLast: true });
        continue;
      }

      if (item.text === '') {
        bufferedTextLength = Math.max(
          0,
          bufferedTextLength - queuedItem.textLength
        );
        notifyProducerForSpace();
        continue;
      }

      let headOffset = 0;
      while (headOffset < item.text.length) {
        const remainingText = item.text.slice(headOffset);
        const pieceLength = item.atomic
          ? remainingText.length
          : findStreamChunkBoundary(
            remainingText,
            computeAdaptivePieceSize(bufferedTextLength, delayMs)
          );
        const pieceEnd = headOffset + pieceLength;
        const piece = item.text.slice(headOffset, pieceEnd);

        bufferedTextLength = Math.max(0, bufferedTextLength - piece.length);
        notifyProducerForSpace();
        await waitForStreamDelay(
          getCadencedStreamDelay({
            targetDelay: hasEmittedText ? delayMs : 0,
            lastVisibleTextAt,
            now: Date.now(),
          }),
          signal
        );
        if (isSignalAborted(signal)) {
          throwAborted();
        }
        hasEmittedText = true;
        lastVisibleTextAt = Date.now();

        yield item.emit({
          text: piece,
          isFirst: headOffset === 0,
          isLast: pieceEnd === item.text.length,
        });
        headOffset = pieceEnd;
      }
    }
  } finally {
    consumerClosed = true;
    if (!producerState.done) {
      abortUpstream?.();
      notifyProducerForSpace();
    }
    await producer;
  }
}
