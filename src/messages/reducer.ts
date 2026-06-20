import { v4 } from 'uuid';
import {
  BaseMessage,
  RemoveMessage,
  BaseMessageLike,
  coerceMessageLikeToMessage,
} from '@langchain/core/messages';

export const REMOVE_ALL_MESSAGES = '__remove_all__';

/**
 * Creates a message that instructs messagesStateReducer to remove ALL
 * existing messages from state.  Messages appearing after this one in
 * the array become the new state.
 *
 * Usage (in a node return value):
 * ```ts
 * return { messages: [createRemoveAllMessage(), ...survivingMessages] };
 * ```
 *
 * This works because the reducer checks for `getType() === 'remove'`
 * with `id === REMOVE_ALL_MESSAGES` and discards everything before it.
 *
 * NOTE: Uses RemoveMessage from @langchain/core with a sentinel id so
 * the reducer can distinguish a "remove-all" marker from a single-message
 * removal.
 */
export function createRemoveAllMessage(): BaseMessage {
  return new RemoveMessage({ id: REMOVE_ALL_MESSAGES });
}

export type Messages =
  | Array<BaseMessage | BaseMessageLike>
  | BaseMessage
  | BaseMessageLike;

/**
 * Coerce each entry to a {@link BaseMessage} in a single pass, skipping
 * null/undefined entries. Providers can emit empty/partial stream chunks that
 * arrive as `undefined`; passing those to `coerceMessageLikeToMessage` throws
 * "Cannot read properties of undefined (reading 'role')" and crashes the run.
 * Folding the null check into the coercion loop avoids a second pass over the
 * array. Refs LibreChat Discussion #12284.
 */
function coerceMessages(
  messages: ReadonlyArray<BaseMessageLike | null | undefined>
): BaseMessage[] {
  const coerced: BaseMessage[] = [];
  for (const message of messages) {
    if (message != null) {
      coerced.push(coerceMessageLikeToMessage(message));
    }
  }
  return coerced;
}

/**
 * Prebuilt reducer that combines returned messages.
 * Can handle standard messages and special modifiers like {@link RemoveMessage}
 * instances.
 */
export function messagesStateReducer(
  left: Messages,
  right: Messages
): BaseMessage[] {
  const leftArray = Array.isArray(left) ? left : [left];
  const rightArray = Array.isArray(right) ? right : [right];
  // coerce to message, skipping null/undefined entries in the same pass
  const leftMessages = coerceMessages(leftArray as BaseMessageLike[]);
  const rightMessages = coerceMessages(rightArray as BaseMessageLike[]);
  // assign missing ids
  for (const m of leftMessages) {
    if (m.id === null || m.id === undefined) {
      m.id = v4();
      m.lc_kwargs.id = m.id;
    }
  }

  let removeAllIdx: number | undefined;
  for (let i = 0; i < rightMessages.length; i += 1) {
    const m = rightMessages[i];
    if (m.id === null || m.id === undefined) {
      m.id = v4();
      m.lc_kwargs.id = m.id;
    }

    if (m.getType() === 'remove' && m.id === REMOVE_ALL_MESSAGES) {
      removeAllIdx = i;
    }
  }

  if (removeAllIdx != null) return rightMessages.slice(removeAllIdx + 1);

  // merge
  const merged = [...leftMessages];
  const mergedById = new Map(merged.map((m, i) => [m.id, i]));
  const idsToRemove = new Set();
  for (const m of rightMessages) {
    const existingIdx = mergedById.get(m.id);
    if (existingIdx !== undefined) {
      if (m.getType() === 'remove') {
        idsToRemove.add(m.id);
      } else {
        idsToRemove.delete(m.id);
        merged[existingIdx] = m;
      }
    } else {
      if (m.getType() === 'remove') {
        throw new Error(
          `Attempting to delete a message with an ID that doesn't exist ('${m.id}')`
        );
      }
      mergedById.set(m.id, merged.length);
      merged.push(m);
    }
  }
  return merged.filter((m) => !idsToRemove.has(m.id));
}
