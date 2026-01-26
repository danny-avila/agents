import {
  BaseMessage,
  BaseMessageLike,
  coerceMessageLikeToMessage,
} from '@langchain/core/messages';
import { v4 } from 'uuid';

export const REMOVE_ALL_MESSAGES = '__remove_all__';

export type Messages =
  | Array<BaseMessage | BaseMessageLike>
  | BaseMessage
  | BaseMessageLike;

/**
 * Prebuilt reducer that combines returned messages.
 * Can handle standard messages and special modifiers like {@link RemoveMessage} instances.
 */
export function messagesStateReducer(
  left: Messages,
  right: Messages
): BaseMessage[] {
  const leftArray = Array.isArray(left) ? left : [left];
  const rightArray = Array.isArray(right) ? right : [right];
  // coerce to message
  const leftMessages = (leftArray as BaseMessageLike[]).map(
    coerceMessageLikeToMessage
  );
  const rightMessages = (rightArray as BaseMessageLike[]).map(
    coerceMessageLikeToMessage
  );
  // assign missing ids
  for (const m of leftMessages) {
    if (m.id == null) {
      m.id = v4();
      m.lc_kwargs.id = m.id;
    }
  }

  let removeAllIdx: number | undefined;
  for (let i = 0; i < rightMessages.length; i += 1) {
    const m = rightMessages[i];
    if (m.id == null) {
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
