import {
  BaseMessage,
  MessageType,
  HumanMessage,
  BaseMessageLike,
  coerceMessageLikeToMessage,
} from '@langchain/core/messages';
import { v4 } from 'uuid';

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
 * NOTE: RemoveMessage from @langchain/core is not re-exported, so we
 * construct a compatible BaseMessage with the correct type/id contract.
 */
export function createRemoveAllMessage(): BaseMessage {
  const msg = new HumanMessage({ content: '', id: REMOVE_ALL_MESSAGES });
  // Override _getType so the reducer recognises this as a removal marker.
  // The reducer only inspects getType() and id — no other fields matter.
  msg._getType = (): MessageType => 'remove';
  return msg;
}

export type Messages =
  | Array<BaseMessage | BaseMessageLike>
  | BaseMessage
  | BaseMessageLike;

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
  // coerce to message
  const leftMessages = (leftArray as BaseMessageLike[]).map(
    coerceMessageLikeToMessage
  );
  const rightMessages = (rightArray as BaseMessageLike[]).map(
    coerceMessageLikeToMessage
  );
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
