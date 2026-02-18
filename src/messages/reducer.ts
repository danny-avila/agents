import {
  BaseMessage,
  BaseMessageLike,
  ToolMessage,
  coerceMessageLikeToMessage,
} from '@langchain/core/messages';
import { v4 } from 'uuid';
import type * as t from '@/types';

export const REMOVE_ALL_MESSAGES = '__remove_all__';

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

  // Preserve and restore artifacts (coerceMessageLikeToMessage loses them)
  const preserveAndCoerce = (msgs: BaseMessageLike[]): BaseMessage[] => {
    return msgs.map((msg) => {
      // Extract artifact before coercion
      let artifact: t.MCPArtifact | undefined;
      if (typeof msg === 'object' && msg !== null) {
        const msgObj = msg as Record<string, unknown>;
        if (
          typeof msgObj._getType === 'function' &&
          msgObj._getType() === 'tool'
        ) {
          const toolMsgLike = msgObj as {
            artifact?: t.MCPArtifact;
            additional_kwargs?: { artifact?: t.MCPArtifact };
          };
          artifact =
            toolMsgLike.artifact ?? toolMsgLike.additional_kwargs?.artifact;
        }
      }

      // Coerce to BaseMessage
      const coerced = coerceMessageLikeToMessage(msg);

      // Restore artifact after coercion
      if (artifact && coerced._getType() === 'tool') {
        const toolMsg = coerced as ToolMessage & { artifact?: t.MCPArtifact };
        toolMsg.artifact = artifact;
        toolMsg.additional_kwargs = toolMsg.additional_kwargs ?? {};
        toolMsg.additional_kwargs.artifact = artifact;
      }

      return coerced;
    });
  };

  const leftMessages = preserveAndCoerce(leftArray as BaseMessageLike[]);
  const rightMessages = preserveAndCoerce(rightArray as BaseMessageLike[]);

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
        // Preserve artifacts when overwriting ToolMessages
        if (
          m.getType() === 'tool' &&
          merged[existingIdx].getType() === 'tool'
        ) {
          const existingToolMsg = merged[existingIdx] as ToolMessage & {
            artifact?: t.MCPArtifact;
          };
          const newToolMsg = m as ToolMessage & { artifact?: t.MCPArtifact };

          // Preserve artifact from existing message if new message doesn't have one
          const existingArtifact = (existingToolMsg.artifact ??
            existingToolMsg.additional_kwargs.artifact) as
            | t.MCPArtifact
            | undefined;
          if (existingArtifact && !newToolMsg.artifact) {
            newToolMsg.artifact = existingArtifact;
            newToolMsg.additional_kwargs = newToolMsg.additional_kwargs ?? {};
            newToolMsg.additional_kwargs.artifact = existingArtifact;
          }
        }
        idsToRemove.delete(m.id);
        merged[existingIdx] = m;
      }
    } else {
      if (m.getType() === 'remove') {
        throw new Error(
          `Attempting to delete a message with an ID that doesn't exist ('${m.id}')`
        );
      }
      // Preserve artifacts when adding new ToolMessages (especially when ID was undefined and got assigned)
      if (m.getType() === 'tool') {
        const toolMsg = m as ToolMessage & { artifact?: t.MCPArtifact };

        // Check if there's an existing ToolMessage with the same tool_call_id that has artifacts
        const existingWithSameToolCallId = merged.find((existing) => {
          if (existing.getType() !== 'tool') return false;
          const existingTool = existing as ToolMessage;
          return existingTool.tool_call_id === toolMsg.tool_call_id;
        }) as (ToolMessage & { artifact?: t.MCPArtifact }) | undefined;

        if (existingWithSameToolCallId) {
          // Preserve artifacts from existing message if new message doesn't have one
          const existingArtifact = (existingWithSameToolCallId.artifact ??
            existingWithSameToolCallId.additional_kwargs.artifact) as
            | t.MCPArtifact
            | undefined;

          if (existingArtifact && !toolMsg.artifact) {
            toolMsg.artifact = existingArtifact;
            toolMsg.additional_kwargs = toolMsg.additional_kwargs ?? {};
            toolMsg.additional_kwargs.artifact = existingArtifact;
          }
        }
      }
      mergedById.set(m.id, merged.length);
      merged.push(m);
    }
  }
  return merged.filter((m) => !idsToRemove.has(m.id));
}
