import type * as t from '@/types';

export type ProgrammaticInvocationParams = {
  code: string;
  tool_manifest?: string[];
  timeout?: number;
  lang?: string;
  runtime?: string;
  language?: string;
};

export function resolveProgrammaticToolDefinitions(
  context: Partial<t.ProgrammaticCache> & { tools?: t.LCTool[] }
): t.LCTool[] | undefined {
  return context.toolDefs ?? context.tools;
}

export function projectProgrammaticToolMap(
  toolMap: t.ToolMap,
  selectedToolDefs: readonly t.LCTool[]
): t.ToolMap {
  const selectedNames = new Set(selectedToolDefs.map((toolDef) => toolDef.name));
  return new Map(
    [...toolMap].filter(([name]) => selectedNames.has(name))
  );
}

export function assertDisallowedToolUsage(
  disallowedNames: ReadonlySet<string>,
  programmaticToolName: string
): void {
  if (disallowedNames.size === 0) {
    return;
  }

  const names = [...disallowedNames];
  throw new Error(
    `Tool${names.length === 1 ? '' : 's'} ${names
      .map((name) => `"${name}"`)
      .join(', ')} cannot be called from "${programmaticToolName}" because ` +
      `the${names.length === 1 ? ' tool is' : 'se tools are'} not marked for code_execution. ` +
      `Call ${names.length === 1 ? 'it' : 'them'} directly instead.`
  );
}

/**
 * Rejects a selection holding two tools that collapse to one runtime identifier.
 *
 * Stub generation binds by identifier, so such tools are indistinguishable once
 * emitted — the generated Python defines both and the later one wins. Failing
 * here beats dispatching by declaration order.
 */
export function assertUnambiguousIdentifiers(
  selectedToolDefs: readonly t.LCTool[],
  normalizeIdentifier: (name: string) => string,
  programmaticToolName: string
): void {
  const seen = new Map<string, string>();
  for (const toolDef of selectedToolDefs) {
    const identifier = normalizeIdentifier(toolDef.name);
    const prior = seen.get(identifier);
    if (prior != null && prior !== toolDef.name) {
      throw new Error(
        `Tools "${prior}" and "${toolDef.name}" both become "${identifier}" in ` +
          `the sandbox, so "${programmaticToolName}" cannot tell which one the ` +
          'code means. Call them directly, or rename one.'
      );
    }
    seen.set(identifier, toolDef.name);
  }
}

export function selectProgrammaticTools(args: {
  requestedToolNames?: string[];
  allowedToolDefs?: t.LCTool[];
  disallowedToolDefs?: t.LCTool[];
  programmaticToolName: string;
  /** Runtime identifier a tool name binds to, used to reject ambiguity. */
  normalizeIdentifier: (name: string) => string;
}): t.LCTool[] {
  const allowedToolDefs = args.allowedToolDefs ?? [];
  const disallowedToolDefs = args.disallowedToolDefs ?? [];
  const requestedToolNames = args.requestedToolNames;

  if (requestedToolNames == null) {
    if (disallowedToolDefs.length > 0) {
      throw new Error(
        `"${args.programmaticToolName}" requires a tool_manifest when direct-only tools are configured. ` +
          'List every registered tool name used by the submitted code in the tool_manifest field.'
      );
    }
    assertUnambiguousIdentifiers(
      allowedToolDefs,
      args.normalizeIdentifier,
      args.programmaticToolName
    );
    return allowedToolDefs;
  }

  const disallowedNames = new Set(
    disallowedToolDefs.map((toolDef) => toolDef.name)
  );
  const requestedDisallowedNames = new Set(
    requestedToolNames.filter((name) => disallowedNames.has(name))
  );
  assertDisallowedToolUsage(
    requestedDisallowedNames,
    args.programmaticToolName
  );

  const allowedByName = new Map(
    allowedToolDefs.map((toolDef) => [toolDef.name, toolDef])
  );
  const unknownNames = new Set(
    requestedToolNames.filter((name) => !allowedByName.has(name))
  );
  if (unknownNames.size > 0) {
    throw new Error(
      `Tool${unknownNames.size === 1 ? '' : 's'} ${[...unknownNames]
        .map((name) => `"${name}"`)
        .join(
          ', '
        )} cannot be used by "${args.programmaticToolName}" because ` +
        `${unknownNames.size === 1 ? 'it is' : 'they are'} not available for code_execution.`
    );
  }

  const selected = [...new Set(requestedToolNames)].map(
    (name) => allowedByName.get(name) as t.LCTool
  );
  assertUnambiguousIdentifiers(
    selected,
    args.normalizeIdentifier,
    args.programmaticToolName
  );
  return selected;
}
