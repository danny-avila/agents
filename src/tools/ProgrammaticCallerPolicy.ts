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

export function selectProgrammaticTools(args: {
  requestedToolNames?: string[];
  allowedToolDefs?: t.LCTool[];
  disallowedToolDefs?: t.LCTool[];
  programmaticToolName: string;
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

  return [...new Set(requestedToolNames)].map(
    (name) => allowedByName.get(name) as t.LCTool
  );
}
