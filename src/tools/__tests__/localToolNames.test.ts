import { describe, it, expect } from '@jest/globals';
import {
  Constants,
  LOCAL_CODING_BUNDLE_NAMES,
  LOCAL_CODING_TOOL_NAMES,
} from '@/common';
import {
  createLocalCodingTools,
  createLocalCodingToolDefinitions,
  LocalEditFileToolName,
  LocalGlobSearchToolName,
  LocalGrepSearchToolName,
  LocalListDirectoryToolName,
  LocalWriteFileToolName,
} from '../local/LocalCodingTools';
import { CompileCheckToolName } from '../local/CompileCheckTool';

/**
 * Pins the tool name surface so a typo upstream — in factories,
 * registry definitions, the policy hook, or LibreChat's icon map —
 * gets caught at build time. The wire-level strings have to match
 * what consumer UIs already special-case (`bash_tool`, `read_file`,
 * `execute_code`, `run_tools_with_code`) and what they may add icons
 * for next (write_file, edit_file, grep_search, glob_search,
 * list_directory, compile_check).
 */
describe('local coding tool names', () => {
  it('the per-file Local*ToolName aliases point at canonical Constants', () => {
    expect(LocalWriteFileToolName).toBe(Constants.WRITE_FILE);
    expect(LocalEditFileToolName).toBe(Constants.EDIT_FILE);
    expect(LocalGrepSearchToolName).toBe(Constants.GREP_SEARCH);
    expect(LocalGlobSearchToolName).toBe(Constants.GLOB_SEARCH);
    expect(LocalListDirectoryToolName).toBe(Constants.LIST_DIRECTORY);
    expect(CompileCheckToolName).toBe(Constants.COMPILE_CHECK);
  });

  it('canonical strings match what consumer UIs (e.g. LibreChat icon map) recognise', () => {
    expect(Constants.READ_FILE).toBe('read_file');
    expect(Constants.WRITE_FILE).toBe('write_file');
    expect(Constants.EDIT_FILE).toBe('edit_file');
    expect(Constants.GREP_SEARCH).toBe('grep_search');
    expect(Constants.GLOB_SEARCH).toBe('glob_search');
    expect(Constants.LIST_DIRECTORY).toBe('list_directory');
    expect(Constants.COMPILE_CHECK).toBe('compile_check');
    expect(Constants.BASH_TOOL).toBe('bash_tool');
    expect(Constants.EXECUTE_CODE).toBe('execute_code');
    expect(Constants.PROGRAMMATIC_TOOL_CALLING).toBe('run_tools_with_code');
    expect(Constants.BASH_PROGRAMMATIC_TOOL_CALLING).toBe('run_tools_with_bash');
  });

  it('LOCAL_CODING_BUNDLE_NAMES matches every name the bundle ships', () => {
    const tools = createLocalCodingTools();
    const bundleNames = tools.map((t) => t.name).sort();
    const advertisedNames = [...LOCAL_CODING_BUNDLE_NAMES].sort();
    expect(bundleNames).toEqual(advertisedNames);
  });

  it('LOCAL_CODING_TOOL_NAMES matches the registry-definitions list (local-only tools)', () => {
    const defs = createLocalCodingToolDefinitions();
    const defNames = defs.map((d) => d.name).sort();
    const advertisedNames = [...LOCAL_CODING_TOOL_NAMES].sort();
    expect(defNames).toEqual(advertisedNames);
  });

  it('LOCAL_CODING_BUNDLE_NAMES is a strict superset of LOCAL_CODING_TOOL_NAMES', () => {
    for (const name of LOCAL_CODING_TOOL_NAMES) {
      expect(LOCAL_CODING_BUNDLE_NAMES).toContain(name);
    }
    expect(LOCAL_CODING_BUNDLE_NAMES.length).toBeGreaterThan(
      LOCAL_CODING_TOOL_NAMES.length
    );
  });
});
