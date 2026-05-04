import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, rm, writeFile } from 'fs/promises';
import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { Constants } from '@/common';
import { createLocalCodingToolBundle } from '../local/LocalCodingTools';
import {
  resolveWorkspacePathSafe,
  getWorkspaceFS,
} from '../local/LocalExecutionEngine';
import { nodeWorkspaceFS } from '../local/workspaceFS';
import type { WorkspaceFS } from '../local/workspaceFS';

describe('workspace seam', () => {
  let workspace: string;
  let extra: string;

  beforeEach(async () => {
    workspace = await mkdtemp(join(tmpdir(), 'lc-ws-'));
    extra = await mkdtemp(join(tmpdir(), 'lc-ws-extra-'));
  });

  afterEach(async () => {
    await rm(workspace, { recursive: true, force: true });
    await rm(extra, { recursive: true, force: true });
  });

  describe('additionalRoots', () => {
    it('lets read tools touch paths inside an additional root', async () => {
      await writeFile(join(extra, 'foo.txt'), 'extra contents\n', 'utf8');
      const path = await resolveWorkspacePathSafe(
        join(extra, 'foo.txt'),
        { workspace: { root: workspace, additionalRoots: [extra] } },
        'read'
      );
      expect(path).toBe(join(extra, 'foo.txt'));
    });

    it('still rejects truly outside paths', async () => {
      await expect(
        resolveWorkspacePathSafe(
          '/etc/passwd',
          { workspace: { root: workspace, additionalRoots: [extra] } },
          'read'
        )
      ).rejects.toThrow(/outside the local workspace/);
    });

    it('honours allowReadOutside split from allowWriteOutside', async () => {
      const cfg = {
        workspace: {
          root: workspace,
          allowReadOutside: true,
          allowWriteOutside: false,
        },
      };
      const readPath = await resolveWorkspacePathSafe(
        '/tmp/whatever.txt',
        cfg,
        'read'
      );
      expect(readPath).toBe('/tmp/whatever.txt');
      await expect(
        resolveWorkspacePathSafe('/tmp/whatever.txt', cfg, 'write')
      ).rejects.toThrow(/outside the local workspace/);
    });

    it('legacy `cwd` + `allowOutsideWorkspace` still works', async () => {
      const cfg = { cwd: workspace, allowOutsideWorkspace: true };
      const p = await resolveWorkspacePathSafe('/tmp/x.txt', cfg, 'write');
      expect(p).toBe('/tmp/x.txt');
    });

    it('legacy `cwd` is honoured when `workspace.root` is absent', async () => {
      const cfg = { cwd: workspace };
      await expect(
        resolveWorkspacePathSafe('/tmp/x.txt', cfg, 'write')
      ).rejects.toThrow(/outside the local workspace/);
      const p = await resolveWorkspacePathSafe(
        join(workspace, 'a.txt'),
        cfg,
        'write'
      );
      expect(p).toBe(join(workspace, 'a.txt'));
    });
  });

  describe('WorkspaceFS seam', () => {
    it('defaults to the Node host fs when nothing is supplied', () => {
      expect(getWorkspaceFS({ workspace: { root: workspace } })).toBe(
        nodeWorkspaceFS
      );
    });

    it('routes file tool calls through a custom WorkspaceFS', async () => {
      // Spy: every read/write/etc. routes through `tracked`. We delegate to
      // the real Node impl so the tool actually completes; the spy just
      // proves the seam.
      const tracked: WorkspaceFS = {
        readFile: jest.fn(nodeWorkspaceFS.readFile) as unknown as WorkspaceFS['readFile'],
        writeFile: jest.fn(nodeWorkspaceFS.writeFile),
        stat: jest.fn(nodeWorkspaceFS.stat),
        readdir: jest.fn(nodeWorkspaceFS.readdir) as unknown as WorkspaceFS['readdir'],
        mkdir: jest.fn(nodeWorkspaceFS.mkdir),
        realpath: jest.fn(nodeWorkspaceFS.realpath),
        unlink: jest.fn(nodeWorkspaceFS.unlink),
        open: jest.fn(nodeWorkspaceFS.open),
      };
      const bundle = createLocalCodingToolBundle({
        workspace: { root: workspace },
        exec: { fs: tracked },
      });
      const writeTool = bundle.tools.find((t) => t.name === 'write_file')!;
      await writeTool.invoke({
        id: 'c1',
        name: 'write_file',
        args: { file_path: 'note.md', content: 'hi\n' },
        type: 'tool_call',
      });
      expect(tracked.writeFile).toHaveBeenCalled();
      expect(tracked.mkdir).toHaveBeenCalled();

      const readTool = bundle.tools.find((t) => t.name === Constants.READ_FILE)!;
      await readTool.invoke({
        id: 'c2',
        name: Constants.READ_FILE,
        args: { file_path: 'note.md' },
        type: 'tool_call',
      });
      expect(tracked.stat).toHaveBeenCalled();
      expect(tracked.readFile).toHaveBeenCalled();
    });
  });
});
