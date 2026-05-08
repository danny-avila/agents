import { dirname } from 'path';
import { nodeWorkspaceFS } from './workspaceFS';
import type { WorkspaceFS } from './workspaceFS';
import type * as t from '@/types';

type Snapshot =
  | { kind: 'absent' }
  | { kind: 'present'; content: Buffer };

/**
 * Per-Run snapshot store for write_file / edit_file. Captures the
 * pre-write byte content of every path the local engine is about to
 * mutate so a later `rewind()` can restore the working tree to its
 * original state. Notes:
 *
 *  - Idempotent per path: subsequent captures preserve the first
 *    snapshot (so rewind always restores the *original* content).
 *  - Captures missing files as `{ kind: 'absent' }`; rewind deletes
 *    those paths so created files are removed.
 *  - In-memory: snapshots live for the lifetime of this instance and
 *    are not persisted across processes. Tie the lifetime to a Run.
 *  - Bounded by `maxBytesPerFile` (default 32 MiB) to bound memory.
 *    A file larger than the cap is recorded but not snapshotted; the
 *    rewind of that path is best-effort and the caller is told via
 *    the result count not to trust it.
 */
export class LocalFileCheckpointerImpl implements t.LocalFileCheckpointer {
  private snapshots = new Map<string, Snapshot>();
  private oversizePaths = new Set<string>();

  constructor(
    private readonly maxBytesPerFile: number = 32 * 1024 * 1024,
    private readonly fs: WorkspaceFS = nodeWorkspaceFS
  ) {}

  async captureBeforeWrite(absolutePath: string): Promise<void> {
    if (this.snapshots.has(absolutePath) || this.oversizePaths.has(absolutePath)) {
      return;
    }
    let info;
    try {
      info = await this.fs.stat(absolutePath);
    } catch {
      this.snapshots.set(absolutePath, { kind: 'absent' });
      return;
    }
    if (!info.isFile()) {
      return;
    }
    if (info.size > this.maxBytesPerFile) {
      this.oversizePaths.add(absolutePath);
      return;
    }
    const content = (await this.fs.readFile(absolutePath)) as Buffer;
    this.snapshots.set(absolutePath, { kind: 'present', content });
  }

  async rewind(): Promise<number> {
    let restored = 0;
    for (const [path, snapshot] of this.snapshots.entries()) {
      if (snapshot.kind === 'absent') {
        await this.fs.unlink(path).catch(() => undefined);
        restored++;
        continue;
      }
      try {
        await this.fs.mkdir(dirname(path), { recursive: true });
        await this.fs.writeFile(path, snapshot.content);
        restored++;
      } catch {
        // Best-effort: ignore individual restore failures so the rest
        // of the rewind continues.
      }
    }
    return restored;
  }

  capturedPaths(): string[] {
    return [...this.snapshots.keys(), ...this.oversizePaths];
  }
}

/**
 * Convenience factory so callers don't have to reach for the impl
 * class directly. Accepts an optional `WorkspaceFS` so a host using a
 * non-default engine (remote sandbox, in-memory test FS, etc.) can
 * route the checkpointer through the same I/O.
 */
export function createLocalFileCheckpointer(
  options: { maxBytesPerFile?: number; fs?: WorkspaceFS } = {}
): t.LocalFileCheckpointer {
  return new LocalFileCheckpointerImpl(options.maxBytesPerFile, options.fs);
}
