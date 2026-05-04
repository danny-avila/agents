import { readFile, rm, writeFile, stat, mkdir } from 'fs/promises';
import { dirname } from 'path';
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

  constructor(private readonly maxBytesPerFile: number = 32 * 1024 * 1024) {}

  async captureBeforeWrite(absolutePath: string): Promise<void> {
    if (this.snapshots.has(absolutePath) || this.oversizePaths.has(absolutePath)) {
      return;
    }
    let info;
    try {
      info = await stat(absolutePath);
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
    const content = await readFile(absolutePath);
    this.snapshots.set(absolutePath, { kind: 'present', content });
  }

  async rewind(): Promise<number> {
    let restored = 0;
    for (const [path, snapshot] of this.snapshots.entries()) {
      if (snapshot.kind === 'absent') {
        await rm(path, { force: true });
        restored++;
        continue;
      }
      try {
        await mkdir(dirname(path), { recursive: true });
        await writeFile(path, snapshot.content);
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
 * class directly.
 */
export function createLocalFileCheckpointer(
  options: { maxBytesPerFile?: number } = {}
): t.LocalFileCheckpointer {
  return new LocalFileCheckpointerImpl(options.maxBytesPerFile);
}
