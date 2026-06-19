import { tmpdir } from 'os';
import { join } from 'path';
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'fs/promises';
import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { LocalFileCheckpointerImpl } from '../FileCheckpointer';
import {
  WorkspaceClientTimeoutError,
  type WorkspaceFS,
} from '../workspaceFS';

/**
 * Pins the LocalFileCheckpointer's per-Run snapshot/rewind contract.
 * Critical because checkpoints are what `--rollback` style features
 * (and the local engine's mid-batch undo) actually rely on — a
 * regression here silently makes rewind a no-op or, worse, restores
 * to the wrong byte content.
 */

describe('LocalFileCheckpointerImpl', () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'lc-fcp-'));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('captures pre-write content and rewind restores it byte-exact', async () => {
    const file = join(dir, 'a.txt');
    await writeFile(file, 'original\n');
    const cp = new LocalFileCheckpointerImpl();

    await cp.captureBeforeWrite(file);
    await writeFile(file, 'overwritten\n');

    const restored = await cp.rewind();
    expect(restored).toBe(1);
    expect(await readFile(file, 'utf8')).toBe('original\n');
  });

  it('captureBeforeWrite is idempotent — second capture preserves the FIRST snapshot', async () => {
    const file = join(dir, 'b.txt');
    await writeFile(file, 'first');
    const cp = new LocalFileCheckpointerImpl();

    await cp.captureBeforeWrite(file);
    // Simulate first write happening between captures, then a second
    // tool wanting to write the same file. The second capture must be
    // a no-op so rewind restores back to the very first content.
    await writeFile(file, 'between');
    await cp.captureBeforeWrite(file);
    await writeFile(file, 'last');

    await cp.rewind();
    expect(await readFile(file, 'utf8')).toBe('first');
  });

  it('captures absent files and rewind deletes any newly-created path', async () => {
    const file = join(dir, 'newly-created.txt');
    const cp = new LocalFileCheckpointerImpl();

    await cp.captureBeforeWrite(file); // file does not exist yet
    await writeFile(file, 'created by some tool');

    const restored = await cp.rewind();
    expect(restored).toBe(1);
    await expect(stat(file)).rejects.toThrow();
  });

  it('rethrows a client-timeout from stat instead of snapshotting as absent', async () => {
    // A stalled stat() must NOT be recorded as "absent" — that would delete an
    // existing file on rewind. The timeout has to surface.
    const fs = {
      stat: async () => {
        throw new WorkspaceClientTimeoutError(
          'cloudflare sandbox listFiles exceeded 6000ms client-side timeout'
        );
      },
    } as unknown as WorkspaceFS;
    const cp = new LocalFileCheckpointerImpl(32 * 1024 * 1024, fs);

    await expect(cp.captureBeforeWrite('/workspace/a.txt')).rejects.toThrow(
      /client-side timeout/
    );
    // Nothing was snapshotted, so rewind is a no-op (no spurious deletion).
    expect(await cp.rewind()).toBe(0);
  });

  it('rewind rethrows a client-timeout from a restore write instead of claiming success', async () => {
    // A stalled restore write leaves the bad bytes on disk — rewind must surface
    // the timeout, not silently count the path as restored.
    const fs = {
      stat: async () => ({ isFile: () => true, size: 5 }),
      readFile: async () => Buffer.from('orig\n'),
      mkdir: async () => undefined,
      writeFile: async () => {
        throw new WorkspaceClientTimeoutError(
          'cloudflare sandbox writeFile exceeded 6000ms client-side timeout'
        );
      },
    } as unknown as WorkspaceFS;
    const cp = new LocalFileCheckpointerImpl(32 * 1024 * 1024, fs);

    await cp.captureBeforeWrite('/workspace/a.txt'); // snapshots 'present'
    await expect(cp.rewind()).rejects.toThrow(/client-side timeout/);
  });

  it('rewinds across multiple files in a single pass', async () => {
    const a = join(dir, 'multi-a.txt');
    const b = join(dir, 'multi-b.txt');
    await writeFile(a, 'A0');
    await writeFile(b, 'B0');
    const cp = new LocalFileCheckpointerImpl();

    await cp.captureBeforeWrite(a);
    await cp.captureBeforeWrite(b);
    await writeFile(a, 'A1');
    await writeFile(b, 'B1');

    const restored = await cp.rewind();
    expect(restored).toBe(2);
    expect(await readFile(a, 'utf8')).toBe('A0');
    expect(await readFile(b, 'utf8')).toBe('B0');
  });

  it('skips snapshotting files larger than maxBytesPerFile but still tracks them', async () => {
    const file = join(dir, 'big.bin');
    // Write 1024 bytes, set the cap to 100 — well under file size.
    await writeFile(file, Buffer.alloc(1024, 0x41));
    const cp = new LocalFileCheckpointerImpl(100);

    await cp.captureBeforeWrite(file);
    expect(cp.capturedPaths()).toContain(file);

    // Mutate the file. Rewind reports 0 restored (nothing snapshotted)
    // but does not throw — best-effort behavior documented in the
    // class JSDoc.
    await writeFile(file, 'mutated');
    const restored = await cp.rewind();
    expect(restored).toBe(0);
    // The file is unchanged from the post-mutation state — there was
    // nothing snapshotted to restore.
    expect(await readFile(file, 'utf8')).toBe('mutated');
  });

  it('rewind of a captured file whose parent directory was removed recreates the directory', async () => {
    const subdir = join(dir, 'nested', 'deep');
    const file = join(subdir, 'x.txt');
    await mkdir(subdir, { recursive: true });
    await writeFile(file, 'kept');
    const cp = new LocalFileCheckpointerImpl();

    await cp.captureBeforeWrite(file);
    // Blow away the subdir entirely — simulates a tool that deleted
    // the parent directory.
    await rm(join(dir, 'nested'), { recursive: true, force: true });

    const restored = await cp.rewind();
    expect(restored).toBe(1);
    expect(await readFile(file, 'utf8')).toBe('kept');
  });
});
