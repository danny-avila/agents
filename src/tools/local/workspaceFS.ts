/**
 * Engine-agnostic filesystem seam for the local-coding tool suite.
 *
 * The current "local" engine maps every operation to Node's
 * `fs/promises` against the host machine. A future engine — e.g. a
 * stateful remote sandbox (e2b / Modal / Daytona / ssh-jail) —
 * supplies its own `WorkspaceFS` implementation and reuses every
 * tool factory unchanged. Same fuzzy-match `edit_file`, same
 * checkpointer, same syntax-check, same image attachments — only
 * the underlying I/O changes.
 *
 * Path semantics belong to the implementation. The local engine
 * interprets paths as host filesystem paths; a remote engine would
 * interpret them as remote-namespace paths. Tool factories don't
 * inspect the strings beyond passing them through.
 *
 * Keep this surface minimal. Add a method only when an existing
 * tool genuinely needs it; resist the temptation to mirror all of
 * `fs/promises`.
 */

import {
  mkdir as fsMkdir,
  open as fsOpen,
  readdir as fsReaddir,
  readFile as fsReadFile,
  realpath as fsRealpath,
  stat as fsStat,
  unlink as fsUnlink,
  writeFile as fsWriteFile,
} from 'fs/promises';
import type { MakeDirectoryOptions, Stats, WriteFileOptions } from 'fs';
import type { FileHandle } from 'fs/promises';

export type ReaddirEntry = {
  name: string;
  isFile(): boolean;
  isDirectory(): boolean;
  isSymbolicLink(): boolean;
};

export interface WorkspaceFS {
  readFile(path: string, encoding: 'utf8'): Promise<string>;
  readFile(path: string): Promise<Buffer>;
  writeFile(
    path: string,
    content: string | Buffer,
    options?: WriteFileOptions
  ): Promise<void>;
  stat(path: string): Promise<Stats>;
  readdir(
    path: string,
    options: { withFileTypes: true }
  ): Promise<ReaddirEntry[]>;
  readdir(path: string): Promise<string[]>;
  mkdir(path: string, options?: MakeDirectoryOptions): Promise<void>;
  realpath(path: string): Promise<string>;
  unlink(path: string): Promise<void>;
  /** Open a file for low-level read access (used by binary detection). */
  open(path: string, flags: 'r'): Promise<FileHandle>;
}

/**
 * Default `WorkspaceFS` backed by Node's `fs/promises` module.
 * Returned by `getWorkspaceFS(config)` when the host hasn't supplied
 * an override on `local.exec.fs`.
 */
export const nodeWorkspaceFS: WorkspaceFS = {
  // The runtime impl ignores the encoding-vs-buffer distinction; the
  // overload signatures above are what callers see.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readFile: ((path: string, encoding?: 'utf8') =>
    encoding != null
      ? fsReadFile(path, encoding)
      : fsReadFile(path)) as WorkspaceFS['readFile'],
  writeFile: (path, content, options) =>
    fsWriteFile(path, content, options ?? 'utf8'),
  stat: (path) => fsStat(path),
  readdir: ((path: string, options?: { withFileTypes: true }) =>
    options?.withFileTypes === true
      ? fsReaddir(path, { withFileTypes: true })
      : fsReaddir(path)) as WorkspaceFS['readdir'],
  mkdir: async (path, options) => {
    await fsMkdir(path, options);
  },
  realpath: (path) => fsRealpath(path),
  unlink: (path) => fsUnlink(path),
  open: (path, flags) => fsOpen(path, flags),
};
