// src/utils/logging.ts
import fs from 'fs';
import util from 'util';

export function setupLogging(logFileName: string): void {
  const logFile = fs.createWriteStream(logFileName, { flags: 'a' });

  const originalConsoleLog = console.log;
  const originalConsoleError = console.error;
  const originalStdoutWrite = process.stdout.write;
  const originalStderrWrite = process.stderr.write;

  console.log = function (...args): void {
    logFile.write(util.format.apply(null, args) + ' ');
    originalConsoleLog.apply(console, args);
  };

  console.error = function (...args): void {
    logFile.write(util.format.apply(null, args) + ' ');
    originalConsoleError.apply(console, args);
  };

  process.stdout.write = function (
    buffer: Uint8Array | string,
    cb?: ((err?: Error | null) => void) | string,
    fd?: (err?: Error | null) => void
  ): boolean {
    if (typeof buffer === 'string') {
      logFile.write(buffer);
    } else {
      logFile.write(buffer.toString());
    }
    return originalStdoutWrite.call(
      process.stdout,
      buffer,
      cb as BufferEncoding | undefined,
      fd
    );
  };

  process.stderr.write = function (
    buffer: Uint8Array | string,
    cb?: ((err?: Error | null) => void) | string,
    fd?: (err?: Error | null) => void
  ): boolean {
    if (typeof buffer === 'string') {
      logFile.write(buffer);
    } else {
      logFile.write(buffer.toString());
    }
    return originalStderrWrite.call(
      process.stderr,
      buffer,
      cb as BufferEncoding | undefined,
      fd
    );
  };
}
