import { tool } from '@langchain/core/tools';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  formatCompletedResponse,
  normalizeToPythonIdentifier,
  ProgrammaticToolCallingDescription,
  ProgrammaticToolCallingName,
  ProgrammaticToolCallingSchema,
  filterToolsByUsage,
} from '@/tools/ProgrammaticToolCalling';
import {
  BashProgrammaticToolCallingDescription,
  BashProgrammaticToolCallingSchema,
  filterBashToolsByUsage,
  normalizeToBashIdentifier,
} from '@/tools/BashProgrammaticToolCalling';
import { Constants } from '@/common';
import {
  executeCloudflareCode,
  getCloudflareWorkspaceRoot,
} from './CloudflareSandboxExecutionEngine';

type ProgrammaticParams = {
  code: string;
  timeout?: number;
  lang?: string;
  runtime?: string;
  language?: string;
};

const DEFAULT_TIMEOUT = 60000;
const MIN_TIMEOUT = 1000;
const MAX_TIMEOUT = 300000;

type TimeoutSchema = {
  type: 'integer';
  minimum: number;
  maximum: number;
  default: number;
  description: string;
};

type CloudflareProgrammaticToolCallingJsonSchema = {
  type: 'object';
  properties: typeof ProgrammaticToolCallingSchema.properties & {
    timeout: TimeoutSchema;
    lang: {
      type: 'string';
      enum: readonly ['py', 'python', 'bash', 'sh'];
      default: 'bash';
      description: string;
    };
  };
  required: readonly ['code'];
};

type CloudflareBashProgrammaticToolCallingJsonSchema = {
  type: 'object';
  properties: typeof BashProgrammaticToolCallingSchema.properties & {
    timeout: TimeoutSchema;
  };
  required: readonly ['code'];
};

const NATIVE_TOOL_NAMES = new Set<string>([
  Constants.READ_FILE,
  Constants.WRITE_FILE,
  Constants.EDIT_FILE,
  Constants.GREP_SEARCH,
  Constants.GLOB_SEARCH,
  Constants.LIST_DIRECTORY,
  Constants.COMPILE_CHECK,
  Constants.BASH_TOOL,
  Constants.EXECUTE_CODE,
]);

function normalizeTimeout(timeoutMs: number | undefined): number {
  if (timeoutMs == null || !Number.isFinite(timeoutMs)) {
    return DEFAULT_TIMEOUT;
  }
  return Math.max(MIN_TIMEOUT, Math.floor(timeoutMs));
}

function formatTimeout(timeoutMs: number): string {
  return timeoutMs % 1000 === 0
    ? `${timeoutMs / 1000} seconds`
    : `${timeoutMs} milliseconds`;
}

function createTimeoutSchema(timeoutMs?: number): TimeoutSchema {
  const defaultTimeout = normalizeTimeout(timeoutMs);
  const maxTimeout = Math.max(MAX_TIMEOUT, defaultTimeout);
  return {
    type: 'integer',
    minimum: MIN_TIMEOUT,
    maximum: maxTimeout,
    default: defaultTimeout,
    description:
      'Maximum Cloudflare Sandbox execution time in milliseconds. ' +
      `Default: ${formatTimeout(defaultTimeout)}. Max: ${formatTimeout(maxTimeout)}.`,
  };
}

function createCloudflareProgrammaticToolCallingSchema(
  config: t.CloudflareSandboxExecutionConfig
): CloudflareProgrammaticToolCallingJsonSchema {
  return {
    ...ProgrammaticToolCallingSchema,
    properties: {
      ...ProgrammaticToolCallingSchema.properties,
      timeout: createTimeoutSchema(config.timeoutMs),
      lang: {
        type: 'string',
        enum: ['py', 'python', 'bash', 'sh'],
        default: 'bash',
        description:
          'Cloudflare Sandbox runtime for orchestration code. Defaults to bash; use py/python for Python orchestration.',
      },
    },
  } as const;
}

function createCloudflareBashProgrammaticToolCallingSchema(
  config: t.CloudflareSandboxExecutionConfig
): CloudflareBashProgrammaticToolCallingJsonSchema {
  return {
    ...BashProgrammaticToolCallingSchema,
    properties: {
      ...BashProgrammaticToolCallingSchema.properties,
      timeout: createTimeoutSchema(config.timeoutMs),
    },
  } as const;
}

function resolveRuntime(params: ProgrammaticParams): 'python' | 'bash' {
  const raw = params.lang ?? params.runtime ?? params.language ?? 'bash';
  return raw === 'py' || raw === 'python' ? 'python' : 'bash';
}

function filterNativeTools(
  toolDefs: t.LCTool[],
  code: string,
  runtime: 'python' | 'bash'
): t.LCTool[] {
  const nativeDefs = toolDefs.filter((def) => NATIVE_TOOL_NAMES.has(def.name));
  const filter =
    runtime === 'bash' ? filterBashToolsByUsage : filterToolsByUsage;
  return filter(nativeDefs, code);
}

function indent(code: string, spaces = 4): string {
  const prefix = ' '.repeat(spaces);
  return code
    .split('\n')
    .map((line) => (line === '' ? line : prefix + line))
    .join('\n');
}

function createPythonNativeToolSource(workspaceRoot: string): string {
  return `
import asyncio, fnmatch, glob, json, os, pathlib, re, shlex, shutil, subprocess, sys, tempfile

WORKSPACE = ${JSON.stringify(workspaceRoot)}

def _resolve(file_path="."):
    raw = file_path or "."
    candidate = raw if os.path.isabs(raw) else os.path.join(WORKSPACE, raw)
    resolved = os.path.abspath(candidate)
    root = os.path.abspath(WORKSPACE)
    if resolved != root and not resolved.startswith(root + os.sep):
        raise ValueError(f"Path is outside the Cloudflare sandbox workspace: {file_path}")
    return resolved

def _line_window(content, offset=None, limit=None):
    start = max((offset or 1) - 1, 0)
    lines = content.split("\\n")
    selected = lines[start:] if not limit or limit <= 0 else lines[start:start + limit]
    return "\\n".join(f"{start + idx + 1:6d}\\t{line}" for idx, line in enumerate(selected))

def _run(command, timeout=None, args=None):
    completed = subprocess.run(
        ["bash", "-lc", command, "--"] + [str(arg) for arg in (args or [])],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=(timeout / 1000 if timeout else None),
    )
    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
    }

def _format_run(result):
    text = ""
    if result.get("stdout"):
        text += f"stdout:\\n{result['stdout']}\\n"
    else:
        text += "stdout: Empty. Ensure you're writing output explicitly.\\n"
    if result.get("stderr"):
        text += f"stderr:\\n{result['stderr']}\\n"
    if result.get("exit_code") not in (None, 0):
        text += f"exit_code: {result['exit_code']}\\n"
    text += f"working_directory: {WORKSPACE}"
    return text.strip()

def _detect_compile_command():
    if os.path.exists(os.path.join(WORKSPACE, "tsconfig.json")):
        return "typescript", "npx --no-install tsc --noEmit", "tsconfig.json present"
    package_json = os.path.join(WORKSPACE, "package.json")
    if os.path.exists(package_json):
        try:
            if '"typescript"' in open(package_json, encoding="utf-8").read():
                return "typescript", "npx --no-install tsc --noEmit", "package.json declares typescript"
        except Exception:
            pass
    if os.path.exists(os.path.join(WORKSPACE, "Cargo.toml")):
        return "rust", "cargo check --message-format=short", "Cargo.toml present"
    if os.path.exists(os.path.join(WORKSPACE, "go.mod")):
        return "go", "go vet ./...", "go.mod present"
    if any(os.path.exists(os.path.join(WORKSPACE, name)) for name in ["pyproject.toml", "setup.py", "setup.cfg"]):
        return "python-compile", "python3 -m py_compile $(find . -name '*.py' -not -path './.venv/*' -not -path './node_modules/*')", "Python project"
    return "unknown", "", "no recognised project marker"

async def bash_tool(command, args=None):
    return _format_run(_run(command, args=args))

async def execute_code(lang, code, args=None):
    args = args or []
    temp_dir = tempfile.mkdtemp(prefix="lc-ptc-", dir=WORKSPACE)
    try:
        def q(value):
            import shlex
            return shlex.quote(str(value))
        arg_text = " ".join(q(arg) for arg in args)
        if lang in ("py", "python"):
            file_path = os.path.join(temp_dir, "main.py")
            open(file_path, "w", encoding="utf-8").write(code)
            return _format_run(_run(f"python3 {q(file_path)} {arg_text}"))
        if lang in ("js", "javascript"):
            file_path = os.path.join(temp_dir, "main.js")
            open(file_path, "w", encoding="utf-8").write(code)
            return _format_run(_run(f"node {q(file_path)} {arg_text}"))
        if lang in ("ts", "typescript"):
            file_path = os.path.join(temp_dir, "main.ts")
            open(file_path, "w", encoding="utf-8").write(code)
            return _format_run(_run(f"npx --no-install tsx {q(file_path)} {arg_text}"))
        if lang == "php":
            file_path = os.path.join(temp_dir, "main.php")
            open(file_path, "w", encoding="utf-8").write(code)
            return _format_run(_run(f"php {q(file_path)} {arg_text}"))
        if lang == "go":
            file_path = os.path.join(temp_dir, "main.go")
            open(file_path, "w", encoding="utf-8").write(code)
            return _format_run(_run(f"go run {q(file_path)} {arg_text}"))
        if lang == "rs":
            file_path = os.path.join(temp_dir, "main.rs")
            open(file_path, "w", encoding="utf-8").write(code)
            binary = os.path.join(temp_dir, "main-rs")
            return _format_run(_run(f"rustc {q(file_path)} -o {q(binary)} && {q(binary)} {arg_text}"))
        if lang == "c":
            file_path = os.path.join(temp_dir, "main.c")
            open(file_path, "w", encoding="utf-8").write(code)
            binary = os.path.join(temp_dir, "main-c")
            return _format_run(_run(f"cc {q(file_path)} -o {q(binary)} && {q(binary)} {arg_text}"))
        if lang == "cpp":
            file_path = os.path.join(temp_dir, "main.cpp")
            open(file_path, "w", encoding="utf-8").write(code)
            binary = os.path.join(temp_dir, "main-cpp")
            return _format_run(_run(f"c++ {q(file_path)} -o {q(binary)} && {q(binary)} {arg_text}"))
        if lang == "java":
            file_path = os.path.join(temp_dir, "Main.java")
            open(file_path, "w", encoding="utf-8").write(code)
            return _format_run(_run(f"javac {q(file_path)} && java -cp {q(temp_dir)} Main {arg_text}"))
        if lang == "r":
            file_path = os.path.join(temp_dir, "main.R")
            open(file_path, "w", encoding="utf-8").write(code)
            return _format_run(_run(f"Rscript {q(file_path)} {arg_text}"))
        if lang == "d":
            file_path = os.path.join(temp_dir, "main.d")
            open(file_path, "w", encoding="utf-8").write(code)
            binary = os.path.join(temp_dir, "main-d")
            return _format_run(_run(f"dmd {q(file_path)} -of={q(binary)} && {q(binary)} {arg_text}"))
        if lang == "f90":
            file_path = os.path.join(temp_dir, "main.f90")
            open(file_path, "w", encoding="utf-8").write(code)
            binary = os.path.join(temp_dir, "main-f90")
            return _format_run(_run(f"gfortran {q(file_path)} -o {q(binary)} && {q(binary)} {arg_text}"))
        if lang in ("bash", "sh"):
            return _format_run(_run(code, args=args))
        raise ValueError(f"Unsupported Cloudflare sandbox runtime: {lang}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def read_file(file_path, offset=None, limit=None):
    resolved = _resolve(file_path)
    with open(resolved, encoding="utf-8") as handle:
        return _line_window(handle.read(), offset, limit)

async def write_file(file_path, content):
    resolved = _resolve(file_path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    existed = os.path.exists(resolved)
    with open(resolved, "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"{'Overwrote' if existed else 'Created'} {resolved} ({len(content)} chars)."

async def edit_file(file_path, old_text=None, new_text=None, edits=None):
    resolved = _resolve(file_path)
    edits = edits or [{"old_text": old_text, "new_text": new_text}]
    content = open(resolved, encoding="utf-8").read()
    for edit in edits:
        old = edit.get("old_text") or ""
        new = edit.get("new_text") or ""
        if content.count(old) != 1:
            raise ValueError(f"Could not locate old_text exactly once in {file_path}")
        content = content.replace(old, new, 1)
    open(resolved, "w", encoding="utf-8").write(content)
    return f"Applied {len(edits)} edit(s) to {resolved}."

async def list_directory(path="."):
    resolved = _resolve(path)
    entries = []
    for name in sorted(os.listdir(resolved)):
        full = os.path.join(resolved, name)
        entries.append(("dir " if os.path.isdir(full) else "file") + "\\t" + name)
    return "\\n".join(entries) or "Directory is empty."

async def grep_search(pattern, path=".", glob=None, max_results=200):
    root = _resolve(path)
    regex = re.compile(pattern)
    out = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "dist", "build"}]
        for name in files:
            rel = os.path.relpath(os.path.join(current, name), root)
            if glob and not fnmatch.fnmatch(rel, glob):
                continue
            try:
                for line_no, line in enumerate(open(os.path.join(current, name), encoding="utf-8", errors="ignore"), 1):
                    if regex.search(line):
                        out.append(f"{os.path.join(current, name)}:{line_no}:{line.rstrip()}")
                        if len(out) >= max_results:
                            return "\\n".join(out)
            except Exception:
                pass
    return "\\n".join(out) if out else "No matches found."

async def glob_search(pattern, path=".", max_results=200):
    root = _resolve(path)
    matches = glob_module.glob(os.path.join(root, pattern), recursive=True)[:max_results]
    return "\\n".join(matches) if matches else "No files found."

async def compile_check(command=None, timeout_ms=None):
    kind, detected, reason = _detect_compile_command()
    command = command or detected
    if not command:
        return f"compile_check: {reason}. Pass an explicit command to override."
    result = _run(command, timeout_ms)
    status = "PASSED" if result["exit_code"] == 0 else "FAILED"
    return f"compile_check ({kind}) {status} via {command}\\n\\nstdout:\\n{result['stdout']}\\nstderr:\\n{result['stderr']}\\nworking_directory: {WORKSPACE}\\nreason: {reason}"

# Avoid shadowing the glob_search function argument named "glob".
glob_module = glob
`.trim();
}

function createNodeNativeToolSource(workspaceRoot: string): string {
  return `
const fs = require("fs");
const fsp = fs.promises;
const path = require("path");
const cp = require("child_process");

const WORKSPACE = ${JSON.stringify(workspaceRoot)};

function resolvePath(filePath) {
  const raw = filePath || ".";
  const candidate = path.isAbsolute(raw) ? raw : path.join(WORKSPACE, raw);
  const resolved = path.resolve(candidate);
  const root = path.resolve(WORKSPACE);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    throw new Error("Path is outside the Cloudflare sandbox workspace: " + filePath);
  }
  return resolved;
}

function lineWindow(content, offset, limit) {
  const start = Math.max((offset || 1) - 1, 0);
  const lines = content.split("\\n");
  const selected = !limit || limit <= 0 ? lines.slice(start) : lines.slice(start, start + limit);
  return selected.map((line, index) => String(start + index + 1).padStart(6, " ") + "\\t" + line).join("\\n");
}

function quote(value) {
  const text = String(value);
  if (text === "") return "''";
  if (/^[A-Za-z0-9_/:=.,@%+-]+$/.test(text)) return text;
  return "'" + text.replace(/'/g, "'\\\\''") + "'";
}

function run(command, timeoutMs, args) {
  return new Promise((resolve) => {
    const child = cp.spawn("bash", ["-lc", command, "--", ...((args || []).map(String))], {
      cwd: WORKSPACE,
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = timeoutMs
      ? setTimeout(() => {
        timedOut = true;
        child.kill("SIGTERM");
      }, timeoutMs)
      : null;

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      if (timer) clearTimeout(timer);
      resolve({ stdout, stderr: stderr + error.message, exit_code: 1, timed_out: timedOut });
    });
    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      resolve({ stdout, stderr, exit_code: timedOut ? null : code, timed_out: timedOut });
    });
  });
}

function formatRun(result) {
  let text = "";
  if (result.stdout) {
    text += "stdout:\\n" + result.stdout + "\\n";
  } else {
    text += "stdout: Empty. Ensure you're writing output explicitly.\\n";
  }
  if (result.stderr) {
    text += "stderr:\\n" + result.stderr + "\\n";
  }
  if (result.timed_out) {
    text += "timed_out: true\\n";
  }
  if (result.exit_code !== null && result.exit_code !== undefined && result.exit_code !== 0) {
    text += "exit_code: " + result.exit_code + "\\n";
  }
  text += "working_directory: " + WORKSPACE;
  return text.trim();
}

async function detectCompileCommand() {
  async function exists(name) {
    try {
      await fsp.access(path.join(WORKSPACE, name));
      return true;
    } catch {
      return false;
    }
  }
  if (await exists("tsconfig.json")) {
    return ["typescript", "npx --no-install tsc --noEmit", "tsconfig.json present"];
  }
  if (await exists("package.json")) {
    try {
      if ((await fsp.readFile(path.join(WORKSPACE, "package.json"), "utf8")).includes('"typescript"')) {
        return ["typescript", "npx --no-install tsc --noEmit", "package.json declares typescript"];
      }
    } catch {}
  }
  if (await exists("Cargo.toml")) return ["rust", "cargo check --message-format=short", "Cargo.toml present"];
  if (await exists("go.mod")) return ["go", "go vet ./...", "go.mod present"];
  if (await exists("pyproject.toml") || await exists("setup.py") || await exists("setup.cfg")) {
    return ["python-compile", "python3 -m py_compile $(find . -name '*.py' -not -path './.venv/*' -not -path './node_modules/*')", "Python project"];
  }
  return ["unknown", "", "no recognised project marker"];
}

function globToRegExp(pattern) {
  const escaped = pattern.replace(/[|\\\\{}()[\\]^$+*?.]/g, "\\\\$&");
  return new RegExp("^" + escaped.replace(/\\\\\\*\\\\\\*/g, ".*").replace(/\\\\\\*/g, "[^/]*") + "$");
}

function globMatch(relativePath, pattern) {
  const matcher = globToRegExp(pattern);
  return matcher.test(relativePath) || matcher.test(path.basename(relativePath));
}

async function walkFiles(root, visit) {
  const entries = await fsp.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) {
      if ([".git", "node_modules", ".venv", "dist", "build"].includes(entry.name)) continue;
      await walkFiles(full, visit);
    } else if (entry.isFile()) {
      await visit(full);
    }
  }
}

async function bash_tool(payload) {
  return formatRun(await run(payload.command, undefined, payload.args));
}

async function execute_code(payload) {
  const lang = payload.lang;
  const code = payload.code;
  const args = payload.args || [];
  const tempDir = await fsp.mkdtemp(path.join(WORKSPACE, "lc-ptc-"));
  try {
    const argText = args.map(quote).join(" ");
    async function writeAndRun(fileName, command) {
      const filePath = path.join(tempDir, fileName);
      await fsp.writeFile(filePath, code, "utf8");
      return formatRun(await run(command(filePath, argText), undefined, []));
    }
    if (lang === "py" || lang === "python") {
      return writeAndRun("main.py", (filePath, argText) => "python3 " + quote(filePath) + " " + argText);
    }
    if (lang === "js" || lang === "javascript") {
      return writeAndRun("main.js", (filePath, argText) => "node " + quote(filePath) + " " + argText);
    }
    if (lang === "ts" || lang === "typescript") {
      return writeAndRun("main.ts", (filePath, argText) => "npx --no-install tsx " + quote(filePath) + " " + argText);
    }
    if (lang === "php") {
      return writeAndRun("main.php", (filePath, argText) => "php " + quote(filePath) + " " + argText);
    }
    if (lang === "go") {
      return writeAndRun("main.go", (filePath, argText) => "go run " + quote(filePath) + " " + argText);
    }
    if (lang === "rs") {
      return writeAndRun("main.rs", (filePath, argText) => {
        const binary = path.join(tempDir, "main-rs");
        return "rustc " + quote(filePath) + " -o " + quote(binary) + " && " + quote(binary) + " " + argText;
      });
    }
    if (lang === "c") {
      return writeAndRun("main.c", (filePath, argText) => {
        const binary = path.join(tempDir, "main-c");
        return "cc " + quote(filePath) + " -o " + quote(binary) + " && " + quote(binary) + " " + argText;
      });
    }
    if (lang === "cpp") {
      return writeAndRun("main.cpp", (filePath, argText) => {
        const binary = path.join(tempDir, "main-cpp");
        return "c++ " + quote(filePath) + " -o " + quote(binary) + " && " + quote(binary) + " " + argText;
      });
    }
    if (lang === "java") {
      return writeAndRun("Main.java", (filePath, argText) => "javac " + quote(filePath) + " && java -cp " + quote(tempDir) + " Main " + argText);
    }
    if (lang === "r") {
      return writeAndRun("main.R", (filePath, argText) => "Rscript " + quote(filePath) + " " + argText);
    }
    if (lang === "d") {
      return writeAndRun("main.d", (filePath, argText) => {
        const binary = path.join(tempDir, "main-d");
        return "dmd " + quote(filePath) + " -of=" + quote(binary) + " && " + quote(binary) + " " + argText;
      });
    }
    if (lang === "f90") {
      return writeAndRun("main.f90", (filePath, argText) => {
        const binary = path.join(tempDir, "main-f90");
        return "gfortran " + quote(filePath) + " -o " + quote(binary) + " && " + quote(binary) + " " + argText;
      });
    }
    if (lang === "bash" || lang === "sh") {
      return formatRun(await run(code, undefined, args));
    }
    throw new Error("Unsupported Cloudflare sandbox runtime: " + lang);
  } finally {
    await fsp.rm(tempDir, { recursive: true, force: true });
  }
}

async function read_file(payload) {
  const resolved = resolvePath(payload.file_path);
  return lineWindow(await fsp.readFile(resolved, "utf8"), payload.offset, payload.limit);
}

async function write_file(payload) {
  const resolved = resolvePath(payload.file_path);
  await fsp.mkdir(path.dirname(resolved), { recursive: true });
  const existed = fs.existsSync(resolved);
  await fsp.writeFile(resolved, payload.content, "utf8");
  return (existed ? "Overwrote " : "Created ") + resolved + " (" + payload.content.length + " chars).";
}

async function edit_file(payload) {
  const resolved = resolvePath(payload.file_path);
  const edits = payload.edits || [{ old_text: payload.old_text, new_text: payload.new_text }];
  let content = await fsp.readFile(resolved, "utf8");
  for (const edit of edits) {
    const oldText = edit.old_text || "";
    const newText = edit.new_text || "";
    if (oldText === "" || content.split(oldText).length - 1 !== 1) {
      throw new Error("Could not locate old_text exactly once in " + payload.file_path);
    }
    content = content.replace(oldText, newText);
  }
  await fsp.writeFile(resolved, content, "utf8");
  return "Applied " + edits.length + " edit(s) to " + resolved + ".";
}

async function list_directory(payload) {
  const resolved = resolvePath(payload.path || ".");
  const entries = await fsp.readdir(resolved, { withFileTypes: true });
  const lines = entries
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((entry) => (entry.isDirectory() ? "dir" : "file") + "\\t" + entry.name);
  return lines.join("\\n") || "Directory is empty.";
}

async function grep_search(payload) {
  const root = resolvePath(payload.path || ".");
  const regex = new RegExp(payload.pattern);
  const maxResults = payload.max_results || 200;
  const out = [];
  await walkFiles(root, async (filePath) => {
    if (out.length >= maxResults) return;
    const relative = path.relative(root, filePath);
    if (payload.glob && !globMatch(relative, payload.glob)) return;
    let text = "";
    try {
      text = await fsp.readFile(filePath, "utf8");
    } catch {
      return;
    }
    text.split("\\n").forEach((line, index) => {
      if (out.length < maxResults && regex.test(line)) {
        out.push(filePath + ":" + (index + 1) + ":" + line);
      }
    });
  });
  return out.join("\\n") || "No matches found.";
}

async function glob_search(payload) {
  const root = resolvePath(payload.path || ".");
  const maxResults = payload.max_results || 200;
  const out = [];
  await walkFiles(root, async (filePath) => {
    if (out.length >= maxResults) return;
    const relative = path.relative(root, filePath);
    if (globMatch(relative, payload.pattern)) out.push(filePath);
  });
  return out.join("\\n") || "No files found.";
}

async function compile_check(payload) {
  const [kind, detected, reason] = await detectCompileCommand();
  const command = payload.command || detected;
  if (!command) {
    return "compile_check: " + reason + ". Pass an explicit command to override.";
  }
  const result = await run(command, payload.timeout_ms);
  const status = result.exit_code === 0 ? "PASSED" : "FAILED";
  return "compile_check (" + kind + ") " + status + " via " + command + "\\n\\nstdout:\\n" + result.stdout + "\\nstderr:\\n" + result.stderr + "\\nworking_directory: " + WORKSPACE + "\\nreason: " + reason;
}

const TOOLS = {
  bash_tool,
  execute_code,
  read_file,
  write_file,
  edit_file,
  list_directory,
  grep_search,
  glob_search,
  compile_check,
};

async function main() {
  const name = process.argv[2];
  const payload = JSON.parse(process.argv[3] || "{}");
  if (!TOOLS[name]) throw new Error("Unknown tool: " + name);
  const result = await TOOLS[name](payload);
  process.stdout.write(typeof result === "string" ? result : JSON.stringify(result));
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
`.trim();
}

function createPythonProgram(
  userCode: string,
  toolDefs: t.LCTool[],
  workspaceRoot: string
): string {
  const aliases = toolDefs
    .map((def) => {
      const pythonName = normalizeToPythonIdentifier(def.name);
      return NATIVE_TOOL_NAMES.has(def.name) && pythonName !== def.name
        ? `${pythonName} = globals()[${JSON.stringify(def.name)}]`
        : '';
    })
    .filter(Boolean)
    .join('\n');
  return `${createPythonNativeToolSource(workspaceRoot)}
${aliases}

async def __lc_user_main__():
${indent(userCode)}

asyncio.run(__lc_user_main__())
`;
}

function createBashProgram(
  userCode: string,
  toolDefs: t.LCTool[],
  workspaceRoot: string
): string {
  const helper = createNodeNativeToolSource(workspaceRoot);
  const functions = toolDefs
    .map((def) => {
      const bashName = normalizeToBashIdentifier(def.name);
      if (!NATIVE_TOOL_NAMES.has(def.name)) {
        return '';
      }
      return `${bashName}() { node "$__LC_TOOL_HELPER" ${JSON.stringify(def.name)} "$1"; }`;
    })
    .filter(Boolean)
    .join('\n');
  return `
set -euo pipefail
command -v node >/dev/null 2>&1 || { echo "Cloudflare programmatic tool calling requires node in the sandbox image." >&2; exit 127; }
__LC_TOOL_HELPER="$(mktemp /tmp/lc-tools.XXXXXX.js)"
cat > "$__LC_TOOL_HELPER" <<'JS'
${helper}
JS
trap 'rm -f "$__LC_TOOL_HELPER"' EXIT
${functions}
${userCode}
`.trim();
}

async function runProgrammatic(args: {
  params: ProgrammaticParams;
  config?: { toolCall?: unknown };
  cloudflareConfig: t.CloudflareSandboxExecutionConfig;
  runtime: 'python' | 'bash';
}): Promise<[string, t.ProgrammaticExecutionArtifact]> {
  const toolCall = (args.config?.toolCall ??
    {}) as Partial<t.ProgrammaticCache>;
  const toolDefs = toolCall.toolDefs ?? [];
  const effectiveTools = filterNativeTools(
    toolDefs,
    args.params.code,
    args.runtime
  );
  const timeoutMs =
    args.params.timeout ?? args.cloudflareConfig.timeoutMs ?? DEFAULT_TIMEOUT;
  const workspaceRoot = getCloudflareWorkspaceRoot(args.cloudflareConfig);
  const result =
    args.runtime === 'bash'
      ? await executeCloudflareCode(
        {
          lang: 'bash',
          code: createBashProgram(
            args.params.code,
            effectiveTools,
            workspaceRoot
          ),
        },
        { ...args.cloudflareConfig, timeoutMs }
      )
      : await executeCloudflareCode(
        {
          lang: 'py',
          code: createPythonProgram(
            args.params.code,
            effectiveTools,
            workspaceRoot
          ),
        },
        { ...args.cloudflareConfig, timeoutMs }
      );

  if (result.exitCode !== 0 || result.timedOut) {
    throw new Error(
      result.stderr !== ''
        ? result.stderr
        : `Cloudflare ${args.runtime} programmatic execution exited with code ${result.exitCode ?? 'unknown'}`
    );
  }

  return formatCompletedResponse({
    status: 'completed',
    session_id: 'cloudflare-sandbox',
    stdout: result.stdout,
    stderr: result.stderr,
    files: [],
  });
}

export function createCloudflareProgrammaticToolCallingTool(
  cloudflareConfig: t.CloudflareSandboxExecutionConfig
): DynamicStructuredTool {
  return tool(
    async (rawParams, config) => {
      const params = rawParams as ProgrammaticParams;
      return runProgrammatic({
        params,
        config,
        cloudflareConfig,
        runtime: resolveRuntime(params),
      });
    },
    {
      name: ProgrammaticToolCallingName,
      description: `${ProgrammaticToolCallingDescription}\n\nCloudflare Sandbox engine: exposes the built-in coding tools inside the sandbox process. Non-coding host tools are not callable from this in-sandbox programmatic runner.`,
      schema: createCloudflareProgrammaticToolCallingSchema(cloudflareConfig),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createCloudflareBashProgrammaticToolCallingTool(
  cloudflareConfig: t.CloudflareSandboxExecutionConfig
): DynamicStructuredTool {
  return tool(
    async (rawParams, config) => {
      const params = rawParams as ProgrammaticParams;
      return runProgrammatic({
        params,
        config,
        cloudflareConfig,
        runtime: 'bash',
      });
    },
    {
      name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      description: `${BashProgrammaticToolCallingDescription}\n\nCloudflare Sandbox engine: exposes the built-in coding tools as bash functions inside the sandbox process. Non-coding host tools are not callable from this in-sandbox programmatic runner.`,
      schema:
        createCloudflareBashProgrammaticToolCallingSchema(cloudflareConfig),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
