/**
 * False-positive guardrails for the bash command policy hard floor.
 *
 * Every test in this file is a command shape an agent legitimately
 * runs in real workloads. If any of these start blocking, we've
 * over-tightened the regex floor and broken real users. Treat any
 * regression here as a P0.
 *
 * What this file is NOT:
 *
 *   - A test of "things the floor catches" — that lives in
 *     `BashExecutor.validation.test.ts`.
 *   - A complete shell-parser conformance suite. The cases here are
 *     curated from realistic agent-loop patterns (git, npm, docker,
 *     test runners, build systems, dev-env init, documentation
 *     generation, etc.), not generated from a bash grammar.
 *
 * Known-accepted false positives (documented at the bottom of the
 * file) are tested as `expect(...).toBe(false)` so the floor's
 * defense-in-depth tradeoff is explicit. The codeapi-side
 * tree-sitter-bash AST validator (ClickHouse/ai#1619) is shipping
 * — once it's the authoritative `/exec`-time gate, these
 * assertions flip to `true` and the agents-lib regex floor relaxes
 * to the minimal fast-fail-UX set.
 */

import { describe, expect, it } from '@jest/globals';
import { validateBashCommandHardFloor } from '../local/bashValidation';

function shouldAllow(command: string, args?: string[]): void {
  const result = validateBashCommandHardFloor(command, args);
  if (!result.valid) {
    throw new Error(
      `FP regression: expected to ALLOW command but floor rejected it.\n  command: ${JSON.stringify(command)}\n  args:    ${JSON.stringify(args)}\n  reasons: ${result.errors.join('; ')}`
    );
  }
  expect(result.valid).toBe(true);
}

describe('FP guardrails — git workflows', () => {
  it('git status / diff / log / show', () => {
    shouldAllow('git status');
    shouldAllow('git status --short');
    shouldAllow('git diff HEAD~5');
    shouldAllow('git log --oneline -20');
    shouldAllow('git show HEAD');
    shouldAllow('git diff --cached');
  });

  it('git add / commit / push', () => {
    shouldAllow('git add .');
    shouldAllow('git add src/');
    shouldAllow('git commit -m "fix: redirect handling"');
    shouldAllow('git commit -am "wip"');
    shouldAllow('git push origin main');
    shouldAllow('git push --force-with-lease');
  });

  it('git branch / checkout / merge / rebase', () => {
    shouldAllow('git branch -a');
    shouldAllow('git checkout -b feature/new-thing');
    shouldAllow('git switch main');
    shouldAllow('git merge --no-ff feature-branch');
    shouldAllow('git rebase main');
    shouldAllow('git cherry-pick abc1234');
  });

  it('git stash / reset / clean (non-destructive shapes)', () => {
    shouldAllow('git stash push -m "wip"');
    shouldAllow('git stash pop');
    shouldAllow('git reset HEAD~1');
    shouldAllow('git reset --hard origin/main');
    shouldAllow('git clean -fd');
  });
});

describe('FP guardrails — package manager workflows', () => {
  it('npm', () => {
    shouldAllow('npm install');
    shouldAllow('npm install --save-dev typescript');
    shouldAllow('npm run build');
    shouldAllow('npm test');
    shouldAllow('npm test -- --coverage');
    shouldAllow('npm publish');
    shouldAllow('npm audit fix');
  });

  it('yarn', () => {
    shouldAllow('yarn install');
    shouldAllow('yarn add react');
    shouldAllow('yarn build');
    shouldAllow('yarn test --watch');
  });

  it('pnpm', () => {
    shouldAllow('pnpm install');
    shouldAllow('pnpm add -D vitest');
    shouldAllow('pnpm run dev');
  });

  it('python package managers', () => {
    shouldAllow('pip install requests');
    shouldAllow('pip install -e .');
    shouldAllow('pip install -r requirements.txt');
    shouldAllow('pip uninstall numpy');
    shouldAllow('uv pip install --upgrade pip');
    shouldAllow('poetry install');
    shouldAllow('poetry add fastapi');
  });

  it('cargo / go / other build tools', () => {
    shouldAllow('cargo build --release');
    shouldAllow('cargo test');
    shouldAllow('cargo run --bin server');
    shouldAllow('go build ./...');
    shouldAllow('go test -race ./...');
    shouldAllow('go mod tidy');
    shouldAllow('make build');
    shouldAllow('make -j4 install');
  });
});

describe('FP guardrails — docker / container workflows', () => {
  it('docker', () => {
    shouldAllow('docker build -t myapp:latest .');
    shouldAllow('docker run --rm -it ubuntu bash');
    shouldAllow('docker ps');
    shouldAllow('docker compose up');
    shouldAllow('docker compose -f docker-compose.yml down');
    shouldAllow('docker exec -it container_name sh');
  });

  it('kubectl', () => {
    shouldAllow('kubectl get pods');
    shouldAllow('kubectl apply -f manifest.yaml');
    shouldAllow('kubectl logs my-pod --tail=50');
    shouldAllow('kubectl exec -it my-pod -- /bin/bash');
  });
});

describe('FP guardrails — test runners', () => {
  it('jest / vitest / mocha', () => {
    shouldAllow('npx jest --watch');
    shouldAllow('jest --coverage --runInBand');
    shouldAllow('vitest run');
    shouldAllow('mocha test/**/*.spec.js');
  });

  it('pytest / unittest', () => {
    shouldAllow('pytest');
    shouldAllow('pytest -xvs tests/');
    shouldAllow('python -m pytest --cov=src');
    shouldAllow('python -m unittest discover');
  });

  it('go test / cargo test', () => {
    shouldAllow('go test -v ./...');
    shouldAllow('cargo test --release');
  });
});

describe('FP guardrails — common file/text utilities', () => {
  it('ls / cat / head / tail / less', () => {
    shouldAllow('ls -la');
    shouldAllow('ls -la /tmp');
    shouldAllow('cat README.md');
    shouldAllow('cat /etc/hosts');
    shouldAllow('head -100 logs/app.log');
    shouldAllow('tail -f /var/log/syslog');
  });

  it('grep / find / sed / awk', () => {
    shouldAllow('grep -r "TODO" src/');
    shouldAllow('grep -E "ERROR|FATAL" app.log');
    shouldAllow('find . -name "*.test.ts" -type f');
    shouldAllow('find /tmp -mtime +7 -delete');
    shouldAllow('sed -i \'s/foo/bar/g\' file.txt');
    shouldAllow('awk \'{print $2}\' data.csv');
  });

  it('sort / uniq / wc / cut / tr', () => {
    shouldAllow('sort -u names.txt');
    shouldAllow('uniq -c | sort -rn');
    shouldAllow('wc -l *.ts');
    shouldAllow('cut -d, -f2 data.csv');
    shouldAllow('tr "[:lower:]" "[:upper:]"');
  });

  it('curl / wget with normal URLs', () => {
    shouldAllow('curl https://api.example.com/users');
    shouldAllow('curl -X POST -d \'{"foo":"bar"}\' https://api.example.com');
    shouldAllow('wget https://example.com/file.tar.gz');
    shouldAllow('curl -fsSL https://install.example.com | bash');
  });
});

describe('FP guardrails — shell setup and env activation', () => {
  it('source bashrc / profile / activate scripts (Codex FP-audit fix)', () => {
    // The whole point of demoting source-from-variable from deny to
    // warn — these were tripping the floor before.
    shouldAllow('source $HOME/.bashrc');
    shouldAllow('source ~/.profile');
    shouldAllow('. $HOME/.bash_profile');
    shouldAllow('source $VIRTUAL_ENV/bin/activate');
    shouldAllow('source $NVM_DIR/nvm.sh');
    shouldAllow('source ${SCRIPT_DIR}/setup.sh');
    shouldAllow('. "${BASH_SOURCE[0]%/*}/lib/utils.sh"');
  });

  it('export / set / unset', () => {
    shouldAllow('export DEBUG=true');
    shouldAllow('export PATH=$HOME/bin:$PATH');
    shouldAllow('export NODE_OPTIONS="--max-old-space-size=4096"');
    shouldAllow('set -euo pipefail');
    shouldAllow('unset DEBUG');
  });
});

describe('FP guardrails — process / system inspection', () => {
  it('ps / top / kill', () => {
    shouldAllow('ps aux');
    shouldAllow('ps -ef | grep node');
    shouldAllow('top -bn1');
    shouldAllow('kill -9 12345');
    shouldAllow('killall -9 node');
  });

  it('df / du / free / lsblk (non-destructive disk inspection)', () => {
    shouldAllow('df -h');
    shouldAllow('du -sh node_modules');
    shouldAllow('du -h --max-depth=1 /var');
    shouldAllow('free -m');
    shouldAllow('lsblk');
  });

  it('non-environ /proc reads', () => {
    shouldAllow('cat /proc/cpuinfo');
    shouldAllow('cat /proc/meminfo');
    shouldAllow('cat /proc/version');
    shouldAllow('cat /proc/loadavg');
    shouldAllow('ls /proc/self/fd/');
    shouldAllow('cat /proc/self/status');
    shouldAllow('cat /proc/self/cmdline');
  });
});

describe('FP guardrails — bash language features', () => {
  it('${var#prefix} parameter expansion (Codex round-5 FP fix)', () => {
    shouldAllow('echo ${PATH#*:}');
    shouldAllow('echo ${file##*/}');
    shouldAllow('VAR=foo.txt; echo ${VAR%.txt}');
    shouldAllow('basename ${path}');
  });

  it('arithmetic $((...)) expansion (Codex round-10/13/14 FP fix)', () => {
    shouldAllow('echo $((1+1))');
    shouldAllow('echo $((2 * 3))');
    shouldAllow('for i in $(seq 1 $((10*5))); do echo $i; done');
    shouldAllow('result=$((a + b))');
  });

  it('mid-word `#` is literal (Codex round-10 FP fix)', () => {
    shouldAllow('echo "file#hash"');
    shouldAllow('echo foo#bar');
    shouldAllow('curl "https://example.com/page#section"');
  });

  it('backslash inside single quotes is literal (Codex round-11 FP fix)', () => {
    shouldAllow('echo \'C:\\Users\\foo\'');
    shouldAllow('echo \'abc\\\'');
  });

  it('command substitution in legitimate cases', () => {
    shouldAllow('echo "Today is $(date)"');
    shouldAllow('VERSION=$(cat VERSION)');
    shouldAllow('result=`uname -s`');
  });
});

describe('FP guardrails — rm / chmod / chown against non-protected paths', () => {
  it('rm -rf against non-protected paths', () => {
    shouldAllow('rm -rf node_modules');
    shouldAllow('rm -rf /tmp/build-artifacts');
    shouldAllow('rm -rf .next dist build');
    shouldAllow('rm -rf $HOME/.cache/pip');
    shouldAllow('rm -rf "$WORKSPACE/build"');
    shouldAllow('rm -rf ./old-data');
  });

  it('chmod against non-protected paths', () => {
    shouldAllow('chmod +x script.sh');
    shouldAllow('chmod -R u+w build/');
    shouldAllow('chmod 644 README.md');
    shouldAllow('chmod -R 755 ./bin');
  });

  it('chown against non-protected paths', () => {
    shouldAllow('chown user:group file.txt');
    shouldAllow('chown -R www-data:www-data /var/www/myapp');
  });

  it('non-protected dd usage', () => {
    shouldAllow('dd if=/tmp/source of=/tmp/dest bs=1M count=10');
    shouldAllow('dd if=/dev/urandom of=/tmp/random bs=1M count=1');
  });
});

describe('FP guardrails — print/echo of destructive-looking strings', () => {
  it('print rm commands in tutorials/warnings/docs', () => {
    shouldAllow('echo "Warning: never run rm -rf / on production"');
    shouldAllow('echo "Example: rm -rf /tmp/old to clean up"');
    shouldAllow('printf "Cleanup: rm -rf node_modules\\n"');
  });

  it('print fork bomb shape (educational)', () => {
    shouldAllow('echo "Classic fork bomb: :(){ :|:& };:"');
  });

  it('print disk-tampering examples', () => {
    shouldAllow('echo "Do NOT run mkfs.ext4 /dev/sda1 without backups"');
  });
});

describe('FP guardrails — shell comments', () => {
  it('comments mentioning destructive patterns are allowed (Codex round-2 FP fix)', () => {
    shouldAllow('echo ok # be careful with rm -rf /');
    shouldAllow('echo ok # mkfs.ext4 is destructive');
    shouldAllow(
      'ls -la\n# TODO: investigate /proc/self/environ usage\necho done'
    );
  });
});

describe('FP guardrails — multi-line scripts', () => {
  it('shebang scripts with multiple commands', () => {
    shouldAllow('#!/bin/bash\nset -e\necho starting\nmake build\necho done');
  });

  it('conditionals and loops', () => {
    shouldAllow('if [ -f config.json ]; then cat config.json; fi');
    shouldAllow('for f in *.ts; do echo "compiling $f"; done');
    shouldAllow('while read line; do echo "$line"; done < input.txt');
  });

  it('functions', () => {
    shouldAllow('greet() { echo "hello $1"; }; greet world');
  });
});

describe('FP guardrails — non-allowlist positional args', () => {
  it('args containing safe paths', () => {
    shouldAllow('cat "$1"', ['/tmp/safe-file']);
    shouldAllow('grep "pattern" "$1"', ['app.log']);
    shouldAllow('ls "$1"', ['$HOME/Documents']);
  });

  it('args containing flags', () => {
    shouldAllow('curl "$1"', ['-fsSL']);
    shouldAllow('grep "$1" "$2"', ['-E', 'pattern.*']);
  });
});

/**
 * KNOWN false-positives we explicitly accept as defense-in-depth.
 * These tests document the trade — they assert the floor REJECTS
 * the input even though bash would treat it as benign.
 *
 * These will flip to `allowed: true` after the codeapi-side
 * tree-sitter-bash AST validator (ClickHouse/ai#1619) ships and
 * becomes the authoritative `/exec`-time gate. When that
 * transition lands, the agents-lib regex floor relaxes here too;
 * each `expect(...).toBe(false)` below becomes `toBe(true)` and
 * the corresponding floor rule is removed.
 */
describe('FP guardrails — KNOWN-accepted false positives', () => {
  it('echo with /proc/<pid>/environ as the literal arg string', () => {
    // Bash would just print the string; we block because the AST
    // scan runs on the comment-stripped form and the regex is
    // command-shape-agnostic. Tree-sitter-bash distinguishes echo
    // (prints) from cat (reads). See codeapi-side fix.
    const result = validateBashCommandHardFloor(
      'echo "see /proc/self/environ"'
    );
    expect(result.valid).toBe(false);
  });

  it('two-arg command with split protected path', () => {
    // `cat "$1" "$2"` with args=['/proc/self', '/environ'] runs two
    // separate cat calls, neither matching the deny pattern. Our
    // args-joined-raw scan reassembles `/proc/self/environ` and
    // blocks. AST would see two separate argv slots.
    const result = validateBashCommandHardFloor('cat "$1" "$2"', [
      '/proc/self',
      '/environ',
    ]);
    expect(result.valid).toBe(false);
  });
});
