import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { cp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, join, relative, sep } from 'node:path';

export const DM_SYSTEM = 'Fix the issue using workspace_read, workspace_write and workspace_test. Only the listed existing files are accessible. Keep a short plan in your response before editing. workspace_test runs only the public tests. Hidden verification runs after you finish. Respect the task changed-file constraints. Finish with a brief summary.';
// shortcut: pinned tasks edit existing setup files; add bounded creation if a future task needs new files.
const IGNORED = new Set(['.git', '.pytest_cache', '__pycache__', '.mypy_cache', '.ruff_cache']);
const COMMAND = ['{python}', '-m', 'pytest', '-q'];

function runTests(directory, python, command, config) {
  if (JSON.stringify(command) !== JSON.stringify(COMMAND)) throw new Error('Unsupported frozen DM verifier command');
  const result = spawnSync(python, [...command.slice(1), ...(config ? ['-c', config] : [])], {
    cwd: directory, windowsHide: true, encoding: 'utf8', maxBuffer: 1024 * 1024,
    env: Object.fromEntries(Object.entries({
      PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: process.env.TEMP, TMP: process.env.TMP,
      PYTHONPATH: directory, PYTHONDONTWRITEBYTECODE: '1', PYTEST_DISABLE_PLUGIN_AUTOLOAD: '1',
      PYTEST_ADDOPTS: '-p no:cacheprovider',
    }).filter(([, value]) => value !== undefined)),
  });
  return { status: result.status, error: result.error?.code ?? result.signal,
    log: (result.stdout ?? '') + (result.stderr ?? '') };
}

export function createDmTestTool(task, workspace, python) {
  return {
    name: 'workspace_test', label: 'workspace_test', description: 'Run the frozen public pytest command in this workspace. Hidden tests are not present yet.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    execute: async () => {
      const result = runTests(workspace, python, task.dm.visible_test_command);
      const text = result.log.length > 10000 ? result.log.slice(0, 5000) + '\n... [truncated] ...\n' + result.log.slice(-5000) : result.log;
      return { content: [{ type: 'text', text: `exit=${result.status}${result.error ? ` error=${result.error}` : ''}\n${text}` }], details: {} };
    },
  };
}

export async function snapshotDm(root) {
  const files = {};
  const symlinks = [];
  async function walk(prefix = '') {
    for (const item of await readdir(join(root, prefix), { withFileTypes: true })) {
      const name = prefix ? `${prefix}/${item.name}` : item.name;
      if (item.isSymbolicLink()) { symlinks.push(name); continue; }
      if (IGNORED.has(item.name)) continue;
      if (item.isDirectory()) await walk(name);
      else files[name] = createHash('sha256').update(await readFile(join(root, name))).digest('hex');
    }
  }
  await walk();
  return { files, symlinks };
}

export async function verifyDm(task, workspace, verification, python, before) {
  const after = await snapshotDm(workspace);
  if (after.symlinks.length) return { outcome: 'policy_violation', passed: 0, diagnostic: 'Symlinks are not allowed in DM workspaces' };
  const changed = [...new Set([...Object.keys(before.files), ...Object.keys(after.files)])].filter(path => before.files[path] !== after.files[path]).sort();
  const allowed = task.dm.allowed_changed_files;
  const violations = allowed.length ? changed.filter(path => !allowed.includes(path)) : [];
  const missing = task.dm.required_changed_files.filter(path => !changed.includes(path));
  await cp(workspace, verification, { recursive: true, filter: path => !relative(workspace, path).split(sep).some(part => IGNORED.has(part)) });
  for (const path of task.dm.hidden_paths) {
    if (path.includes('\\') || path.includes(':') || path.split('/').some(part => ['', '.', '..'].includes(part))) throw new Error('Invalid frozen hidden-test path');
    const source = join(task.directory, 'hidden_tests', path);
    const target = join(verification, path);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, await readFile(source));
  }
  const config = join(verification, '.dm-verifier.ini');
  await writeFile(config, '[pytest]\n');
  const result = runTests(verification, python, task.dm.hidden_test_command, config);
  const infra = result.error || ![0, 1, 2].includes(result.status);
  const details = { changed_files: changed, scope_violations: violations, required_missing: missing,
    hidden_passed: infra ? null : result.status === 0, hidden_exit: result.status,
    hidden_test_command: task.dm.hidden_test_command, pytest_config: 'trusted-empty' };
  if (infra) return { outcome: 'infra_error', passed: null, diagnostic: result.error ?? `pytest_exit_${result.status}`, verifier: result.log, dm_verification: details };
  const scopeFailed = violations.length || missing.length;
  const passed = !scopeFailed && result.status === 0;
  return { outcome: scopeFailed ? 'scope_violation' : passed ? 'passed' : 'failed', passed: passed ? 1 : 0,
    verifier: result.log, dm_verification: details };
}
