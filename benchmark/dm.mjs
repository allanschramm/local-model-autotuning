import { createHash } from 'node:crypto';
import { cp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, join, relative, sep } from 'node:path';
import { runPytest } from './verifier.mjs';

export const DM_SYSTEM = 'Fix the issue using workspace_read and workspace_write. Only the listed existing files are accessible. Keep a short plan in your response before editing. Read the public tests as specifications. Hidden verification runs after you finish. Respect the task changed-file constraints. Finish with a brief summary.';
// shortcut: pinned tasks edit existing setup files; add bounded creation if a future task needs new files.
const IGNORED = new Set(['.git', '.pytest_cache', '__pycache__', '.mypy_cache', '.ruff_cache']);
const COMMAND = ['{python}', '-m', 'pytest', '-q'];

function runTests(directory, python, command, config) {
  if (JSON.stringify(command) !== JSON.stringify(COMMAND)) throw new Error('Unsupported frozen DM verifier command');
  return runPytest(directory, python, [...command.slice(3), ...(config ? ['-c', config] : [])]);
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
    hidden_test_command: task.dm.hidden_test_command, pytest_config: 'trusted-empty', pytest_completion: result.report };
  if (infra) return { outcome: 'infra_error', passed: null, diagnostic: result.error ?? `pytest_exit_${result.status}`, verifier: result.log, dm_verification: details };
  const scopeFailed = violations.length || missing.length;
  const passed = !scopeFailed && result.status === 0;
  return { outcome: scopeFailed ? 'scope_violation' : passed ? 'passed' : 'failed', passed: passed ? 1 : 0,
    verifier: result.log, dm_verification: details };
}
