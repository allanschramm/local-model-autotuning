import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const RUNNER = join(dirname(fileURLToPath(import.meta.url)), 'pytest_runner.py');

export function runPytest(directory, python, args) {
  // This directory is created here and never points to operator workspaces.
  const reports = mkdtempSync(join(tmpdir(), 'pi-verifier-report-'));
  try {
    const path = join(reports, 'completion.json');
    const result = spawnSync(python, [RUNNER, path, ...args], {
      cwd: directory, windowsHide: true, encoding: 'utf8', maxBuffer: 1024 * 1024,
      env: Object.fromEntries(Object.entries({
        PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: process.env.TEMP, TMP: process.env.TMP,
        PYTHONPATH: directory, PYTHONDONTWRITEBYTECODE: '1', PYTEST_DISABLE_PLUGIN_AUTOLOAD: '1',
        PYTEST_ADDOPTS: '-p no:cacheprovider',
      }).filter(([, value]) => value !== undefined)),
    });
    let report;
    try { report = JSON.parse(readFileSync(path, 'utf8')); } catch { report = { complete: false, success: false }; }
    return {
      status: result.status === 0 && !report.success ? 1 : result.status,
      error: result.error?.code ?? result.signal,
      log: (result.stdout ?? '') + (result.stderr ?? ''), report,
    };
  } finally { rmSync(reports, { recursive: true, force: true }); }
}
