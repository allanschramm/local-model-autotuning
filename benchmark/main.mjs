#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { cp, mkdir, mkdtemp, readFile, readdir, realpath, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { DatabaseSync } from 'node:sqlite';
import { parse } from 'yaml';
import {
  createAgentSession, createExtensionRuntime, createReadTool, createWriteTool,
  ModelRuntime, SessionManager, SettingsManager, VERSION,
} from '@earendil-works/pi-coding-agent';

const ROOT = dirname(fileURLToPath(import.meta.url));
const PACKS = new Map([
  ['workspace-smoke', join(ROOT, 'tasks')],
  ['humaneval-base10', join(ROOT, 'packs/humaneval-base10/tasks')],
  ['mbpp-base10', join(ROOT, 'packs/mbpp-base10/tasks')],
]);
const PROTOCOL = 'pi-workspace-v2';
const PI_VERSION = '1.4.0';
if (VERSION !== PI_VERSION) throw new Error(`Pi version drift: expected ${PI_VERSION}, installed ${VERSION}; run npm ci`);
const MAX_OUTPUT_TOKENS = 4096;
const SYSTEM_PROMPT = 'Fix the issue using workspace_read and workspace_write. Only the listed files are accessible. Hidden verification runs after you finish. Do not invent tools. Finish with a brief summary.';

async function files(root, prefix = '') {
  const found = [];
  for (const item of await readdir(join(root, prefix), { withFileTypes: true })) {
    if (item.name === '__pycache__' || item.name === '.pytest_cache') continue;
    const name = prefix ? `${prefix}/${item.name}` : item.name;
    if (item.isSymbolicLink()) throw new Error('Symlinks are not allowed in frozen tasks');
    if (item.isDirectory()) found.push(...await files(root, name));
    else found.push(name);
  }
  return found.sort();
}

async function loadTasks(selected, pack) {
  const taskRoot = PACKS.get(pack);
  if (!taskRoot) throw new Error(`Unknown pack: ${pack}`);
  const tasks = [];
  for (const name of (await readdir(taskRoot)).sort()) {
    if (selected?.length && !selected.includes(name)) continue;
    const directory = join(taskRoot, name);
    const task = parse(await readFile(join(directory, 'task.yaml'), 'utf8'));
    if (task.id !== name || typeof task.issue !== 'string' || !task.issue.trim()
      || !Number.isSafeInteger(task.max_turns) || task.max_turns < 1
      || !Array.isArray(task.allowlist) || !task.allowlist.length) throw new Error(`Invalid task: ${name}`);
    const workspace = await realpath(join(directory, 'workspace'));
    for (const path of task.allowlist) {
      if (typeof path !== 'string' || isAbsolute(path) || path.includes('\\')
        || path.split('/').some(p => !p || p === '.' || p === '..')) throw new Error(`Invalid allowlist: ${name}`);
      const actual = await realpath(join(workspace, path));
      if (relative(workspace, actual).startsWith('..') || isAbsolute(relative(workspace, actual))) throw new Error(`Escaping task: ${name}`);
    }
    const hash = createHash('sha256');
    for (const path of await files(directory)) {
      hash.update(path + '\0').update(await readFile(join(directory, path))).update('\0');
    }
    tasks.push({ ...task, directory, hash: hash.digest('hex') });
  }
  if (!tasks.length || selected?.some(id => !tasks.some(t => t.id === id))) throw new Error('No tasks found or unknown task ID');
  return tasks;
}

function openStore(path) {
  const db = new DatabaseSync(path);
  db.exec(`PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS benchmark_runs (
      run_id TEXT PRIMARY KEY, protocol TEXT NOT NULL, pi_version TEXT NOT NULL,
      provider TEXT NOT NULL, model TEXT NOT NULL, config_json TEXT NOT NULL,
      started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS benchmark_tasks (
      run_id TEXT NOT NULL REFERENCES benchmark_runs(run_id), task_id TEXT NOT NULL,
      outcome TEXT NOT NULL, passed INTEGER CHECK(passed IN (0,1) OR passed IS NULL),
      elapsed_sec REAL NOT NULL, task_hash TEXT NOT NULL, artifact TEXT NOT NULL,
      PRIMARY KEY(run_id,task_id)
    );`);
  return db;
}

function verify(directory, python) {
  const result = spawnSync(python, ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', '_hidden_tests'], {
    cwd: directory, windowsHide: true, encoding: 'utf8', maxBuffer: 1024 * 1024,
    // The verifier runs model-written Python; credentials are not inherited.
    env: Object.fromEntries(Object.entries({
      PATH: process.env.PATH, SystemRoot: process.env.SystemRoot,
      TEMP: process.env.TEMP, TMP: process.env.TMP,
      PYTHONPATH: directory, PYTHONDONTWRITEBYTECODE: '1', PYTEST_DISABLE_PLUGIN_AUTOLOAD: '1',
    }).filter(([, v]) => v !== undefined)),
  });
  const log = (result.stdout ?? '') + (result.stderr ?? '');
  // Pytest exit 2 includes collection failures caused by model-written syntax/import errors.
  if (result.error || result.signal || ![0, 1, 2].includes(result.status)) {
    return { outcome: 'infra_error', passed: null, verifier: log, diagnostic: result.error?.code ?? `pytest_exit_${result.status}` };
  }
  return { outcome: result.status === 0 ? 'passed' : 'failed', passed: result.status === 0 ? 1 : 0, verifier: log };
}

async function runTask(task, runtime, model, python, secrets) {
  const started = performance.now();
  const temp = await mkdtemp(join(tmpdir(), 'pi-benchmark-task-'));
  const workspace = join(temp, 'workspace');
  let session;
  let turns = 0;
  let stopReason;
  let result;
  const evidence = {
    protocol: PROTOCOL, pi_version: PI_VERSION, task: task.id, task_hash: task.hash,
    provider: model.provider, model: model.id, messages: [], workspace: {},
  };
  try {
    await cp(join(task.directory, 'workspace'), workspace, { recursive: true });
    const tools = [createReadTool(workspace), createWriteTool(workspace)].map(tool => ({
      ...tool, name: `workspace_${tool.name}`, label: `workspace_${tool.name}`,
      execute: async (...args) => {
        const path = args[1]?.path;
        const target = typeof path === 'string' ? resolve(workspace, path) : undefined;
        if (!target || !task.allowlist.includes(relative(workspace, target).split(sep).join('/'))
          || await realpath(target) !== target) {
          stopReason = 'policy_violation';
          throw new Error('Only allowlisted workspace files are accessible');
        }
        return tool.execute(...args);
      },
    }));
    const resourceLoader = {
      getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
      getSkills: () => ({ skills: [], diagnostics: [] }),
      getPrompts: () => ({ prompts: [], diagnostics: [] }),
      getThemes: () => ({ themes: [], diagnostics: [] }),
      getAgentsFiles: () => ({ agentsFiles: [] }),
      getSystemPrompt: () => SYSTEM_PROMPT,
      getSystemPromptSource: () => undefined,
      getAppendSystemPrompt: () => [], getAppendSystemPromptSources: () => [],
      extendResources: () => {}, reload: async () => {},
    };
    ({ session } = await createAgentSession({
      cwd: workspace, agentDir: join(temp, 'profile'), model, modelRuntime: runtime,
      resourceLoader, tools: tools.map(t => t.name), customTools: tools,
      thinkingLevel: 'off', sessionManager: SessionManager.inMemory(workspace),
      settingsManager: SettingsManager.inMemory({ cacheWarming: 'off', compaction: { enabled: false }, retry: { enabled: false } }),
    }));
    evidence.thinking_level = session.thinkingLevel;
    const stream = session.agent.streamFunction;
    session.agent.streamFunction = (requestedModel, context, options) => {
      if (stopReason) throw new Error(stopReason);
      if (turns >= task.max_turns) {
        stopReason = 'turn_limit';
        throw new Error(stopReason);
      }
      turns++;
      return stream(requestedModel, context, { ...options, maxTokens: MAX_OUTPUT_TOKENS, temperature: 0, maxRetries: 0 });
    };
    await session.prompt(`${task.issue}\n\nAccessible workspace files:\n${task.allowlist.join('\n')}`);
    const last = session.messages.filter(m => m.role === 'assistant').at(-1);
    if (stopReason) result = { outcome: stopReason, passed: 0 };
    else if (!last || last.stopReason === 'error' || last.stopReason === 'aborted') {
      result = { outcome: 'infra_error', passed: null, diagnostic: last?.errorMessage ?? 'Missing assistant response' };
    } else {
      const verification = join(temp, 'verification');
      await cp(workspace, verification, { recursive: true });
      await cp(join(task.directory, 'hidden_tests'), join(verification, '_hidden_tests'), { recursive: true });
      result = verify(verification, python);
    }
  } catch (error) {
    result = { outcome: stopReason ?? 'infra_error', passed: stopReason ? 0 : null, diagnostic: String(error.message) };
  } finally {
    if (session) {
      evidence.messages = session.messages.map(message => {
        if (!message.usage) return message;
        const { cost, ...usage } = message.usage;
        return { ...message, usage };
      });
      session.dispose();
    }
    for (const path of task.allowlist) {
      try { evidence.workspace[path] = await readFile(join(workspace, path), 'utf8'); }
      catch { evidence.workspace[path] = null; }
    }
    // This exact directory was created above, never an operator model/workspace directory.
    await rm(temp, { recursive: true, force: true });
  }
  Object.assign(evidence, result, { turns, elapsed_sec: (performance.now() - started) / 1000 });
  let serialized = JSON.stringify(evidence, null, 2);
  for (const secret of secrets) serialized = serialized.replaceAll(secret, '[REDACTED]');
  return JSON.parse(serialized);
}

async function main() {
  const { values } = parseArgs({ options: {
    help: { type: 'boolean' }, list: { type: 'boolean' },
    pack: { type: 'string', default: 'workspace-smoke' },
    provider: { type: 'string' }, model: { type: 'string' },
    'base-url': { type: 'string' }, api: { type: 'string', default: 'openai-completions' },
    'context-window': { type: 'string' }, 'api-key-env': { type: 'string' },
    python: { type: 'string' }, task: { type: 'string', multiple: true },
    db: { type: 'string', default: 'results.db' },
  } });
  if (values.help) {
    console.log('node main.mjs --provider <provider> --model <id> --python <venv-python> [--pack workspace-smoke|humaneval-base10|mbpp-base10] [--task <id>] [--db results.db]\nCustom endpoint: --base-url <url> --context-window <tokens> [--api <Pi protocol>] [--api-key-env <VARIABLE>]\n--list lists frozen tasks in the selected pack. No tuning configuration or local server launch is required.');
    return 0;
  }
  const tasks = await loadTasks(values.task, values.pack);
  if (values.list) { console.log(tasks.map(t => t.id).join('\n')); return 0; }
  if (!values.provider || !values.model || !values.python) throw new Error('--provider, --model and --python are required');
  const python = resolve(values.python);
  const verifier = spawnSync(python, ['-c', 'import pytest'], { windowsHide: true, encoding: 'utf8' });
  if (verifier.error || verifier.status !== 0) throw new Error('--python must point to an environment with pytest installed');
  const key = values['api-key-env'] ? process.env[values['api-key-env']] : undefined;
  if (values['api-key-env'] && !key) throw new Error('The selected API key environment variable is empty');
  const profile = await mkdtemp(join(tmpdir(), 'pi-benchmark-profile-'));
  let db;
  try {
    const runtime = await ModelRuntime.create({
      authPath: join(profile, 'auth.json'), modelsPath: null,
      modelsStorePath: join(profile, 'models.json'), allowModelNetwork: false,
    });
    if (values['base-url']) {
      const url = new URL(values['base-url']);
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('Endpoint must be HTTP(S), without credentials, query, or fragment');
      const context = Number(values['context-window']);
      if (!Number.isSafeInteger(context) || context <= MAX_OUTPUT_TOKENS) throw new Error('--context-window must exceed the 4096-token output budget');
      runtime.registerProvider(values.provider, {
        baseUrl: url.href.replace(/\/$/, ''), api: values.api, apiKey: key ?? 'local-no-key',
        models: [{ id: values.model, name: values.model, reasoning: false, input: ['text'],
          contextWindow: context, maxTokens: MAX_OUTPUT_TOKENS,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
      });
    } else if (key) await runtime.setRuntimeApiKey(values.provider, key);
    const model = runtime.getModel(values.provider, values.model);
    if (!model) throw new Error('Unknown Pi model; configure --base-url for a custom endpoint');
    if (model.maxTokens < MAX_OUTPUT_TOKENS) throw new Error('Model cannot support this protocol output budget');
    const database = resolve(values.db);
    await mkdir(dirname(database), { recursive: true });
    db = openStore(database);
    const runId = randomUUID();
    const artifactDir = join(dirname(database), 'benchmark-artifacts', runId);
    await mkdir(artifactDir, { recursive: true });
    const config = {
      pack: values.pack,
      tools: ['workspace_read', 'workspace_write'], thinking_requested: 'off', temperature: 0,
      max_output_tokens: MAX_OUTPUT_TOKENS, auto_retry: false, compaction: false,
      context_window: model.contextWindow, context_source: values['base-url'] ? 'operator-declared' : 'pi-catalog',
      system_prompt: SYSTEM_PROMPT, task_hashes: Object.fromEntries(tasks.map(t => [t.id, t.hash])),
      turn_limits: Object.fromEntries(tasks.map(t => [t.id, t.max_turns])),
    };
    db.prepare('INSERT INTO benchmark_runs (run_id,protocol,pi_version,provider,model,config_json,started_at,finished_at,status) VALUES (?,?,?,?,?,?,?,?,?)').run(
      runId, PROTOCOL, PI_VERSION, values.provider, values.model, JSON.stringify(config), new Date().toISOString(), null, 'running');
    const secrets = Object.entries(process.env).filter(([name, value]) => /KEY|TOKEN|SECRET|PASSWORD/i.test(name) && value?.length >= 8).map(([, value]) => value);
    let exitCode = 0;
    for (const task of tasks) {
      const result = await runTask(task, runtime, model, python, secrets);
      const artifact = join(artifactDir, `${task.id}.json`);
      await writeFile(artifact, JSON.stringify(result, null, 2));
      db.prepare('INSERT INTO benchmark_tasks (run_id,task_id,outcome,passed,elapsed_sec,task_hash,artifact) VALUES (?,?,?,?,?,?,?)').run(
        runId, task.id, result.outcome, result.passed, result.elapsed_sec, task.hash, artifact);
      console.log(`${task.id}: ${result.outcome} | ${artifact}`);
      exitCode = Math.max(exitCode, result.outcome === 'infra_error' ? 2 : result.passed ? 0 : 1);
    }
    db.prepare('UPDATE benchmark_runs SET status=?,finished_at=? WHERE run_id=?').run('complete', new Date().toISOString(), runId);
    console.log(`run_id=${runId} protocol=${PROTOCOL} results=${database}`);
    return exitCode;
  } finally {
    db?.close();
    await rm(profile, { recursive: true, force: true });
  }
}

main().then(code => { process.exitCode = code; }).catch(error => {
  console.error(error.message);
  process.exitCode = 2;
});
