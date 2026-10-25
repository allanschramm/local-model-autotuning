// Failure modes: fake passes, missing evidence, leaked hidden tests/credentials,
// path escapes, allowed absolute SDK paths rejected, absolute hidden-test access,
// infinite turns, infra errors scored as model failures, and repo dependencies.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { cp, mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';

const source = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.BENCHMARK_PYTHON;
assert(python, 'Set BENCHMARK_PYTHON to the project venv Python');
const artifacts = process.env.BENCHMARK_E2E_DIR
  ? resolve(process.env.BENCHMARK_E2E_DIR)
  : await mkdtemp(join(tmpdir(), 'pi-benchmark-e2e-'));
await mkdir(artifacts, { recursive: true });
const isolated = join(artifacts, 'standalone');
await cp(source, isolated, { recursive: true, filter: p => !/[\\/](node_modules|artifacts)([\\/]|$)/.test(p) });
// Installed dependencies are external inputs; source code and fixtures come only from benchmark/.
await cp(join(source, 'node_modules'), join(isolated, 'node_modules'), { recursive: true });
const database = join(artifacts, 'results.db');
const key = 'e2e-key-must-not-appear-in-artifacts';
let calls = 0;
const requests = [];
const server = createServer(async (req, res) => {
  let raw = '';
  for await (const chunk of req) raw += chunk;
  const body = JSON.parse(raw);
  requests.push(body);
  calls++;
  assert.equal(req.url, '/v1/chat/completions');
  assert.equal(req.headers.authorization, `Bearer ${key}`);
  assert(!raw.includes('def test_inclusive_max'), 'hidden test leaked');
  assert(!raw.includes('PERSONAL_RESOURCE_SENTINEL'), 'personal Pi resource leaked');
  if (body.model === 'infra-error') {
    res.writeHead(401, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'fixture provider unavailable' } }));
    return;
  }
  let tool;
  const usedTools = body.messages.some(m => m.role === 'tool');
  if (body.model.startsWith('absolute-')) {
    const prompt = body.messages.map(m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n');
    const cwd = prompt.match(/<cwd>\s*([\s\S]*?)\s*<\/cwd>/)?.[1];
    assert(cwd, 'Pi must advertise its workspace directory');
    const turns = body.messages.filter(m => m.role === 'tool').length;
    if (body.model === 'absolute-escape') {
      tool = { name: 'workspace_write', arguments: JSON.stringify({ path: resolve(cwd, '../escaped.txt'), content: 'escape' }) };
    } else if (body.model === 'absolute-hidden') {
      tool = { name: 'workspace_read', arguments: JSON.stringify({ path: join(cwd, '_hidden_tests/test_clamp.py') }) };
    } else if (turns === 0) {
      tool = { name: 'workspace_read', arguments: JSON.stringify({ path: join(cwd, 'clamp.py') }) };
    } else if (turns === 1) {
      tool = { name: 'workspace_write', arguments: JSON.stringify({ path: join(cwd, 'clamp.py'), content: 'def clamp(value, lo, hi):\n    return min(max(value, lo), hi)\n' }) };
    }
  } else if (['pass', 'syntax'].includes(body.model) && !usedTools) {
    tool = { name: 'workspace_write', arguments: JSON.stringify({ path: 'clamp.py', content: body.model === 'syntax' ? 'def broken syntax!\n' : 'def clamp(value, lo, hi):\n    return min(max(value, lo), hi)\n' }) };
  } else if (body.model === 'escape') {
    tool = { name: 'workspace_write', arguments: JSON.stringify({ path: '../escaped.txt', content: 'escape' }) };
  } else if (body.model === 'hidden') {
    tool = { name: 'workspace_read', arguments: JSON.stringify({ path: '_hidden_tests/test_clamp.py' }) };
  } else if (body.model === 'loop') {
    tool = { name: 'workspace_read', arguments: JSON.stringify({ path: 'clamp.py' }) };
  }
  res.writeHead(200, { 'Content-Type': 'text/event-stream' });
  const send = data => res.write(`data: ${JSON.stringify(data)}\n\n`);
  send({ id: 'fixture', object: 'chat.completion.chunk', model: body.model,
    choices: [{ index: 0, delta: { role: 'assistant', ...(tool ? { tool_calls: [{ index: 0, id: `call_${calls}`, type: 'function', function: tool }] } : { content: 'Done.' }) }, finish_reason: null }] });
  send({ id: 'fixture', object: 'chat.completion.chunk', model: body.model,
    choices: [{ index: 0, delta: {}, finish_reason: tool ? 'tool_calls' : 'stop' }],
    usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 } });
  res.end('data: [DONE]\n\n');
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const profile = join(artifacts, 'personal-profile');
await mkdir(profile, { recursive: true });
await writeFile(join(profile, 'AGENTS.md'), 'PERSONAL_RESOURCE_SENTINEL');
const run = args => new Promise((accept, reject) => {
  const child = spawn(process.execPath, [join(isolated, 'main.mjs'), ...args], {
    cwd: artifacts, env: { ...process.env, BENCHMARK_TEST_KEY: key, PI_CODING_AGENT_DIR: profile },
    windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  child.stdout.on('data', x => output += x);
  child.stderr.on('data', x => output += x);
  child.on('error', reject);
  child.on('close', code => accept({ code, output }));
});
try {
  const list = await run(['--list']);
  assert.equal(list.code, 0, list.output);
  assert(list.output.includes('issue-clamp-inclusive'));
  for (const model of ['pass', 'fail', 'infra-error', 'escape', 'hidden', 'loop', 'syntax', 'absolute-pass', 'absolute-escape', 'absolute-hidden']) {
    const result = await run([
      '--provider', 'fixture', '--model', model,
      '--base-url', `http://127.0.0.1:${server.address().port}/v1`,
      '--context-window', '65536', '--api-key-env', 'BENCHMARK_TEST_KEY',
      '--python', python, '--task', 'issue-clamp-inclusive', '--db', database,
    ]);
    await writeFile(join(artifacts, `${model}.log`), result.output);
    assert.equal(result.code, ['pass', 'absolute-pass'].includes(model) ? 0 : model === 'infra-error' ? 2 : 1, result.output);
  }
  const db = new DatabaseSync(database);
  const rows = db.prepare('SELECT * FROM benchmark_tasks ORDER BY rowid').all();
  assert.equal(rows.length, 10);
  assert.equal(rows[0].outcome, 'passed');
  assert.equal(rows[1].outcome, 'failed');
  assert.equal(rows[2].outcome, 'infra_error');
  assert.equal(rows[2].passed, null);
  assert.equal(rows[3].outcome, 'policy_violation');
  assert.equal(rows[4].outcome, 'policy_violation');
  assert.equal(rows[5].outcome, 'turn_limit');
  assert.equal(rows[6].outcome, 'failed', 'invalid model-written Python is a model failure');
  assert.equal(rows[7].outcome, 'passed', 'Pi read/write tools accept allowed absolute paths');
  assert.equal(rows[8].outcome, 'policy_violation');
  assert.equal(rows[9].outcome, 'policy_violation');
  for (const row of rows) {
    const evidence = JSON.parse(await readFile(row.artifact, 'utf8'));
    assert.equal(evidence.task, 'issue-clamp-inclusive');
    assert.equal(evidence.protocol, 'pi-workspace-v2');
    assert.equal(evidence.pi_version, '1.4.0');
    assert.equal(evidence.outcome, row.outcome);
    assert.equal(evidence.passed, row.passed);
    assert.equal(typeof evidence.task_hash, 'string');
    assert(!JSON.stringify(evidence).includes(key));
    assert(evidence.messages.filter(m => m.usage).every(m => !('cost' in m.usage)));
  }
  const runs = db.prepare('SELECT * FROM benchmark_runs').all();
  assert(runs.every(r => r.status === 'complete'));
  assert(requests.every(r => r.model && (r.max_tokens ?? r.max_completion_tokens) === 4096));
  assert(!requests.some(r => 'min_p' in r || 'chat_template_kwargs' in r));
  db.close();
  const summary = { status: 'passed', cases: rows.map(r => ({ task: r.task_id, outcome: r.outcome })), requests: calls, database, standalone: isolated };
  await writeFile(join(artifacts, 'verification.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await new Promise(r => server.close(r));
}
