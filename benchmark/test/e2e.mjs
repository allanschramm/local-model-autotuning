// Failure modes: fake passes, missing evidence, leaked hidden tests/credentials,
// path escapes, allowed absolute SDK paths rejected, absolute hidden-test access,
// infinite turns, infra errors scored as model failures, and repo dependencies.
// Pack selection must preserve the default five tasks; HumanEval check(candidate)
// must reject every empty stub and accept every upstream oracle without EvalPlus.
// MBPP must preserve the legacy's non-contiguous IDs and execute base assertions,
// including imported entrypoints/helpers, without changing either existing pack.
// LiveCodeBench must execute private stdin cases, reject sample-only solutions,
// preserve strict whitespace comparison, and isolate each program's cwd.
// BigCodeBench must run every original unittest against stubs/oracles, share
// solution globals with tests, keep plots headless, and fail infra on missing deps.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { cp, mkdir, mkdtemp, readFile, rename, writeFile } from 'node:fs/promises';
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
const credentialRequests = [];
const server = createServer(async (req, res) => {
  let raw = '';
  for await (const chunk of req) raw += chunk;
  const body = JSON.parse(raw);
  requests.push(body);
  calls++;
  assert.equal(req.url, '/v1/chat/completions');
  const credential = req.headers.authorization?.slice(7);
  if (body.model.startsWith('credential-')) credentialRequests.push(credential);
  else assert.equal(req.headers.authorization, `Bearer ${key}`);
  assert(!raw.includes('def test_inclusive_max'), 'hidden test leaked');
  assert(!raw.includes('def test_original_mbpp_assertions'), 'MBPP hidden test leaked');
  assert(!raw.includes('private_cases.json.gz'), 'LCB hidden verifier leaked');
  assert(!raw.includes('test_original_bigcode_suite'), 'BigCode hidden verifier leaked');
  assert(!raw.includes('PERSONAL_RESOURCE_SENTINEL'), 'personal Pi resource leaked');
  if (body.model === 'infra-error' || body.model === 'credential-error') {
    res.writeHead(401, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: body.model === 'credential-error' ? `Rejected credential: ${credential}` : 'fixture provider unavailable' } }));
    return;
  }
  let tool;
  const usedTools = body.messages.some(m => m.role === 'tool');
  if (body.model === 'credential-echo' && !usedTools) {
    tool = { name: 'workspace_read', arguments: JSON.stringify({ path: 'clamp.py', [credential]: 'fixture-value' }) };
  } else if (body.model.startsWith('lcb-') && body.model !== 'lcb-empty' && !usedTools) {
    let code = "from pathlib import Path\nassert Path.cwd() != Path(__file__).resolve().parent\nmarker = Path('case-marker.txt')\nassert not marker.exists()\nmarker.write_text('seen')\nx = int(input())\nanswer = sum(i*j for i in range(1,10) for j in range(1,10) if i*j != x)\nprint(answer)\n";
    if (body.model === 'lcb-samples-only') code = "x = int(input())\nprint({1: 2024, 11: 2025, 24: 1929}.get(x, 0))\n";
    if (body.model === 'lcb-whitespace') code = code.replace('print(answer)', "print(answer, end=' \\n')");
    if (body.model === 'lcb-syntax') code = 'def broken syntax!\n';
    tool = { name: 'workspace_write', arguments: JSON.stringify({ path: 'solution.py', content: code }) };
  } else if (['human-oracle', 'mbpp-oracle', 'bigcode-oracle'].includes(body.model) && !usedTools) {
    const pack = body.model === 'human-oracle' ? 'humaneval-base10' : body.model === 'mbpp-oracle' ? 'mbpp-base10' : 'bigcode-hard10';
    const sourceId = raw.match(/(?:HumanEval|Mbpp|BigCodeBench)\/\d+/)?.[0];
    const oracles = JSON.parse(await readFile(join(isolated, `packs/${pack}/oracles.json`), 'utf8'));
    assert(sourceId && oracles[sourceId], 'Expected a frozen source task');
    tool = { name: 'workspace_write', arguments: JSON.stringify({ path: 'solution.py', content: oracles[sourceId] }) };
  } else if (body.model.startsWith('absolute-')) {
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
  } else if (body.model === 'exit-zero' && !usedTools) {
    tool = { name: 'workspace_write', arguments: JSON.stringify({ path: 'clamp.py', content: 'import os\nos._exit(0)\n' }) };
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
    choices: [{ index: 0, delta: { role: 'assistant', ...(tool ? { tool_calls: [{ index: 0, id: `call_${calls}`, type: 'function', function: tool }] } : { content: body.model === 'credential-echo' ? `Echo: ${credential}; JSON: ${JSON.stringify(credential)}` : 'Done.' }) }, finish_reason: null }] });
  send({ id: 'fixture', object: 'chat.completion.chunk', model: body.model,
    choices: [{ index: 0, delta: {}, finish_reason: tool ? 'tool_calls' : 'stop' }],
    usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 } });
  res.end('data: [DONE]\n\n');
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const profile = join(artifacts, 'personal-profile');
await mkdir(profile, { recursive: true });
await writeFile(join(profile, 'AGENTS.md'), 'PERSONAL_RESOURCE_SENTINEL');
const run = (args, environment = {}) => new Promise((accept, reject) => {
  const child = spawn(process.execPath, [join(isolated, 'main.mjs'), ...args], {
    cwd: artifacts, env: { ...process.env, BENCHMARK_TEST_KEY: key, PI_CODING_AGENT_DIR: profile, ...environment },
    windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  child.stdout.on('data', x => output += x);
  child.stderr.on('data', x => output += x);
  child.on('error', reject);
  child.on('close', code => accept({ code, output }));
});
try {
  const integrity = join(artifacts, 'integrity');
  await mkdir(integrity);
  const integrityDatabase = join(integrity, 'results.db');
  const failures = [];
  for (const [index, secret] of ['fixture-"quoted-key', 'fixture-\\slash-key', 's!7', 'fixture-arbitrary-name-key', 'null', 'passed'].entries()) {
    const response = await run(['--provider', 'fixture', '--model', 'credential-echo',
      '--base-url', `http://127.0.0.1:${server.address().port}/v1`, '--context-window', '65536',
      '--api-key-env', 'MODEL_AUTH', '--python', python, '--task', 'issue-clamp-inclusive', '--db', integrityDatabase], { MODEL_AUTH: secret });
    assert.equal(credentialRequests.at(-1), secret, 'Selected credential must reach endpoint');
    assert.equal(response.code, 1, 'Credential redaction must preserve model-failure outcomes');
    await writeFile(join(integrity, `credential-${index}.log`), response.output);
    const check = new DatabaseSync(integrityDatabase);
    const row = check.prepare('SELECT artifact FROM benchmark_tasks ORDER BY rowid DESC LIMIT 1').get(); check.close();
    const evidence = JSON.parse(await readFile(row.artifact, 'utf8'));
    const text = evidence.messages.filter(m => m.role === 'assistant').flatMap(m => m.content.filter(c => c.type === 'text').map(c => c.text)).join('\n');
    if (text.includes(secret) || text.includes(JSON.stringify(secret).slice(1, -1))) failures.push(`Credential ${index} leaked`);
    const args = evidence.messages.flatMap(m => m.content ?? []).filter(c => c.type === 'toolCall').map(c => c.arguments);
    if (args.some(a => Object.keys(a).includes(secret))) failures.push(`Credential ${index} leaked through JSON property name`);
  }
  const errorSecret = 'fixture-"error\\key';
  await run(['--provider', 'fixture', '--model', 'credential-error', '--base-url', `http://127.0.0.1:${server.address().port}/v1`,
    '--context-window', '65536', '--api-key-env', 'MODEL_AUTH', '--python', python, '--task', 'issue-clamp-inclusive', '--db', integrityDatabase], { MODEL_AUTH: errorSecret });
  const premature = await run(['--provider', 'fixture', '--model', 'exit-zero', '--base-url', `http://127.0.0.1:${server.address().port}/v1`,
    '--context-window', '65536', '--api-key-env', 'BENCHMARK_TEST_KEY', '--python', python, '--task', 'issue-clamp-inclusive', '--db', integrityDatabase]);
  const check = new DatabaseSync(integrityDatabase);
  const integrityRows = check.prepare('SELECT * FROM benchmark_tasks ORDER BY rowid').all(); check.close();
  const errorEvidence = JSON.parse(await readFile(integrityRows.at(-2).artifact, 'utf8'));
  if (errorEvidence.diagnostic?.includes(errorSecret) || errorEvidence.diagnostic?.includes(JSON.stringify(errorSecret).slice(1, -1))) failures.push('HTTP error credential leaked');
  if (premature.code !== 1 || integrityRows.at(-1).passed !== 0) failures.push('Early process exit incorrectly passed hidden verification');
  await writeFile(join(integrity, 'verification.json'), JSON.stringify({ status: failures.length ? 'failed' : 'passed', cases: integrityRows.length, failures }, null, 2));
  assert.deepEqual(failures, [], `Integrity regressions: ${integrity}`);
  const list = await run(['--list']);
  assert.equal(list.code, 0, list.output);
  assert(list.output.includes('issue-clamp-inclusive'));
  assert.equal(list.output.trim().split('\n').length, 5, 'Default pack stays frozen');
  const packList = await run(['--list', '--pack', 'humaneval-base10']);
  assert.equal(packList.code, 0, packList.output);
  const humanIds = packList.output.trim().split(/\r?\n/);
  assert.equal(humanIds.length, 10);
  const mbppList = await run(['--list', '--pack', 'mbpp-base10']);
  assert.equal(mbppList.code, 0, mbppList.output);
  const mbppIds = mbppList.output.trim().split(/\r?\n/);
  assert.deepEqual(mbppIds, [2, 3, 4, 6, 7, 8, 9, 11, 12, 14].map(id => `mbpp-${String(id).padStart(3, '0')}`));
  const lcbList = await run(['--list', '--pack', 'lcb-v6-stdin10']);
  assert.equal(lcbList.code, 0, lcbList.output);
  const lcbIds = lcbList.output.trim().split(/\r?\n/);
  assert.deepEqual(lcbIds, ['abc387_b', 'abc387_f', 'abc387_a', 'abc387_c', 'abc388_d', 'abc388_b', 'abc388_g', 'abc388_c', 'abc388_a', 'abc388_f'].map((id, i) => `lcb-${String(i).padStart(3, '0')}-${id}`));
  assert.equal((await run(['--list', '--pack', '../tasks'])).code, 2);
  const bigcodeList = await run(['--list', '--pack', 'bigcode-hard10']);
  assert.equal(bigcodeList.code, 0, bigcodeList.output);
  const bigcodeIds = bigcodeList.output.trim().split(/\r?\n/);
  assert.deepEqual(bigcodeIds, [13, 15, 17, 19, 34, 37, 82, 89, 92, 93].map(id => `bigcode-${String(id).padStart(3, '0')}`));
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
  for (const model of ['human-empty', 'human-oracle', 'mbpp-empty', 'mbpp-oracle']) {
    const pack = model.startsWith('human-') ? 'humaneval-base10' : 'mbpp-base10';
    const result = await run([
      '--provider', 'fixture', '--model', model, '--pack', pack,
      '--base-url', `http://127.0.0.1:${server.address().port}/v1`,
      '--context-window', '65536', '--api-key-env', 'BENCHMARK_TEST_KEY',
      '--python', python, '--db', database,
    ]);
    await writeFile(join(artifacts, `${model}.log`), result.output);
    assert.equal(result.code, model.endsWith('-oracle') ? 0 : 1, result.output);
  }
  for (const model of ['lcb-empty', 'lcb-reference', 'lcb-samples-only', 'lcb-whitespace', 'lcb-syntax', 'lcb-missing-cases']) {
    const archive = join(isolated, 'packs/lcb-v6-stdin10/tasks', lcbIds[0], 'hidden_tests/private_cases.json.gz');
    if (model === 'lcb-missing-cases') await rename(archive, `${archive}.saved`);
    let result;
    try { result = await run([
      '--provider', 'fixture', '--model', model, '--pack', 'lcb-v6-stdin10',
      '--base-url', `http://127.0.0.1:${server.address().port}/v1`,
      '--context-window', '65536', '--api-key-env', 'BENCHMARK_TEST_KEY',
      '--python', python, '--db', database,
      ...(model === 'lcb-empty' ? [] : ['--task', lcbIds[0]]),
    ]); } finally {
      if (model === 'lcb-missing-cases') await rename(`${archive}.saved`, archive);
    }
    await writeFile(join(artifacts, `${model}.log`), result.output);
    assert.equal(result.code, model === 'lcb-reference' ? 0 : model === 'lcb-missing-cases' ? 2 : 1, result.output);
  }
  for (const model of ['bigcode-empty', 'bigcode-oracle', 'bigcode-missing-dependency']) {
    const requirements = join(isolated, 'packs/bigcode-hard10/tasks', bigcodeIds[0], 'hidden_tests/dependencies.json');
    const saved = await readFile(requirements);
    if (model === 'bigcode-missing-dependency') await writeFile(requirements, '["missing_benchmark_dependency_fixture"]');
    let result;
    try { result = await run([
      '--provider', 'fixture', '--model', model, '--pack', 'bigcode-hard10',
      '--base-url', `http://127.0.0.1:${server.address().port}/v1`,
      '--context-window', '65536', '--api-key-env', 'BENCHMARK_TEST_KEY',
      '--python', python, '--db', database,
      ...(model === 'bigcode-missing-dependency' ? ['--task', bigcodeIds[0]] : []),
    ]); } finally { await writeFile(requirements, saved); }
    await writeFile(join(artifacts, `${model}.log`), result.output);
    assert.equal(result.code, model === 'bigcode-oracle' ? 0 : model === 'bigcode-missing-dependency' ? 2 : 1, result.output);
  }
  const db = new DatabaseSync(database);
  const allRows = db.prepare('SELECT * FROM benchmark_tasks ORDER BY rowid').all();
  assert.equal(allRows.length, 86);
  const rows = allRows.slice(0, 10);
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
  for (const [start, outcome, ids] of [[10, 'failed', humanIds], [20, 'passed', humanIds], [30, 'failed', mbppIds], [40, 'passed', mbppIds]]) {
    const packRows = allRows.slice(start, start + 10);
    assert.deepEqual(packRows.map(r => r.task_id), ids);
    assert(packRows.every(r => r.outcome === outcome), 'Every coding verifier must execute its assertions');
  }
  assert.deepEqual(allRows.slice(50, 60).map(r => r.task_id), lcbIds);
  assert(allRows.slice(50, 60).every(r => r.outcome === 'failed'));
  assert.equal(allRows[60].outcome, 'passed');
  assert(allRows.slice(61, 64).every(r => r.outcome === 'failed'));
  assert.equal(allRows[64].outcome, 'infra_error', 'Missing private fixtures are not model failures');
  assert.equal(allRows[64].passed, null);
  assert.deepEqual(allRows.slice(65, 75).map(r => r.task_id), bigcodeIds);
  assert(allRows.slice(65, 75).every(r => r.outcome === 'failed'));
  assert.deepEqual(allRows.slice(75, 85).map(r => r.task_id), bigcodeIds);
  assert(allRows.slice(75, 85).every(r => r.outcome === 'passed'));
  assert.equal(allRows[85].outcome, 'infra_error');
  assert.equal(allRows[85].passed, null);
  for (const row of allRows) {
    const evidence = JSON.parse(await readFile(row.artifact, 'utf8'));
    assert.equal(evidence.task, row.task_id);
    assert.equal(evidence.protocol, 'pi-workspace-v3');
    assert.equal(evidence.pi_version, '1.4.0');
    assert.equal(evidence.outcome, row.outcome);
    assert.equal(evidence.passed, row.passed);
    assert.equal(typeof evidence.task_hash, 'string');
    assert(!JSON.stringify(evidence).includes(key));
    assert(evidence.messages.filter(m => m.usage).every(m => !('cost' in m.usage)));
  }
  const runs = db.prepare('SELECT * FROM benchmark_runs').all();
  assert(runs.every(r => r.status === 'complete'));
  assert(runs.slice(10, 12).every(r => JSON.parse(r.config_json).pack === 'humaneval-base10'));
  assert(runs.slice(12, 14).every(r => JSON.parse(r.config_json).pack === 'mbpp-base10'));
  assert(runs.slice(14, 20).every(r => JSON.parse(r.config_json).pack === 'lcb-v6-stdin10'));
  assert(runs.slice(20).every(r => JSON.parse(r.config_json).pack === 'bigcode-hard10'));
  assert(runs.slice(0, 10).every(r => JSON.parse(r.config_json).pack === 'workspace-smoke'));
  assert(requests.every(r => r.model && (r.max_tokens ?? r.max_completion_tokens) === 4096));
  assert(!requests.some(r => 'min_p' in r || 'chat_template_kwargs' in r));
  db.close();
  const summary = { status: 'passed', cases: allRows.map(r => ({ task: r.task_id, outcome: r.outcome })), requests: calls, database, standalone: isolated, integrity: 'integrity/verification.json' };
  await writeFile(join(artifacts, 'verification.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await new Promise(r => server.close(r));
}
