// Failure modes: hidden fixtures exposed during the agent, missing/corrupt
// fixtures treated as passes, scope ignored, required edits omitted, tests never
// run, cross-workspace access, source/repo coupling and generation budget drift.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { cp, mkdtemp, readFile, rename, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import { parse } from 'yaml';

const python = process.env.BENCHMARK_PYTHON;
assert(python, 'Set BENCHMARK_PYTHON to the project venv Python');
const source = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directory = await mkdtemp(join(tmpdir(), 'pi-dm-e2e-'));
const isolated = join(directory, 'standalone');
await cp(source, isolated, { recursive: true, filter: p => !/[\\/](node_modules|artifacts)([\\/]|$)/.test(p) });
await cp(join(source, 'node_modules'), join(isolated, 'node_modules'), { recursive: true });
const pack = join(isolated, 'packs/dm-code30');
const dbPath = join(directory, 'results.db');
let requests = 0;
const server = createServer(async (request, response) => {
  let raw = ''; for await (const chunk of request) raw += chunk;
  const body = JSON.parse(raw); requests++;
  assert(!raw.includes('test_punctuation_and_repeated_spaces'), 'Hidden assertion leaked');
  assert(!raw.includes('fixture-secret-must-stay-private'), 'Credential leaked');
  assert.equal(body.max_tokens ?? body.max_completion_tokens, 4096);
  assert(!body.tools.some(t => t.function.name === 'workspace_test'), 'Interactive code execution must not be exposed');
  const id = raw.match(/Task source: ([a-z0-9_]+)/)?.[1]; assert(id);
  const used = body.messages.filter(m => m.role === 'tool').length;
  const oracles = JSON.parse(await readFile(join(pack, 'oracles.json'), 'utf8'));
  const manifest = parse(await readFile(join(pack, 'tasks', id, 'task.yaml'), 'utf8'));
  const oracle = oracles[id];
  const plan = [];
  if (body.model === 'pytest-filter') {
    for (const path of manifest.allowlist) {
      let content = await readFile(join(pack, 'tasks', id, 'workspace', path), 'utf8');
      content += '\n# Fixture edit\n';
      if (path === 'pyproject.toml') content += '\n[tool.pytest.ini_options]\naddopts = "-k test_ci_runs_pytest"\n';
      plan.push({ name: 'workspace_write', arguments: JSON.stringify({ path, content }) });
    }
  } else if (body.model === 'public-only') plan.push({ name: 'workspace_test', arguments: '{}' });
  else if (['escape', 'hidden'].includes(body.model)) plan.push({ name: 'workspace_read', arguments: JSON.stringify({ path: body.model === 'escape' ? '../private.txt' : 'tests/test_hidden_slugify.py' }) });
  else if (['partial-exit', 'partial-skip'].includes(body.model)) {
    const content = oracles.slugify_cleanup['text_utils.py'] + `\nimport os, pytest\n_original = slugify\n_first = None\ndef slugify(text):\n    global _first\n    current = os.environ.get('PYTEST_CURRENT_TEST')\n    if _first is None: _first = current\n    elif current != _first: ${body.model === 'partial-exit' ? "pytest.exit('fixture partial exit', returncode=0)" : "pytest.skip('fixture partial skip')"}\n    return _original(text)\n`;
    plan.push({ name: 'workspace_write', arguments: JSON.stringify({ path: 'text_utils.py', content }) });
  }
  else if (['syntax', 'exit-zero', 'public-probe'].includes(body.model)) {
    const content = body.model === 'syntax' ? 'def invalid syntax!\n' : body.model === 'exit-zero' ? 'import os\nos._exit(0)\n' : `from pathlib import Path\nraise RuntimeError(Path(${JSON.stringify(join(pack, 'tasks/slugify_cleanup/hidden_tests/tests/test_hidden_slugify.py'))}).read_text())\n`;
    plan.push({ name: 'workspace_write', arguments: JSON.stringify({ path: 'text_utils.py', content }) });
    if (body.model === 'public-probe') plan.push({ name: 'workspace_test', arguments: '{}' });
  }
  else if (body.model !== 'empty') {
    assert(oracle, 'Expected an upstream reference fixture');
    for (const [path, content] of Object.entries(oracle)) {
      if (body.model === 'missing-required' && path.startsWith('tests/')) continue;
      plan.push({ name: 'workspace_write', arguments: JSON.stringify({ path, content }) });
    }
    if (body.model === 'forbidden-edit') {
      const path = 'tests/test_public_config_loader.py';
      const original = await readFile(join(pack, 'tasks', id, 'workspace', path), 'utf8');
      plan.push({ name: 'workspace_write', arguments: JSON.stringify({ path, content: original + '\n# Out-of-scope fixture edit\n' }) });
    }
  }
  const tool = used < plan.length ? plan[used] : undefined;
  response.writeHead(200, { 'Content-Type': 'text/event-stream' });
  const chunk = (delta, finish_reason) => response.write(`data: ${JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', choices: [{ index: 0, delta, finish_reason }] })}\n\n`);
  chunk({ role: 'assistant', ...(tool ? { tool_calls: [{ index: 0, id: `call_${used}`, type: 'function', function: tool }] } : { content: 'Finished fixture.' }) }, null);
  chunk({}, tool ? 'tool_calls' : 'stop'); response.end('data: [DONE]\n\n');
  assert.equal(manifest.max_turns, manifest.dm.max_steps);
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const run = args => new Promise((done, reject) => {
  const child = spawn(process.execPath, [join(isolated, 'main.mjs'), ...args], {
    cwd: directory, windowsHide: true, env: { ...process.env, FIXTURE_SECRET: 'fixture-secret-must-stay-private' },
  });
  let output = ''; child.stdout.on('data', x => output += x); child.stderr.on('data', x => output += x);
  child.on('error', reject); child.on('close', code => done({ code, output }));
});
try {
  const list = await run(['--pack', 'dm-code30', '--list']);
  assert.equal(list.code, 0, list.output);
  const ids = list.output.trim().split(/\r?\n/);
  assert.equal(ids.length, 30); assert.equal(ids[0], 'slugify_cleanup'); assert.equal(ids[15], 'config_precedence');
  const referenceIds = ['slugify_cleanup', 'config_precedence', 'cli_config_docs_contract', 'packaging_ci_contract'];
  const manifestPath = join(pack, 'tasks/slugify_cleanup/task.yaml');
  const originalManifest = await readFile(manifestPath, 'utf8');
  for (const change of ['missing-contract', 'empty-hidden-list']) {
    const broken = JSON.parse(originalManifest);
    if (change === 'missing-contract') delete broken.dm; else broken.dm.hidden_paths = [];
    try {
      await writeFile(manifestPath, JSON.stringify(broken));
      assert.equal((await run(['--pack', 'dm-code30', '--list'])).code, 2, 'Incomplete DM metadata must fail before scoring');
    } finally { await writeFile(manifestPath, originalManifest); }
  }
  for (const [model, selected, expected] of [
    ['pytest-filter', ['packaging_ci_contract'], 1],
    ['empty', [], 1], ['reference', referenceIds, 0],
    ['forbidden-edit', ['config_precedence'], 1], ['missing-required', ['cli_config_docs_contract'], 1],
    ['escape', ['slugify_cleanup'], 1], ['hidden', ['slugify_cleanup'], 1],
    ['syntax', ['slugify_cleanup'], 1], ['public-only', ['slugify_cleanup'], 1],
    ['missing-fixture', ['slugify_cleanup'], 2],
    ['exit-zero', ['slugify_cleanup'], 1], ['public-probe', ['slugify_cleanup'], 1],
    ['partial-exit', ['slugify_cleanup'], 1], ['partial-skip', ['slugify_cleanup'], 1],
  ]) {
    const fixture = join(pack, 'tasks/slugify_cleanup/hidden_tests/tests/test_hidden_slugify.py');
    if (model === 'missing-fixture') await rename(fixture, `${fixture}.saved`);
    let result;
    try { result = await run(['--pack', 'dm-code30', '--provider', 'fixture', '--model', model,
      '--base-url', `http://127.0.0.1:${server.address().port}/v1`, '--context-window', '65536',
      '--python', python, '--db', dbPath, ...selected.flatMap(id => ['--task', id])]);
    } finally { if (model === 'missing-fixture') await rename(`${fixture}.saved`, fixture); }
    await writeFile(join(directory, `${model}.log`), result.output);
    assert.equal(result.code, expected, result.output);
  }
  const db = new DatabaseSync(dbPath);
  const allRows = db.prepare('SELECT * FROM benchmark_tasks ORDER BY rowid').all();
  assert.equal(allRows[0].outcome, 'failed', 'Edited pytest config must not suppress hidden tests');
  assert.equal(JSON.parse(allRows[0].score_json).hidden_passed, false);
  const rows = allRows.slice(1);
  assert.equal(rows.length, 45);
  assert.deepEqual(rows.slice(0, 30).map(r => r.task_id), ids);
  assert(rows.slice(0, 30).every(r => r.passed === 0));
  assert(rows.slice(30, 34).every(r => r.passed === 1));
  assert(rows.slice(34, 36).every(r => r.outcome === 'scope_violation'));
  assert(rows.slice(34, 36).every(r => JSON.parse(r.score_json).hidden_passed));
  assert(rows.slice(36, 38).every(r => r.outcome === 'policy_violation'));
  assert(rows.slice(38, 40).every(r => r.outcome === 'failed'));
  assert.equal(rows[40].outcome, 'infra_error'); assert.equal(rows[40].passed, null);
  assert(rows.slice(41).every(r => r.passed === 0), 'Early exit and public execution probes must not pass');
  for (const row of [rows[41], rows[43], rows[44]]) assert.equal(JSON.parse(row.score_json).pytest_completion.complete, false, 'Partial or missing pytest execution must be incomplete');
  const probe = JSON.parse(await readFile(rows[42].artifact, 'utf8'));
  assert(!JSON.stringify(probe.messages).includes('test_punctuation_and_repeated_spaces'), 'Hidden assertions must not reach agent messages');
  const first = JSON.parse(await readFile(rows[30].artifact, 'utf8'));
  const testMessages = first.messages.filter(m => m.role === 'toolResult' && m.toolName === 'workspace_test');
  assert.equal(testMessages.length, 0, 'File-only DM exposes no public code execution');
  assert(!JSON.stringify(first.messages).includes('test_punctuation_and_repeated_spaces'));
  for (const row of rows) {
    const evidence = JSON.parse(await readFile(row.artifact, 'utf8'));
    assert.equal(evidence.protocol, 'pi-dm-v2');
    assert.equal(evidence.outcome, row.outcome);
    assert(!JSON.stringify(evidence).includes('fixture-secret-must-stay-private'));
  }
  db.close();
  const summary = { status: 'passed', cases: allRows.length, requests, database: dbPath, standalone: isolated };
  await writeFile(join(directory, 'verification.json'), JSON.stringify(summary, null, 2)); console.log(JSON.stringify(summary));
} finally { await new Promise(done => server.close(done)); }
