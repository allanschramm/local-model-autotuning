// Failure modes: repo coupling, rubric leakage, fake tool execution, lost partial
// scores, ignored forbidden calls, state leakage, startup leaks and paid web use.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { cp, mkdtemp, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import { parse } from 'yaml';

const python = process.env.BENCHMARK_PYTHON;
assert(python, 'Set BENCHMARK_PYTHON to the project venv Python');
const source = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directory = await mkdtemp(join(tmpdir(), 'pi-claw-e2e-'));
const isolated = join(directory, 'standalone');
await cp(source, isolated, { recursive: true, filter: p => !/[\\/](node_modules|artifacts)([\\/]|$)/.test(p) });
await cp(join(source, 'node_modules'), join(isolated, 'node_modules'), { recursive: true });
const dbPath = join(directory, 'results.db');
const pack = join(isolated, 'packs/claw-tiers');
const calls = {
  gmail_list_messages: {}, calendar_list_events: { date: '2026-03-02' },
  calendar_create_event: { title: 'Fixture meeting', start_time: '2026-03-03T14:00:00', end_time: '2026-03-03T14:30:00' },
  gmail_get_message: { message_id: 'msg_101' },
  gmail_save_draft: { to: 'client@bigcorp.com', subject: 'Project update', body: 'Fixture draft' },
  todo_list_tasks: {}, todo_update_task: { task_id: 'todo_001', priority: 'high' },
  contacts_search: { query: 'David' }, finance_list_transactions: {}, notes_list: {},
  notes_get: { note_id: 'note_001' }, kb_search: { query: 'VPN' },
  kb_get_article: { article_id: 'kb_001' }, helpdesk_get_ticket: { ticket_id: 'TK-1001' },
  helpdesk_update_ticket: { ticket_id: 'TK-1001', priority: 'high' },
};
let requests = 0;
const server = createServer(async (request, response) => {
  let raw = ''; for await (const chunk of request) raw += chunk;
  const body = JSON.parse(raw); requests++;
  assert(!raw.includes('scoring_components') && !raw.includes('reference_solution'), 'Hidden rubric leaked');
  assert(!raw.includes('fixture-secret-never-inherit'), 'Environment secret leaked');
  assert.equal(body.max_tokens ?? body.max_completion_tokens, 4096);
  const visible = body.messages.map(m => typeof m.content === 'string' ? m.content : (m.content ?? []).map(c => c.text ?? '').join('\n')).join('\n');
  const manifest = (await Promise.all((await readdir(join(pack, 'upstream/tasks'))).map(async id => {
    const task = parse(await readFile(join(pack, 'upstream/tasks', id, 'task.yaml'), 'utf8'));
    return task;
  }))).find(t => visible.includes(t.prompt.text.trim()));
  assert(manifest, 'Expected original task prompt');
  const used = body.messages.filter(m => m.role === 'tool').length;
  let plan = [];
  for (const component of manifest.scoring_components) {
    const check = component.check;
    if (check.type === 'tool_called') for (let n = 0; n < (check.min_calls ?? 1); n++) plan.push([check.tool_name, calls[check.tool_name]]);
  }
  if (!plan.length) plan.push(['web_search', { query: 'CloudPay' }]);
  if (body.model === 'unsafe') plan.push(['gmail_send_message', { to: 'fixture@example.test', subject: 'Fixture', body: 'Local mock only' }]);
  let tool;
  if (body.model !== 'empty' && used < plan.length) {
    const [name, args] = plan[used]; assert(args);
    tool = { id: `call_${used}`, type: 'function', function: { name, arguments: JSON.stringify(args) } };
  }
  const words = manifest.scoring_components.flatMap(c => c.check.keywords ?? c.check.categories ?? []).join(' ');
  const answer = body.model === 'empty' ? '' : words + ' Fixture summary.'.repeat(100);
  response.writeHead(200, { 'Content-Type': 'text/event-stream' });
  const chunk = (delta, finish_reason) => response.write(`data: ${JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', choices: [{ index: 0, delta, finish_reason }] })}\n\n`);
  chunk({ role: 'assistant', ...(tool ? { tool_calls: [{ index: 0, ...tool }] } : { content: answer }) }, null);
  chunk({}, tool ? 'tool_calls' : 'stop'); response.end('data: [DONE]\n\n');
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const run = args => new Promise((done, reject) => {
  const child = spawn(process.execPath, [join(isolated, 'main.mjs'), ...args], {
    cwd: directory, windowsHide: true, env: { ...process.env, FIXTURE_SECRET: 'fixture-secret-never-inherit' },
  });
  let output = ''; child.stdout.on('data', x => output += x); child.stderr.on('data', x => output += x);
  child.on('error', reject); child.on('close', code => done({ code, output }));
});
try {
  const security = await new Promise((done, reject) => {
    const child = spawn(python, [join(isolated, 'test/claw_web_security.py'), '--output', join(directory, 'web-security')], { windowsHide: true });
    let output = ''; child.stdout.on('data', x => output += x); child.stderr.on('data', x => output += x);
    child.on('error', reject); child.on('close', code => done({ code, output }));
  });
  await writeFile(join(directory, 'web-security.log'), security.output);
  assert.equal(security.code, 0, security.output);
  const historical = new DatabaseSync(dbPath);
  historical.exec(`CREATE TABLE benchmark_runs (run_id TEXT PRIMARY KEY,protocol TEXT,pi_version TEXT,provider TEXT,model TEXT,config_json TEXT,started_at TEXT,finished_at TEXT,status TEXT);
    CREATE TABLE benchmark_tasks (run_id TEXT,task_id TEXT,outcome TEXT,passed INTEGER,elapsed_sec REAL,task_hash TEXT,artifact TEXT,PRIMARY KEY(run_id,task_id));
    INSERT INTO benchmark_runs VALUES ('history','pi-claw-v1','1.4.0','fixture','history','{}','before','before','complete');
    INSERT INTO benchmark_tasks VALUES ('history','historical-task','passed',1,1.5,'frozen','historical-artifact');`);
  historical.close();
  const quick = await run(['--pack', 'claw-quick5', '--list']);
  assert.equal(quick.code, 0, quick.output);
  const ids = quick.output.trim().split(/\r?\n/); assert.equal(ids.length, 5);
  const full = await run(['--pack', 'claw-full15', '--list']);
  assert.equal(full.code, 0, full.output); assert.equal(full.output.trim().split(/\r?\n/).length, 15);
  for (const [model, corpus, task, expected] of [
    ['positive', 'claw-quick5', null, 0], ['empty', 'claw-quick5', null, 1],
    ['unsafe', 'claw-quick5', ids[0], 1], ['positive', 'claw-full15', null, 2],
    ['positive', 'claw-quick5', ids[0], 0],
  ]) {
    const result = await run(['--pack', corpus, '--provider', 'fixture', '--model', model,
      '--base-url', `http://127.0.0.1:${server.address().port}/v1`, '--context-window', '65536',
      '--python', python, '--db', dbPath, ...(task ? ['--task', task] : [])]);
    await writeFile(join(directory, `${model}-${corpus}-${task ?? 'all'}.log`), result.output);
    assert.equal(result.code, expected, result.output);
  }
  const fixture = join(pack, 'upstream/tasks', ids[0], 'fixtures/gmail/inbox.json');
  await rename(fixture, `${fixture}.saved`);
  try {
    const result = await run(['--pack', 'claw-quick5', '--provider', 'fixture', '--model', 'positive',
      '--base-url', `http://127.0.0.1:${server.address().port}/v1`, '--context-window', '65536',
      '--python', python, '--db', dbPath, '--task', ids[0]]);
    assert.equal(result.code, 2, result.output);
  } finally { await rename(`${fixture}.saved`, fixture); }
  const db = new DatabaseSync(dbPath);
  const history = db.prepare("SELECT * FROM benchmark_tasks WHERE run_id='history'").get();
  assert.equal(db.prepare("SELECT protocol FROM benchmark_runs WHERE run_id='history'").get().protocol, 'pi-claw-v1');
  assert.equal(history.passed, 1); assert.equal(history.task_hash, 'frozen'); assert.equal(history.elapsed_sec, 1.5);
  assert.equal(history.score, null); assert.equal(history.score_json, null);
  const rows = db.prepare("SELECT * FROM benchmark_tasks WHERE run_id!='history' ORDER BY rowid").all();
  assert.equal(rows.length, 28);
  assert(rows.slice(0, 5).every(r => r.passed === 1 && r.score === 1));
  assert(rows.slice(5, 10).every(r => r.passed === 0 && r.score === 0));
  assert.equal(rows[10].outcome, 'safety_violation'); assert.equal(rows[10].score, 1);
  assert(rows.slice(11, 21).every(r => r.passed === 1 && r.score === 1));
  assert(rows.slice(21, 26).every(r => r.outcome === 'infra_error' && r.passed === null && r.score === null));
  assert.equal(rows[26].passed, 1); assert.equal(rows[27].passed, null);
  for (const row of rows) {
    const evidence = JSON.parse(await readFile(row.artifact, 'utf8'));
    assert.equal(evidence.protocol, 'pi-claw-v2');
    assert(!JSON.stringify(evidence).includes('fixture-secret-never-inherit'));
    assert(evidence.services_closed, 'Service child must close before recording outcome');
    if (row.passed === 1) assert(evidence.tool_calls.every(c => c.status < 400 && !c.result.error), JSON.stringify(evidence.tool_calls));
  }
  const todoRuns = await Promise.all([rows[3], rows[14]].map(async row => JSON.parse(await readFile(row.artifact, 'utf8'))));
  assert.deepEqual(todoRuns[0].tool_calls.find(c => c.tool === 'todo_list_tasks').result, todoRuns[1].tool_calls.find(c => c.tool === 'todo_list_tasks').result, 'Task mutations must not survive into another run');
  db.close();
  const summary = { status: 'passed', cases: rows.length, requests, database: dbPath, standalone: isolated, security: 'web-security/verification.json' };
  await writeFile(join(directory, 'verification.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary));
} finally { await new Promise(done => server.close(done)); }
