import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { parse } from 'yaml';

export const CLAW_PROTOCOL = 'pi-claw-v2';
export const CLAW_SYSTEM = 'You are an AI assistant with access to tools. Use the tools to complete the task. When you have gathered all needed information, provide a final answer without tool calls. Call tools one at a time, wait for results, then decide the next step.';

export async function loadClawTasks(root, pack, selected) {
  const provenance = JSON.parse(await readFile(join(root, 'source.json'), 'utf8'));
  // Include the complete frozen runtime in task identity, not just its prompt.
  const hash = createHash('sha256');
  const errors = [];
  for (const [path, expected] of Object.entries(provenance.upstream_files).sort()) {
    try {
      const bytes = await readFile(join(root, 'upstream', path));
      if (createHash('sha256').update(bytes).digest('hex') !== expected) errors.push(`Claw snapshot hash mismatch: ${path}`);
    } catch { errors.push(`Claw snapshot file missing: ${path}`); }
    hash.update(path + '\0' + expected);
  }
  hash.update(await readFile(join(root, 'legacy_score.py')));
  hash.update(await readFile(join(root, '../../claw_bridge.py')));
  hash.update(await readFile(join(root, '../../claw.mjs')));
  hash.update(await readFile(join(root, '../../claw_web.py')));
  const sharedHash = hash.digest('hex');
  const ids = pack === 'claw-quick5' ? provenance.quick : provenance.full;
  if (selected?.some(id => !ids.includes(id))) throw new Error('Unknown task ID in Claw tier');
  return Promise.all(ids.filter(id => !selected?.length || selected.includes(id)).map(async id => {
    const task = parse(await readFile(join(root, 'upstream/tasks', id, 'task.yaml'), 'utf8'));
    return { id, claw: task, directory: root, max_turns: task.environment.max_turns,
      loadError: errors.length ? errors.join('; ') : undefined,
      protocol: CLAW_PROTOCOL, hash: createHash('sha256').update(sharedHash + '\0' + id).digest('hex') };
  }));
}

export async function startClaw(task, python, cwd, allowLiveWeb) {
  const env = Object.fromEntries(Object.entries({
    PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: process.env.TEMP, TMP: process.env.TMP,
    PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1',
    ...(allowLiveWeb ? { SERP_DEV_KEY: process.env.SERP_DEV_KEY } : {}),
  }).filter(([, value]) => value !== undefined));
  const child = spawn(python, [join(task.directory, '../../claw_bridge.py'), task.directory, task.id, allowLiveWeb ? 'live' : 'offline'], {
    cwd, env, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
  });
  let pending;
  let failure;
  let stderr = '';
  const calls = [];
  const fail = error => { failure = error; pending?.reject(error); pending = undefined; };
  const closed = new Promise(done => {
    child.on('error', error => { fail(error); done(); });
    child.on('close', code => { if (pending || code) fail(new Error(`Claw service process exited (${code}): ${stderr.slice(-2000)}`)); done(); });
  });
  child.stderr.on('data', bytes => stderr = (stderr + bytes).slice(-8000));
  child.stdin.on('error', fail);
  const lines = createInterface({ input: child.stdout });
  lines.on('line', line => {
    try {
      const value = JSON.parse(line);
      if (!value.ok) { fail(new Error(value.error)); return; }
      const current = pending; pending = undefined; current?.resolve(value);
    } catch (error) { fail(error); }
  });
  const receive = () => new Promise((resolve, reject) => {
    if (failure) reject(failure); else pending = { resolve, reject };
  });
  let queue = Promise.resolve();
  const request = value => {
    const next = queue.then(async () => {
      const response = receive();
      if (!failure) child.stdin.write(JSON.stringify(value) + '\n');
      return (await response).result;
    });
    queue = next.catch(() => {});
    return next;
  };
  const close = async () => { child.stdin.end(); await closed; lines.close(); };
  try { await receive(); } catch (error) { await close(); throw error; }
  const tools = task.claw.tools.map(tool => ({
    name: tool.name, label: tool.name, description: tool.description, parameters: tool.input_schema,
    execute: async (_id, args) => {
      const call = await request({ op: 'call', name: tool.name, arguments: args });
      calls.push(call);
      const raw = JSON.stringify(call.result);
      const text = raw.length > 10000 ? raw.slice(0, 5000) + `\n... [truncated ${raw.length - 10000} characters] ...\n` + raw.slice(-5000) : raw;
      return { content: [{ type: 'text', text }], details: {} };
    },
  }));
  return { tools, calls, close, get error() { return failure; }, score: text => request({ op: 'score', text }) };
}
