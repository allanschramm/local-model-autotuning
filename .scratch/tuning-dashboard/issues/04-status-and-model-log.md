# 04 — Status “Em execução” + log do modelo

**What to build:** The dashboard shows whether a Trial's model server looks active (Idle vs Em execução) based on recent growth of `llama_server.log`, and a live-ish tail of that log so the human can watch the model/server work and spot errors without the agent terminal.

**Blocked by:** 01 — Shell do dashboard localhost

**Status:** ready-for-agent

- [ ] Status badge is Em execução when `llama_server.log` mtime/size grew recently (~10s window); otherwise Idle
- [ ] Log panel tails the current/last Trial log at `autoresearch/runners/llama_server.log`
- [ ] Missing log file shows a clear empty state (not a crash)
- [ ] UI does not start/stop `llama-server` or any Search process; log is display-only
