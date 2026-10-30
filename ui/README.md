# Auto-tuning dashboard

Run python -m ui with the project venv and open http://127.0.0.1:18765.

The read-only localhost view shows Baseline, recent tuning attempts, context, TPS, PPL, fixed reference PPL, frontier membership and the latest server log. It neither changes Baseline nor controls processes.

An empty or historical-quality-only database produces an empty tuning view; tables are never migrated or rewritten. [AGENTS.md](AGENTS.md) owns the contract.
