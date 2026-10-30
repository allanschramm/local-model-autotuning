"""Freeze the existing Claw tier selection, services and legacy completion scorer."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

REVISION = "18f4e07d5c03b24dcd5ebfbb705f234a4ea844ff"
SCORE_SHA256 = "553636565324a78301cea372b2fab5c1554127158e1713499466b57e2f1db44a"
TASKS = [
    "T002_email_triage",
    "T004_calendar_scheduling",
    "T006_email_reply_draft",
    "T008_todo_management",
    "T010_contact_lookup",
    "T012_expense_report",
    "T014_meeting_notes",
    "T016_kb_search",
    "T018_ticket_triage",
    "T044_service_outage_research",
    "T046_cve_research",
    "T048_oss_comparison",
    "T050_regulatory_research",
    "T053_finance_us_steel_merger",
    "T054_finance_nflx_arppu_trend",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Claw checkout containing pinned commit"
    )
    parser.add_argument("--legacy-scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "packs/claw-tiers")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    import yaml

    source = args.legacy_scorer.read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "score_task"
    )
    scorer = ast.get_source_segment(source, node)
    if hashlib.sha256(scorer.encode()).hexdigest() != SCORE_SHA256:
        raise ValueError("Legacy completion scorer hash mismatch")

    def git(*command):
        return subprocess.check_output(["git", "-C", str(args.source), *command])

    paths = git("ls-tree", "-r", "--name-only", REVISION).decode().splitlines()
    snapshots = {}
    adapter_sources = {}
    services = {"_base.py"}
    for task_id in TASKS:
        raw = git("show", f"{REVISION}:tasks/{task_id}/task.yaml")
        task = yaml.safe_load(raw)
        if task["task_id"] != task_id:
            raise ValueError("Task identity mismatch")
        for component in task["scoring_components"]:
            if component["check"]["type"] not in (
                "tool_called",
                "keywords_present",
                "categories_present",
                "min_length",
            ):
                raise ValueError("Unsupported completion rule")
        for service in task["services"]:
            services.add(service["name"] + "/")
        for path in paths:
            if path == f"tasks/{task_id}/task.yaml" or path.startswith(
                f"tasks/{task_id}/fixtures/"
            ):
                snapshots[path] = git("show", f"{REVISION}:{path}")
    for path in paths:
        relative = path.removeprefix("mock_services/")
        if (
            path.startswith("mock_services/")
            and path.endswith(".py")
            and any(relative == s or relative.startswith(s) for s in services)
        ):
            raw = git("show", f"{REVISION}:{path}")
            if path == "mock_services/web_real/server.py":
                adapter_sources[path] = hashlib.sha256(raw).hexdigest()
            else:
                snapshots[path] = raw
    snapshots["LICENSE"] = git("show", f"{REVISION}:LICENSE")
    args.output.mkdir(parents=True)
    hashes = {}
    for path, raw in sorted(snapshots.items()):
        destination = args.output / "upstream" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        hashes[path] = hashlib.sha256(raw).hexdigest()
    (args.output / "legacy_score.py").write_text(
        "from pathlib import Path\n\n" + scorer + "\n", encoding="utf-8", newline="\n"
    )
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "upstream": "https://github.com/claw-eval/claw-eval",
                "checkout_commit": REVISION,
                "quick": TASKS[:5],
                "full": TASKS,
                "upstream_files": hashes,
                "adapter_sources": adapter_sources,
                "legacy_score_sha256": SCORE_SHA256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output / "NOTICE.md").write_text(
        "# Claw tier snapshot\n\nClaw-Eval (2026), MIT; see upstream/LICENSE.\n"
        "Source: https://github.com/claw-eval/claw-eval\n"
        "Pinned checkout includes local upstream integration. Selected task/fixture bytes are unchanged.\n"
        "The live-web server is maintained as ../../claw_web.py with SSRF protection, bounded extraction and generic errors; adapter_sources records its original hash.\n"
        "Adaptation: Pi tools, per-task ASGI process, legacy deterministic completion scorer and separate safety diagnostics/veto.\n"
        "This is a single-trial legacy-selected slice, not the official Pass^3 or full upstream grader.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Frozen {len(TASKS)} tasks and {len(hashes)} upstream files: {args.output}")


if __name__ == "__main__":
    main()
