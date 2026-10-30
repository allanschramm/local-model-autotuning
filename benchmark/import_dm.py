"""Freeze DM-Code-Agent's pinned 30-task coding/maintenance contract for Pi."""

import argparse
import ast
import hashlib
import json
import sys
import types
from pathlib import Path

REVISION = "e4530be37e6f972d9735381e524cab4b5f732ddd"
FILES = {
    "tasks.py": "329a587c70a257e937598e08ec5d71d8e3dff1b73f981a8a3b530206d70c234f",
    "models.py": "99cbb69f9876185791a8f7629c29bce8ab7dcc28cb93ae0fb371e890e15d6ba1",
    "LICENSE": "a3d1b7c3780ba82745de850a64423f29d22d6570e2f519197fe5f276ea9769af",
    "test_coding_benchmarks.py": "3ecf4fba5fbbc91ce4877568dcf30ce36211b131afb92a8371f926acde1b46d7",
}
CONTRACT_SHA256 = "d0ede70da0ea2e1a000ff2487cfe89736e98453aa9154643be1ed551726c063f"
REFERENCES = {
    "test_hidden_tests_pass_for_known_slugify_solution": "slugify_cleanup",
    "test_maintenance_hidden_tests_pass_for_known_config_solution": "config_precedence",
    "test_maintenance_hidden_tests_pass_for_known_cli_docs_solution": "cli_config_docs_contract",
    "test_maintenance_hidden_tests_pass_for_known_packaging_ci_solution": "packaging_ci_contract",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def relative(value):
    if (
        not isinstance(value, str)
        or "\\" in value
        or ":" in value
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError(f"Invalid frozen relative path: {value!r}")
    return value


def reference_files(source):
    oracles = {}

    def path(node):
        if isinstance(node, ast.Name) and node.id == "tmp_path":
            return ""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return "/".join(filter(None, (path(node.left), path(node.right))))
        raise ValueError("Unknown upstream reference path expression")

    for function in ast.parse(source).body:
        if not isinstance(function, ast.FunctionDef) or function.name not in REFERENCES:
            continue
        values, files = {}, {}
        for statement in function.body:
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
                values[statement.targets[0].id] = statement.value.value
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "write_text":
                continue
            argument = call.args[0]
            if isinstance(argument, ast.JoinedStr):
                text = "".join(
                    part.value if isinstance(part, ast.Constant) else str(values[part.value.id])
                    for part in argument.values
                )
            else:
                text = ast.literal_eval(argument)
            files[relative(path(call.func.value.args[0]))] = text
        oracles[REFERENCES[function.name]] = files
    if set(oracles) != set(REFERENCES.values()):
        raise ValueError("Missing upstream reference controls")
    return oracles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Directory containing four pinned source files"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "packs/dm-code30")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    raw = {name: (args.source / name).read_bytes() for name in FILES}
    for name, expected in FILES.items():
        if digest(raw[name]) != expected:
            raise ValueError(f"Pinned DM source hash mismatch: {name}")
    package = types.ModuleType("_dm_export")
    package.__path__ = []
    sys.modules[package.__name__] = package
    # Execute only hash-verified stdlib-only data modules, without installing the agent.
    for name in ("models", "tasks"):
        module = types.ModuleType(f"_dm_export.{name}")
        sys.modules[module.__name__] = module
        exec(compile(raw[f"{name}.py"], f"<frozen DM {name}>", "exec"), module.__dict__)
    rows = [vars(task) for task in module.get_benchmark_tasks("all")]
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if len(rows) != 30 or digest(encoded) != CONTRACT_SHA256:
        raise ValueError("DM task contract/order mismatch")
    oracles = reference_files(raw["test_coding_benchmarks.py"].decode("utf-8"))
    args.output.mkdir(parents=True)

    def write(path, value):
        target = args.output / relative(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8", newline="\n")

    for row in rows:
        prefix = f"tasks/{relative(row['task_id'])}"
        for name, contents in row["setup_files"].items():
            write(f"{prefix}/workspace/{relative(name)}", contents)
        for name, contents in row["hidden_files"].items():
            write(f"{prefix}/hidden_tests/{relative(name)}", contents)
        for command in (row["visible_test_command"], row["hidden_test_command"]):
            if command != ["{python}", "-m", "pytest", "-q"]:
                raise ValueError("Unexpected command in pinned DM contract")
        contract = {
            key: row[key]
            for key in (
                "max_steps",
                "tags",
                "visible_test_command",
                "hidden_test_command",
                "allowed_changed_files",
                "required_changed_files",
            )
        }
        contract["hidden_paths"] = list(row["hidden_files"])
        issue = f"Task source: {row['task_id']}.\n\n" + row["prompt"]
        if row["allowed_changed_files"]:
            issue += (
                "\n\nOnly these files may be changed: "
                + ", ".join(row["allowed_changed_files"])
                + "."
            )
        if row["required_changed_files"]:
            issue += (
                "\nThese files must be changed: " + ", ".join(row["required_changed_files"]) + "."
            )
        write(
            f"{prefix}/task.yaml",
            json.dumps(
                {
                    "id": row["task_id"],
                    "name": row["name"],
                    "issue": issue,
                    "max_turns": row["max_steps"],
                    "allowlist": list(row["setup_files"]),
                    "dm": contract,
                },
                indent=2,
            )
            + "\n",
        )
    write("oracles.json", json.dumps(oracles, indent=2) + "\n")
    write(
        "source.json",
        json.dumps(
            {
                "upstream": "https://github.com/hwfengcs/DM-Code-Agent",
                "revision": REVISION,
                "source_sha256": FILES,
                "contract_sha256": CONTRACT_SHA256,
                "task_ids": [r["task_id"] for r in rows],
                "coding": [r["task_id"] for r in rows[:15]],
                "maintenance": [r["task_id"] for r in rows[15:]],
                "reference_functions": REFERENCES,
            },
            indent=2,
        )
        + "\n",
    )
    for name, contents in raw.items():
        write(f"upstream/{name}", contents.decode("utf-8"))
    write(
        "NOTICE.md",
        "# DM-Code-Agent attribution\n\nDM-Code-Agent, MIT; see upstream/LICENSE.\nSource: https://github.com/hwfengcs/DM-Code-Agent\n\nChanges: frozen 30-task Pi file/test protocol with original setup/hidden files, commands, step limits and changed-file constraints. Four reference controls are extracted from pinned upstream tests; they are not a gold solution set for all 30 tasks.\n",
    )
    print(f"Frozen {len(rows)} DM tasks: {args.output}")


if __name__ == "__main__":
    main()
