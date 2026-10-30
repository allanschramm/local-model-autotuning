"""Freeze the legacy first ten BigCodeBench Hard v0.1.4 tasks for Pi."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

REVISION = "298d2cc7b96612e15e47313c3603ee124cee0c1f"
SUBSET_SHA256 = "7a44b29b7578abde447ccf0385b746413f6bd6abe1bde3db0d3ed96019f1f0d8"
IDS = [13, 15, 17, 19, 34, 37, 82, 89, 92, 93]
REQUIREMENTS = """psutil==7.2.2
matplotlib==3.11.2
wordcloud==1.9.6
pandas==3.1.0
scikit-learn==1.9.1
seaborn==0.13.2
numpy==2.5.4
scipy==1.18.1
Flask==3.1.3
Flask-Login==0.6.3
Flask-WTF==1.3.0
WTForms==3.2.2
Werkzeug==3.1.8
"""
DEPENDENCIES = {
    13: [],
    15: [],
    17: ["psutil"],
    19: [],
    34: ["matplotlib", "wordcloud"],
    37: ["pandas", "sklearn", "seaborn", "matplotlib"],
    82: ["flask", "flask_login", "flask_wtf", "wtforms", "werkzeug"],
    89: ["numpy", "matplotlib", "scipy", "sklearn"],
    92: ["pandas", "matplotlib", "sklearn", "numpy"],
    93: ["numpy", "pandas", "matplotlib", "sklearn"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned v0.1.4 parquet")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "packs/bigcode-hard10"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    from datasets import Dataset

    rows = list(Dataset.from_parquet(str(args.source)).select(range(10)))
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != SUBSET_SHA256 or [r["task_id"] for r in rows] != [
        f"BigCodeBench/{i}" for i in IDS
    ]:
        raise ValueError("BigCodeBench v0.1.4 subset hash/order mismatch")
    verifier = Path(__file__).with_name("bigcode_verifier.py").read_text(encoding="utf-8")
    target = args.output
    target.mkdir(parents=True)
    sources, oracles = {}, {}

    def write(path, value):
        (target / path).write_text(value, encoding="utf-8", newline="\n")

    for number, entry in zip(IDS, rows, strict=True):
        task_id = f"bigcode-{number:03}"
        directory = target / "tasks" / task_id
        (directory / "workspace").mkdir(parents=True)
        (directory / "hidden_tests").mkdir()
        base = f"tasks/{task_id}"
        manifest = {
            "id": task_id,
            "source_issue": entry["task_id"],
            "max_turns": 12,
            "allowlist": ["solution.py"],
            "issue": f"Task source: {entry['task_id']}. Implement the complete solution in solution.py.\n\n"
            + entry["instruct_prompt"],
        }
        write(f"{base}/task.yaml", json.dumps(manifest, indent=2) + "\n")
        write(
            f"{base}/workspace/solution.py",
            entry["code_prompt"] + "    raise NotImplementedError('Complete this function')\n",
        )
        write(f"{base}/hidden_tests/test_bigcode.py", verifier)
        write(f"{base}/hidden_tests/original_tests.txt", entry["test"])
        write(f"{base}/hidden_tests/dependencies.json", json.dumps(DEPENDENCIES[number]) + "\n")
        oracles[entry["task_id"]] = entry["code_prompt"] + entry["canonical_solution"]
        sources[task_id] = {
            "source_id": entry["task_id"],
            "libs": ast.literal_eval(entry["libs"]),
            "dependencies": DEPENDENCIES[number],
            "unittest_cases": sum(
                isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                for n in ast.walk(ast.parse(entry["test"]))
            ),
            **{
                key + "_sha256": hashlib.sha256(entry[key].encode()).hexdigest()
                for key in ("instruct_prompt", "code_prompt", "test")
            },
            "verifier_sha256": hashlib.sha256(verifier.encode()).hexdigest(),
        }
    write(
        "source.json",
        json.dumps(
            {
                "pack": "bigcode-hard10",
                "dataset": "bigcode/bigcodebench-hard",
                "split": "v0.1.4",
                "revision": REVISION,
                "source_subset_sha256": SUBSET_SHA256,
                "verification": "Original unittest TestCases in shared solution namespace; legacy Windows shell-message compatibility; headless plots",
                "tasks": sources,
            },
            indent=2,
        )
        + "\n",
    )
    write("oracles.json", json.dumps(oracles, indent=2) + "\n")
    write("requirements.txt", REQUIREMENTS)
    # Reuse the standard Apache text without EvalPlus's package-specific appendix.
    apache = (Path(__file__).parent / "packs/mbpp-base10/LICENSE").read_text(encoding="utf-8")
    write("LICENSE", apache.split("\n---", 1)[0].rstrip() + "\n")
    write(
        "NOTICE.md",
        "# BigCodeBench attribution\n\nBigCodeBench, Zhuo et al. (2024).\n"
        "Source: https://huggingface.co/datasets/bigcode/bigcodebench-hard\n"
        "Original dataset declares Apache-2.0: https://huggingface.co/datasets/bigcode/bigcodebench\n"
        "Upstream: https://github.com/bigcode-project/bigcodebench\n\n"
        "Changes: first ten legacy-selected v0.1.4 tasks, Pi workspace stubs, hidden unittest bridge and dependency preflight. Original test text is unchanged.\n",
    )
    print(f"Frozen {len(rows)} BigCodeBench tasks: {target}")


if __name__ == "__main__":
    main()
