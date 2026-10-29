"""Freeze legacy HumanEval IDs as a standalone Pi workspace pack (base tests only)."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

SUBSET_SHA256 = "828cd06f7b778dabda38314556656766b89ce42be25b578f362194936325813f"
LICENSE = """The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "packs/humaneval-base10"
    )
    args = parser.parse_args()
    from evalplus.data import get_human_eval_plus
    from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION

    if version("evalplus") != "0.3.1" or HUMANEVAL_PLUS_VERSION != "v0.1.10":
        raise ValueError("Reproduction requires evalplus==0.3.1 and dataset v0.1.10")
    entries = dict(list(get_human_eval_plus().items())[:10])
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if hashlib.sha256(payload.encode()).hexdigest() != SUBSET_SHA256:
        raise ValueError("Upstream subset drift; refusing to replace the frozen task contract")
    if list(entries) != [f"HumanEval/{i}" for i in range(10)]:
        raise ValueError("Legacy task order changed")
    target = args.output
    target.mkdir(parents=True, exist_ok=False)
    sources = {}
    oracles = {}
    for index, (source_id, entry) in enumerate(entries.items()):
        task_id = f"humaneval-{index:03d}"
        directory = target / "tasks" / task_id
        (directory / "workspace").mkdir(parents=True)
        (directory / "hidden_tests").mkdir()
        prompt = entry["prompt"]
        tests = entry["test"]
        task = {
            "id": task_id,
            "source_issue": source_id,
            "max_turns": 12,
            "allowlist": ["solution.py"],
            "issue": f"Complete the Python function in solution.py. Task source: {source_id}. "
            "Preserve its signature and satisfy the specification in its docstring.",
        }
        (directory / "task.yaml").write_text(
            json.dumps(task, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (directory / "workspace/solution.py").write_text(
            prompt + "    raise NotImplementedError('Complete this function')\n",
            encoding="utf-8",
            newline="\n",
        )
        verifier = (
            f"from solution import {entry['entry_point']} as _candidate\n\n"
            + tests
            + "\n\ndef test_original_humaneval_assertions():\n    check(_candidate)\n"
        )
        (directory / "hidden_tests/test_original.py").write_text(
            verifier, encoding="utf-8", newline="\n"
        )
        oracles[source_id] = prompt + entry["canonical_solution"]
        sources[task_id] = {
            "source_id": source_id,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "original_test_sha256": hashlib.sha256(tests.encode()).hexdigest(),
            "verifier_sha256": hashlib.sha256(verifier.encode()).hexdigest(),
        }
    provenance = {
        "pack": "humaneval-base10",
        "source": "https://github.com/evalplus/evalplus/releases/tag/v0.1.10",
        "upstream": "https://github.com/openai/human-eval",
        "evalplus_version": "0.3.1",
        "dataset_version": "v0.1.10",
        "source_subset_sha256": SUBSET_SHA256,
        "verification": "Original base check(candidate) only; plus_input is not evaluated",
        "tasks": sources,
    }
    (target / "source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (target / "oracles.json").write_text(
        json.dumps(oracles, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (target / "LICENSE").write_text(LICENSE, encoding="utf-8", newline="\n")
    print(f"Frozen {len(entries)} HumanEval base tasks: {target}")


if __name__ == "__main__":
    main()
