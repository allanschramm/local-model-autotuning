"""Freeze legacy HumanEval or MBPP IDs as Pi workspace packs (base tests only)."""

import argparse
import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path
from textwrap import indent

HUMANEVAL_SHA256 = "828cd06f7b778dabda38314556656766b89ce42be25b578f362194936325813f"
MBPP_SHA256 = "ffb1a6f17bdb5661ef65958c07808726f1eb4b3e649bd18a11d161018cd87a43"
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
        "--pack", choices=("humaneval-base10", "mbpp-base10"), default="humaneval-base10"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if version("evalplus") != "0.3.1":
        raise ValueError("Reproduction requires evalplus==0.3.1")
    humaneval = args.pack == "humaneval-base10"
    if humaneval:
        from evalplus.data import get_human_eval_plus
        from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION

        entries = dict(list(get_human_eval_plus().items())[:10])
        dataset_version = HUMANEVAL_PLUS_VERSION
        expected_version, expected_hash = "v0.1.10", HUMANEVAL_SHA256
        expected_ids = [f"HumanEval/{i}" for i in range(10)]
        license_text = LICENSE
    else:
        from evalplus.data import get_mbpp_plus
        from evalplus.data.mbpp import MBPP_PLUS_VERSION

        entries = dict(list(get_mbpp_plus().items())[:10])
        dataset_version = MBPP_PLUS_VERSION
        expected_version, expected_hash = "v0.2.0", MBPP_SHA256
        expected_ids = [f"Mbpp/{i}" for i in (2, 3, 4, 6, 7, 8, 9, 11, 12, 14)]
        license_text = distribution("evalplus").read_text("LICENSE")
        if not license_text or "Apache License" not in license_text:
            raise ValueError("EvalPlus license is missing from the export environment")
    if dataset_version != expected_version:
        raise ValueError(f"Reproduction requires dataset {expected_version}")
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if hashlib.sha256(payload.encode()).hexdigest() != expected_hash:
        raise ValueError("Upstream subset drift; refusing to replace the frozen task contract")
    if list(entries) != expected_ids:
        raise ValueError("Legacy task order changed")
    target = args.output or Path(__file__).parent / "packs" / args.pack
    target.mkdir(parents=True, exist_ok=False)
    sources = {}
    oracles = {}
    for source_id, entry in entries.items():
        number = int(source_id.split("/")[1])
        task_id = f"{'humaneval' if humaneval else 'mbpp'}-{number:03d}"
        directory = target / "tasks" / task_id
        (directory / "workspace").mkdir(parents=True)
        (directory / "hidden_tests").mkdir()
        prompt = entry["prompt"]
        tests = entry["test"] if humaneval else entry["assertion"]
        stub = prompt + "    raise NotImplementedError('Complete this function')\n"
        issue = (
            f"Complete the Python function in solution.py. Task source: {source_id}. "
            "Preserve its signature and satisfy the specification in its docstring."
        )
        if not humaneval:
            stub = (
                prompt + f"\ndef {entry['entry_point']}(*args, **kwargs):\n"
                "    raise NotImplementedError('Complete this function')\n"
            )
            issue = (
                f"Implement {entry['entry_point']} in solution.py. Task source: {source_id}. "
                "Follow the specification and example in the module docstring."
            )
        task = {
            "id": task_id,
            "source_issue": source_id,
            "max_turns": 12,
            "allowlist": ["solution.py"],
            "issue": issue,
        }
        (directory / "task.yaml").write_text(
            json.dumps(task, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (directory / "workspace/solution.py").write_text(
            stub,
            encoding="utf-8",
            newline="\n",
        )
        verifier = (
            f"from solution import {entry['entry_point']} as _candidate\n\n"
            + tests
            + "\n\ndef test_original_humaneval_assertions():\n    check(_candidate)\n"
        )
        if not humaneval:
            verifier = (
                f"from solution import {entry['entry_point']}\n\n\ndef test_original_mbpp_assertions():\n"
                + indent(tests, "    ")
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
        "pack": args.pack,
        "source": "https://github.com/evalplus/evalplus/releases/tag/v0.1.10"
        if humaneval
        else "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
        "upstream": "https://github.com/openai/human-eval"
        if humaneval
        else "https://github.com/google-research/google-research/tree/master/mbpp",
        "evalplus_version": "0.3.1",
        "dataset_version": dataset_version,
        "source_subset_sha256": expected_hash,
        "verification": "Original base check(candidate) only; plus_input is not evaluated"
        if humaneval
        else "Original base assertion field only; plus_input is not evaluated",
        "tasks": sources,
    }
    (target / "source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (target / "oracles.json").write_text(
        json.dumps(oracles, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (target / "LICENSE").write_text(license_text, encoding="utf-8", newline="\n")
    if not humaneval:
        (target / "NOTICE.md").write_text(
            "# MBPP attribution\n\n"
            "Original work: Mostly Basic Python Problems, Austin et al. (Google Research, 2021).\n"
            "Source: https://github.com/google-research/google-research/tree/master/mbpp\n"
            "Dataset card and original license: https://huggingface.co/datasets/google-research-datasets/mbpp (CC BY 4.0).\n"
            "License link: https://creativecommons.org/licenses/by/4.0/\n\n"
            "Adapted source: EvalPlus MBPP+ v0.2.0, https://github.com/evalplus/evalplus (Apache-2.0; see LICENSE).\n"
            "Changes: first ten legacy-selected tasks, workspace stubs and hidden pytest wrappers around base assertions; no plus-input execution.\n",
            encoding="utf-8",
            newline="\n",
        )
    print(f"Frozen {len(entries)} {args.pack} tasks: {target}")


if __name__ == "__main__":
    main()
