"""Freeze the legacy ten-task LiveCodeBench v6 stdin slice from a pinned source."""

import argparse
import base64
import gzip
import hashlib
import io
import json
import pickletools
import zlib
from pathlib import Path

SOURCE_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
SOURCE_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
QUESTION_IDS = (
    "abc387_b",
    "abc387_f",
    "abc387_a",
    "abc387_c",
    "abc388_d",
    "abc388_b",
    "abc388_g",
    "abc388_c",
    "abc388_a",
    "abc388_f",
)
MAX_DECODED_BYTES = 64 * 1024 * 1024


def decode_private_cases(raw: str) -> list[dict]:
    """Read one literal JSON string from pickle's wire format without unpickling."""
    try:
        compressed = base64.b64decode(raw, validate=True)
        decoder = zlib.decompressobj()
        data = decoder.decompress(compressed, MAX_DECODED_BYTES + 1)
        if len(data) > MAX_DECODED_BYTES or not decoder.eof or decoder.unused_data:
            raise ValueError("Oversized or incomplete compressed cases")
        operations = list(pickletools.genops(data))
        allowed = {
            "PROTO",
            "FRAME",
            "BINUNICODE",
            "BINUNICODE8",
            "SHORT_BINUNICODE",
            "MEMOIZE",
            "BINPUT",
            "LONG_BINPUT",
            "STOP",
        }
        if not operations or any(op.name not in allowed for op, _, _ in operations):
            raise ValueError("Only literal-string pickle opcodes are supported")
        if operations[-1][0].name != "STOP" or operations[-1][2] + 1 != len(data):
            raise ValueError("Trailing pickle data")
        strings = [
            arg
            for op, arg, _ in operations
            if op.name in {"BINUNICODE", "BINUNICODE8", "SHORT_BINUNICODE"}
        ]
        if len(strings) != 1:
            raise ValueError("Expected one literal JSON string")
        cases = json.loads(strings[0])
        if (
            not isinstance(cases, list)
            or not cases
            or any(
                not isinstance(case, dict)
                or case.get("testtype") != "stdin"
                or not isinstance(case.get("input"), str)
                or not isinstance(case.get("output"), str)
                for case in cases
            )
        ):
            raise ValueError("Expected nonempty stdin input/output cases")
        return cases
    except (ValueError, TypeError, zlib.error, UnicodeError) as exc:
        raise ValueError(f"Invalid private cases: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned test6.jsonl file")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "packs/lcb-v6-stdin10"
    )
    args = parser.parse_args()
    with args.source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Source SHA256 drift; refusing to parse private case serialization")
    entries = []
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            cases = decode_private_cases(entry["private_test_cases"])
            if len(cases) != 40:
                raise ValueError("Expected the pinned 40 private cases per task")
            entries.append((entry, cases))
            if len(entries) == 10:
                break
    if tuple(entry["question_id"] for entry, _ in entries) != QUESTION_IDS:
        raise ValueError("Legacy task selection/order changed")
    template = Path(__file__).with_name("stdin_verifier.py").read_text(encoding="utf-8")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for index, (entry, cases) in enumerate(entries):
        task_id = f"lcb-{index:03d}-{entry['question_id']}"
        directory = args.output / "tasks" / task_id
        (directory / "workspace").mkdir(parents=True)
        (directory / "hidden_tests").mkdir()
        issue = f"# {entry['question_title']}\n\n{entry['question_content']}"
        if entry.get("starter_code"):
            issue += f"\n\nStarter code:\n{entry['starter_code']}"
        issue += "\n\nWrite a complete Python program in solution.py that reads standard input and writes the required answer to standard output."
        task = {
            "id": task_id,
            "source_issue": entry["question_id"],
            "max_turns": 12,
            "allowlist": ["solution.py"],
            "issue": issue,
        }
        (directory / "task.yaml").write_text(
            json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        (directory / "workspace/solution.py").write_text(
            "raise NotImplementedError('Implement the stdin/stdout program')\n",
            encoding="utf-8",
            newline="\n",
        )
        (directory / "hidden_tests/test_stdin.py").write_text(
            template, encoding="utf-8", newline="\n"
        )
        logical_cases = json.dumps(
            cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as archive:
            archive.write(logical_cases)
        bundle = buffer.getvalue()
        (directory / "hidden_tests/private_cases.json.gz").write_bytes(bundle)
        sources[task_id] = {
            "question_id": entry["question_id"],
            "title": entry["question_title"],
            "platform": entry["platform"],
            "difficulty": entry["difficulty"],
            "contest_id": entry["contest_id"],
            "contest_date": entry["contest_date"],
            "question_sha256": hashlib.sha256(entry["question_content"].encode()).hexdigest(),
            "private_count": len(cases),
            "public_count": len(json.loads(entry["public_test_cases"])),
            "private_logical_sha256": hashlib.sha256(logical_cases).hexdigest(),
            "private_archive_sha256": hashlib.sha256(bundle).hexdigest(),
            "verifier_sha256": hashlib.sha256(template.encode()).hexdigest(),
        }
    provenance = {
        "pack": "lcb-v6-stdin10",
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "source": f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{SOURCE_REVISION}/test6.jsonl",
        "selection": "First ten legacy test6.jsonl records with private tests, in source order; not the full release_v6 aggregate",
        "verification": "All 40 private stdin cases; fresh cwd/process per case; exact stdout after rstrip newline; no automatic command timeout",
        "public_cases": "Examples remain in the original statement; not separately scored, matching the legacy selection",
        "decoder": "Pinned source hash plus literal string opcode extraction; no pickle execution",
        "tasks": sources,
    }
    (args.output / "source.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (args.output / "NOTICE.md").write_text(
        "# LiveCodeBench attribution\n\n"
        "Source: LiveCodeBench code_generation_lite, test6.jsonl at revision "
        + SOURCE_REVISION
        + ".\n"
        "Dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite\n"
        "Dataset card: https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/"
        + SOURCE_REVISION
        + "/README.md\n"
        "The source dataset card declares `license: cc` without a variant; preserve the source attribution and do not infer an MIT dataset license from the separate LiveCodeBench harness.\n"
        "Problems originate from AtCoder; source question/contest IDs and dates are retained in source.json.\n"
        "Changes: first ten legacy records, Pi workspace instructions and private pytest stdin verification; gzip case storage. No gold solutions are supplied by this dataset.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Frozen {len(entries)} LiveCodeBench tasks / {sum(len(cases) for _, cases in entries)} private cases: {args.output}"
    )


if __name__ == "__main__":
    main()
