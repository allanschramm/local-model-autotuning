import base64
import json
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoresearch.benchmarks import benchmark_coding


def _lcb_pickled_json(payload_str: str) -> str:
    """Encode a JSON string the way LiveCodeBench v6 encodes its private test cases."""
    return base64.b64encode(zlib.compress(pickle.dumps(payload_str))).decode()


class TestBenchmarkCoding(unittest.TestCase):
    # ------------------------------------------------------------------ weights

    @patch("autoresearch.benchmarks.benchmark_coding.run_coding_eval")
    def test_run_benchmark_weights(self, mock_eval):
        """
        val_score = 0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode
        mock_eval returns (HE, MBPP, LCB, BigCode) pass rates.
        """
        mock_eval.side_effect = [
            (0.6, 100, 10.0),  # HumanEval -> val_pass2
            (0.4, 200, 5.0),  # MBPP     -> val_pass3
            (0.8, 300, 8.0),  # LCB      -> val_pass1
            (0.2, 50, 4.0),  # BigCode  -> val_pass4
        ]

        from autoresearch.core.llama_client import LlamaClient

        client = MagicMock(spec=LlamaClient)
        client.port = 1234

        result = benchmark_coding.run_benchmark(client, task_limit=10)

        # val_pass1=LCB, val_pass2=HE, val_pass3=MBPP, val_pass4=BigCode
        self.assertAlmostEqual(result.val_pass1, 0.8, places=4)
        self.assertAlmostEqual(result.val_pass2, 0.6, places=4)
        self.assertAlmostEqual(result.val_pass3, 0.4, places=4)
        self.assertAlmostEqual(result.val_pass4, 0.2, places=4)
        # weights: 0.35*0.8 + 0.25*0.6 + 0.25*0.4 + 0.15*0.2
        # = 0.28 + 0.15 + 0.10 + 0.03 = 0.56
        self.assertAlmostEqual(result.val_score, 0.56, places=4)
        self.assertAlmostEqual(result.total_seconds, 27.0, places=4)
        # TPS = 650 tokens / 27.0 s
        self.assertAlmostEqual(result.avg_tps, round(650 / 27.0, 2), places=2)

    @patch("autoresearch.benchmarks.benchmark_coding._load_problems")
    @patch("autoresearch.benchmarks.benchmark_coding._run_tests")
    def test_run_coding_eval_pass_at_1_humaneval(self, mock_run_tests, mock_load):
        """Verify pass@1 calculation for the evalplus path."""
        mock_load.return_value = {
            f"HE/{i}": {
                "prompt": f"def f{i}(x):",
                "test": f"assert f{i}(1) == 1",
                "entry_point": f"f{i}",
            }
            for i in range(4)
        }
        mock_run_tests.side_effect = [True, False, True, False]

        from autoresearch.core.llama_client import LlamaClient

        client = MagicMock(spec=LlamaClient)
        client.complete.return_value = {
            "content": "def f(x): return x",
            "usage": {"total_tokens": 10},
        }

        pass_rate, tokens, elapsed = benchmark_coding.run_coding_eval(
            client, benchmark_coding.EvalplusTask("humaneval", 0.25), task_limit=4
        )
        self.assertEqual(pass_rate, 0.5)
        self.assertGreater(tokens, 0)

    @patch("autoresearch.benchmarks.benchmark_coding._load_problems")
    @patch("autoresearch.benchmarks.benchmark_coding._run_tests")
    def test_run_coding_eval_indentation_handling(self, mock_run_tests, mock_load):
        """Verify that run_coding_eval properly dedents and re-indents raw code."""
        mock_load.return_value = {
            "HE/0": {
                "prompt": 'def f0(x):\n    """docstring"""',
                "test": "assert f0(1) == 1",
                "entry_point": "f0",
            }
        }
        mock_run_tests.return_value = True

        from autoresearch.core.llama_client import LlamaClient

        client = MagicMock(spec=LlamaClient)
        # Mock completion response with extra 8-space indentation (common in thinking blocks)
        client.complete.return_value = {
            "content": "        return x",
            "usage": {"total_tokens": 10},
        }

        benchmark_coding.run_coding_eval(
            client, benchmark_coding.EvalplusTask("humaneval", 0.25), task_limit=1
        )

        # Inspect the code passed to _run_tests
        mock_run_tests.assert_called_once()
        actual_code = mock_run_tests.call_args[0][0]
        # Should have the prompt_sig, followed by exactly 4 spaces + return x
        expected_code = 'def f0(x):\n    """docstring"""\n    return x'
        self.assertEqual(actual_code, expected_code)

    # ------------------------------------------------------------------ LCB

    def test_lcb_private_case_decode(self):
        """Pickle+zlib+base64 roundtrip."""
        raw_json = json.dumps([{"input": "1\n", "output": "1\n", "testtype": "stdin"}])
        encoded = _lcb_pickled_json(raw_json)
        decoded = benchmark_coding._decode_lcb_private_cases(encoded)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["input"], "1\n")
        self.assertEqual(decoded[0]["output"], "1\n")

    def test_lcb_tests_pass(self):
        """A program that doubles n passes all tests."""
        code = "n = int(input())\nprint(n*2)\n"
        private_json = json.dumps(
            [
                {"input": "5\n", "output": "10\n"},
                {"input": "21\n", "output": "42\n"},
            ]
        )
        entry = {
            "_private_tests_decoded": json.loads(private_json),
        }
        self.assertTrue(benchmark_coding._run_lcb_tests(code, entry))

    def test_lcb_tests_fail_on_wrong_output(self):
        code = "n = int(input())\nprint(n+1)\n"
        private_json = json.dumps([{"input": "5\n", "output": "10\n"}])
        entry = {"_private_tests_decoded": json.loads(private_json)}
        self.assertFalse(benchmark_coding._run_lcb_tests(code, entry))

    def test_lcb_tests_fail_on_runtime_error(self):
        code = "raise Exception('boom')\n"
        private_json = json.dumps([{"input": "1\n", "output": "2\n"}])
        entry = {"_private_tests_decoded": json.loads(private_json)}
        self.assertFalse(benchmark_coding._run_lcb_tests(code, entry))

    @patch("huggingface_hub.hf_hub_download")
    def test_download_lcb_file_copies_not_symlinks(self, mock_hf):
        """Windows without Developer Mode cannot symlink; copy2 must materialize test6.jsonl."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "hf_src" / "test6.jsonl"
            src.parent.mkdir(parents=True)
            src.write_text('{"question_title":"x"}\n', encoding="utf-8")
            mock_hf.return_value = str(src)

            cache = tmp_path / "lcb_cache"
            with (
                patch.object(benchmark_coding, "LCB_CACHE_DIR", cache),
                patch.object(benchmark_coding, "DATA_DIR", tmp_path / "data"),
            ):
                out = benchmark_coding._download_lcb_file(force=True)

            self.assertEqual(out, cache / "test6.jsonl")
            self.assertTrue(out.is_file())
            self.assertFalse(out.is_symlink())
            self.assertGreater(out.stat().st_size, 0)
            self.assertEqual(out.read_text(encoding="utf-8"), src.read_text(encoding="utf-8"))

    @patch("autoresearch.benchmarks.benchmark_coding._download_lcb_file")
    def test_lcb_loader_filters_and_decodes(self, mock_dl):
        """Loader: reads JSONL, decodes private tests, applies platform filter."""
        records = [
            {
                "question_title": "T1",
                "question_content": "q1",
                "platform": "atcoder",
                "starter_code": "",
                "private_test_cases": _lcb_pickled_json(
                    json.dumps(
                        [
                            {"input": "1\n", "output": "1\n", "testtype": "stdin"},
                        ]
                    )
                ),
            },
            {
                "question_title": "T2",
                "question_content": "q2",
                "platform": "leetcode",
                "starter_code": "",
                "private_test_cases": _lcb_pickled_json(json.dumps([{"input": "", "output": ""}])),
            },
            {
                "question_title": "T3",
                "question_content": "q3",
                "platform": "atcoder",
                "starter_code": "",
                "private_test_cases": "garbage-data",
            },
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            tmp_path = f.name
        mock_dl.return_value = Path(tmp_path)

        loaded = benchmark_coding._load_livecodebench(task_limit=10, platform_filter="atcoder")
        self.assertEqual(len(loaded), 1, "should keep only atcoder with decodable tests")
        self.assertEqual(loaded[0]["question_title"], "T1")
        self.assertEqual(len(loaded[0]["_private_tests_decoded"]), 1)

    # ------------------------------------------------------------------ BigCode

    def test_bigcode_tests_pass(self):
        code = "def task_func(n):\n    return 1 if n<=1 else n*task_func(n-1)\n"
        entry = {
            "entry_point": "task_func",
            "test": (
                "import unittest\n"
                "class TestCases(unittest.TestCase):\n"
                "    def test_one(self): self.assertEqual(task_func(5), 120)\n"
                "    def test_two(self): self.assertEqual(task_func(0), 1)\n"
            ),
        }
        self.assertTrue(benchmark_coding._run_bigcode_tests(code, entry))

    def test_bigcode_tests_fail(self):
        code = "def task_func(n):\n    return 0\n"
        entry = {
            "entry_point": "task_func",
            "test": (
                "import unittest\n"
                "class TestCases(unittest.TestCase):\n"
                "    def test_one(self): self.assertEqual(task_func(5), 120)\n"
            ),
        }
        self.assertFalse(benchmark_coding._run_bigcode_tests(code, entry))

    def test_bigcode_tests_missing_test_field(self):
        code = "def task_func(n): return n\n"
        self.assertFalse(
            benchmark_coding._run_bigcode_tests(code, {"entry_point": "task_func", "test": ""})
        )

    # ------------------------------------------------------------------ evalplus strict

    @patch("autoresearch.benchmarks.benchmark_coding.bench_config")
    def test_strict_mode_prefers_test_field(self, mock_bench_cfg):
        """When `test` field exists, strict mode should use it (don't fall back to I/O)."""
        mock_bench_cfg.EVALPLUS_STRICT = True
        entry = {
            "test": "assert candidate([1,2,3]) == True",
            "base_input_output_tests": [([4, 5], False)],
            "plus_input_output_tests": [([100, 200], True), ([300, 400], False)],
            "entry_point": "has_close_elements",
        }
        test_code = benchmark_coding._get_test_code(entry, "humaneval")
        self.assertIn("candidate([1,2,3])", test_code)
        # In strict mode, do NOT generate extra asserts from base/plus pairs when test is present
        self.assertNotIn("[100, 200]", test_code)

    @patch("autoresearch.benchmarks.benchmark_coding.bench_config")
    def test_non_strict_mode_falls_back_to_io_pairs(self, mock_bench_cfg):
        """When strict=False and test is absent, build from base+plus pairs."""
        mock_bench_cfg.EVALPLUS_STRICT = False
        entry = {
            "base_input_output_tests": [([1, 2], True)],
            "plus_input_output_tests": [([10, 20], False)],
            "entry_point": "f",
        }
        test_code = benchmark_coding._get_test_code(entry, "humaneval")
        self.assertIn("f(*[1, 2]) == True", test_code)

    @patch("autoresearch.benchmarks.benchmark_coding.bench_config")
    def test_strict_mode_falls_back_to_io_pairs_when_no_test(self, mock_bench_cfg):
        """Strict mode but no `test` field: build asserts from plus then base pairs."""
        mock_bench_cfg.EVALPLUS_STRICT = True
        entry = {
            "base_input_output_tests": [([1, 2], True)],
            "plus_input_output_tests": [([10, 20], False)],
            "entry_point": "f",
        }
        test_code = benchmark_coding._get_test_code(entry, "humaneval")
        # Strict -> plus first, then base
        self.assertIn("f(*[10, 20]) == False", test_code)
        self.assertIn("f(*[1, 2]) == True", test_code)

    # ------------------------------------------------------------------ code extraction

    def test_strip_code_strips_think_tags(self):
        text = "<think>reasoning</think>```python\nprint('hi')\n```"
        self.assertEqual(benchmark_coding._strip_code(text), "print('hi')")

    def test_strip_code_handles_plain_text(self):
        self.assertEqual(benchmark_coding._strip_code("print(1)"), "print(1)")

    def test_strip_code_handles_generic_code_block(self):
        text = "```\nx = 1\n```"
        self.assertEqual(benchmark_coding._strip_code(text), "x = 1")

    def test_strip_code_ifm_think_tags(self):
        """K2-Horizon <ifm|think> reasoning scratchpads must be cleanly stripped."""
        text = "<ifm|think>\nAnalyzing problem...\nNeed to double n.\n</ifm|think>\n```python\ndef solve(n):\n    return n * 2\n```"
        self.assertEqual(benchmark_coding._strip_code(text), "def solve(n):\n    return n * 2")

    def test_strip_code_ifm_think_fast_tags(self):
        """K2-Horizon <ifm|think_fast> reasoning scratchpads must be cleanly stripped."""
        text = "<ifm|think_fast>\nQuick reasoning.\n</ifm|think_fast>\n```python\nprint(42)\n```"
        self.assertEqual(benchmark_coding._strip_code(text), "print(42)")

    def test_strip_code_ifm_think_faster_tags(self):
        """K2-Horizon <ifm|think_faster> reasoning scratchpads must be cleanly stripped."""
        text = "<ifm|think_faster>Immediate thought.</ifm|think_faster>\n```python\nx = 1\n```"
        self.assertEqual(benchmark_coding._strip_code(text), "x = 1")

    def test_strip_code_ifm_think_plain_code_no_syntax_error(self):
        """Plain code following <ifm|think> should compile as valid Python without SyntaxError."""
        import ast

        text = "<ifm|think>scratchpad with <invalid> python syntax</ifm|think>\ndef foo():\n    return 'valid'"
        code = benchmark_coding._strip_code(text)
        self.assertEqual(code, "def foo():\n    return 'valid'")
        ast.parse(code)

    def test_strip_code_with_comparison_operators(self):
        """Code containing < and > comparisons alongside <ifm|think> must not be corrupted."""
        import ast

        text = "<ifm|think>need a < b and c > d</ifm|think>\n```python\ndef check(a, b, c, d):\n    return a < b and c > d\n```"
        code = benchmark_coding._strip_code(text)
        self.assertEqual(code, "def check(a, b, c, d):\n    return a < b and c > d")
        ast.parse(code)

    def test_strip_code_compact_comparison_not_corrupted(self):
        """Compact comparison expressions like x<think_limit or (x<think and y>0) must not be corrupted."""
        import ast

        text = "<ifm|think_fast>reasoning</ifm|think_fast>\n```python\ndef solve(x, think_limit, think, y):\n    if (x<think_limit):\n        return y > 0\n    if (x<think and y>0):\n        return 42\n    return 0\n```"
        code = benchmark_coding._strip_code(text)
        self.assertIn("x<think_limit", code)
        self.assertIn("x<think and y>0", code)
        ast.parse(code)

    def test_strip_code_ifm_think_mixed_closing_tag(self):
        """<ifm|think_fast> closed with </ifm|think> must be stripped cleanly."""
        import ast

        text = "<ifm|think_fast>Quick thinking</ifm|think>\n```python\ndef foo():\n    return 'bar'\n```"
        code = benchmark_coding._strip_code(text)
        self.assertEqual(code, "def foo():\n    return 'bar'")
        ast.parse(code)

    # ------------------------------------------------------------------ _strip_code (bug fix)

    def test_strip_code_think_plus_plain_code(self):
        """Think block followed by plain code (no fence) extracts the code."""
        text = "<think>Need to read n and double it.</think>\nn = int(input())\nprint(n*2)"
        result = benchmark_coding._strip_code(text)
        self.assertEqual(result, "n = int(input())\nprint(n*2)")

    def test_strip_code_multiple_think_blocks(self):
        """Multiple think blocks are all stripped."""
        text = (
            "<think>First thought.</think>\n"
            "<think>Wait, reconsidering.</think>\n"
            "n = int(input())\nprint(n*2)"
        )
        result = benchmark_coding._strip_code(text)
        self.assertEqual(result, "n = int(input())\nprint(n*2)")
        self.assertNotIn("<think>", result)
        self.assertNotIn("</think>", result)

    def test_strip_code_prose_prefix_then_code(self):
        """Prose prefix preserved (no fence to extract from); code still in output."""
        text = "Here is the solution:\n\nn = int(input())\nprint(n*2)"
        result = benchmark_coding._strip_code(text)
        self.assertIn("n = int(input())", result)
        self.assertIn("print(n*2)", result)

    def test_strip_code_function_definition_extraction(self):
        """BigCodeBench-style response: prose + think + bare function definition."""
        text = (
            "<think>The user wants task_func to return 42.</think>\n\n"
            "def task_func(*args, **kwargs):\n    return 42"
        )
        result = benchmark_coding._strip_code(text)
        self.assertIn("def task_func", result)
        self.assertNotIn("<think>", result)

    def test_strip_code_indented_continuation_lines(self):
        """Indented continuation lines are preserved after def."""
        text = "<think>thinking</think>\ndef add(a, b):\n    return a + b\n"
        result = benchmark_coding._strip_code(text)
        self.assertEqual(result, "def add(a, b):\n    return a + b")

    def test_strip_code_mid_prose_think_then_code(self):
        """Think blocks stripped, prose prefix preserved, code in output."""
        text = (
            "Sure, here's the code:\n\n"
            "<think>Wait, do I need the empty case?</think>\n"
            "import sys\n"
            "n = int(sys.stdin.readline())\n"
            "print(n * 2)"
        )
        result = benchmark_coding._strip_code(text)
        self.assertIn("import sys", result)
        self.assertIn("n = int", result)
        self.assertNotIn("<think>", result)

    def test_strip_code_fenced_with_language_tag(self):
        """```py and ```python fences both work."""
        text = "<think>x</think>\n```py\nprint(1)\n```"
        self.assertEqual(benchmark_coding._strip_code(text), "print(1)")

    def test_strip_code_preserves_indentation(self):
        """Verify that horizontal indentation is preserved even when empty lines are stripped."""
        text = "    def solution():\n        return True\n\n"
        self.assertEqual(
            benchmark_coding._strip_code(text), "    def solution():\n        return True"
        )

        # Fenced code with indentation
        text_fence = "```python\n    x = 1\n    return x\n```"
        self.assertEqual(benchmark_coding._strip_code(text_fence), "    x = 1\n    return x")

    # ------------------------------------------------------------------ run_coding_eval: content + reasoning_content

    @patch("autoresearch.benchmarks.benchmark_coding._run_lcb_tests")
    @patch("autoresearch.benchmarks.benchmark_coding._load_bigcodebench_hard")
    @patch("autoresearch.benchmarks.benchmark_coding._load_livecodebench")
    @patch("autoresearch.benchmarks.benchmark_coding._load_problems")
    def test_eval_combines_content_and_reasoning(self, mock_load, mock_lcb, mock_bcb, mock_run_lcb):
        """
        If content is empty but reasoning_content has the code (thinking model
        ran out of tokens for content but emitted code in reasoning), we still
        extract a passing code.
        """
        # Empty HE/MBPP loads; LCB runs the mocked test
        mock_load.side_effect = [{}, {}]
        mock_lcb.return_value = [{"_private_tests_decoded": [{"input": "1\n", "output": "2\n"}]}]
        mock_bcb.return_value = []
        mock_run_lcb.return_value = True

        from autoresearch.core.llama_client import LlamaClient

        client = MagicMock(spec=LlamaClient)
        client.port = 1234

        # LCB task: model returns empty content but the code lives in reasoning_content
        def fake_complete(prompt, **kwargs):
            if "standard input" in prompt.lower() or "read from" in prompt.lower():
                return {
                    "content": "",  # model ran out of tokens mid-think
                    "reasoning_content": "n = int(input())\nprint(n*2)",
                    "usage": {"total_tokens": 10},
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": "n = int(input())\nprint(n*2)",
                                "tool_calls": [],
                            }
                        }
                    ],
                }
            return {
                "content": "def f():\n    return 1",
                "usage": {"total_tokens": 10},
                "choices": [{"message": {"content": "def f():\n    return 1", "tool_calls": []}}],
            }

        client.complete.side_effect = fake_complete

        result = benchmark_coding.run_benchmark(
            client, task_limit=1, lcb_task_limit=1, bigcode_task_limit=0
        )
        # LCB pass should be 1.0 since reasoning_content was extracted and ran
        self.assertEqual(result.val_pass1, 1.0)


if __name__ == "__main__":
    unittest.main()
