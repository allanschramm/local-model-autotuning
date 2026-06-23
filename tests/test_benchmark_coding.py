import unittest
from unittest.mock import patch, MagicMock, mock_open
from autoresearch.benchmarks import benchmark_coding
from pathlib import Path
import sys
import json
import base64
import zlib
import pickle
import tempfile


def _lcb_pickled_json(payload_str: str) -> str:
    """Encode a JSON string the way LiveCodeBench v6 encodes its private test cases."""
    return base64.b64encode(zlib.compress(pickle.dumps(payload_str))).decode()


class TestBenchmarkCoding(unittest.TestCase):

    def test_parse_args(self):
        test_args = ["prog", "--ctx-size", "2048", "--model", "test-model.gguf"]
        with patch.object(sys, "argv", test_args):
            args = benchmark_coding.parse_args()
            self.assertEqual(args.ctx_size, 2048)
            self.assertEqual(args.model, "test-model.gguf")

    # ------------------------------------------------------------------ weights

    @patch("autoresearch.benchmarks.benchmark_coding._load_bigcodebench_hard")
    @patch("autoresearch.benchmarks.benchmark_coding._load_livecodebench")
    @patch("autoresearch.benchmarks.benchmark_coding.run_coding_eval")
    def test_run_benchmark_weights(self, mock_eval, mock_lcb_loader, mock_bigcode_loader):
        """
        val_score = 0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode
        mock_eval returns (lcb, he, mbpp, bigcode) pass rates.
        """
        mock_lcb_loader.return_value = ["p1"] * 10  # 10 LCB problems
        mock_bigcode_loader.return_value = ["p2"] * 10
        mock_eval.side_effect = [
            (0.6, 100, 10.0),  # HumanEval -> val_pass2
            (0.4, 200, 5.0),   # MBPP     -> val_pass3
            (0.8, 300, 8.0),   # LCB      -> val_pass1
            (0.2, 50,  4.0),   # BigCode  -> val_pass4
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
        self.assertAlmostEqual(result.avg_tps, round(650/27.0, 2), places=2)

    @patch("autoresearch.benchmarks.benchmark_coding._load_bigcodebench_hard")
    @patch("autoresearch.benchmarks.benchmark_coding._load_livecodebench")
    @patch("autoresearch.benchmarks.benchmark_coding.run_coding_eval")
    def test_run_benchmark_passes_gen_kwargs(self, mock_eval, mock_lcb_loader, mock_bigcode_loader):
        """Verify gen_kwargs are forwarded to all sub-eval calls."""
        mock_lcb_loader.return_value = ["p1"] * 10
        mock_bigcode_loader.return_value = ["p2"] * 10
        mock_eval.return_value = (0.5, 50, 5.0)

        from autoresearch.core.llama_client import LlamaClient
        client = MagicMock(spec=LlamaClient)
        client.port = 1234

        benchmark_coding.run_benchmark(
            client, task_limit=5,
            temperature=0.6, top_p=0.95, top_k=20,
        )
        # 4 sub-evals: HE, MBPP, LCB, BigCode
        self.assertEqual(mock_eval.call_count, 4)
        for call in mock_eval.call_args_list:
            _, kwargs = call
            self.assertEqual(kwargs.get("temperature"), 0.6)
            self.assertEqual(kwargs.get("top_p"), 0.95)
            self.assertEqual(kwargs.get("top_k"), 20)

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
            client, "humaneval", task_limit=4
        )
        self.assertEqual(pass_rate, 0.5)
        self.assertGreater(tokens, 0)

    # ------------------------------------------------------------------ LCB

    def test_lcb_private_case_decode(self):
        """Pickle+zlib+base64 roundtrip."""
        raw_json = json.dumps([{"input": "1\n", "output": "1\n", "testtype": "stdin"}])
        encoded = _lcb_pickled_json(raw_json)
        decoded = benchmark_coding._decode_lcb_private_cases(encoded)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["input"], "1\n")
        self.assertEqual(decoded[0]["output"], "1\n")

    def test_lcb_private_case_decode_garbage(self):
        """Bad data returns [] silently rather than raising."""
        self.assertEqual(benchmark_coding._decode_lcb_private_cases("not-base64!@#"), [])

    def test_lcb_prompt_contains_question_and_io_hint(self):
        entry = {
            "question_title": "A. Sum",
            "question_content": "Given n, print its double.",
            "starter_code": "def solve(): pass\n",
            "platform": "atcoder",
        }
        prompt = benchmark_coding._build_lcb_prompt(entry)
        self.assertIn("Sum", prompt)
        self.assertIn("Given n, print its double.", prompt)
        self.assertIn("standard input", prompt)
        self.assertIn("def solve(): pass", prompt)

    def test_lcb_tests_pass(self):
        """A program that doubles n passes all tests."""
        code = "n = int(input())\nprint(n*2)\n"
        private_json = json.dumps([
            {"input": "5\n",  "output": "10\n"},
            {"input": "21\n", "output": "42\n"},
        ])
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

    @patch("autoresearch.benchmarks.benchmark_coding._download_lcb_file")
    def test_lcb_loader_filters_and_decodes(self, mock_dl):
        """Loader: reads JSONL, decodes private tests, applies platform filter."""
        records = [
            {
                "question_title": "T1",
                "question_content": "q1",
                "platform": "atcoder",
                "starter_code": "",
                "private_test_cases": _lcb_pickled_json(json.dumps([
                    {"input": "1\n", "output": "1\n", "testtype": "stdin"},
                ])),
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

    def test_bigcode_prompt_uses_instruct_split(self):
        entry = {
            "instruct_prompt": "Compute the factorial of n.",
            "complete_prompt": "def task_func(n):",
        }
        prompt = benchmark_coding._build_bigcode_prompt(entry)
        self.assertIn("factorial of n", prompt)
        self.assertIn("code block", prompt.lower())

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
        self.assertFalse(benchmark_coding._run_bigcode_tests(code, {"entry_point": "task_func", "test": ""}))

    @patch("autoresearch.benchmarks.benchmark_coding._load_bigcodebench_hard")
    @patch("autoresearch.benchmarks.benchmark_coding._load_livecodebench")
    @patch("autoresearch.benchmarks.benchmark_coding.run_coding_eval")
    def test_bigcode_via_run_benchmark(self, mock_eval, mock_lcb_loader, mock_bigcode_loader):
        """BigCode tasks feed through run_coding_eval via the `problems` list."""
        mock_lcb_loader.return_value = ["l"] * 5
        mock_bigcode_loader.return_value = [{"task_id": f"BCB/{i}", "instruct_prompt": "noop", "test": ""} for i in range(3)]
        mock_eval.side_effect = [
            (1.0, 0, 1.0),  # HE
            (1.0, 0, 1.0),  # MBPP
            (0.0, 0, 1.0),  # LCB
            (0.0, 0, 1.0),  # BigCode (3 tasks, no test field -> all skip -> 0%)
        ]
        from autoresearch.core.llama_client import LlamaClient
        client = MagicMock(spec=LlamaClient)
        client.port = 1234

        result = benchmark_coding.run_benchmark(client, task_limit=5, lcb_task_limit=5, bigcode_task_limit=3)
        # The bigcode call should have received the 3 problems as the `problems` arg
        bigcode_call = mock_eval.call_args_list[3]
        passed_problems = bigcode_call.kwargs.get("problems") or bigcode_call.args[2] if len(bigcode_call.args) > 2 else bigcode_call.kwargs.get("problems")
        self.assertEqual(len(passed_problems), 3)

    # ------------------------------------------------------------------ evalplus strict

    @patch("autoresearch.benchmarks.benchmark_coding.config")
    def test_strict_mode_prefers_test_field(self, mock_cfg):
        """When `test` field exists, strict mode should use it (don't fall back to I/O)."""
        mock_cfg.EVALPLUS_STRICT = True
        entry = {
            "test": "assert candidate([1,2,3]) == True",
            "base_input_output_tests": [([4,5], False)],
            "plus_input_output_tests": [([100,200], True), ([300,400], False)],
            "entry_point": "has_close_elements",
        }
        test_code = benchmark_coding._get_test_code(entry, "humaneval")
        self.assertIn("candidate([1,2,3])", test_code)
        # In strict mode, do NOT generate extra asserts from base/plus pairs when test is present
        self.assertNotIn("[100, 200]", test_code)

    @patch("autoresearch.benchmarks.benchmark_coding.config")
    def test_non_strict_mode_falls_back_to_io_pairs(self, mock_cfg):
        """When strict=False and test is absent, build from base+plus pairs."""
        mock_cfg.EVALPLUS_STRICT = False
        entry = {
            "base_input_output_tests": [([1,2], True)],
            "plus_input_output_tests": [([10,20], False)],
            "entry_point": "f",
        }
        test_code = benchmark_coding._get_test_code(entry, "humaneval")
        self.assertIn("f(*[1, 2]) == True", test_code)

    @patch("autoresearch.benchmarks.benchmark_coding.config")
    def test_strict_mode_falls_back_to_io_pairs_when_no_test(self, mock_cfg):
        """Strict mode but no `test` field: build asserts from plus then base pairs."""
        mock_cfg.EVALPLUS_STRICT = True
        entry = {
            "base_input_output_tests": [([1,2], True)],
            "plus_input_output_tests": [([10,20], False)],
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


    # ------------------------------------------------------------------ _strip_code (bug fix)

    def test_strip_code_empty_input(self):
        """Empty string returns empty string."""
        self.assertEqual(benchmark_coding._strip_code(""), "")
        self.assertEqual(benchmark_coding._strip_code(None), "")

    def test_strip_code_think_only_truncated(self):
        """Truncated think block (no closing tag, no code) returns empty."""
        text = f"<think>Let me think about this carefully. The problem is asking me to"
        self.assertEqual(benchmark_coding._strip_code(text), "")

    def test_strip_code_think_closed_no_code(self):
        """Closed think block with no code after returns empty."""
        text = f"<think>Need to solve the doubling problem.</think>"
        self.assertEqual(benchmark_coding._strip_code(text), "")

    def test_strip_code_think_plus_plain_code(self):
        """Think block followed by plain code (no fence) extracts the code."""
        text = f"<think>Need to read n and double it.</think>\nn = int(input())\nprint(n*2)"
        result = benchmark_coding._strip_code(text)
        self.assertEqual(result, "n = int(input())\nprint(n*2)")

    def test_strip_code_multiple_think_blocks(self):
        """Multiple think blocks are all stripped."""
        text = (
            f"<think>First thought.</think>\n"
            f"<think>Wait, reconsidering.</think>\n"
            "n = int(input())\nprint(n*2)"
        )
        result = benchmark_coding._strip_code(text)
        self.assertEqual(result, "n = int(input())\nprint(n*2)")
        self.assertNotIn("<think>", result)
        self.assertNotIn("</think>", result)

    def test_strip_code_prose_prefix_then_code(self):
        """Prose prefix is stripped; code is extracted from the first code-looking line."""
        text = "Here is the solution:\n\nn = int(input())\nprint(n*2)"
        result = benchmark_coding._strip_code(text)
        self.assertEqual(result, "n = int(input())\nprint(n*2)")
        self.assertNotIn("Here is", result)

    def test_strip_code_function_definition_extraction(self):
        """BigCodeBench-style response: prose + think + bare function definition."""
        text = (
            f"<think>The user wants task_func to return 42.</think>\n\n"
            "def task_func(*args, **kwargs):\n    return 42"
        )
        result = benchmark_coding._strip_code(text)
        self.assertIn("def task_func", result)
        self.assertNotIn("<think>", result)

    def test_strip_code_indented_continuation_lines(self):
        """Indented continuation lines are preserved after def."""
        text = (
            f"<think>thinking</think>\n"
            "def add(a, b):\n"
            "    return a + b\n"
        )
        result = benchmark_coding._strip_code(text)
        self.assertEqual(result, "def add(a, b):\n    return a + b")

    def test_strip_code_mid_prose_think_then_code(self):
        """Code is found even when surrounded by prose + think."""
        text = (
            "Sure, here's the code:\n\n"
            f"<think>Wait, do I need the empty case?</think>\n"
            "import sys\n"
            "n = int(sys.stdin.readline())\n"
            "print(n * 2)"
        )
        result = benchmark_coding._strip_code(text)
        self.assertIn("import sys", result)
        self.assertIn("n = int", result)
        self.assertNotIn("<think>", result)
        self.assertNotIn("Sure", result)

    def test_strip_code_fenced_with_language_tag(self):
        """```py and ```python fences both work."""
        text = f"<think>x</think>\n```py\nprint(1)\n```"
        self.assertEqual(benchmark_coding._strip_code(text), "print(1)")

    def test_strip_code_unclosed_fence_falls_through(self):
        """Unclosed fence (model truncated) -> extract via code-line scan."""
        text = f"<think>x</think>\n```python\nn = 1\n"
        result = benchmark_coding._strip_code(text)
        self.assertIn("n = 1", result)

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
            if 'standard input' in prompt.lower() or 'read from' in prompt.lower():
                return {
                    "content": "",  # model ran out of tokens mid-think
                    "reasoning_content": "n = int(input())\nprint(n*2)",
                    "usage": {"total_tokens": 10},
                    "choices": [{"message": {"content": "", "reasoning_content": "n = int(input())\nprint(n*2)", "tool_calls": []}}],
                }
            return {"content": "def f():\n    return 1", "usage": {"total_tokens": 10}, "choices": [{"message": {"content": "def f():\n    return 1", "tool_calls": []}}]}
        client.complete.side_effect = fake_complete

        result = benchmark_coding.run_benchmark(client, task_limit=1, lcb_task_limit=1, bigcode_task_limit=0)
        # LCB pass should be 1.0 since reasoning_content was extracted and ran
        self.assertEqual(result.val_pass1, 1.0)

    @patch("autoresearch.benchmarks.benchmark_coding._load_bigcodebench_hard")
    @patch("autoresearch.benchmarks.benchmark_coding._load_livecodebench")
    @patch("autoresearch.benchmarks.benchmark_coding._run_lcb_tests", return_value=True)
    @patch("autoresearch.benchmarks.benchmark_coding._run_bigcode_tests", return_value=True)
    @patch("autoresearch.benchmarks.benchmark_coding._run_tests", return_value=True)
    @patch("autoresearch.benchmarks.benchmark_coding._load_problems")
    def test_eval_uses_per_dataset_max_tokens(self, mock_load, mock_run_tests, mock_run_bigcode, mock_run_lcb, mock_lcb, mock_bcb):
        """LCB and BigCodeBench sub-evals receive 2048 max_tokens."""
        mock_load.side_effect = [{}, {}]
        mock_lcb.return_value = [{"question_title": "T", "question_content": "q", "platform": "atcoder", "starter_code": "", "_private_tests_decoded": [{"input": "1\n", "output": "2\n"}]}]
        mock_bcb.return_value = [{"task_id": "BCB/0", "instruct_prompt": "noop", "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_one(self): self.assertTrue(True)\n", "entry_point": "task_func"}]

        from autoresearch.core.llama_client import LlamaClient
        client = MagicMock(spec=LlamaClient)
        client.port = 1234
        client.complete.return_value = {
            "content": "x = 1", "reasoning_content": "",
            "usage": {"total_tokens": 5},
            "choices": [{"message": {"content": "x = 1", "tool_calls": []}}],
        }

        benchmark_coding.run_benchmark(client, task_limit=1, lcb_task_limit=1, bigcode_task_limit=1, max_tokens=512)
        # Find the LCB and BigCode calls; check their max_tokens kwarg
        seen_max = []
        for call in client.complete.call_args_list:
            seen_max.append(call.kwargs.get("max_tokens"))
        # LCB and BigCodeBench calls should have 2048 (overridden). HE+MBPP
        # return early when their problem dict is empty, so they never reach
        # client.complete; only LCB/BigCode appear here.
        self.assertIn(2048, seen_max)
        self.assertEqual(seen_max.count(2048), 2)

if __name__ == "__main__":
    unittest.main()
