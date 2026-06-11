import unittest
from unittest.mock import MagicMock, patch
from autoresearch.benchmarks.benchmark_harness import BenchmarkHarness, EvalTask, BenchmarkResult
from autoresearch.core.llama_client import LlamaClient

class DummyTask(EvalTask):
    def __init__(self, id="test-task"):
        self.id = id
        self.p1_called = 0
        self.p2_called = 0

    def get_initial_prompt(self, pass_num, padding=""):
        if pass_num == 1: self.p1_called += 1
        else: self.p2_called += 1
        return "Initial Prompt"

    def get_tools(self, pass_num):
        return []

    def process_step(self, pass_num, response, tool_calls):
        return None

    def get_final_score(self, pass_num):
        return 1.0 if pass_num == 1 else 0.8

class TestBenchmarkHarness(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock(spec=LlamaClient)
        self.harness = BenchmarkHarness(self.mock_client, target_tps=10.0, p1_weight=0.5)

    def test_run_task_loop_no_tools(self):
        task = DummyTask()
        self.mock_client.complete.return_value = {
            "content": "Final Answer",
            "usage": {"total_tokens": 100},
            "choices": [{"message": {"tool_calls": []}}]
        }
        
        score, tokens = self.harness._run_task_loop(task, pass_num=1)
        
        self.assertEqual(score, 1.0)
        self.assertEqual(tokens, 100)

    def test_system_prefix(self):
        task = DummyTask()
        self.mock_client.complete.return_value = {
            "content": "Done",
            "usage": {"total_tokens": 10},
            "choices": [{"message": {"tool_calls": []}}]
        }
        self.harness._run_task_loop(task, pass_num=1, system_prefix="<|think|>")
        self.mock_client.complete.assert_called_once()
        args, kwargs = self.mock_client.complete.call_args
        self.assertTrue(args[0].startswith("<|think|>Initial Prompt"))

    @patch("time.time")
    def test_evaluate_success(self, mock_time):
        # 1. t_start = 0
        # 2. t0 = 10
        # 3. dur = 11 - 10 = 1s
        # 4. total = 12 - 0 = 12s
        mock_time.side_effect = [0, 10, 11, 12] 
        
        task = DummyTask()
        self.mock_client.complete.return_value = {
            "content": "Done",
            "usage": {"total_tokens": 20}, # 20 TPS
            "choices": [{"message": {"tool_calls": []}}]
        }
        
        result = self.harness.evaluate([task])
        self.assertEqual(result.val_score, 0.9)
        self.assertEqual(result.avg_tps, 20.0)

    @patch("time.time")
    def test_evaluate_slow_tps(self, mock_time):
        # 1. t_start = 0
        # 2. t0 = 10
        # 3. dur = 14 - 10 = 4s
        # 4. total = 15 - 0 = 15s
        mock_time.side_effect = [0, 10, 14, 15]
        
        task = DummyTask()
        self.mock_client.complete.return_value = {
            "content": "Done",
            "usage": {"total_tokens": 20}, # 5 TPS
            "choices": [{"message": {"tool_calls": []}}]
        }

        # speed_factor = 0.5 + 0.5 * (5/10) = 0.75
        # p2 = 0.8 * 0.75 = 0.6
        # val = 0.5 * 1.0 + 0.5 * 0.6 = 0.8
        
        result = self.harness.evaluate([task])
        self.assertEqual(result.val_score, 0.8)
        self.assertEqual(result.avg_tps, 5.0)

if __name__ == "__main__":
    unittest.main()
