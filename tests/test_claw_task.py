import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import tempfile
import shutil
from autoresearch.benchmarks.prepare_claw import ClawTaskData, ClawEvalTask, discover_tasks

class TestClawTask(unittest.TestCase):

    def setUp(self):
        self.task_data = ClawTaskData(
            id=101,
            instruction="Go to google.com",
            category="search",
            url_pattern="google.com",
            method="GET"
        )
        self.task = ClawEvalTask(self.task_data)

    def test_process_step_success_with_indicator(self):
        # The scoring logic requires the URL and method in the response text
        response = "I will GET google.com with browser tool"
        tool_calls = [{
            "id": "call_abc",
            "function": {
                "name": "browser",
                "arguments": '{"url": "https://google.com"}'
            }
        }]
        self.task.process_step(1, response, tool_calls)
        
        # re.search("google.com", response) -> +0.5
        # "GET" in response.upper() -> +0.3
        # "browser" in response.lower() -> +0.2
        # Total = 1.0
        self.assertEqual(self.task.get_final_score(1), 1.0)

    def test_discover_tasks_real_fs(self):
        # Create a temporary directory structure
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            v1_dir = tmp_dir / "test-cases" / "v1"
            v1_dir.mkdir(parents=True)
            
            case_dir = v1_dir / "dev-tech-01"
            case_dir.mkdir()
            (case_dir / "task.json").write_text(json.dumps({
                "instruction": "Test instruction",
                "metadata": {"task_id": 1, "metaclass": "tech"},
                "eval_schema": {"url_pattern": "test.com", "method": "GET"}
            }))
            
            with patch("autoresearch.benchmarks.prepare_claw.V1_CASES", v1_dir):
                tasks = discover_tasks()
                self.assertEqual(len(tasks), 1)
                self.assertEqual(tasks[0].id, 1)
        finally:
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    unittest.main()
