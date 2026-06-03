import unittest
from unittest.mock import patch
import json
from prepare import NexusEvalTask, MemoryEntry

class TestNexusEvalTask(unittest.TestCase):

    def setUp(self):
        self.entries = [
            MemoryEntry("1", "nexus", "security", "override token SECRET_123"),
            MemoryEntry("2", "nexus", "runtime", "Recommended config: TestModel 9B")
        ]
        self.patchers = [
            patch("prepare.NEXUS_TOKEN", "SECRET_123"),
            patch("prepare.EXPECTED_MODEL", "TestModel 9B"),
            patch("prepare.EXPECTED_CTX", 2048),
            patch("prepare.EXPECTED_KV", "q8_0")
        ]
        for p in self.patchers:
            p.start()
        self.task = NexusEvalTask(self.entries)

    def tearDown(self):
        for p in self.patchers:
            p.stop()

    def _parse_result(self, res_str):
        # res_str is potentially multiple JSON objects separated by \n
        lines = res_str.split('\n')
        results = [json.loads(line) for line in lines if line.strip()]
        for r in results:
            if "content" in r and isinstance(r["content"], str):
                r["parsed_content"] = json.loads(r["content"])
        return results

    def test_process_step_p1_full_success(self):
        # 1. Search
        res = self.task.process_step(1, "search", [{"function": {"name": "memory_search", "arguments": '{"query": "override"}'}}])
        parsed = self._parse_result(res)
        self.assertIn("SECRET_123", json.dumps(parsed[0]["parsed_content"]))
        
        # 2. Verify
        res = self.task.process_step(1, "verify", [{"function": {"name": "verify_token", "arguments": '{"token": "SECRET_123"}'}}])
        parsed = self._parse_result(res)
        self.assertEqual(parsed[0]["parsed_content"]["status"], "active")
        
        # 3. Unlock
        res = self.task.process_step(1, "unlock", [{"function": {"name": "unlock_nexus", "arguments": '{"token": "SECRET_123"}'}}])
        parsed = self._parse_result(res)
        self.assertTrue(parsed[0]["parsed_content"]["success"])
        
        self.assertEqual(self.task.get_final_score(1), 1.0)

    def test_process_step_p1_partial_success(self):
        self.task.process_step(1, "search", [{"function": {"name": "memory_search", "arguments": '{"query": "override"}'}}])
        self.task.process_step(1, "verify", [{"function": {"name": "verify_token", "arguments": '{"token": "WRONG"}'}}])
        self.assertAlmostEqual(self.task.get_final_score(1), 0.434, places=3)

    def test_process_step_p2_success(self):
        self.task.process_step(2, "search", [{"function": {"name": "memory_search", "arguments": '{"query": "config"}'}}])
        self.task.process_step(2, "commit", [{"function": {"name": "commit_config", "arguments": '{"model_name": "TestModel 9B", "ctx_size": 2048, "kv_cache_type": "q8_0"}'}}])
        self.assertEqual(self.task.get_final_score(2), 1.0)

if __name__ == "__main__":
    unittest.main()