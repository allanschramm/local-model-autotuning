import unittest
from unittest.mock import patch

from autoresearch.benchmarks.agentic_benchmarks import (
    AGENTIC_BENCHMARKS,
    format_agentic_benchmarks,
    list_agentic_benchmarks,
)
from autoresearch.runners import run


class TestAgenticBenchmarkCatalog(unittest.TestCase):
    def test_catalog_includes_claw_eval(self):
        self.assertEqual(len(AGENTIC_BENCHMARKS), 1)
        self.assertEqual(AGENTIC_BENCHMARKS[0].key, "claw-eval")
        self.assertEqual(AGENTIC_BENCHMARKS[0].status, "adopt-next")

    def test_specs_have_sources_and_harnesses(self):
        for spec in AGENTIC_BENCHMARKS:
            self.assertTrue(spec.source_url.startswith("https://"))
            self.assertTrue(spec.harness)
            self.assertTrue(spec.scope)

    def test_status_filter(self):
        specs = list_agentic_benchmarks(status="adopt-next")
        self.assertEqual([spec.key for spec in specs], ["claw-eval"])

    def test_cli_format(self):
        self.assertEqual(
            format_agentic_benchmarks(),
            "key\tstatus\tharness\tname\nclaw-eval\tadopt-next\tclaw-eval\tClaw-Eval",
        )

    def test_parse_args_list_agentic_benchmarks(self):
        with patch("sys.argv", ["prog", "--list-agentic-benchmarks"]):
            args = run.parse_args()
        self.assertTrue(args.list_agentic_benchmarks)


if __name__ == "__main__":
    unittest.main()