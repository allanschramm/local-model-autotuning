import unittest
from unittest.mock import patch

from autoresearch.benchmarks.agentic_benchmarks import (
    AGENTIC_BENCHMARKS,
    format_agentic_benchmarks,
    list_agentic_benchmarks,
)
from autoresearch.runners import run


class TestAgenticBenchmarkCatalog(unittest.TestCase):
    def test_catalog_starts_empty(self):
        self.assertEqual(AGENTIC_BENCHMARKS, ())

    def test_specs_have_sources_and_harnesses(self):
        for spec in AGENTIC_BENCHMARKS:
            self.assertTrue(spec.source_url.startswith("https://"))
            self.assertTrue(spec.harness)
            self.assertTrue(spec.scope)

    def test_status_filter(self):
        self.assertEqual(list_agentic_benchmarks(status="adopt-next"), [])

    def test_cli_format(self):
        self.assertEqual(format_agentic_benchmarks(), "key\tstatus\tharness\tname")

    def test_parse_args_list_agentic_benchmarks(self):
        with patch("sys.argv", ["prog", "--list-agentic-benchmarks"]):
            args = run.parse_args()
        self.assertTrue(args.list_agentic_benchmarks)


if __name__ == "__main__":
    unittest.main()