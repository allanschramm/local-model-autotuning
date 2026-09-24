"""Tests for detect_hardware_capabilities (issue #17).

All probes are mocked so the suite stays green on CPU-only / CI hosts.
"""

import unittest
from unittest.mock import MagicMock, mock_open, patch

from autoresearch.core import hardware


class TestDetectPhysicalCores(unittest.TestCase):
    def test_darwin_reads_sysctl_hw_physicalcpu(self):
        proc = MagicMock(stdout="10\n")
        with patch("autoresearch.core.hardware.sys.platform", "darwin"):
            with patch("autoresearch.core.hardware.subprocess.run", return_value=proc) as run:
                self.assertEqual(hardware.detect_physical_cores(), 10)
        run.assert_called_once()

    def test_linux_dedupes_physical_core_pairs(self):
        cpuinfo = (
            "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0\n\n"
            "processor\t: 1\nphysical id\t: 0\ncore id\t\t: 0\n\n"
            "processor\t: 2\nphysical id\t: 0\ncore id\t\t: 1\n\n"
        )
        with patch("autoresearch.core.hardware.sys.platform", "linux"):
            with patch("builtins.open", mock_open(read_data=cpuinfo)):
                self.assertEqual(hardware.detect_physical_cores(), 2)


class TestDetectSimdHints(unittest.TestCase):
    def test_linux_parses_cpuinfo_flags(self):
        cpuinfo = "processor\t: 0\nflags\t\t: fpu vme sse4_2 avx2 avx512f avx512_vnni\n"
        with patch("autoresearch.core.hardware.sys.platform", "linux"):
            with patch("builtins.open", mock_open(read_data=cpuinfo)):
                self.assertEqual(
                    hardware.detect_simd_hints(),
                    ["avx512_vnni", "avx512f", "avx2", "sse4_2"],
                )

    def test_darwin_normalizes_mac_style_flags(self):
        proc = MagicMock(stdout="SSE4.2 AVX1.0 AVX2.0 AVX512F F16C")
        with patch("autoresearch.core.hardware.sys.platform", "darwin"):
            with patch("autoresearch.core.hardware.subprocess.run", return_value=proc) as run:
                self.assertEqual(
                    hardware.detect_simd_hints(),
                    ["avx512f", "avx2", "avx", "sse4_2", "f16c"],
                )
        run.assert_called_once()

    def test_linux_read_failure_returns_empty(self):
        with patch("autoresearch.core.hardware.sys.platform", "linux"):
            with patch("builtins.open", side_effect=OSError("no /proc/cpuinfo")):
                self.assertEqual(hardware.detect_simd_hints(), [])

    def test_returns_empty_when_no_relevant_flags(self):
        cpuinfo = "flags\t\t: fpu vme pse\n"
        with patch("autoresearch.core.hardware.sys.platform", "linux"):
            with patch("builtins.open", mock_open(read_data=cpuinfo)):
                self.assertEqual(hardware.detect_simd_hints(), [])


if __name__ == "__main__":
    unittest.main()
