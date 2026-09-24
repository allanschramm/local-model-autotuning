import unittest
from pathlib import Path

from autoresearch.core.llama_runner import (
    VRAM_QUANT_FACTORS,
    estimate_vram_mb,
)


class TestAdversarialChallenger(unittest.TestCase):
    def test_vram_quant_factors_monotonicity(self):
        """3. Validate VRAM quant factors: turbo4 > turbo3 > turbo2."""
        t4 = VRAM_QUANT_FACTORS.get("turbo4")
        t3 = VRAM_QUANT_FACTORS.get("turbo3")
        t2 = VRAM_QUANT_FACTORS.get("turbo2")

        self.assertIsNotNone(t4, "turbo4 factor is missing")
        self.assertIsNotNone(t3, "turbo3 factor is missing")
        self.assertIsNotNone(t2, "turbo2 factor is missing")

        # Verify strictly monotonic descending (greater factors mean more memory)
        self.assertGreater(t4, t3, "VRAM factor for turbo4 must be greater than turbo3")
        self.assertGreater(t3, t2, "VRAM factor for turbo3 must be greater than turbo2")

        # Print for verification logs
        print(f"[Monotonicity Check] turbo4={t4}, turbo3={t3}, turbo2={t2}")

    def test_vram_factor_resolution(self):
        """Validate resolution of VRAM factors under different cases."""
        # 1. Standard resolution
        v_turbo4 = estimate_vram_mb(
            Path("non-existent"), 2048, kv_cache_k="turbo4", kv_cache_v="turbo4"
        )
        v_turbo3 = estimate_vram_mb(
            Path("non-existent"), 2048, kv_cache_k="turbo3", kv_cache_v="turbo3"
        )
        v_turbo2 = estimate_vram_mb(
            Path("non-existent"), 2048, kv_cache_k="turbo2", kv_cache_v="turbo2"
        )

        self.assertGreater(v_turbo4, v_turbo3)
        self.assertGreater(v_turbo3, v_turbo2)

        # 2. Mixed case string resolution
        v_mixed = estimate_vram_mb(
            Path("non-existent"), 2048, kv_cache_k="TuRbO4", kv_cache_v="tUrBo4"
        )
        self.assertEqual(v_turbo4, v_mixed)

        # 3. Fallback resolution for unknown quantization types
        v_unknown = estimate_vram_mb(
            Path("non-existent"), 2048, kv_cache_k="unknown-format", kv_cache_v="unknown-format"
        )
        factor_est = estimate_vram_mb(
            Path("non-existent"), 2048, kv_cache_k="nonexistent", kv_cache_v="nonexistent"
        )

        model_size_mb = 4000.0
        kv_base_mb = 2048 * 80.0 / 1024.0  # 160.0 MB
        expected_kv = (160.0 / 2.0) * 0.3 + (160.0 / 2.0) * 0.3  # 48.0 MB
        expected_total = model_size_mb + expected_kv + 300.0  # 4348.0 MB
        self.assertAlmostEqual(factor_est, expected_total)


if __name__ == "__main__":
    unittest.main()
