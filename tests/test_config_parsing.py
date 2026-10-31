import unittest

from autoresearch.core.config import load_config
from autoresearch.core.llama_runner import validate_config


class TestRuntimeInvariants(unittest.TestCase):
    def test_validate_config_accepts_custom_tps_floor(self):
        cfg = load_config()
        cfg["N_CPU_MOE"] = None
        cfg["TPS_FLOOR"] = 15.0
        out = validate_config(cfg)
        self.assertEqual(out["TPS_FLOOR"], 15.0)


if __name__ == "__main__":
    unittest.main()
