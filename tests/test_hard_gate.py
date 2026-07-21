import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "scripts" / "hooks" / "block-adhoc-eval.ps1"


class TestHardGate(unittest.TestCase):
    @unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell required")
    def test_rejects_baseline_cli_overrides(self):
        payload = json.dumps({
            "command": (
                'venv/Scripts/python.exe benchmark_search.py --desc "ornith" '
                '--model "Ornith-1.0-35B-UD-Q3_K_XL.gguf" --threads 8 '
                '--threads-batch 8 --batch-size 1024 --ubatch-size 256'
            ),
            "cwd": str(ROOT),
            "workspace_roots": [str(ROOT)],
        })
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HOOK)],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
        )

        decision = json.loads(result.stdout)
        self.assertEqual(decision["permission"], "deny")
        self.assertIn("config.py", decision["user_message"])


if __name__ == "__main__":
    unittest.main()
