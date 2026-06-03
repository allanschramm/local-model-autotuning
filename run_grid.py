#!/usr/bin/env python3
import subprocess
import csv
import os
import re
from pathlib import Path

# Configuration
KV_CACHES = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
MAX_TOKENS_LIST = [1024, 2048]
MODEL = "gemma-4-E4B-it-Q4_K_M.gguf"
CTX_SIZE = 131072

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "grid_results.csv"
SCRIPTS = {
    "nexus": BASE_DIR / "benchmark_search.py",
    "claw": BASE_DIR / "benchmark_search_claw.py",
    "coding": BASE_DIR / "benchmark_coding.py",
}


def parse_metrics(output):
    metrics = {}
    patterns = {
        "val_score": r"val_score:\s+([\d\.]+)",
        "val_retrieval": r"val_retrieval:\s+([\d\.]+)",
        "val_agency": r"val_agency:\s+([\d\.]+)",
        "tokens_per_sec": r"tokens_per_sec:\s+([\d\.]+)",
        "peak_vram_mb": r"peak_vram_mb:\s+([\d\.]+)",
        "humaneval_pass1": r"humaneval:\s+.*'pass1':\s+([\d\.]+)",
        "mbpp_pass1": r"mbpp:\s+.*'pass1':\s+([\d\.]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = match.group(1)
    return metrics

def run_benchmark(script, kv_cache, max_tokens):
    cmd = [
        "python3", script,
        "--model", MODEL,
        "--ctx-size", str(CTX_SIZE),
        "--kv-cache", kv_cache,
        "--max-tokens", str(max_tokens)
    ]
    
    # Se for o script de coding, passamos o range lite
    if "benchmark_coding.py" in script:
        # Usamos uma variável de ambiente ou argumento se suportado
        # Vou assumir que vamos rodar as tasks 0 a 14 (15 tasks)
        cmd += ["--id_range", "[0, 15]"]
        
    print(f"\n>>> Running: {' '.join(cmd)}")
    try:
        # We use -u for unbuffered output to see it in real-time if redirected
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout + "\n" + result.stderr
    except Exception as e:
        return f"ERROR: {e}"

def health_check(kv_cache):
    """Verifica se o servidor sequer sobe com esse KV cache (previne crashes fatais)."""
    # Usamos o benchmark_search.py apenas para tentar o boot
    cmd = [
        "python3", SCRIPTS["nexus"],
        "--model", MODEL,
        "--ctx-size", str(CTX_SIZE),
        "--kv-cache", kv_cache,
        "--max-tokens", "128" # maxtok baixo para boot rápido
    ]
    print(f"  [HealthCheck] Testing KV={kv_cache}...")
    try:
        # Timeout curto para o boot (se não subir em 120s, algo está errado)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if "FAIL: Server crashed" in result.stdout or "FAIL:" in result.stdout:
            return False, result.stdout
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Timeout during health check"
    except Exception as e:
        return False, str(e)

def main():
    headers = [
        "kv_cache", "max_tokens", "status",
        "nexus_val_score", "nexus_retrieval", "nexus_agency", "nexus_tps", "nexus_vram",
        "claw_val_score", "claw_retrieval", "claw_agency", "claw_tps", "claw_vram",
        "humaneval_pass1", "mbpp_pass1", "coding_vram"
    ]
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    for kv in KV_CACHES:
        # Roda o Health Check uma vez por tipo de KV
        supported, reason = health_check(kv)
        if not supported:
            print(f"  [SKIP] KV={kv} is not supported or crashed: {reason.splitlines()[-1] if reason else 'Unknown'}")
            # Registra como UNSUPPORTED
            with open(RESULTS_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow({"kv_cache": kv, "max_tokens": "N/A", "status": "UNSUPPORTED"})
            continue

        for mt in MAX_TOKENS_LIST:
            print(f"\n{'='*80}")
            print(f"CONFIG: KV={kv}, MAX_TOKENS={mt}")
            print(f"{'='*80}")
            
            row = {"kv_cache": kv, "max_tokens": mt, "status": "OK"}
            
            # 1. Nexus
            nexus_out = run_benchmark(SCRIPTS["nexus"], kv, mt)
            nexus_metrics = parse_metrics(nexus_out)
            row.update({
                "nexus_val_score": nexus_metrics.get("val_score"),
                "nexus_retrieval": nexus_metrics.get("val_retrieval"),
                "nexus_agency": nexus_metrics.get("val_agency"),
                "nexus_tps": nexus_metrics.get("tokens_per_sec"),
                "nexus_vram": nexus_metrics.get("peak_vram_mb")
            })
            
            # 2. Claw
            claw_out = run_benchmark(SCRIPTS["claw"], kv, mt)
            claw_metrics = parse_metrics(claw_out)
            row.update({
                "claw_val_score": claw_metrics.get("val_score"),
                "claw_retrieval": claw_metrics.get("val_retrieval"),
                "claw_agency": claw_metrics.get("val_agency"),
                "claw_tps": claw_metrics.get("tokens_per_sec"),
                "claw_vram": claw_metrics.get("peak_vram_mb")
            })
            
            # 3. Coding
            coding_out = run_benchmark(SCRIPTS["coding"], kv, mt)
            coding_metrics = parse_metrics(coding_out)
            row.update({
                "humaneval_pass1": coding_metrics.get("humaneval_pass1"),
                "mbpp_pass1": coding_metrics.get("mbpp_pass1"),
                "coding_vram": coding_metrics.get("peak_vram_mb")
            })
            
            # Save row
            with open(RESULTS_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
            
            print(f"\nCompleted configuration: KV={kv}, MT={mt}")
            print(f"Nexus Score: {row['nexus_val_score']}, Claw Score: {row['claw_val_score']}, HumanEval: {row['humaneval_pass1']}")

if __name__ == "__main__":
    main()