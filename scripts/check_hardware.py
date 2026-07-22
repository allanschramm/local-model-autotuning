"""
check_hardware.py — Diagnóstico Automático de Hardware para IA Local
Recomenda modelos GGUF, tamanho de contexto e flags para llama.cpp com base no seu PC.
"""

import os
import sys
import subprocess

def get_system_info():
    info = {
        "ram_gb": 0.0,
        "vram_gb": 0.0,
        "gpu_name": "Não detectada (CPU)",
        "has_cuda": False
    }

    # RAM do Sistema
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        if sys.platform == "win32":
            res = subprocess.run(["wmic", "computersystem", "get", "TotalPhysicalMemory"], capture_output=True, text=True)
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip().isdigit()]
            if lines:
                info["ram_gb"] = round(int(lines[0]) / (1024**3), 1)

    # VRAM da GPU via nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                info["gpu_name"] = parts[0]
                info["vram_gb"] = round(float(parts[1]) / 1024.0, 1)
                info["has_cuda"] = True
    except Exception:
        pass

    return info

def generate_recommendations(info):
    vram = info["vram_gb"]
    ram = info["ram_gb"]

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print(" [PC] RECOMENDADOR DE IA LOCAL (Diagnóstico de Hardware)")
    print("=" * 60)
    print(f" * Processador / RAM: {ram} GB RAM")
    print(f" * Placa de Vídeo (GPU): {info['gpu_name']}")
    print(f" * VRAM Dedicada: {vram} GB VRAM")
    print("-" * 60)

    if vram >= 15.0:
        tier = "Alto Desempenho (16GB+ VRAM)"
        model = "Qwen2.5-Coder-7B-Instruct (GGUF Q8_0 ou Q4_K_M) / Llama-3.1-8B"
        ngl = "99 (Carregar 100% na GPU)"
        ctx = "32768"
        tps_est = "40-80 TPS"
    elif vram >= 7.5:
        tier = "Intermediário (8GB-12GB VRAM)"
        model = "Qwen2.5-Coder-7B-Instruct (GGUF Q4_K_M) ou DeepSeek-R1-Distill-Qwen-7B"
        ngl = "33 (Todas as camadas na GPU)"
        ctx = "16384"
        tps_est = "25-50 TPS"
    elif vram >= 3.5:
        tier = "Básico GPU (4GB-6GB VRAM)"
        model = "Qwen2.5-Coder-3B-Instruct (GGUF Q4_K_M) ou Phi-3.5-mini"
        ngl = "24 (Parcial na GPU)"
        ctx = "8192"
        tps_est = "15-35 TPS"
    else:
        tier = "Modo CPU / Sem GPU Dedicada"
        model = "Qwen2.5-Coder-1.5B / Llama-3.2-1B-Instruct (GGUF Q4_K_M)"
        ngl = "0 (Somente CPU)"
        ctx = "4096"
        tps_est = "5-12 TPS"

    print(f"\n🎯 CLASSIFICAÇÃO: {tier}")
    print(f" 📦 Modelo Recomendado: {model}")
    print(f" ⚙️ Flag recomendada -ngl: {ngl}")
    print(f" 📝 Tamanho de Contexto -c: {ctx}")
    print(f" 🚀 Velocidade Estimada: {tps_est}")
    print("\n💡 Exemplo de comando recomendado para o llama.cpp:")
    print(f"  llama-server.exe -m models/seu-modelo.gguf -ngl {ngl.split()[0]} -c {ctx} --host 127.0.0.1 --port 8080")
    print("=" * 60)

if __name__ == "__main__":
    sys_info = get_system_info()
    generate_recommendations(sys_info)
