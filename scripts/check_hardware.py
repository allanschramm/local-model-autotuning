"""
check_hardware.py — Diagnóstico Automático de Hardware para IA Local
Recomenda modelos GGUF, tamanho de contexto e flags para llama.cpp com base no seu PC.
"""

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
        model = "GGUF denso que caiba integralmente na VRAM física"
        ngl = "99 (100% GPU; modelo denso deve caber na VRAM física)"
        ctx = "32768"
    elif vram >= 7.5:
        tier = "Intermediário (8GB-12GB VRAM)"
        model = "GGUF denso compacto que caiba integralmente na VRAM física"
        ngl = "99 (100% GPU; modelo denso deve caber na VRAM física)"
        ctx = "16384"
    elif vram >= 3.5:
        tier = "Básico GPU (4GB-6GB VRAM)"
        model = "GGUF denso pequeno que caiba integralmente na VRAM física"
        ngl = "99 (100% GPU; modelo denso deve caber na VRAM física)"
        ctx = "8192"
    else:
        tier = "Modo CPU / Sem GPU Dedicada"
        model = "GGUF pequeno adequado à RAM disponível"
        ngl = "0 (Somente CPU)"
        ctx = "4096"

    print(f"\n🎯 CLASSIFICAÇÃO: {tier}")
    print(f" 📦 Modelo Recomendado: {model}")
    print(f" ⚙️ Flag recomendada -ngl: {ngl}")
    print(f" 📝 Tamanho de Contexto -c: {ctx}")
    print(" 🚀 Velocidade: meça no seu hardware; não use estimativas como resultado")
    print("\n💡 Aplique estes valores em autoresearch/core/config.py e inicie com:")
    print("  .\\venv\\Scripts\\python.exe scripts\\serve-config.py serve")
    print("=" * 60)

if __name__ == "__main__":
    sys_info = get_system_info()
    generate_recommendations(sys_info)
