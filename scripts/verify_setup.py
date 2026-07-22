"""
verify_setup.py — Validação Automática de IA Local e Medição de TPS
Testa se o seu servidor de IA Local (llama-server ou LM Studio) está rodando e mede o desempenho real (TPS).
"""

import argparse
import sys
import time
import urllib.request
import urllib.error
import json

def performance_advice(tps):
    if tps >= 30:
        return "🎉 EXCELENTE! Desempenho alto. Pronto para agente e dev!"
    if tps >= 15:
        return "👍 BOM! Velocidade confortável para chat e código."
    return "⚠️ TPS baixo: reduza o contexto ou a KV cache; se necessário, escolha um modelo menor que caiba na VRAM física."


def check_server(port=18080, host="127.0.0.1"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    url_models = f"http://{host}:{port}/v1/models"
    url_chat = f"http://{host}:{port}/v1/chat/completions"

    print("=" * 60)
    print(f" 🧪 TESTANDO CONEXÃO COM IA LOCAL (http://{host}:{port})")
    print("=" * 60)

    # 1. Verificar se a porta/servidor está respondendo
    try:
        req = urllib.request.Request(url_models)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            model_id = models[0]["id"] if models else "modelo-local"
            print(f" [OK] Servidor ativo! Modelo detectado: {model_id}")
    except urllib.error.URLError:
        print(f" [X] Servidor NÃO encontrado na porta {port}.")
        print("\n💡 Como resolver:")
        print(" 1. Inicie o llama-server ou LM Studio Local Server.")
        print(f" 2. Verifique se a porta configurada é a {port}.")
        print(" 3. Configure autoresearch/core/config.py e inicie com scripts/serve-config.py serve.")
        print("=" * 60)
        sys.exit(1)

    # 2. Enviar prompt de teste e medir TPS
    print("\n 🚀 Medindo velocidade de geração (TPS)...")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Escreva uma lista rápida de 5 frutas e seus benefícios em português."}
        ],
        "max_tokens": 150,
        "temperature": 0.2
    }

    start_time = time.time()
    try:
        req = urllib.request.Request(
            url_chat,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.time() - start_time
            res_data = json.loads(resp.read().decode("utf-8"))

            usage = res_data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            if not completion_tokens and "choices" in res_data:
                text = res_data["choices"][0]["message"]["content"]
                completion_tokens = len(text.split()) * 1.3 # estimativa se omitido

            tps = completion_tokens / elapsed if elapsed > 0 else 0

            print(f" [OK] Resposta recebida com sucesso!")
            print(f" • Tempo total: {elapsed:.2f}s")
            print(f" • Tokens gerados: {completion_tokens}")
            print(f" • Desempenho real: {tps:.1f} TPS (tokens por segundo)")
            print("-" * 60)

            print(f" {performance_advice(tps)}")
            print("=" * 60)

    except Exception as e:
        print(f" [X] Erro ao testar geração: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida uma API local compatível com OpenAI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    check_server(port=args.port, host=args.host)
