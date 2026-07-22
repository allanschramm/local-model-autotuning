from scripts.check_hardware import generate_recommendations
from scripts.verify_setup import performance_advice


def test_dense_gpu_recommendation_never_uses_partial_offload(capsys):
    generate_recommendations(
        {
            "ram_gb": 16.0,
            "vram_gb": 4.0,
            "gpu_name": "Test GPU",
            "has_cuda": True,
        }
    )

    output = capsys.readouterr().out
    assert "-ngl: 99" in output
    assert "Parcial na GPU" not in output
    assert "deve caber" in output


def test_cpu_recommendation_remains_cpu_only(capsys):
    generate_recommendations(
        {
            "ram_gb": 16.0,
            "vram_gb": 0.0,
            "gpu_name": "Não detectada (CPU)",
            "has_cuda": False,
        }
    )

    output = capsys.readouterr().out
    assert "-ngl: 0" in output
    assert "Somente CPU" in output


def test_low_tps_advice_does_not_suggest_dense_layer_spill():
    advice = performance_advice(10)

    assert "aumente a flag -ngl" not in advice
    assert "contexto" in advice.lower()
    assert "modelo menor" in advice.lower()
