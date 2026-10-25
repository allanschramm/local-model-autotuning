"""Median-of-reps llama-cli bench helper."""

from autoresearch.core import runtime_measurements as evaluation


def test_median_llama_bench_validation(monkeypatch):
    seq = [10.0, 30.0, 20.0]

    def fake_run(*_a, **_k):
        return seq.pop(0)

    monkeypatch.setattr(evaluation, "run_llama_bench_validation", fake_run)
    monkeypatch.setattr(evaluation, "wait_gpu_near_idle", lambda **_k: 40.0)
    median, values, temp = evaluation.median_llama_bench_validation(reps=3)
    assert median == 20.0
    assert values == [10.0, 30.0, 20.0]
    assert temp == 40.0
    assert evaluation._tps_spread(values, median) == 100.0
