# 06 — CPU Inference Guide

**What to build:** A user-facing guide at `docs/discovery/cpu-inference-guide.md` covering CPU-optimized inference with `llama.cpp`. Sections: llama.cpp CPU build instructions (AVX-512, AMX flags), Intel vs AMD CPU architecture differences for inference, NUMA configuration (`--numa isolate` / `distribute`), thread affinity best practices (`-t` = physical cores only), memory allocator optimization (`tcmalloc` / `jemalloc`), and recommended GGUF quantizations for CPU cache efficiency. Update `docs/discovery/AGENTS.md` Child DOX Index with the new entry.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `docs/discovery/cpu-inference-guide.md` created with all sections
- [ ] Intel AMX / AVX-512 build instructions included
- [ ] AMD Zen 4/5 AVX-512 benefits documented
- [ ] NUMA, tcmalloc, and thread affinity best practices covered
- [ ] `docs/discovery/AGENTS.md` Child DOX Index updated
