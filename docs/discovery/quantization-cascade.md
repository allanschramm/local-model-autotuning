# Quantization Format Selection Guide

Best Choice Cascade for selecting a GGUF quantization format. Prioritizes **quality first**, then **size efficiency**, then **hardware constraints**.

## The Golden Rule

**Always choose the highest bit-depth that fits in your VRAM.**
If you have room, **Unsloth Dynamic (UD)** versions are **always superior** to standard versions of the same bit-depth.

## The Best Choice Cascade (Top to Bottom)

Use this decision tree. Stop at the first option that fits in your VRAM.

### Tier 1: Maximum Fidelity (massive VRAM)

1. **BF16 / FP16** (Original Precision)
   - Best for: research, training, or 80GB+ VRAM (A100/H100). Zero quality loss, but huge file size.
2. **UD-Q8_0** (Unsloth Dynamic 8-bit)
   - Best for: near-perfect accuracy with slightly better speed than BF16.
   - Note: `UD` here is rare; standard `Q8_0` is usually fine.
3. **Q8_0** (Standard 8-bit)
   - Best for: "lossless" experience for most users. Indistinguishable from BF16 for chat/coding.

### Tier 2: The Sweet Spot (High Quality, Reasonable Size)

4. **UD-Q5_K_M** or **UD-Q5_K_XL** — **Highly Recommended**
   - Why: the absolute best balance for most users. `UD` optimizes layers to match or beat standard Q6/Q7.
   - Use when: you have 16GB–24GB VRAM.
5. **Q5_K_M** (Standard 5-bit)
   - Why: excellent accuracy, slightly larger than `UD-Q5` but very stable.
6. **UD-Q4_K_XL** / **UD-Q4_K_M** — **Best 4-bit**
   - Why: the new standard. `UD` makes these smarter than standard Q5s in some cases.
   - Use when: you have 12GB–16GB VRAM.

### Tier 3: The Efficiency Zone (Aggressive but Smart)

7. **UD-IQ4_NL** / **UD-IQ4_XS**
   - Why: best 4-bit compression. `NL` is highest quality, `XS` is smallest.
   - Use when: you need to save space but refuse to drop to 3-bit.
8. **Q4_K_M** (Standard 4-bit)
   - Why: the classic "good enough" choice. Reliable, but `UD` versions beat it.

### Tier 4: The Last Resort (Tiny VRAM)

9. **UD-IQ3_XS** / **UD-IQ2_M**
   - Why: only choose these if 4-bit doesn't fit. `UD` saves them from being terrible.
   - Warning: expect some quality drop in reasoning and logic.
10. **Q3_K_M** / **Q2_K**
    - Why: avoid if possible. Only use if forced to run a 70B model on 8GB VRAM.

## Quick Decision Matrix

| Your VRAM | Model Size | Best Choice | Why? |
|:---|:---|:---|:---|
| 24GB+ | 70B | `UD-Q5_K_M` or `Q5_K_M` | Max intelligence without BF16 bloat. |
| 16GB–20GB | 35B–70B | `UD-Q4_K_XL` or `UD-IQ4_NL` | Best 4-bit quality. Fits comfortably. |
| 12GB–16GB | 30B–35B | `UD-Q4_K_M` or `UD-IQ4_XS` | Great balance of speed and smarts. |
| 8GB–10GB | 13B–30B | `UD-IQ4_XS` (if fits) else `UD-IQ3_XS` | Squeeze 4-bit in first; drop to 3-bit only if needed. |
| < 8GB | 7B–13B | `UD-Q4_K_M` (7B) or `UD-IQ3_XS` (13B) | Limited by bit-depth. |

## Summary Rules

1. **Bit-Depth Wins:** 4-bit (`Q4`/`IQ4`) > 3-bit (`Q3`/`IQ3`) > 2-bit.
2. **UD Wins:** `UD-Q4` > `Standard Q4`. `UD-IQ4` > `Standard IQ4`.
3. **The XL Trick:** `Q4_K_XL` is often better than `Q5_K_M` for file size, but `Q5_K_M` is safer for logic.
4. **VRAM is King:** If `UD-Q4` makes your GPU crash, drop to `UD-IQ4_XS`. If that crashes, drop to `UD-Q3_K_M`. Never run a model that offloads to CPU unless you have no choice.

## Final Recommendation

If you see a file named **`UD-Q4_K_XL`** or **`UD-IQ4_NL`**, download that one first. It is the current state-of-the-art for local LLMs.

## Related Docs

- `docs/models/` — per-model GGUF specs and architecture notes
- `docs/discovery/discover-models.md` — end-to-end model selection workflow
