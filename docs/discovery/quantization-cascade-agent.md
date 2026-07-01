# Quant Selection — Agent Quick Reference

Read full version: [`quantization-cascade.md`](./quantization-cascade.md)

## Rules

1. Pick highest bit-depth that fits VRAM
2. UD beats standard at same bit-depth
3. Never offload to CPU if avoidable
4. Flash attention always on

## Decision

```
VRAM >= 24GB  → UD-Q5_K_M (or Q5_K_M)
VRAM 16-24GB → UD-Q4_K_XL or UD-IQ4_NL
VRAM 12-16GB → UD-Q4_K_M or UD-IQ4_XS
VRAM 8-12GB  → UD-IQ4_XS, else UD-IQ3_XS
VRAM < 8GB   → UD-Q4_K_M (7B) or UD-IQ3_XS (13B)
```

## Priority

```
UD-Q4_K_XL > UD-Q4_K_M > Q4_K_M > Q3_K_M > Q2_K
UD-IQ4_NL  > UD-IQ4_XS > IQ4 > IQ3
```

## Gotchas

- Q4_K_XL often beats Q5_K_M on size, but Q5 safer for logic
- UD-IQ3_XS: last resort, expect reasoning drop
- Q3_K_M / Q2_K: avoid, only if forced
- Model name must end `.gguf` — nothing else works with llama.cpp
