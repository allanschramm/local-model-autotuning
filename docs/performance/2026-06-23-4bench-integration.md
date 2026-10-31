# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-06-23-4bench-integration.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

**Root Causes Identified**:
1. **Think-only responses**: When a model used all tokens thinking (`<think>...</think>`), dropping the `<think>` block resulted in an empty string.
2. **Prose prefix leakage**: When the model returned `Here is the solution:\n\n<code>` without fences, the text was executed directly in Python, causing `SyntaxError`s but masked as FAIL.
3. **Truncated think blocks**: Cut-off thinking generated unclosed tags, executing partial thought as code.
4. **MTP Reasoning Field**: `llama-server` emits think tokens into a separate `reasoning_content` field (when using `--jinja`). `LlamaClient` was completely ignoring `reasoning_content`, meaning think-only responses were treated as empty `content=""`.
