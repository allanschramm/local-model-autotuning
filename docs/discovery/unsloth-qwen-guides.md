
=== /home/shark/.gemini/antigravity-cli/brain/74f162d7-085f-4761-8abd-e730ac5027a2/.system_generated/steps/383/content.md ===

For the complete documentation index, see
llms.txt
. This page is also available as
Markdown
.
Copy
On this page
Models
💜
Qwen3.6 - How to Run Locally
Run the new Qwen3.6-27B and 35B-A3B models locally!
Qwen3.6 is Alibaba’s new family of multimodal hybrid-thinking models, including:
Qwen3.6-27B
and
35B-A3B
. It delivers top performance for its size, supports 256K context across 201 languages. It excels in agentic coding, vision, chat tasks. Qwen3.6-27B runs on
18GB RAM
setups and 35B-A3B runs on
22GB
. You can now run and train the models in
Unsloth Studio
.
NEW:
Qwen3.6 MTP is here
! MTP enables 1.4-2.2x faster inference without accuracy loss. Run MTP directly in
Unsloth Studio
.
We conducted
Qwen3.6 GGUF Benchmarks
to help you pick the best quant.
Run Qwen3.6 Tutorials
MTP Guide
Qwen3.6 GGUFs use Unsloth
Dynamic 2.0
for SOTA quant performance - so quants are calibrated on real world use-case datasets and important layers are upcasted.
Thank you Qwen for day zero access.
Developer Role Support
for Codex, OpenCode and more:
Our uploads now support the
developer role
for agentic coding tools.
Tool calling:
Like
Qwen3.5
, we improved parsing nested objects to make tool calling succeed more.
Qwen3.6 running in
Unsloth Studio
.
⚙️
Usage Guide
Table: Inference hardware requirements
(units = total memory: RAM + VRAM, or unified memory)
Qwen3.6
3-bit
4-bit
6-bit
8-bit
BF16
27B
15 GB
18 GB
24 GB
30 GB
55 GB
35B-A3B
17 GB
23 GB
30 GB
38 GB
70 GB
For best performance, make sure your total available memory (VRAM + system RAM) exceeds the size of the quantized model file you’re downloading. If it doesn’t, llama.cpp can still run via SSD/HDD offloading, but inference will be slower.
Do NOT use CUDA 13.2 as you may get gibberish outputs. Use below CUDA 13.2 or CUDA 13.3.
To train Qwen3.6, you can refer to our previous
Qwen3.5 fine-tuning guide
.
Recommended Settings
Maximum context window:
262,144
(can be extended to 1M via YaRN)
presence_penalty = 0.0 to 2.0
default this is off, but to reduce repetitions, you can use this, however using a higher value may result in
slight decrease in performance
Adequate Output Length
:
32,768
tokens for most queries
If you're getting gibberish, your context length might be set too low. Or try using
--cache-type-k bf16 --cache-type-v bf16
which might help.
As Qwen3.6 is hybrid reasoning, thinking and non-thinking mode have different settings:
Thinking mode:
Qwen3.6 now has
Preserve Thinking
.
General tasks
Precise coding tasks (e.g. WebDev)
temperature = 1.0
temperature = 0.6
top_p = 0.95
top_p = 0.95
top_k = 20
top_k = 20
min_p = 0.0
min_p = 0.0
presence_penalty = 0.0
presence_penalty = 0.0
repeat_penalty = disabled or 1.0
repeat_penalty = disabled or 1.0
Thinking mode for general tasks:
Thinking mode for precise coding tasks:
Instruct (non-thinking) mode settings:
General tasks
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repeat_penalty = disabled or 1.0
To
disable thinking / reasoning
, use
--chat-template-kwargs '{"enable_thinking":false}'
If you're on
Windows
Powershell, use:
--chat-template-kwargs "{\"enable_thinking\":false}"
Use 'true' and 'false' interchangeably.
Instruct (non-thinking) for general tasks:
Qwen3.6 Inference Tutorials:
We'll be using Dynamic 4-bit
UD-Q4_K_XL
GGUF variants for inference workloads. Click below to navigate to designated model instructions:
Run in Unsloth Studio
Run in llama.cpp
MTP Guide
NVFP4 Guide
Do NOT use CUDA 13.2 as you may get gibberish outputs. Use below CUDA 13.2 or CUDA 13.3.
🦥 Unsloth Studio Guide
Qwen3.6 and Qwen3.6 MTP can now be run in
Unsloth Studio
, our new open-source web UI for local AI. Unsloth Studio lets you run models locally on
MacOS, Windows
, Linux and:
Search, download,
run GGUFs
and safetensor models
Self-healing
tool calling
+
web search
Code execution
(Python, Bash)
Automatic inference
parameter tuning (temp, top-p, etc.)
Fast CPU + GPU inference via llama.cpp
Train LLMs
2x faster with 70% less VRAM
1
Install Unsloth
Run in your terminal:
MacOS, Linux, WSL:
Windows PowerShell:
Installation will be quick and take approx 20 sec - 1 mins.
2
Launch Unsloth
MacOS, Linux, WSL and Windows:
Then open
http://127.0.0.1:8888
(or your specific URL) in your browser.
3
Search and download Qwen3.6 or Qwen3.6 MTP
On first launch you will need to create a password to secure your account and sign in again later. Then go to the
Studio Chat
tab and search for Qwen3.6 or Qwen3.6 MTP in the search bar and download your desired model and quant.
4
Run Qwen3.6
Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings.
For more information, you can view our
Unsloth Studio inference guide
. Below, the 2-bit Qwen3.6 GGUF made 30+ tool calls, searched 20 sites and executed Python code:
⚡ MTP Guide
MTP (Multi Token Prediction)
speculative decoding enables models like Qwen3.6 to have
~1.4-2.2x faster generation with
no change in accuracy
. This enables Qwen3.6 27B and 35B-A3B to have
>1.4x speed-up
over the original baseline which is especially useful for local models.
Unsloth Qwen3.6 MTP GGUFs are no longer in experimental mode, and llama.cpp has merged MTP support. Run directly in
Unsloth Studio’s UI
or via llama.cpp.
Qwen3.6 27B MTP now runs at 160 tokens/s generation and Qwen3.6 35B-A3B at 240 tokens/s on a RTX 6000 GPU.
See
MTP Benchmarks
.
Unsloth Studio automatically sets the ideal MTP settings optimized for your specific hardware (Mac, CPU, GPU etc.) - you can still change it later.
MTP uses slightly more VRAM than standard GGUFs
, so plan for ~1 GB additional RAM/VRAM headroom.
Run in Unsloth Studio
Run in llama.cpp
Run NVFP4
Qwen3.6-27B-MTP-GGUF
Qwen3.6-35B-A3B-MTP-GGUF
In practice, MTP predicts several future tokens, then the main model verifies those tokens in parallel. This reduces the number of forward passes needed during generation and make output faster.
We found
--spec-draft-n-max 2
to work best in most setups.
However, do not assume
2
is optimal, as performance is hardware-dependent. Try values from
1
through
6
and use whichever is fastest for your system.
We also
uploaded MTP GGUFs
for the
Qwen3.5
model family
including: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B and 397B-A17B. Llama.cpp is continually improving MTP performance, so expect it to get faster overtime!
Table: MTP hardware requirements
(units = total memory: RAM + VRAM, or unified memory)
Qwen3.6
3-bit
4-bit
6-bit
8-bit
BF16
27B
16 GB
19 GB
25 GB
31 GB
56 GB
35B-A3B
18 GB
24 GB
31 GB
39 GB
71 GB
🦥 Unsloth Studio MTP Guide
Unsloth Studio automatically sets the ideal MTP settings optimized for your specific hardware (Mac, CPU, GPU etc.) - you can still change it later.
1
Install Unsloth
Run in your terminal:
MacOS, Linux, WSL:
Windows PowerShell:
2
Launch Unsloth
MacOS, Linux, WSL and Windows:
Then open
http://127.0.0.1:8888
(or your specific URL) in your browser.
3
Search and download Qwen3.6 MTP
On first launch you will need to create a password to secure your account and sign in again later. Then go to the
Studio Chat
tab and search for Qwen3.6 MTP in the search bar and download your desired model and quant.
4
Run Qwen3.6 MTP
Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings.
For more information, you can view our
Unsloth Studio inference guide
. Below, the 2-bit Qwen3.6 MTP GGUF made 10+ tool calls, searched 10 sites and executed Python code:
🦙 Llama.cpp MTP Guide
1
Install the latest version of
llama.cpp
on
GitHub here
. You can follow the build instructions below as well. Change
-DGGML_CUDA=ON
to
-DGGML_CUDA=OFF
if you don't have a GPU or just want CPU inference.
For Apple Mac / Metal devices
, set
-DGGML_CUDA=OFF
then continue as usual - Metal support is on by default.
2
If you want to use
llama.cpp
directly to load models, you can do the below: (:
Q4_K_XL
) is the quantization type. You can also download via Hugging Face (point 3). This is similar to
ollama run
. Use
export LLAMA_CACHE="folder"
to force
llama.cpp
to save to a specific location. The model has a maximum of 256K context length.
Follow one of the commands for the specific models:
27B MTP
35-A3B MTP
MTP Qwen3.6-27B:
Thinking mode:
Please see Qwen3.6's new
Preserved Thinking
.
General tasks:
For precise coding tasks, change:
temperature=0.6
Non-thinking mode:
General tasks:
MTP Qwen3.6-35B-A3B:
Thinking mode:
Please see Qwen3.6's new
Preserved Thinking
.
General tasks:
For precise coding tasks, change:
temperature=0.6
Non-thinking mode:
General tasks:
3
You can also download the model manually as well via the code below (after installing
pip install huggingface_hub
). You can choose Q4_K_M or other quantized versions like
UD-Q4_K_XL
. We recommend using at least 2-bit dynamic quant
UD-Q2_K_XL
to balance size and accuracy. If downloads get stuck, see:
Hugging Face Hub, XET debugging
4
Then run the model in conversation mode:
🦙 Llama.cpp Guide
For this guide we will be utilizing Dynamic 4-bit which works great on a 24GB RAM / Mac device for fast inference on
llama.cpp
. Because the model is only around 72GB at full F16 precision, we won't need to worry much about performance.
See our GGUF collection
.
27B
35-A3B
1
Obtain the latest
llama.cpp
on
GitHub here
. You can follow the build instructions below as well. Change
-DGGML_CUDA=ON
to
-DGGML_CUDA=OFF
if you don't have a GPU or just want CPU inference.
For Apple Mac / Metal devices
, set
-DGGML_CUDA=OFF
then continue as usual - Metal support is on by default.
2
If you want to use
llama.cpp
directly to load models, you can do the below: (:
Q4_K_XL
) is the quantization type. You can also download via Hugging Face (point 3). This is similar to
ollama run
. Use
export LLAMA_CACHE="folder"
to force
llama.cpp
to save to a specific location. The model has a maximum of 256K context length.
Follow one of the commands for the specific models:
27B
35-A3B
Qwen3.6-27B:
Thinking mode:
Please see Qwen3.6's new
Preserved Thinking
.
General tasks:
For precise coding tasks, change:
temperature=0.6
Non-thinking mode:
General tasks:
Qwen3.6-35B-A3B:
Thinking mode:
Please see Qwen3.6's new
Preserved Thinking
.
General tasks:
For precise coding tasks, change:
temperature=0.6
Non-thinking mode:
General tasks:
3
You can also download the model manually as well via the code below (after installing
pip install huggingface_hub
). You can choose Q4_K_M or other quantized versions like
UD-Q4_K_XL
. We recommend using at least 2-bit dynamic quant
UD-Q2_K_XL
to balance size and accuracy. If downloads get stuck, see:
Hugging Face Hub, XET debugging
4
Then run the model in conversation mode:
Llama-server & OpenAI completion library
To deploy Qwen3.6 for production, we use
llama-server
In a new terminal say via tmux, deploy the model via:
Then in a new terminal, after doing
pip install openai
, do:
🍎 MLX Dynamic Quants
We also uploaded dynamic Qwen3.6 4bit and 8bit quants for MacOS devices! Our MLX quant algorithm is still evolving, and we’re actively refining it wherever improvements can be made.
You can run all MLX models in
Unsloth Studio
!
Qwen3.6-27B MLX:
3-bit
4-bit
MXFP4
NVFP4
6-bit
8-bit
Qwen3.6-35B-A3B MLX:
3-bit
4-bit
8-bit
To try them out use:
See below for Qwen3.6-27B KL Divergence (KLD) and Perplexity (PPL) scores (lower is better):
Model
Mean KLD
Median KLD
PPL
P90 KLD
P99.9 KLD
Size
8-bit
0.0028
0.0003
4.812
0.0019
0.192
34.7 GB
6-bit
0.0037
0.0007
4.809
0.0032
0.343
30.5 GB
4-bit
0.0227
0.0053
4.821
0.0293
2.339
26.2 GB
NVFP4
0.0325
0.0087
4.843
0.0466
3.693
26.2 GB
MXFP4
0.0479
0.0153
4.902
0.0769
4.035
25.6 GB
3-bit
0.0734
0.0223
4.976
0.1261
5.529
24.1 GB
⚡️NVFP4
You can now run our NVFP4 quants with MTP tensors directly integrated inside the NVFP4 quant. Both
vLLM
and
SGLang
work for this. We tried
vllm==0.22.0
and
sglang==0.5.9
(you may need to use SGLang main).
Qwen3.6-35B-A3B NVFP4:
huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4
Qwen3.6-27B NVFP4:
huggingface.co/unsloth/Qwen3.6-27B-NVFP4
vLLM:
SGLang:
💡 Thinking: Enable/Disable + Preserve Thinking
Qwen3.6 also has
Preserve Thinking
which leaves the thinking trace from the previous conversation. This increases the number of tokens you use, but could increase accuracy in continued conversations. Unsloth Studio has 'Think' and Preserved Thinking toggles for Qwen3.6:
Unsloth Studio has Think toggle by default and a new
Preserved Thinking
toggle
To enable
preserve thinking
in llama.cpp use (change to 'true' or 'false') '
preserve_thinking
' instead of '
enable_thinking
' or '
disable_thinking
'.
For normal thinking, you can enable / disable thinking in llama.cpp by following the below commands. Use '
true
' and '
false
' interchangeably.
llama-server OS:
Enable Thinking
Disable Thinking
Linux, MacOS, WSL:
Windows / Powershell:
As an example for Qwen3.6-35B-A3B to enable preserve thinking (default is enabled):
And then in Python:
👨‍💻 OpenAI Codex & Claude Code
To run the model via local coding agentic workloads, you can
follow our guide
. Use the
llama-server
we just set up just then, and set the model name to the exact id it reports at
GET /v1/models
(the
--alias
value above, e.g.
unsloth/Qwen3.6-35B-A3B-GGUF
). Follow the correct Qwen3.6 parameters and usage instructions.
Claude Code
OpenAI Codex
After following the instructions for Claude Code for example you will see:
We can then ask say
Create a Python game for Chess
:
📊 Benchmarks
Unsloth GGUF Benchmarks
We conducted Mean KL Divergence benchmarks for Qwen3.6-35-A3B GGUFs across providers to help you pick the best quant.
KL Divergence puts nearly all Unsloth GGUFs on the SOTA Pareto frontier
KLD shows how well a quantized model matches the original BF16 output distribution, indicating retained accuracy.
This makes Unsloth the top-performing in 21 of 22 sizes
Only Q6_K was updated for more Dynamic layers and we introduced a new
UD-IQ4_NL_XL
quant
35B-A3B - KLD benchmarks (lower is better)
MTP Benchmarks
We benchmarked the new quants we made for 27B and 35B MoE. In general, dense models are much more accelerated with MTP (1.4-2x) vs MoE models (1.15-1.25x).
With this, Qwen3.6 27B can now do 140 tokens / s generation with UD-Q2_K_XL and Qwen3.6 35B-A3B 220 tokens / s generation! Some of the throughput numbers are noisy, so don't infer some quants are slower than others.
In terms of average speedup, we see a 1.4x for dense models at draft tokens = 2 and for the MoE around 1.15 to 1.2x.
We do not recommend more than 2 draft tokens because the acceptance rate drops precipitously from 83% to 50% with 4 draft tokens, and the forward passes for MTP become less beneficial.
Official Qwen Benchmarks
Qwen3.6-27B
Qwen3.6-35B-A3B
Previous
Fine-tune Gemma 4
Next
Kimi K2.7 Code
Last updated
5 minutes ago
Was this helpful?

=== /home/shark/.gemini/antigravity-cli/brain/74f162d7-085f-4761-8abd-e730ac5027a2/.system_generated/steps/385/content.md ===

For the complete documentation index, see
llms.txt
. This page is also available as
Markdown
.
Copy
On this page
Models
💜
Qwen3.5 - How to Run Locally
Run the new Qwen3.5 LLMs including Medium: Qwen3.5-35B-A3B, 27B, 122B-A10B, Small: Qwen3.5-0.8B, 2B, 4B, 9B and 397B-A17B on your local device!
Qwen3.5 is Alibaba’s new model family, including Qwen3.5-
35B
-A3B,
27B
,
122B
-A10B and
397B
-A17B and the new
Small
series: Qwen3.5-0.8B, 2B, 4B and 9B. The multimodal hybrid reasoning LLMs deliver the strongest performances for their sizes. They support
256K context
across 201 languages, have
thinking
+
non-
thinking, and excel in agentic coding, vision, chat, and long-context tasks. The 35B and 27B models work on a 22GB Mac / RAM device. See all
GGUFs here
.
Run Qwen3.5 Tutorials
Fine-tune Qwen3.5
Mar 17 Update:
You can now run Qwen3.5 in
Unsloth Studio
.
Mar 5 Update:
Redownload Qwen3.5-
35B
,
27B
,
122B
and
397B
.
All GGUFs now updated with an
improved quantization
algorithm.
All use our
new imatrix data
. See some improvements in chat, coding, long context, and tool-calling use-cases.
Tool-calling improved
following our chat template fixes.
Fix is universal
and applies to
any
Qwen3.5 format and
any
uploader.
Check new GGUF benchmarks
for Unsloth performance results + our
MXFP4 investigation
.
We're retiring MXFP4 layers from 3 Qwen3.5 GGUFs: Q2_K_XL, Q3_K_XL and Q4_K_XL.
All uploads use Unsloth
Dynamic 2.0
for SOTA quantization performance - so 4-bit has important layers upcasted to 8 or 16-bit. Thank you Qwen for providing Unsloth with day zero access. You can also
fine-tune
Qwen3.5
with Unsloth.
To enable or disable thinking see
How to enable or disable reasoning & thinking
.Qwen3.5 Small models disables by default.
⚙️
Usage Guide
Table: Inference hardware requirements
(units = total memory: RAM + VRAM, or unified memory)
Qwen3.5
3-bit
4-bit
6-bit
8-bit
BF16
0.8B
+
2B
3 GB
3.5 GB
5 GB
7.5 GB
9 GB
4B
4.5 GB
5.5 GB
7 GB
10 GB
14 GB
9B
5.5 GB
6.5 GB
9 GB
13 GB
19 GB
27B
14 GB
17 GB
24 GB
30 GB
54 GB
35B-A3B
17 GB
22 GB
30 GB
38 GB
70 GB
122B-A10B
60 GB
70 GB
106 GB
132 GB
245 GB
397B-A17B
180 GB
214 GB
340 GB
512 GB
810 GB
For best performance, make sure your total available memory (VRAM + system RAM) exceeds the size of the quantized model file you’re downloading. If it doesn’t, llama.cpp can still run via SSD/HDD offloading, but inference will be slower.
Between
27B
and
35B-A3B
, use 27B if you want slightly more accurate results and can't fit in your device. Go for 35B-A3B if you want much faster inference.
Recommended Settings
Maximum context window:
262,144
(can be extended to 1M via YaRN)
presence_penalty = 0.0 to 2.0
default this is off, but to reduce repetitions, you can use this, however using a higher value may result in
slight decrease in performance
Adequate Output Length
:
32,768
tokens for most queries
If you're getting gibberish, your context length might be set too low. Or try using
--cache-type-k bf16 --cache-type-v bf16
which might help.
As Qwen3.5 is hybrid reasoning, thinking and non-thinking mode have different settings:
Thinking mode:
General tasks
Precise coding tasks (e.g. WebDev)
temperature = 1.0
temperature = 0.6
top_p = 0.95
top_p = 0.95
top_k = 20
top_k = 20
min_p = 0.0
min_p = 0.0
presence_penalty = 1.5
presence_penalty = 0.0
repeat_penalty = disabled or 1.0
repeat_penalty = disabled or 1.0
Thinking mode for general tasks:
Thinking mode for precise coding tasks:
Instruct (non-thinking) mode settings:
General tasks
Reasoning tasks
temperature = 0.7
temperature = 1.0
top_p = 0.8
top_p = 0.95
top_k = 20
top_k = 20
min_p = 0.0
min_p = 0.0
presence_penalty = 1.5
presence_penalty = 1.5
repeat_penalty = disabled or 1.0
repeat_penalty = disabled or 1.0
To
disable thinking / reasoning
, use
--chat-template-kwargs '{"enable_thinking":false}'
If you're on
Windows
Powershell, use:
--chat-template-kwargs "{\"enable_thinking\":false}"
Use 'true' and 'false' interchangeably.
For Qwen3.5 0.8B, 2B, 4B and 9B, reasoning is disabled by default
. To enable it, use:
--chat-template-kwargs '{"enable_thinking":true}'
Instruct (non-thinking) for general tasks:
Instruct (non-thinking) for reasoning tasks:
Qwen3.5 Inference Tutorials:
Because Qwen3.5 comes in many different sizes, we'll be using Dynamic 4-bit
MXFP4_MOE
GGUF variants for all inference workloads. Click below to navigate to designated model instructions:
Run in Unsloth Studio
Qwen3.5-35B-A3B
27B
122B-A10B
397B-A17B
Small (0.8B - 9B)
Unsloth Dynamic GGUF uploads:
Qwen3.5-
35B-A3B
Qwen3.5-
27B
Qwen3.5-
122B-A10B
Qwen3.5-
397B-A17B
Qwen3.5-
0.8B
Qwen3.5-
2B
Qwen3.5-
4B
Qwen3.5-
9B
presence_penalty = 0.0 to 2.0
default this is off, but to reduce repetitions, you can use this, however using a higher value may result in
slight decrease in performance.
Currently no Qwen3.5 GGUF works in Ollama due to separate mmproj vision files. Use llama.cpp compatible backends.
🦥 Unsloth Studio Guide
Qwen3.5 can be run and fine-tuned in
Unsloth Studio
, our new open-source web UI for local AI. Unsloth Studio lets you run models locally on
MacOS, Windows
, Linux and:
Search, download,
run GGUFs
and safetensor models
Self-healing
tool calling
+
web search
Code execution
(Python, Bash)
Automatic inference
parameter tuning (temp, top-p, etc.)
Fast CPU + GPU inference via llama.cpp
Train LLMs
2x faster with 70% less VRAM
1
Install Unsloth
Run in your terminal:
MacOS, Linux, WSL:
Windows PowerShell:
Installation will be quick and take approx 1-2 mins.
2
Launch Unsloth
MacOS, Linux, WSL and Windows:
Then open
http://localhost:8888
in your browser.
3
Search and download Qwen3.5
On first launch you will need to create a password to secure your account and sign in again later. Then go to the
Studio Chat
tab and search for Qwen3.5 in the search bar and download your desired model and quant.
4
Run Qwen3.5
Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings.
For more information, you can view our
Unsloth Studio inference guide
.
🦙 Llama.cpp Guides
Qwen3.5-35B-A3B
For this guide we will be utilizing Dynamic 4-bit which works great on a 24GB RAM / Mac device for fast inference. Because the model is only around 72GB at full F16 precision, we won't need to worry much about performance. GGUF:
Qwen3.5-35B-A3B-GGUF
For these tutorials, we will using
llama.cpp
for fast local inference, especially if you have a CPU.
1
Obtain the latest
llama.cpp
on
GitHub here
. You can follow the build instructions below as well. Change
-DGGML_CUDA=ON
to
-DGGML_CUDA=OFF
if you don't have a GPU or just want CPU inference.
For Apple Mac / Metal devices
, set
-DGGML_CUDA=OFF
then continue as usual - Metal support is on by default.
2
If you want to use
llama.cpp
directly to load models, you can do the below: (:Q4_K_M) is the quantization type. You can also download via Hugging Face (point 3). This is similar to
ollama run
. Use
export LLAMA_CACHE="folder"
to force
llama.cpp
to save to a specific location. The model has a maximum of 256K context length.
Follow one of the specific commands below, according to your use-case:
Thinking mode:
Precise coding tasks (e.g. WebDev):
General tasks:
Non-thinking mode:
General tasks:
Reasoning tasks:
3
Download the model via (after installing
pip install huggingface_hub hf_transfer
). You can choose Q4_K_M or other quantized versions like
UD-Q4_K_XL
. We recommend using at least 2-bit dynamic quant
UD-Q2_K_XL
to balance size and accuracy. If downloads get stuck, see:
Hugging Face Hub, XET debugging
4
Then run the model in conversation mode:
Qwen3.5 Small (0.8B • 2B • 4B • 9B)
For Qwen3.5 0.8B, 2B, 4B and 9B,
reasoning is disabled
by default
. To enable it, use:
--chat-template-kwargs '{"enable_thinking":true}'
On Windows use:
--chat-template-kwargs "{\"enable_thinking\":true}"
For the Qwen3.5 Small series, because they're so small, all you need to do is change the model name in the scripts to desired variant. For this specific guide we'll be using the 9B parameter variant. To run them all in near full precision, you'll just need 12GB of RAM / VRAM / unified memory device. GGUFs:
Qwen3.5-
0.8B
Qwen3.5-
2B
Qwen3.5-
4B
Qwen3.5-
9B
1
Obtain the latest
llama.cpp
on
GitHub here
. You can follow the build instructions below as well. Change
-DGGML_CUDA=ON
to
-DGGML_CUDA=OFF
if you don't have a GPU or just want CPU inference.
2
If you want to use
llama.cpp
directly to load models, you can do the below: (:Q4_K_XL) is the quantization type. You can also download via Hugging Face (point 3). This is similar to
ollama run
. Use
export LLAMA_CACHE="folder"
to force
llama.cpp
to save to a specific location. The model has a maximum of 256K context length.
Follow one of the specific commands below, according to your use-case:
To use another variant other than 9B, you can change the '9B' to: 0.8B, 2B or 4B etc.
Thinking mode (disabled by default)
Qwen3.5 Small models disable thinking by default. Use llama-server to enable it.
General tasks:
To use another variant other than 9B, you can change the '9B' to: 0.8B, 2B or 4B etc.
Non-thinking mode is already on by default
General tasks:
Reasoning tasks:
3
Download the model via (after installing
pip install huggingface_hub hf_transfer
). You can choose Q4_K_M or other quantized versions like
UD-Q4_K_XL
. We recommend using at least 2-bit dynamic quant
UD-Q2_K_XL
to balance size and accuracy. If downloads get stuck, see:
Hugging Face Hub, XET debugging
4
Then run the model in conversation mode:
Qwen3.5-27B
For this guide we will be utilizing Dynamic 4-bit which works great on a 18GB RAM / Mac device for fast inference. GGUF:
Qwen3.5-27B-GGUF
1
Obtain the latest
llama.cpp
on
GitHub here
. You can follow the build instructions below as well. Change
-DGGML_CUDA=ON
to
-DGGML_CUDA=OFF
if you don't have a GPU or just want CPU inference.
2
If you want to use
llama.cpp
directly to load models, you can do the below: (:Q4_K_M) is the quantization type. You can also download via Hugging Face (point 3). This is similar to
ollama run
. Use
export LLAMA_CACHE="folder"
to force
llama.cpp
to save to a specific location. The model has a maximum of 256K context length.
Follow one of the specific commands below, according to your use-case:
Thinking mode:
Precise coding tasks (e.g. WebDev):
General tasks:
Non-thinking mode:
General tasks:
Reasoning tasks:
3
Download the model via (after installing
pip install huggingface_hub hf_transfer
). You can choose
MXFP4_MOE
or other quantized versions like
UD-Q4_K_XL
. We recommend using at least 2-bit dynamic quant
UD-Q2_K_XL
to balance size and accuracy. If downloads get stuck, see:
Hugging Face Hub, XET debugging
4
Then run the model in conversation mode:
Qwen3.5-122B-A10B
For this guide we will be utilizing Dynamic 4-bit which works great on a 70GB RAM / Mac device for fast inference. GGUF:
Qwen3.5-122B-A10B-GGUF
1
Obtain the latest
llama.cpp
on
GitHub here
. You can follow the build instructions below as well. Change
-DGGML_CUDA=ON
to
-DGGML_CUDA=OFF
if you don't have a GPU or just want CPU inference.
2
If you want to use
llama.cpp
directly to load models, you can do the below: (:Q4_K_M) is the quantization type. You can also download via Hugging Face (point 3). This is similar to
ollama run
. Use
export LLAMA_CACHE="folder"
to force
llama.cpp
to save to a specific location. The model has a maximum of 256K context length.
Follow one of the specific commands below, according to your use-case:
Thinking mode:
Precise coding tasks (e.g. WebDev):
General tasks:
Non-thinking mode:
General tasks:
Reasoning tasks:
3
Download the model via (after installing
pip install huggingface_hub hf_transfer
). You can choose
MXFP4_MOE
(dynamic 4bit) or other quantized versions like
UD-Q4_K_XL
. We recommend using at least 2-bit dynamic quant
UD-Q2_K_XL
to balance size and accuracy. If downloads get stuck, see:
Hugging Face Hub, XET debugging
4
Then run the model in conversation mode:
Qwen3.5-397B-A17B
Qwen3.5-397B-A17B is in the same performance tier as Gemini 3 Pro, Claude Opus 4.5, and GPT-5.2. The full 397B checkpoint is ~807GB on disk, but via
Unsloth's 397B GGUFs
you can run:
3-bit
: fits on
192GB RAM
systems (e.g., a 192GB Mac)
4-bit (MXFP4)
: fits on
256GB RAM
. Unsloth
4-bit dynamic
UD-Q4_K_XL
is
~214GB on disk
- loads directly on a
256GB M3 Ultra
Runs on a
single 24GB GPU + 256GB system RAM
via
MoE offloading
, reaching
25+ tokens/s
8-bit
needs
~512GB RAM/VRAM
See
397B quantization benchmarks
on how Unsloth GGUFs perform.
1
Obtain the latest
llama.cpp
on
GitHub here
. You can follow the build instructions below as well. Change
-DGGML_CUDA=ON
to
-DGGML_CUDA=OFF
if you don't have a GPU or just want CPU inference.
2
If you want to use
llama.cpp
directly to load models, you can do the below: (:Q4_K_M) is the quantization type. You can also download via Hugging Face (point 3). This is similar to
ollama run
. Use
export LLAMA_CACHE="folder"
to force
llama.cpp
to save to a specific location. Remember the model has only a maximum of 256K context length.
Follow this for
thinking
mode:
Follow this for
non-thinking
mode:
3
Download the model via (after installing
pip install huggingface_hub hf_transfer
). You can choose
MXFP4_MOE
(dynamic 4bit) or other quantized versions like
UD-Q4_K_XL
. We recommend using at least 2-bit dynamic quant
UD-Q2_K_XL
to balance size and accuracy. If downloads get stuck, see:
Hugging Face Hub, XET debugging
4
You can edit
--threads 32
for the number of CPU threads,
--n-gpu-layers 2
for GPU offloading on how many layers. Try adjusting it if your GPU goes out of memory. Also remove it if you have CPU only inference.
👾 LM Studio Guide
For this guide, we'll be using
LM Studio
, a unified UI interface for running LLMs. The '💡Thinking' and 'Non-thinking' toggle may not appear by default so we'll need some extra steps to get it working.
1
Download
LM Studio
for your device. Then open Model Search, search for 'unsloth/qwen3.5', and download the GGUF (quant) that you desire.
2
Thinking Toggle instructions:
After downloading, Open your Terminal / PowerShell and try:
lms --help
. Then if LM Studio appears normally with many commands, run:
This will get a yaml file which enables your GGUF to have the '💡Thinking' and 'Non-thinking' toggle appear. You can change
4b
to the desired quant you'd like to have.
Otherwise, you can go to
our LM Studio page
and download the specific yaml file.
3
Restart LM Studio, then load your downloaded model (with the specific thinking toggle you downloaded). You should now see the Thinking toggle enabled. Don't forget to set the
correct parameters
.
🦙 Llama-server serving & OpenAI's completion library
To deploy Qwen3.5-397B-A17B for production, we use
llama-server
In a new terminal say via tmux, deploy the model via:
Then in a new terminal, after doing
pip install openai
, do:
🤔
How to enable or disable reasoning & thinking
For the below commands, you can use '
true
' and '
false
' interchangeably.
Unsloth Studio
automatically has a 'Think' Toggle for thinking models.
To have Think toggle for LM Studio,
read our guide
.
Unsloth Studio has Think toggle by default
To
disable
thinking / reasoning, use within llama-server:
If you're on
Windows
or Powershell, use:
--chat-template-kwargs "{\"enable_thinking\":false}"
To
enable
thinking / reasoning, use within llama-server:
If you're on
Windows
or Powershell, use:
--chat-template-kwargs "{\"enable_thinking\":true}"
For Qwen3.5 0.8B, 2B, 4B and 9B, reasoning is disabled by default
. To enable it, use:
--chat-template-kwargs '{"enable_thinking":true}'
And on Windows or Powershell:
--chat-template-kwargs "{\"enable_thinking\":true}"
As an example for Qwen3.5-9B to enable thinking (default is disabled):
And then in Python:
👨‍💻 OpenAI Codex & Claude Code
To run the model via local coding agentic workloads, you can
follow our guide
. Use the
llama-server
we just set up just then, and set the model name to the exact id it reports at
GET /v1/models
(the
--alias
value above, e.g.
unsloth/Qwen3.5-9B-GGUF
). Follow the correct Qwen3.5 parameters and usage instructions.
Claude Code
OpenAI Codex
After following the instructions for Claude Code for example you will see:
We can then ask say
Create a Python game for Chess
:
🔨
Tool Calling with Qwen3.5
See
Tool Calling Guide
for more details on how to do tool calling. In a new terminal (if using tmux, use CTRL+B+D), we create some tools like adding 2 numbers, executing Python code, executing Linux functions and much more:
We then use the below functions (copy and paste and execute) which will parse the function calls automatically and call the OpenAI endpoint for any model:
After launching Qwen3.5 via
llama-server
like in
Qwen3.5
or see
Tool Calling Guide
for more details, we then can do some tool calls.
📊 Benchmarks
Unsloth GGUF Benchmarks
We updated Qwen3.5-35B Unsloth Dynamic quants
being SOTA
on nearly all bits. We did over 150 KL Divergence benchmarks, totally
9TB of GGUFs
. We uploaded all research artifacts. We also fixed a
tool calling
chat template
bug
(affects all quant uploaders)
All GGUFs now updated with an
improved quantization
algorithm.
All use our
new imatrix data
. See some improvements in chat, coding, long context, and tool-calling use-cases.
Qwen3.5-35B-A3B GGUFs are updated to use new fixes (112B, 27B still converting, re-download once they are updated)
99.9% KL Divergence shows SOTA
on Pareto Frontier for UD-Q4_K_XL, IQ3_XXS & more.
Retiring MXFP4
from all GGUF quants: Q2_K_XL, Q3_K_XL and Q4_K_XL, except for pure MXFP4_MOE.
35B-A3B - KLD benchmarks (lower is better)
122B-A10B - KLD benchmarks (lower is better)
READ OUR DETAILED QWEN3.5 ANALYSIS + BENCHMARKS HERE:
Qwen3.5 GGUF Benchmarks
Qwen3.5-397B-A17B Benchmarks
Benjamin Marie (third-party) benchmarked
Qwen3.5-397B-A17B
using Unsloth GGUFs on a
750-prompt mixed suite
(LiveCodeBench v6, MMLU Pro, GPQA, Math500), reporting both
overall accuracy
and
relative error increase
(how much more often the quantized model makes mistakes vs. the original).
Key results (accuracy; change vs. original; relative error increase):
Original weights:
81.3%
UD-Q4_K_XL:
80.5%
(−0.8 points; +4.3% relative error increase)
UD-Q3_K_XL:
80.7%
(−0.6 points; +3.5% relative error increase)
UD-Q4_K_XL
and
UD-Q3_K_XL
stay extremely close to the original,
well under a 1-point accuracy drop
on this suite, which Ben insinuates that you can
sharply reduce memory footprint
(
~500 GB less
) with little to no practical loss on the tested tasks.
How to choose:
Q3 scoring slightly higher than Q4 here is completely plausible as normal run-to-run variance at this scale, so treat
Q3 and Q4 as effectively similar quality
in this benchmark:
Pick
Q3
if you want the
smallest footprint / best memory savings
Pick
Q4
if you want a
slightly more conservative
option with
similar
results
All listed quants utilize our dynamic metholodgy. Even
UD-IQ2_M
uses a the same methodology of dynamic however the conversion process is different to
UD-Q2-K-XL
where K-XL is usually faster than
UD-IQ2_M
even though it's bigger, so that is why
UD-IQ2_M
may perform better than
UD-Q2-K-XL
.
Official Qwen Benchmarks
Qwen3.5-35B-A3B, 27B and 122B-A10B Benchmarks
Qwen3.5-4B and 9B Benchmarks
Qwen3.5-397B-A17B Benchmarks
Previous
MiniMax M3
Next
Fine-tune Qwen3.5
Last updated
5 minutes ago
Was this helpful?
