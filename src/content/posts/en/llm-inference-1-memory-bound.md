---
title: Doubling FLOPS won't make ChatGPT faster
description: An LLM writes its answer only as fast as it can read its weights from memory. Part 1 of the series "LLM inference, from compute to memory".
pubDatetime: 2026-10-10T00:00:00+09:00
category: Engineering
type: tech-essay
tags:
  - llm
  - inference
  - gpu
  - kv-cache
translationKey: llm-inference-1-memory-bound
draft: false
---

> Series **LLM inference, from compute to memory** (1/3)
> 1. Doubling FLOPS won't make ChatGPT faster (this post)
> 2. Five ways to make ChatGPT faster are all the same idea (coming soon)
> 3. Why NVIDIA started splitting inference into three pieces (coming soon)

## The small model I gave up on

A while ago I entered a competition to build an AI-powered self-service kiosk in Korea. I wanted to put a small language model (sLLM) inside the kiosk, so I tried a small model like Gemma on Colab. The answers took far too long. Inference speed was one of the main judging criteria, so I dropped the sLLM and went with Rasa (an open-source framework for intent-based chatbots) and other NLU models instead.

Back then I only used the Hugging Face transformers library in a way that returned the whole result at once. Nothing showed up on screen until the model had produced its last token. Later, when I ran a similar model through ollama or streamed the output, text started appearing almost right away. I didn't even know the term TTFT at the time. (ollama was also faster for a separate reason, which I'll cover in another post.)

I had seen something similar before. When training Transformer models with PyTorch, a bad `num_workers` setting on the `DataLoader` would tank GPU utilization. The CPU couldn't prepare and ship data to the GPU fast enough, so the GPU sat idle.

That made me wonder. My setup was one desktop running one model. What about ChatGPT, where huge data center GPUs are shared by a lot of users at once? Where does that get stuck?

## Memory sets the writing speed

Here is the question this series started from.

> Suppose NVIDIA ships a GPU tomorrow that has **twice the compute (FLOPS)** but the same memory bandwidth. Does ChatGPT get twice as fast?

It doesn't. The time before the answer **starts** goes down, but the speed at which the answer **is written** barely moves. How fast an LLM writes its answer, token by token, depends on how fast it can read data from memory, not on how fast it can compute.


## Background

### 1) LLMs write one token at a time

LLMs work with tokens. You can think of a token as a word or a piece of a word. An LLM doesn't produce its answer in one go. It looks at the text so far, picks the next token, appends it, looks again, and picks the one after that. Feeding each output back in as input like this is called autoregressive generation.

→ Token 2 can't be computed until token 1 exists. Within one user's answer, the steps have to happen in order.

Some newer models such as [Mercury](https://arxiv.org/abs/2506.17298) use diffusion to refine many tokens at once. Most LLMs, including GPT and Gemini, still write one token at a time, and that's the kind this post is about.

### 2) FLOPS and memory bandwidth

Strip a GPU's job down and it does two things:

1. Fetch numbers from memory
2. Compute with them

FLOPS (floating-point operations per second) measures step 2: how many arithmetic operations the GPU can do per second. It's the number GPU marketing puts front and center. Memory bandwidth measures step 1: how much data can move from memory to where the math happens, per second.

A kitchen makes this easier to picture. FLOPS is how fast the chef's hands are. Memory bandwidth is the truck that brings ingredients from the warehouse. A fast chef still has to wait if the ingredients arrive late.

Depending on which side holds things back, people describe a workload as:

- **Compute-bound**: ingredients arrive with room to spare, and the chef is the busy one
- **Memory-bandwidth-bound**: the chef has free hands, and the truck can't keep up

### 3) Prefill, decode, TTFT and ITL

An LLM handles a request in two phases that behave very differently.

- **Prefill** processes the entire input at once. Thousands of input tokens can be packed into one big matrix computation. The first output token comes out at the end of this phase.
- **Decode** then produces tokens one by one and appends them. A longer answer means a longer decode.

Users feel these two phases as two different kinds of delay:

- **TTFT (time to first token)**: from sending the request to seeing the first word. Prefill mostly decides this.
- **ITL (inter-token latency)**: the gap between tokens as text streams in. Decode decides this.

![Comparison of when the user first sees text when the response is returned all at once versus streamed](/images/posts/llm-inference-1-memory-bound/request-timeline-en.png)

My kiosk experience is the top row. If you take the result all at once, the user waits for the whole generation, which is much longer than TTFT. The GPU does exactly the same work, but with streaming the text starts appearing as soon as prefill ends.

### 4) KV cache: don't redo work you've already done

Each new token in decode has to look back at every earlier token. That's the job of [self-attention](https://arxiv.org/abs/1706.03762), which computes three values for each token: Q (query), K (key) and V (value).

Think of a library. Q is the search you're running right now, K is the catalog entry for each book, and V is what's inside the book. Every new token brings a new search (Q), scans every catalog entry so far (K), and pulls in the relevant contents (V).

The K and V of past tokens are used again by every future search. The Q of a past token is used once, while that token is being processed, and never again. So only K and V are stored and reused. That store is the **KV cache**, and it's why there is no "Q cache."

→ Prefill builds the KV cache for the whole input. Decode reads it for every new token and appends the new K and V.

## Why decode is tied to memory

### The ceiling, in numbers

Take a 70B model (70 billion parameters). Stored in FP16 (16 bits, 2 bytes per parameter), the weights alone come to about 140 GB.

To produce **one** token in decode, the model has to go through all of those weights once, because every layer's weights feed into the next token. (That holds for typical dense models that use every layer. Mixture-of-experts models only use some expert layers per token, so they read less.) The [NVIDIA H100 SXM](https://www.nvidia.com/en-us/data-center/h100/) has 3.35 TB/s of memory bandwidth.

```text
140 GB ÷ 3.35 TB/s ≈ 0.042 s
```

→ However fast the compute is, one user gets at most about 24 tokens per second. In practice a 70B model doesn't fit on one GPU and gets split across several, but each token still has to read all of the weights.

### I first thought the weights came over NVLink

When I first ran this calculation, I assumed the weights traveled from HBM over NVLink. So I figured NVLink bandwidth would need to grow along with compute. That picture was wrong.

![The path weights take from HBM to the compute units during decode, compared with NVLink and PCIe bandwidth](/images/posts/llm-inference-1-memory-bound/gpu-memory-path-en.png)

HBM (High Bandwidth Memory) sits right next to the GPU die, in the same package. The decode bottleneck is **the path from HBM to the compute units inside the GPU**. NVLink connects one GPU to another, and PCIe connects the GPU to the CPU side.

| Path | Connects | H100 SXM bandwidth |
|---|---|---:|
| HBM → compute units | inside the GPU | **3.35 TB/s** |
| NVLink | GPU ↔ GPU | 900 GB/s |
| PCIe Gen5 | CPU ↔ GPU | 128 GB/s |

→ PCIe mostly matters when the model is first loaded onto the GPU. Once the whole model sits in GPU memory, decode barely touches it.

### I first thought the weights only had to be read once

The second thing I misread was what the 0.042 seconds meant. I took it as the time to load 140 GB onto the GPU in the first place. But the weights are already in HBM. Those 0.042 seconds are spent **every time one token is made**, streaming the weights from HBM to the compute units again.

The weights never change, so why not read them once and keep them next to the compute units? I wondered the same thing. The problem is that there's nowhere to put them. The fast memory right beside the compute units is the [H100's L2 cache](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), and it holds 50 MB. That's about 2,800 times smaller than 140 GB. So the weights have to be read from HBM again for every token.

### I first thought the bandwidth was sitting idle

The third mistake had the direction backwards. I'd heard that a GPU serving a single user doesn't reach its full performance, and I took that to mean bandwidth was going unused. It's the opposite.

Back in the kitchen, the truck is running nonstop at top speed. Bandwidth is maxed out. The chef is the one standing around. Each truckload is enough for one plate, so the chef cooks one serving and waits for the next truck. A chef who could cook for a hundred is cooking for one.

→ "Decode is memory-bandwidth-bound" means bandwidth is saturated and compute has capacity to spare.

Prefill is different. It handles thousands of input tokens at once, so a single truckload of weights feeds thousands of servings. That's why prefill leans compute-bound.

### Back to the opening question

What happens with a GPU that has twice the FLOPS and nothing else?

- Prefill is limited by compute, so it speeds up → TTFT drops
- Decode is limited by bandwidth, so it stays about the same → ITL stays the same

→ The first word shows up sooner, but the text streams at the same pace. It would be hard to feel that ChatGPT got twice as fast.

### The same problem as my DataLoader

Compare this with the `num_workers` story from the start. On my desktop, the path from CPU to GPU was jammed, so the GPU sat idle. In data center decode, the jam is **inside the GPU**, on the path from HBM to the compute units, and so the compute sits idle.

The symptom is the same: the GPU has compute to spare. Only the location of the jam changes.

## One wall, two directions

There are two broad ways to deal with this wall. Both are easier to follow once you know the two main kinds of memory.

### SRAM and DRAM

Computer memory comes in two main kinds, depending on how it's built.

- **SRAM (static RAM)** holds each bit with a handful of transistors. It's very fast, but each cell is large and expensive, so you can't fit much of it. It goes in small amounts right next to the compute units. A GPU's L2 cache is SRAM.
- **DRAM (dynamic RAM)** stores each bit as charge in a tiny capacitor. It's dense and cheap, so you can build large capacities, but it's slower than SRAM. HBM is DRAM stacked in layers and placed beside the GPU.

→ SRAM is a **type** of memory. "L2 cache" names a **role** that SRAM plays. A GPU uses a little SRAM as a cache in front of a large HBM. Groq, below, uses SRAM itself as the main memory that holds the weights.

### Two directions

| Direction | Example | How | Cost |
|---|---|---|---|
| Grow compute | Each new general-purpose GPU generation | Keep raising FLOPS, and raise HBM bandwidth with it | Decode only speeds up as much as bandwidth does |
| Bring memory onto the chip | Groq LPU | Keep all the weights in on-chip SRAM | Each chip has little SRAM, so a large model has to be spread over many chips |

The first-generation [Groq LPU](https://groq.com/blog/the-groq-lpu-explained) has 230 MB of SRAM per chip with about 80 TB/s of on-chip bandwidth, more than an order of magnitude above the H100's HBM. The catch is capacity: holding 140 GB of weights in 230 MB chunks takes more than 600 chips by simple division. It pushes past the "nowhere to put them" wall by adding chips.

This design now sits inside NVIDIA. In March 2026, NVIDIA announced the Groq 3 LPU, built on Groq's technology, and positioned it to handle the decode phase within its next-generation Vera Rubin platform ([IEEE Spectrum](https://spectrum.ieee.org/nvidia-groq-3)). Part 3 goes into this.

The first direction also has a structural problem. According to [AI and Memory Wall (Gholami et al., 2024)](https://arxiv.org/abs/2403.14123), peak server hardware FLOPS grew 3.0x every two years over the past 20 years, while DRAM bandwidth grew only 1.6x. The gap between compute and memory keeps widening.

## Coming up

If decode is tied to memory, how is the industry working around it? Batching several users together, shrinking the KV cache with GQA, cutting memory waste with PagedAttention, skipping repeated work with prefix caching: these look like different techniques, but they are all doing the same thing. That's Part 2.

Part 3 looks at why NVIDIA started splitting inference into stages and handing each one to a different chip.

## My take

When I gave up on the sLLM for the kiosk, I decided the model was too slow. Looking back, half of that was how I was receiving the output. If I had known that TTFT and total generation time are different things, I might have reached a different conclusion with the same model.

Knowing how inference works inside also shapes product decisions. Which model to use, which hardware to pick, and what to show the user first all rest on this structure.

## Sources

- NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/) (HBM 3.35 TB/s, NVLink 900 GB/s, PCIe Gen5 128 GB/s)
- NVIDIA, [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) (50 MB L2 cache)
- Gholami et al., [AI and Memory Wall (arXiv:2403.14123)](https://arxiv.org/abs/2403.14123)
- Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670) (different bottlenecks in prefill and decode)
- Groq, [What is a Language Processing Unit?](https://groq.com/blog/the-groq-lpu-explained)
- [AI Accelerators for LLM Inference: Architecture Analysis and Scaling Strategies (arXiv:2506.00008)](https://arxiv.org/abs/2506.00008) (LPU SRAM capacity)
- Vaswani et al., [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)
- Inception Labs, [Mercury: Ultra-Fast Language Models Based on Diffusion (arXiv:2506.17298)](https://arxiv.org/abs/2506.17298)
- IEEE Spectrum, [Nvidia Groq 3](https://spectrum.ieee.org/nvidia-groq-3) (2026-03-16, decode in Vera Rubin)
