---
title: Five ways to make ChatGPT faster are all the same idea
description: Batching, GQA, PagedAttention, prefix caching, and Groq. None of them makes the math faster. Part 2 of the series "LLM inference, from compute to memory".
pubDatetime: 2026-10-12T00:00:00+09:00
category: Engineering
type: tech-essay
tags:
  - llm
  - inference
  - kv-cache
  - vllm
  - groq
translationKey: llm-inference-2-five-ways
draft: false
---

> Series **LLM inference, from compute to memory** (2/3)
> 1. [Doubling FLOPS won't make ChatGPT faster](/en/posts/llm-inference-1-memory-bound/)
> 2. Five ways to make ChatGPT faster are all the same idea (this post)
> 3. [Why NVIDIA started splitting inference into three pieces](/en/posts/llm-inference-3-three-pieces/)

## How many meals does the cook need per delivery?

[Part 1](/en/posts/llm-inference-1-memory-bound/) ended here. During decode, when an LLM writes its answer one token at a time, every new token requires reading all of the model's weights from HBM into the compute units. For a 70B model that is 140GB per token. In the kitchen analogy from Part 1, the delivery truck (memory bandwidth) is driving flat out, and the cook (compute) makes one meal from each delivery and then waits for the next truck.

So how many meals would the cook have to make from each delivery to stay busy? With the H100 numbers from Part 1:

```text
H100 compute ≈ 1,000 TFLOPS (FP16, dense) ÷ bandwidth 3.35 TB/s ≈ 300
→ the GPU needs about 300 operations per byte it reads to keep its compute busy
```

Decoding for a single user does about one operation per byte of weights. Each parameter is 2 bytes in FP16, and producing one token takes one multiply and one add per parameter. To keep the cook busy, each delivery has to feed **about 300 users**.

Now put a 70B model on 8 GPUs and give every user a 100,000-token conversation. The number of users you can actually serve at the same time is about **15**. With the older attention design it would have been **1**. The hardware wants 300 and gets 15, or 1. This post is about how the industry has been closing that gap.

## Five methods, one sentence

The five methods are batching, GQA, PagedAttention, prefix caching, and Groq. They come from different places and have different names. My claim about them is this:

> None of the five makes the math faster. Every one of them changes where the data sits and how little of it has to move.

I got this wrong twice before landing on it. My first guess was that they all "widen the bandwidth to cut loading time." None of them widens anything. They all move less data over the same bandwidth. My second guess was that they "use every resource as fully as possible while serving many users quickly." That is true, and it is equally true of database indexes and factory assembly lines, so it explains nothing about LLMs.

What helped was writing down, for each method, what it lets the system do less of.

| Method | What it does less of |
|---|---|
| Batching | Moving the weights separately for each user |
| GQA | Storing and reading KV cache |
| PagedAttention | Reserving memory that never gets used |
| Prefix caching | Recomputing the same prompt prefix (in exchange for storing more) |
| Groq | Moving weights from HBM to the compute units |

→ The verbs left over are *store* and *move*. Nothing on the list makes computation faster. That also makes the claim falsifiable: if any of the five actually wins by computing faster, it fails. I test it against two objections at the end.

## Background

A quick refresher on the KV cache from Part 1. Attention uses three values per token: Q (query), K (key), and V (value). Think of a library: Q is the search query the current token is asking, K is the index card attached to each earlier token, and V is that token's actual content. Every later token needs the K and V of every earlier token, so they are saved instead of recomputed. That saved data is the KV cache.

The rest of this post turns on one difference.

- Weights are shared by every user. Everyone talks to the same model.
- KV cache belongs to one user. The K and V of my conversation are useless for yours.

## Method 1. Batching: many meals per delivery

The obvious first move is to process many users' requests together. If one read of the weights computes the next token for 64 users, the data moved stays the same and the useful work goes up 64 times. The closer you get to 300 operations per byte, the less the cook sits idle.

I first thought batching increased bandwidth. It doesn't. The truck is the same size and drives at the same speed. It just feeds more people per trip. The kitchen could always cook for a hundred, and cooking for one was the waste.

Real serving systems batch at the level of a single decoding step, not a whole request. Finished requests leave and new ones join at every step. [Orca (Yu et al., OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) introduced this as iteration-level scheduling.

### The catch: KV takes up space

Batching has a limit. Weights are shared, but KV cache is per user, so every extra user adds their own KV to GPU memory.

Here is the KV size per token, from [Llama 3.1 70B's config file](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json). The model has 80 layers, 8 heads that hold K and V, and 128 numbers per head.

```text
80 layers × 2 (K and V) × 8 heads × 128 × 2 bytes (FP16) = 327,680 bytes ≈ 0.33MB
```

In library terms, it's an 80-floor building where every floor files two cards (K and V) per token, each card holding 1,024 numbers (8 × 128) at 2 bytes each. A third of a megabyte per token sounds small, but a 100,000-token conversation takes about 33GB for one user.

Eight H100s have 640GB of memory. Take away 140GB for the weights, and about 500GB is left. At 33GB per user, that fits 15 users.

![Comparison of how many 100K-token users fit in 8 H100s next to the weights: 1 with MHA, 15 with GQA](/images/posts/llm-inference-2-five-ways/kv-capacity-en.png)

→ The target is 300 and we have 15. Three of the remaining four methods work on this gap.

## Method 2. GQA: sharing K and V

The top row of the figure is the same model built the older way. Llama 3.1 70B has 64 query heads but only 8 key/value heads. The original Transformer's multi-head attention (MHA) would have had 64 K/V heads too. That makes the KV 8 times larger, about 2.6MB per token or about 262GB for 100,000 tokens. Only one user fits in 500GB.

The method that cut 64 down to 8 is GQA (grouped-query attention). Query heads are grouped eight at a time, and each group shares a single pair of K and V heads. The extreme case, where all query heads share one K/V pair, is called MQA (multi-query attention).

![How query heads connect to key/value heads in MHA, GQA, and MQA](/images/posts/llm-inference-2-five-ways/gqa-groups-en.png)

### "Doesn't that lose perspectives?"

That was my first reaction. I had learned that each attention head looks at the sentence from a different angle. Sharing K and V seemed to cut 64 angles down to 8.

Half of that worry holds. **The 64 query heads are untouched**, so the model can still ask 64 different questions. What shrinks is the variety of index cards and content that answer them. So quality can drop somewhat, and the question is how much.

In the [GQA paper (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245), experiments on T5-XXL gave GQA with 8 K/V heads an average score of 47.1, against 47.2 for MHA. Time per sample was 0.28 seconds, close to MQA's 0.24. MQA lost quality, and training it from scratch produced loss spikes and divergence. The paper also showed that an existing MHA checkpoint can be converted to GQA with about 5% of the original pre-training compute ("uptraining"), so models don't have to be retrained from zero.

### Why stop at 8 instead of 1?

There is a second reason besides quality. A 70B model doesn't fit on one GPU, so it's split across several. Each GPU computes attention for its own query heads, so the K and V those heads need must sit in that GPU's memory.

I got confused here too. I assumed MHA's 64 K/V heads were copied onto each of the 8 GPUs, so 64 × 8 in total. They aren't. The 64 heads are **divided** among the 8 GPUs, 8 each. Copies only appear when there are fewer K/V heads than GPUs.

| Design | K/V heads | Per GPU, with 8 GPUs | Total across 8 |
|---|---:|---|---:|
| MHA | 64 | 8 (divided) | 64 |
| GQA | 8 | 1 (divided) | 8 |
| MQA | 1 | 1 (copied) | 8 |

→ On 8 GPUs, GQA and MQA use the same amount of memory. MQA fills it with 8 identical copies and GQA with 8 different heads. The GQA paper makes the same point: when a large model is split across devices, MQA's single K/V head gets replicated on every partition. I couldn't find a statement from Meta that this is why Llama 3 uses 8, but as long as the model is split across 8 GPUs, there is little reason to go lower.

## Method 3. PagedAttention: stop reserving space

GQA shrank each user's KV. But the 15-user estimate hides an assumption: that every user uses exactly 100,000 tokens. In practice some people send three messages and leave, others keep going, and nobody knows how long a reply will be until it's finished.

Older serving systems reserved one contiguous chunk of memory per request, sized for the maximum length. The [vLLM paper (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) measured that only **20.4 to 38.2%** of the KV cache memory in existing systems held actual tokens. It lists three kinds of waste.

1. Reserved slots: space held for tokens that haven't been generated yet. No other request can use it until this one finishes.
2. Internal fragmentation: space left over when a request reserved for the maximum length and ended early.
3. External fragmentation: gaps between chunks of different sizes. There is enough free memory in total, but no single gap is big enough.

My first take was that this was a static allocation problem: you reserve 100,000 tokens' worth even if you use a fraction, and you run out if you need more. So why not allocate dynamically? But as long as each request's KV lives in one contiguous block, a growing conversation gets stuck as soon as the next block over belongs to someone else, and external fragmentation stays.

### The answer was on my OS exam

I took operating systems in my second year of university and walked into the exam barely prepared. What I remember now is an analogy that compared deadlock to people waiting for restroom stalls, and round-robin scheduling. The answer to this problem was in that same course.

An operating system lets each process see one contiguous range of memory, while the physical memory behind it is split into fixed-size pages scattered anywhere. A page table connects the two. PagedAttention applies the same idea to the KV cache.

- KV cache is cut into fixed-size blocks, 16 tokens per block by default.
- Each request gets a block table, a mapping like "my block 0 is physical slot 5, my block 1 is slot 1."
- When a block fills up, the request gets one more free slot. Since every slot is the same size, any free slot works.

![Two users' block tables pointing to scattered physical blocks in GPU memory, with the shared prefix stored once](/images/posts/llm-inference-2-five-ways/paged-blocks-en.png)

The waste per request is now at most one partly filled block. In its [launch post](https://vllm.ai/blog/2023-06-20-vllm), the vLLM team wrote that existing systems wasted 60 to 80% of memory and PagedAttention wastes under 4%. Fitting more requests onto the same GPU raised throughput by 2 to 4 times at the same latency.

For a while I thought vLLM was a tool like ollama. Both are inference engines, but ollama is built for one person on their own machine, and vLLM is built for a server handling many requests at once. The question in this post, how many users fit, belongs to vLLM's world.

## Method 4. Prefix caching: store the shared beginning once

Since a block table is just a set of pointers, another trick follows. A company's chatbot might start every conversation with the same system prompt. The KV for that prefix is the same for everyone. Compute it once, store it once, and point every user's block table at the same physical blocks. Blocks 0 to 2 in the figure above work this way.

I once mixed this up with caching the weights. Prefix caching reuses **the KV of the prompt's opening tokens**, and it has nothing to do with the weights. The prefix also has to match exactly. A token's K and V depend on every token before it, so once two prompts differ by even one token, nothing after that point can be shared, even if the remaining text is identical.

### The half-full last block

The tricky part is the boundary. With a 5,000-token system prompt and 16 tokens per block:

```text
5,000 ÷ 16 = 312 full blocks + 1 block holding 8 tokens
```

When two users continue their conversations, both want to write into the 8 empty slots of that last block. They can't write different things into the same slots. So whoever writes first gets a copy of that one block and continues in the copy. The other user is now the only one pointing at the original and can keep writing there. Picture two people sharing a notebook whose last page is half full: the first person to write photocopies that page and writes on the copy. Operating systems call this copy-on-write, and the vLLM paper uses the same term. The 312 full blocks stay shared.

Prefix caching is the only one of the five that cuts **prefill** work. Skipping the recomputation shortens the time to the first token (TTFT), at the cost of more storage. API pricing shows this directly: [Claude's prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) charges 0.1 times the base input price for cached reads.

## The other bet: Groq

The four methods so far all bet on the same thing. They keep weights in HBM and try to serve more users per read. The fifth bets the other way.

[Groq](https://groq.com/blog/the-groq-lpu-explained)'s LPU keeps the entire model's weights in on-chip SRAM and has no HBM at all. At first I understood this as something like a GPU's L2 cache, loading weights faster. It is the same kind of memory, used very differently.

| | GPU L2 cache | Groq SRAM |
|---|---|---|
| Role | Temporary holding area in front of a large HBM | Main memory that holds the weights |
| When data isn't there | Fetch it again from HBM | Doesn't happen; there's no HBM behind it |
| Who decides what goes in | Hardware, at run time | The compiler, ahead of time |

In kitchen terms, a GPU brings ingredients from a warehouse by truck and keeps a small shelf in the kitchen. Groq moved the whole warehouse into the kitchen and got rid of the truck.

The numbers from the opening make the difference clear. According to [Groq's product brief](https://groq.com/wp-content/uploads/2024/08/GroqChip™-Processor-Product-Brief-v1.7.pdf), the first-generation chip has 188 TFLOPS of FP16 compute and 80 TB/s of SRAM bandwidth.

```text
H100:     ≈ 1,000 TFLOPS ÷ 3.35 TB/s ≈ 300
GroqChip:     188 TFLOPS ÷   80 TB/s ≈ 2.4
```

→ A GPU needs about 300 operations per byte to keep its compute busy, so it works hard to batch hundreds of users. Groq needs about 2. Its deliveries are so fast that cooking one or two meals each time keeps the cook busy.

The number to look at is 188. A Groq chip has less than a fifth of an H100's compute. **The chip that computes more slowly answers a single user faster.** I can't think of a cleaner piece of evidence for this post's claim.

The cost is chip count. One chip has 230MB of SRAM. Holding 140GB of weights takes roughly 600 chips by simple division, where two H100s would do. One user's 100,000-token KV cache (33GB) is also more than 100 times a single chip's SRAM. I couldn't find public material on where Groq actually keeps the KV cache or how many users it batches.

### Two directions, plus one more

GQA isn't the only way to shrink KV. Instead of cutting the number of K/V heads, DeepSeek compresses K and V into a small latent vector and stores that. This is MLA (multi-head latent attention). According to the [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434), it cut the KV cache by 93.3% compared with the earlier DeepSeek 67B and raised maximum generation throughput 5.76 times.

| Approach | How it saw the bottleneck | Fix | Cost |
|---|---|---|---|
| Serve more per read (batching, GQA, PagedAttention, prefix caching) | Each read of the weights serves only one user | Serve many users per read, and save KV space to make room | Per-user speed stays capped by bandwidth |
| Compress the KV (MLA) | Storing K and V as-is is too big | Store a small compressed latent instead | The model has to be trained with MLA |
| Keep weights close (Groq) | The path from HBM is slow | Keep weights in SRAM next to the compute | Hundreds of chips; weak at long contexts and many users |

→ The first two aim at throughput and cost per token. The last one aims at the speed a single user feels (inter-token latency).

## Why now

An [analysis of 100 trillion tokens by a16z and OpenRouter](https://arxiv.org/abs/2601.10088) found that over the 13 months to November 2025, the average input per request grew from about 1,500 tokens to over 6,000, and the average output from about 150 to 400. Over the same period, the share of tokens going to reasoning models went from almost zero to more than half.

Four times the input means four times the KV cache per user, which means a quarter as many users per GPU. Longer outputs hold that memory for longer. As more usage looks like agents that keep pasting documents and tool results into the context, I expect these five methods to matter more.

## What's next

Once the KV cache no longer fits in HBM, I think the next step is to give it room outside HBM. In January 2026 NVIDIA announced [storage infrastructure built on BlueField-4](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai) that keeps KV cache in a storage tier outside the GPU and pulls it back when needed. Groq's approach has also moved inside NVIDIA as a chip that handles decode. Part 3 covers this trend of splitting inference into stages and giving each stage its own chip and memory.

There is also a signal that would prove this wrong. Today's competition assumes the KV cache grows with the length of the conversation. If architectures whose stored state doesn't grow with length, like state space models or linear attention, become the default for mainstream models, the fight over KV space loses most of its force.

## My take

Here are the two objections I expect.

> "Batching uses more compute. And Groq is a chip that computes, so isn't it fast because it computes fast?"

Batching uses compute that was already sitting idle, without adding any. As for Groq, its compute is less than a fifth of an H100's. It is not fast because it computes fast.

> "What about FlashAttention? It's famous for making attention several times faster. Isn't that a case of faster math?"

The original title of [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) includes the word IO-Awareness. Standard attention wrote a large intermediate table (tokens × tokens) to HBM and read it back. FlashAttention cuts Q, K, and V into tiles small enough for SRAM and finishes the work there without sending intermediate results to HBM. The output is identical. What drops is the number of trips to HBM, so this objection ends up as one more example for the claim.

I have a personal memory of FlashAttention. I once tried to run the MKQA benchmark with a multilingual embedding model, `jinaai/jina-embeddings-v3`. The model is set up to use `flash_attn`, but the GPU on our lab server didn't support it, so I ran without it and the benchmark took more than 30 hours. I thought the GPU was an A6000. Looking it up for this post, [FlashAttention-2 supports Ampere and newer](https://github.com/Dao-AILab/flash-attention), and the RTX A6000 is Ampere, so it would have worked. The server actually had a Quadro RTX 6000, a similarly named card from the previous Turing generation. I never measured how much of those 30 hours was down to FlashAttention.

Quantization sits on the boundary. Cutting weights to 4 bits reduces the bytes to move, and lower precision also makes the math itself faster (an H100 does twice as many FP8 operations as FP16). I suspect the first effect dominates in decode and the second in prefill. I plan to check that in a separate post on quantization.

In the end I've returned to the sentence I started this series with: inference speed is a contest over where the data sits and how little of it has to move. Part 1 showed that it holds for a single decode step. This post found that five methods the industry actually uses all fit inside it. For someone who barely studied for the OS exam, it's a little embarrassing that the answer was in the OS textbook.

## References

- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245)
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
- vLLM, [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm) (2023-06-20; 60 to 80% waste vs. under 4%)
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
- DeepSeek-AI, [DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) (MLA, 93.3% smaller KV cache)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135), [flash-attention README](https://github.com/Dao-AILab/flash-attention) (supported GPUs)
- Meta Llama 3.1 70B [config.json](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json) (NousResearch mirror)
- NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/) (FP16 1,979 TFLOPS with sparsity, FP8 3,958, HBM 3.35 TB/s, 80GB)
- NVIDIA, [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) (FP16 1,000 dense / 2,000 sparse TFLOPS, preliminary)
- Groq, [What is a Language Processing Unit?](https://groq.com/blog/the-groq-lpu-explained), [GroqChip Processor Product Brief v1.7](https://groq.com/wp-content/uploads/2024/08/GroqChip™-Processor-Product-Brief-v1.7.pdf) (188 TFLOPS FP16, 230MB SRAM, 80 TB/s)
- a16z, OpenRouter, [State of AI: An Empirical 100 Trillion Token Study (arXiv:2601.10088)](https://arxiv.org/abs/2601.10088)
- Anthropic, [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- NVIDIA, [BlueField-4 AI-native storage infrastructure](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai) (2026-01)
