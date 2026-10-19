---
title: Why NVIDIA started splitting inference into three pieces
description: A GPU company is now selling racks of chips that aren't GPUs. The reason is that each part of inference reads different data. Part 3 of the series "LLM inference, from compute to memory".
pubDatetime: 2026-10-19T00:00:00+09:00
category: Engineering
type: tech-essay
tags:
  - llm
  - inference
  - nvidia
  - groq
  - kv-cache
translationKey: llm-inference-3-three-pieces
draft: false
---

> Series **LLM inference, from compute to memory** (3/3)
> 1. [Doubling FLOPS won't make ChatGPT faster](/en/posts/llm-inference-1-memory-bound/)
> 2. [Five ways to make ChatGPT faster are all the same idea](/en/posts/llm-inference-2-five-ways/)
> 3. Why NVIDIA started splitting inference into three pieces (this post)

## A GPU company shipped a rack with no GPUs in it

In March 2026 NVIDIA announced a rack called Groq 3 LPX ([The Register](https://www.theregister.com/2026/03/16/nvidia_lpx_groq_3/)). The chips inside are not GPUs. They are the third generation of Groq's LPU, the chip from Part 2 that keeps weights in on-chip SRAM and has no HBM. Three months earlier, on Christmas Eve 2025, NVIDIA had signed an inference technology licensing deal with Groq, and Groq founder Jonathan Ross and other key staff moved to NVIDIA ([Groq announcement](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale)). Neither company disclosed the price. The $20 billion figure comes from press reports.

LPX wasn't the only new piece. Six months before it, NVIDIA announced Rubin CPX, a separate GPU for processing long inputs. In January 2026 it announced a storage tier that keeps the KV cache outside the GPU. A company that used one kind of GPU for both training and inference split its inference hardware three ways within a year.

Why would a GPU company sell chips that aren't GPUs? The questions from Parts 1 and 2 point to the answer.

## The claim

> NVIDIA is splitting inference into three bottlenecks (compute, bandwidth, and capacity) and assigning a different chip to each.

Prefill, which reads the whole input at once, goes to a chip with a lot of compute. The weights, which every user reads, go to a chip with a lot of bandwidth. The KV cache, which grows separately for each user, goes to a chip with a lot of capacity and to a storage tier behind it. This claim can be wrong. If the split products see little real use, or if a single general-purpose GPU keeps beating the split setups on cost per token, it is wrong. I come back to these signals at the end.

## What you need first

### The two parts inside a layer

An LLM is a stack of identical layers. Llama 3.1 70B has 80 of them. Each layer has two parts in sequence:

- **Attention** decides which earlier tokens the current token should draw on, and mixes in their content.
- **FFN** (feed-forward network) takes the single vector that attention produced and multiplies it by a few large matrices. It never looks at earlier tokens.

To produce one token during decode, the model runs this attention-then-FFN pair 80 times. At the end, a matrix called the LM head and a softmax pick the next token. The LM head gives a score to every token in the tokenizer's vocabulary. [Llama 3.1 70B](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json) has 128,256 vocabulary entries, so the LM head is 8,192 × 128,256, and it is also a weight that everyone shares. The vocabulary is made of subword tokens, which are smaller than words.

### Weights and K are different things

Part 2 said that attention uses Q, K, and V for each token, and that the stored K and V of past tokens are the KV cache. While preparing this post I got confused here. K changes with the user's input, so doesn't the weight that produces K also differ per user?

I had learned this before and forgotten it. I was mixing up two things with similar names:

| | What it is | Different per user? |
|---|---|---|
| W_K, W_V | Matrices fixed by training. Weights stored in the model file | No |
| K, V | An input vector x multiplied by those matrices. `K = x × W_K` | Yes |

Think of a recipe and a dish. Everyone reads the same recipe (W_K), but each customer brings different ingredients (x), so each dish (K) comes out different. The KV cache stores the dishes.

This distinction carries the rest of the post.

- **Weights** are shared by everyone and fixed in size. A 70B model is 140GB in FP16 and 70GB in 8-bit.
- **The KV cache** builds up separately for each user. As calculated in Part 2, one 100,000-token conversation takes about 33GB, and the total grows with the number of users and the length of their conversations. Fifteen such users need about 500GB, more than the weights.

## Piece 1. Separating prefill from decode

As Part 1 showed, prefill and decode have different bottlenecks. Prefill processes the whole prompt at once, so each read of the weights serves thousands of tokens and compute is the limit. Decode produces one token at a time, so the limit is how fast the weights can be read.

When one GPU does both, they get in each other's way. A long prompt's prefill can stall every other user's decode while it runs. [DistServe (Zhong et al., OSDI 2024)](https://arxiv.org/abs/2401.09670) put prefill and decode on different GPUs and served up to 7.4 times more requests than existing systems under the same latency targets.

NVIDIA built this approach into its serving software [Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models) in March 2025. It reported more than double the throughput for Llama 70B on Hopper GPUs and up to 30 times more requests served for DeepSeek-R1 on GB200 NVL72.

Up to this point the GPUs were the same and only their roles differed. The next step was to build a different chip for each role. [Rubin CPX](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference), announced in September 2025, is a GPU dedicated to long inputs. It has 30 PFLOPS of NVFP4 compute, and it uses 128GB of GDDR7 in place of HBM. Prefill is limited by compute, so it doesn't need HBM-class bandwidth. My reading is that NVIDIA left expensive memory out of a place where bandwidth would go unused. It is scheduled for the end of 2026.

## Piece 2. Splitting decode one more time

### I thought Groq 3 would handle decode

In Part 1 I wrote that NVIDIA had set up Groq 3 to "handle the decode phase," and Part 2 said the same thing. My picture was GPU for prefill, LPU for decode.

That picture was one of several setups. NVIDIA's [technical blog](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) from August 2026 describes three ways to combine GPUs and LPX:

1. Prefill-decode disaggregation: the GPU runs prefill and hands the KV cache over once per turn, and LPX runs the entire decode. This was my picture.
2. Speculative decoding: LPX runs a small draft model quickly, and the large model on the GPU checks its output.
3. **Attention-FFN Disaggregation (AFD)**: inside decode, the GPU runs attention and LPX runs the FFN.

The third one is the subject of this post. In NVIDIA's words: "Vera Rubin NVL72 computes attention and holds the KV cache in DRAM, while Groq 3 LPX executes the FFN layers." Only an intermediate vector travels between the two racks, once per layer.

### Why attention stays on the GPU

When I first saw this setup, I assumed attention creates the KV cache and the FFN reads it. That's wrong. Attention both writes and reads the KV cache. Comparing the new token's Q against every stored K and mixing the V values is the attention step, and the KV cache exists so this step doesn't have to recompute them. The FFN sees one vector and the weights, nothing else.

So the two parts read different data:

| Part of decode | Data it reads | Size | Chip |
|---|---|---|---|
| Attention | Per-user KV cache | Grows with users × conversation length | Vera Rubin GPU (288GB HBM) |
| FFN | Shared weights | Fixed | Groq 3 LPU (500MB SRAM, 150 TB/s) |

The numbers show why you can't swap them. According to [IEEE Spectrum](https://spectrum.ieee.org/nvidia-groq-3), each Groq 3 LPU has 500MB of SRAM, and an LPX rack holds 256 of them for 128GB in total. One 100,000-token user needs about 33GB of KV cache, so a whole rack can't hold four such users. Weights are a fixed size and can be spread across hundreds of chips. A 70B model stored in 8-bit is 70GB and fits in one rack.

In Part 2's terms, the large, per-user meeting notes (the KV cache) go to the big warehouse (HBM), and the small recipe everyone shares (the weights) goes on the shelf right next to the cutting board (SRAM). You might ask why the weights can't just sit in HBM. I asked the same thing. HBM is a separate chip next to the compute cores, so even data that is "loaded" there has to travel to the cores before it can be used. Every decode step hauls all the weights from HBM to the cores again. SRAM sits on the same chip as the cores. It takes up a lot of die area, though, which caps it at 500MB per chip. So instead of building a bigger shelf, Groq chains hundreds of chips that each have one.

### What batching doesn't fix

There is an obvious objection here.

> "The FFN already runs efficiently with batching. The weights are read once and shared by many users. Why move it to a different chip?"

Batching raises throughput. It doesn't change how fast the answer arrives for any one person. If a dish takes 30 minutes to cook, it takes 30 minutes whether one person orders it or a hundred. Batching only lets you cook a hundred portions in the same pan.

In decode, those 30 minutes are set like this:

```text
time per step ≥ weight size ÷ memory bandwidth
```

No matter how many users share a step, the step has to read all the weights once. Batching puts more people in that window, and bandwidth shrinks the window itself. Rubin's HBM delivers 22 TB/s, and Groq 3's SRAM delivers 150 TB/s. The word in NVIDIA's blog title is interactivity, the speed one person feels, rather than throughput.

Running Part 2's calculation on both chips shows how different they are:

```text
Rubin GPU:  50 PFLOPS (FP4) ÷  22 TB/s ≈ 2,300
Groq 3 LPU: 1.2 PFLOPS (FP8) ÷ 150 TB/s ≈ 8
```

→ A chip that needs 2,300 operations per byte read to keep busy and a chip that needs 8 split the same request. The precisions differ (FP4 versus FP8), so this isn't an exact comparison, but the gap in orders of magnitude is far larger than the gap in precision. The GPU needs hundreds of users to reach its full performance. The LPU stays busy with a handful.

The performance figures come from Artificial Analysis measurements quoted in NVIDIA's blog. Gemma 4 31B running on LPX with a 100,000-token context produced a median of 3,431 output tokens per second. The same blog says the fastest public endpoint managed 870. These are numbers a company chose to publish on its own blog, which is worth keeping in mind.

### Not a new idea

NVIDIA didn't invent the idea of separating attention from the FFN. [MegaScale-Infer (April 2025)](https://arxiv.org/abs/2504.02263) from ByteDance Seed put attention and FFN on separate GPU groups when serving MoE models and raised per-GPU throughput by up to 1.90 times. It even proposed mixing GPU types, with attention on GPUs that have plenty of memory and FFN on GPUs with cheaper compute. StepFun's [Step-3 (July 2025)](https://arxiv.org/abs/2507.19427) gave the approach the name AFD. Both picked from among GPUs. NVIDIA attached a chip with a completely different memory design to the FFN side. Like the five methods in Part 2, an idea that started in software has moved down into hardware.

What about larger models? OpenAI, Anthropic, and other frontier labs don't publish the sizes of their latest models. Among open models, [DeepSeek-V3](https://arxiv.org/abs/2412.19437) has 671B parameters, nearly ten times a 70B model. Models this size were always spread across dozens of GPUs. A single matrix can be split across GPUs, layers can be split across servers, and in an MoE model the experts can be split across GPUs. Each GPU then reads only its own share, so the bandwidth of many GPUs adds up. One Vera Rubin NVL72 rack has 72 GPUs and about 20TB of HBM (72 × 288GB). DeepSeek-V3 is also an MoE model, so producing one token uses only 37B of its parameters. NVIDIA's blog includes a projection of a GPT-OSS model scaled to 2 trillion parameters running on GPUs plus LPX, but those numbers are projected; no measurements were published.

## Piece 3. Moving the KV cache off the GPU

The last piece is capacity. Even if AFD frees the GPU's HBM for the KV cache alone, more users and longer conversations eventually fill it. Agents in particular keep going for many turns, appending tool results as they go. Like the prefix cache in Part 2, earlier turns' KV gets reused often, and keeping all of it on the GPU takes space away from the users who are generating right now.

When NVIDIA announced its [BlueField-4 storage tier](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai) in January 2026, it wrote: "A KV cache cannot be stored on GPUs long term, as this would create a bottleneck for real-time inference in multi-agent systems." KV that isn't needed right now moves to a storage tier outside the GPU and comes back for the next turn. NVIDIA claims up to 5 times more tokens per second and up to 5 times better power efficiency than traditional storage. It ships in the second half of 2026. On the software side, Dynamo's KV cache manager moves older KV out to CPU memory, local storage, and networked object storage.

Putting the three pieces on one page:

![One request flowing through a prefill chip, round trips between a GPU running attention and LPX running the FFN, and a KV storage tier outside the GPU](/images/posts/llm-inference-3-three-pieces/three-pieces-en.png)

NVIDIA didn't announce these four as one configuration. In the setups the LPX blog describes, Vera Rubin NVL72 handles prefill. The figure collects where each announcement fits.

## Different bets

| Approach | How it saw the bottleneck | Fix | Cost |
|---|---|---|---|
| One general-purpose GPU | One chip can handle every stage | Batching and the techniques from Part 2 | Compute or bandwidth sits idle depending on the stage. Per-user speed stays capped by HBM bandwidth |
| Same GPUs, separate roles (DistServe, Dynamo, MegaScale-Infer) | Stages with different needs interfere | Separate GPU pools per stage | KV or intermediate vectors must move between GPUs. Pool ratios need tuning |
| Different chips (Rubin CPX, Groq 3 LPX, BlueField-4) | Each stage runs short of a different resource | A chip with the right memory for each bottleneck | AFD makes one rack-to-rack round trip per layer for every token. More chip types mean harder operations |
| One chip for everything (Groq on its own) | The path from HBM is itself too slow | Keep the weights in SRAM | Hundreds of chips. No good place for the KV cache |

→ Going down the table, the unit of splitting gets finer. The third row covers the last row's weakness (where to keep the KV cache) with GPUs, and covers the GPU's weakness (per-user speed) with LPUs.

## Why now

I see two reasons. The first is a long-running trend. According to [AI and Memory Wall (Gholami et al., 2024)](https://arxiv.org/abs/2403.14123), peak server compute grew 3.0 times every two years, while DRAM bandwidth grew 1.6 times and interconnect bandwidth 1.4 times. The wider that gap gets, the harder it is for one chip to be good at both.

The second is how models are used. In the [OpenRouter 100-trillion-token study](https://arxiv.org/abs/2601.10088) cited in Part 2, the average input per request grew from about 1,500 tokens to more than 6,000 over 13 months. Longer inputs make prefill heavier and the KV cache bigger, so compute runs short on one side while capacity runs short on the other. Once one chip couldn't cover both, splitting the work was the way forward.

## What's next

I think the splitting will continue on the memory side. One candidate is HBF (High Bandwidth Flash), NAND flash stacked like HBM. Sandisk and SK hynix [announced](https://www.businesswire.com/news/home/20250806259749/en) a standardization effort in August 2025, targeting 8 to 16 times the capacity of HBM at a similar cost, with first samples in the second half of 2026. Flash wears out with repeated writes, but in inference the weights are loaded once and then only read. By this post's distinction, HBF suits the weights and probably suits the constantly rewritten KV cache less well.

Two signals would show this forecast is wrong:

1. Split products like LPX and CPX see little use in real data centers, and a single general-purpose GPU keeps winning on cost per token.
2. As mentioned at the end of Part 2, architectures like State Space Models, whose state doesn't grow with conversation length, become mainstream. If the KV piece shrinks, so does the room for a capacity chip.

## My take

Looking back, the conclusion of Parts 1 and 2, that inference speed comes down to where data lives and how little of it moves, has now become the unit by which chips are divided. GQA and PagedAttention in Part 2 rearranged data inside one chip. This time a separate chip was built for each kind of data.

> "Isn't this just NVIDIA buying a competitor and dressing it up as a technical story?"

I can't know the motives behind the deal. What I can check is that, technically, the two chips need each other. A Groq chip alone has nowhere to keep the KV cache (in Part 2 I couldn't find public material on where Groq keeps it), and a GPU alone leaves per-user speed tied to HBM bandwidth. AFD sits exactly where those two weaknesses meet.

> "Won't a rack-to-rack round trip at every layer slow things down?"

What travels is one vector per token. For a 70B model that's 8,192 numbers, about 16KB in FP16. One layer's weights are about 1.75GB in FP16 (140GB ÷ 80), so the vector is tiny by comparison. Eighty round trips still add latency, though. The 3,431 tokens per second is NVIDIA's own showcase, and whether other models and setups see the same gain will depend on measurements still to come.

Finally, a few words about Groq itself. Groq made chips, but the business it put forward publicly was selling inference run on those chips. On [GroqCloud](https://console.groq.com/docs/models), developers call models through an API and pay per token processed. For example, gpt-oss-120b costs $0.15 per million input tokens and $0.60 per million output tokens (as of October 2026). [Batch requests](https://console.groq.com/docs/batch) that can finish anywhere from 24 hours to 7 days later get a 50% discount. The other line of business is national-scale deals. In February 2025 Saudi Arabia committed $1.5 billion to expand Groq's LPU inference infrastructure there, and Groq runs a data center in Dammam and became the inference provider for HUMAIN, a Saudi AI company ([Groq](https://groq.com/blog/groq-solidifies-status-as-emerging-hyperscaler-with-new-global-deployment)). Revenue figures haven't been disclosed. When you sell by the token, cost per token decides the margin, which is also why this post's falsification signal is framed in cost per token.

At the end of 2025 the chip technology was licensed to NVIDIA, and the founder and other designers moved there. In August 2026 the remaining Groq raised $350 million, and the [press release](https://groq.com/newsroom/groq-closes-usd350-million-series-a-building-the-world-s-leading-ai-inference-cloud) headline promised to build "the world's leading AI inference cloud." It listed 13 data centers, six million developers, and plans to grow from 54 megawatts of power to more than 200 megawatts in 2027. Measuring growth in megawatts of data center power is how a cloud operator talks. More striking is where the money goes. The release says it will support customers who want clusters of "NVIDIA accelerated computing." LPUs come up in one line about "experience operating LPUs at scale."

The design that splits chips by data type moved inside a chip company, and the company that first pushed that design moved over to running chips. I don't know yet whether more of inference's value will accrue to those who sell chips or to those who run inference. But the way memory makers (HBM, and now HBF) and inference chip makers are all drawing attention at once points the same way as this series' conclusion. I plan to write about that industry and its supply chain separately.

That wraps up the series, but one question from Part 2 is still open: does quantization work by reducing the bytes to move, or by making the math faster? That's the next post.

## Sources

- NVIDIA, [How NVIDIA Groq 3 LPX Unlocks Ultrafast Interactivity at Long Context on NVIDIA Vera Rubin](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) (2026-08; AFD, three configurations, Gemma 4 31B measurement)
- IEEE Spectrum, [Nvidia Groq 3](https://spectrum.ieee.org/nvidia-groq-3) (Groq 3 LPU: 500MB SRAM, 1.2 PFLOPS 8-bit, 150 TB/s; Rubin GPU: 288GB HBM, 50 PFLOPS 4-bit, 22 TB/s)
- The Register, [Nvidia slaps $20B Groq tech into massive new LPX racks](https://www.theregister.com/2026/03/16/nvidia_lpx_groq_3/) (2026-03-16)
- Groq, [Groq and NVIDIA Enter Non-Exclusive Inference Technology Licensing Agreement](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale) (2025-12)
- Groq, [GroqCloud Models](https://console.groq.com/docs/models), [Batch API](https://console.groq.com/docs/batch) (per-token pricing, 50% batch discount, checked 2026-10-19)
- Groq, [Groq Solidifies Status as Emerging Hyperscaler with New Global Deployment](https://groq.com/blog/groq-solidifies-status-as-emerging-hyperscaler-with-new-global-deployment) (2025-05-27; Saudi $1.5B, HUMAIN)
- Groq, [Groq closes $350 million Series A](https://groq.com/newsroom/groq-closes-usd350-million-series-a-building-the-world-s-leading-ai-inference-cloud) (2026-08-17)
- NVIDIA, [NVIDIA Unveils Rubin CPX](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) (2025-09)
- NVIDIA, [BlueField-4 AI-native storage infrastructure](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai) (2026-01)
- NVIDIA, [Introducing NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models) (2025-03)
- Zhong et al., [DistServe (arXiv:2401.09670, OSDI 2024)](https://arxiv.org/abs/2401.09670)
- Zhu et al., [MegaScale-Infer (arXiv:2504.02263)](https://arxiv.org/abs/2504.02263) (ByteDance Seed, up to 1.90x per-GPU throughput)
- StepFun, [Step-3 is Large yet Affordable (arXiv:2507.19427)](https://arxiv.org/abs/2507.19427)
- DeepSeek-AI, [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) (671B total, 37B per token)
- Meta Llama 3.1 70B [config.json](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json) (NousResearch mirror; 80 layers, hidden size 8,192, vocabulary 128,256)
- Gholami et al., [AI and Memory Wall (arXiv:2403.14123)](https://arxiv.org/abs/2403.14123)
- a16z, OpenRouter, [State of AI: An Empirical 100 Trillion Token Study (arXiv:2601.10088)](https://arxiv.org/abs/2601.10088)
- Sandisk, [Sandisk and SK hynix HBF standardization announcement](https://www.businesswire.com/news/home/20250806259749/en) (2025-08)
