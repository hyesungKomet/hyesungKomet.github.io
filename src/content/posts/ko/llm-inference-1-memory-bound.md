---
title: FLOPS를 2배로 올려도 ChatGPT는 빨라지지 않는다
description: LLM이 답을 쓰는 속도는 계산이 아니라 메모리에서 읽어 오는 속도가 정한다. 시리즈 "LLM 추론, 연산에서 메모리로"의 1편.
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

> 시리즈 **LLM 추론, 연산에서 메모리로** (1/3)
> 1. FLOPS를 2배로 올려도 ChatGPT는 빨라지지 않는다 (이 글)
> 2. ChatGPT를 빠르게 만드는 다섯 가지 방법은 전부 같은 얘기다 (준비 중)
> 3. NVIDIA는 왜 추론을 세 조각으로 쪼개기 시작했나 (준비 중)

## 내가 기각했던 sLLM

예전에 무인 키오스크를 주제로 한 대회에 나간 적이 있다. 키오스크 안에 작은 언어 모델(sLLM)을 넣어 보려고 Colab에서 Gemma 같은 작은 모델을 돌려 봤는데, 응답이 너무 늦게 나왔다. 그 대회는 추론 속도가 평가의 주요 요소였다. 결국 sLLM은 쓰지 못하고 Rasa와 다른 NLU 모델로 방향을 틀었다.

그때 나는 transformers 라이브러리로 결과를 **한 번에** 돌려받는 방식만 썼다. 모델이 마지막 글자까지 다 만든 뒤에야 화면에 뭔가가 떴다. 나중에 ollama로 같은 종류의 모델을 돌리거나 스트리밍으로 받아 보니 글자가 금방 나오기 시작했다. 그때는 TTFT라는 말도 몰랐다. ollama가 실제로도 더 빨랐던 이유는 따로 있는데, 그건 다른 글에서 다루려 한다.

비슷한 기억이 하나 더 있다. PyTorch로 Transformer 계열 모델을 학습시킬 때 `DataLoader`의 `num_workers`를 잘못 잡으면 GPU 사용률이 바닥을 쳤다. CPU가 데이터를 준비해서 GPU로 올리는 속도가 못 따라가니 GPU가 놀고 있었던 것이다.

그래서 궁금해졌다. 내 데스크탑 한 대에서 모델 하나를 돌리는 게 아니라, ChatGPT처럼 데이터센터의 거대한 GPU를 수많은 사용자가 나눠 쓰는 곳에서는 어디가 막힐까?

## 답을 쓰는 속도는 메모리가 정한다

이 시리즈의 출발점이 된 질문이 있다.

> NVIDIA가 내일 **연산 성능(FLOPS)만 2배**로 올리고 메모리 대역폭은 그대로인 GPU를 낸다면, ChatGPT는 2배 빨라질까?

결론부터 말하면 그렇지 않다. 답이 **시작되는** 시간은 줄어들지만, 답이 **써지는** 속도는 거의 그대로다. LLM이 답을 한 글자씩 써 내려가는 속도는 계산이 얼마나 빠른지가 아니라 메모리에서 데이터를 얼마나 빨리 읽어 오는지가 정하기 때문이다.

이 주장을 이해하려면 개념 몇 가지가 필요하다. 하나씩 뜯어보자.

## 먼저 알아야 할 것들

### 1) LLM은 한 토큰씩 쓴다

LLM은 문장을 토큰(token) 단위로 다룬다. 토큰은 단어나 단어 조각이라고 생각하면 된다. 그리고 답을 한 번에 만들지 않는다. 지금까지의 글을 보고 다음 토큰 하나를 고르고, 그 토큰을 붙인 글을 다시 보고 그다음 토큰을 고르는 일을 반복한다. 이렇게 앞의 결과를 다음 입력으로 쓰는 방식을 autoregressive(자기회귀) 생성이라고 한다.

→ 토큰 2는 토큰 1이 나와야 만들 수 있다. 한 사용자의 답 안에서는 순서대로 할 수밖에 없다.

최근에는 [Mercury](https://arxiv.org/abs/2506.17298)처럼 diffusion 방식으로 여러 토큰을 한꺼번에 다듬어 가는 언어 모델도 나오고 있다. 하지만 GPT나 Gemini 같은 대부분의 LLM은 여전히 한 토큰씩 쓰는 방식이고, 이 글은 그쪽을 다룬다.

### 2) FLOPS와 메모리 대역폭

GPU가 하는 일을 아주 단순하게 나누면 두 가지다.

1. 메모리에서 숫자를 가져온다
2. 가져온 숫자로 계산한다

FLOPS(Floating-point Operations Per Second)는 2번, 즉 GPU가 1초에 할 수 있는 실수 연산 횟수다. GPU 광고에서 가장 크게 내세우는 숫자다. 메모리 대역폭(memory bandwidth)은 1번, 즉 1초에 메모리에서 계산하는 곳까지 옮길 수 있는 데이터의 양이다.

주방에 비유하면 이해가 쉽다. FLOPS는 요리사의 손 빠르기이고, 메모리 대역폭은 창고에서 주방까지 재료를 실어 나르는 트럭이다. 요리사가 아무리 빨라도 재료가 늦게 오면 요리사는 기다려야 한다.

어느 쪽이 발목을 잡느냐에 따라 상태를 두 가지로 부른다.

- **Compute-bound(연산 병목)**: 재료는 넉넉히 오는데 요리사가 바쁘다
- **Memory-bandwidth-bound(대역폭 병목)**: 요리사는 손이 비는데 트럭이 못 따라온다

### 3) prefill과 decode, TTFT와 ITL

LLM이 요청 하나를 처리하는 과정은 성격이 다른 두 단계로 나뉜다.

- **prefill**: 사용자가 넣은 입력 전체를 한 번에 처리하는 단계다. 입력 토큰 수천 개를 큰 행렬 계산 하나로 묶어서 처리할 수 있고, 이 단계 끝에서 첫 번째 출력 토큰이 나온다.
- **decode**: 그 뒤로 토큰을 하나씩 만들어 붙이는 단계다. 답이 길면 이 단계도 길어진다.

사용자 입장에서는 이 두 단계가 서로 다른 지연으로 느껴진다.

- **TTFT(Time To First Token)**: 요청을 보내고 첫 글자가 보일 때까지의 시간. 대부분 prefill이 정한다.
- **ITL(Inter-Token Latency)**: 글자가 흘러나올 때 토큰과 토큰 사이의 간격. decode가 정한다.

![같은 요청을 한꺼번에 받을 때와 스트리밍으로 받을 때 사용자가 첫 글자를 보는 시점 비교](/images/posts/llm-inference-1-memory-bound/request-timeline.svg)

키오스크에서 내가 겪은 일이 이 그림의 윗줄이다. 결과를 한꺼번에 받으면 사용자가 기다리는 시간은 TTFT보다 훨씬 긴 전체 생성 시간이 된다. GPU가 하는 일은 똑같은데도 스트리밍으로 받으면 prefill이 끝나는 순간부터 글자가 보인다.

### 4) KV Cache: 이미 계산한 것을 다시 하지 않기

decode에서 새 토큰을 하나 만들 때마다 모델은 앞의 모든 토큰을 참고해야 한다. 이때 쓰는 것이 [Self-Attention](https://arxiv.org/abs/1706.03762)이고, 각 토큰에서 Q(Query), K(Key), V(Value) 세 가지 값을 뽑아 쓴다.

도서관에 비유하면 Q는 지금 하는 검색어, K는 책의 색인, V는 책의 내용이다. 새 토큰이 들어올 때마다 새 검색어(Q)로 지금까지 쌓인 모든 색인(K)을 훑고, 관련 있는 내용(V)을 가져온다.

여기서 중요한 점이 있다. 과거 토큰의 K와 V는 앞으로 올 모든 검색에서 계속 쓰이지만, 과거 토큰의 Q는 그 토큰을 처리할 때 한 번 쓰고 끝난다. 그래서 K와 V만 저장해 두고 재사용한다. 이것이 **KV Cache**다. Q Cache가 없는 이유가 여기에 있다.

→ prefill이 입력 전체의 KV Cache를 만들고, decode는 매 토큰마다 그 KV Cache를 읽고 새 K, V를 덧붙인다.

## decode는 왜 메모리에 묶이나

### 숫자로 보는 천장

70B(700억 개 파라미터) 모델을 예로 든다. 파라미터 하나를 FP16(16bit, 2바이트)으로 저장하면 weight(가중치)만 약 140GB다.

decode에서 토큰 **하나**를 만들려면 이 weight 전체를 한 번씩 거쳐야 한다. 모든 층의 모든 가중치를 써서 다음 토큰을 계산하기 때문이다(모든 층을 다 쓰는 일반적인 dense 모델 기준이다. 일부 전문가 층만 골라 쓰는 MoE 모델은 읽는 양이 다르다). [NVIDIA H100 SXM](https://www.nvidia.com/en-us/data-center/h100/)의 메모리 대역폭은 3.35 TB/s다.

```text
140 GB ÷ 3.35 TB/s ≈ 0.042초
```

→ 연산이 아무리 빨라도, 사용자 한 명에게는 초당 약 24토큰이 천장이다. 실제로 70B 모델은 GPU 한 장에 다 안 들어가서 여러 장에 나눠 올리지만, 토큰마다 weight 전체를 읽어야 한다는 구조는 같다.

### 처음엔 NVLink에서 가져오는 줄 알았다

이 계산을 처음 봤을 때 나는 weight를 HBM에서 NVLink로 가져오는 줄 알았다. 그래서 NVLink 대역폭이 연산 속도에 맞춰 커져야 하는 게 아닌가 생각했다. 틀린 그림이었다.

![decode 때 weight가 HBM에서 연산 유닛으로 오는 경로와 NVLink, PCIe의 대역폭 비교](/images/posts/llm-inference-1-memory-bound/gpu-memory-path.svg)

HBM(High Bandwidth Memory)은 GPU 칩 바로 옆, 같은 패키지 안에 붙어 있는 메모리다. decode의 병목은 **HBM에서 GPU 안의 연산 유닛으로 오는 길**이다. NVLink는 GPU와 GPU 사이를 잇는 선이고, PCIe는 CPU 쪽과 GPU를 잇는 선이다.

| 경로 | 무엇을 잇나 | H100 SXM 대역폭 |
|---|---|---:|
| HBM → 연산 유닛 | GPU 내부 | **3.35 TB/s** |
| NVLink | GPU ↔ GPU | 900 GB/s |
| PCIe Gen5 | CPU ↔ GPU | 128 GB/s |

→ PCIe는 모델을 처음 GPU에 올릴 때 주로 쓰인다. 모델이 GPU 메모리에 다 올라가 있다면 decode 중에는 거의 쓰이지 않는다.

### 처음엔 한 번만 읽으면 되는 줄 알았다

두 번째로 헷갈린 지점은 0.042초의 의미였다. 나는 이걸 "140GB를 GPU에 처음 올리는 시간"으로 읽었다. 그런데 weight는 이미 HBM에 올라가 있다. 0.042초는 토큰을 **하나 만들 때마다** HBM에서 연산 유닛으로 weight를 다시 흘려보내는 시간이다.

그럼 weight는 바뀌지 않는 값이니 한 번 읽어서 연산 유닛 옆에 캐시해 두면 되지 않을까? 나도 그렇게 생각했다. 문제는 담을 자리가 없다는 것이다. 연산 유닛 바로 옆에 있는 빠른 메모리가 [H100의 L2 캐시](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)인데, 50MB다. 140GB와는 약 2,800배 차이가 난다. 그래서 매 토큰마다 HBM에서 다시 읽을 수밖에 없다.

### 처음엔 대역폭이 놀고 있는 줄 알았다

세 번째는 방향을 거꾸로 이해한 것이다. 사용자 한 명만 처리할 때 GPU가 제 성능을 못 낸다는 얘기를 듣고, 나는 대역폭을 다 못 쓰고 있다는 뜻인 줄 알았다. 실제로는 반대다.

주방으로 돌아가 보면 트럭은 쉬지 않고 최고 속도로 달리고 있다. 대역폭은 한계까지 쓰이고 있다. 놀고 있는 건 요리사다. 트럭이 재료를 한 번 실어 올 때마다 요리사는 1인분만 만들고 다음 트럭을 기다린다. 100인분을 만들 수 있는 요리사가 1인분만 만들고 있는 셈이다.

→ "decode는 memory-bandwidth-bound다" = 대역폭은 꽉 차 있고 연산이 남아도는 상태다.

prefill은 사정이 다르다. 입력 토큰 수천 개를 한 번에 처리하니, 트럭 한 번에 실어 온 weight로 수천 인분을 만든다. 그래서 prefill은 compute-bound에 가깝다.

### 처음 질문으로 돌아가면

FLOPS만 2배인 GPU가 나오면 어떻게 될까?

- prefill은 연산이 병목이라 빨라진다 → TTFT가 줄어든다
- decode는 대역폭이 병목이라 거의 그대로다 → ITL은 그대로다

→ 첫 글자는 더 빨리 나오지만, 글자가 흘러나오는 속도는 똑같다. ChatGPT가 2배 빨라졌다고 느끼기는 어렵다.

### 데스크탑의 DataLoader와 같은 문제

글 처음에 꺼낸 `num_workers` 이야기와 비교해 보면 재미있다. 데스크탑에서 학습할 때는 CPU에서 GPU로 데이터를 올리는 길이 막혀서 GPU가 놀았다. 데이터센터의 decode에서는 막히는 곳이 **GPU 안쪽**, HBM에서 연산 유닛으로 가는 길이다. 그래서 연산이 논다.

증상은 같다. GPU의 연산 능력이 남는다. 막히는 위치만 다르다. 규모가 커져도 결국 데이터를 옮기는 길이 문제라는 점은 같았다.

## 같은 벽, 두 방향

이 벽 앞에서 갈 수 있는 방향은 크게 두 가지다. 그 전에 메모리 종류 두 가지를 짚고 가야 한다.

### SRAM과 DRAM

컴퓨터의 메모리는 만드는 방식에 따라 크게 둘로 나뉜다.

- **SRAM(Static RAM)**: 트랜지스터 여러 개로 1비트를 붙잡아 두는 메모리다. 아주 빠르지만 한 칸이 크고 비싸서 많이 넣을 수 없다. 그래서 연산 유닛 바로 옆에 조금만 둔다. GPU의 L2 캐시가 바로 SRAM이다.
- **DRAM(Dynamic RAM)**: 작은 축전기에 전하를 담아 1비트를 기억하는 메모리다. 촘촘하고 싸서 용량을 크게 만들 수 있지만 SRAM보다 느리다. HBM은 이 DRAM을 여러 층 쌓아 GPU 옆에 붙인 것이다.

→ SRAM은 메모리의 **종류**이고, L2 캐시는 그 SRAM을 어디에 어떻게 쓰느냐에 붙은 **용도**의 이름이다. GPU는 작은 SRAM을 큰 HBM 앞의 캐시로 쓰고, 아래 Groq은 SRAM 자체를 weight를 담는 주 메모리로 쓴다.

### 두 방향


| 방향 | 대표 | 어떻게 | 대가 |
|---|---|---|---|
| 연산을 키운다 | 범용 GPU의 세대 교체 | FLOPS를 계속 올리고 HBM 대역폭도 함께 올린다 | decode 속도는 대역폭이 느는 만큼만 빨라진다 |
| 메모리를 칩 안으로 | Groq LPU | weight를 칩 위 SRAM에 통째로 올린다 | 칩 하나의 SRAM이 작아 큰 모델은 칩 여러 장에 나눠야 한다 |

[Groq LPU](https://groq.com/blog/the-groq-lpu-explained)는 첫 세대 기준 칩 하나에 SRAM 230MB를 두고, 내부 대역폭이 약 80TB/s다. H100의 HBM보다 한 자릿수 이상 크다. 대신 230MB로는 140GB짜리 weight를 담으려면 단순 계산으로도 칩 600장 넘게 필요하다. 아까 "캐시할 자리가 없다"는 벽을 칩 개수로 밀어붙인 셈이다.

이 구조는 이제 NVIDIA 안으로 들어갔다. 2026년 3월 NVIDIA는 Groq 기술을 바탕으로 한 Groq 3 LPU를 발표했고, 자사 차세대 플랫폼 Vera Rubin 안에서 decode 단계를 맡기는 구성을 내놓았다([IEEE Spectrum](https://spectrum.ieee.org/nvidia-groq-3)). 이 이야기는 3편에서 자세히 다룬다.

그리고 첫 번째 방향에는 구조적인 어려움이 있다. [AI and Memory Wall (Gholami 등, 2024)](https://arxiv.org/abs/2403.14123)에 따르면 지난 20년 동안 서버 하드웨어의 최대 FLOPS는 2년마다 3.0배씩 늘었지만, DRAM 대역폭은 1.6배씩만 늘었다. 연산과 메모리의 격차가 계속 벌어지고 있다는 뜻이다.

## 다음 글에서

decode가 메모리에 묶여 있다면, 업계는 이 문제를 어떻게 풀고 있을까? 사용자 여러 명을 묶어 처리하는 batching, KV Cache를 작게 만드는 GQA, 메모리 낭비를 줄이는 PagedAttention, 같은 계산을 건너뛰는 Prefix Cache까지 서로 달라 보이는 방법들이 사실은 같은 얘기를 하고 있다. 2편에서 다룬다.

3편에서는 NVIDIA가 왜 추론을 단계별로 쪼개서 서로 다른 칩에 맡기기 시작했는지를 본다.

## 내 생각

키오스크 대회에서 sLLM을 기각했을 때, 나는 모델이 느리다고 판단했다. 지금 보면 절반은 내가 결과를 받는 방식의 문제였다. TTFT와 전체 생성 시간이 다르다는 걸 알았다면 같은 모델로도 다른 결론을 냈을 것 같다.

추론이 내부에서 어떻게 돌아가는지 알면 성능 튜닝뿐 아니라 제품 판단도 달라진다고 생각한다. 어떤 모델을 쓸지, 어떤 하드웨어를 고를지, 사용자에게 무엇을 먼저 보여 줄지가 전부 이 구조 위에서 정해진다.

## 출처

- NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/) (HBM 3.35 TB/s, NVLink 900 GB/s, PCIe Gen5 128 GB/s)
- NVIDIA, [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) (L2 캐시 50MB)
- Gholami 등, [AI and Memory Wall (arXiv:2403.14123)](https://arxiv.org/abs/2403.14123)
- Zhong 등, [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670) (prefill과 decode의 병목 차이)
- Groq, [What is a Language Processing Unit?](https://groq.com/blog/the-groq-lpu-explained)
- [AI Accelerators for LLM Inference: Architecture Analysis and Scaling Strategies (arXiv:2506.00008)](https://arxiv.org/abs/2506.00008) (LPU SRAM 용량)
- Vaswani 등, [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)
- Inception Labs, [Mercury: Ultra-Fast Language Models Based on Diffusion (arXiv:2506.17298)](https://arxiv.org/abs/2506.17298)
- IEEE Spectrum, [Nvidia Groq 3](https://spectrum.ieee.org/nvidia-groq-3) (2026-03-16, Vera Rubin에서 decode 담당)
