---
title: NVIDIA는 왜 추론을 세 조각으로 쪼개기 시작했나
description: GPU 회사가 GPU가 아닌 칩을 랙째로 팔기 시작했다. 추론의 각 부분이 읽는 데이터가 다르기 때문이다. 시리즈 "LLM 추론, 연산에서 메모리로"의 3편.
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

> 시리즈 **LLM 추론, 연산에서 메모리로** (3/3)
> 1. [FLOPS를 2배로 올려도 ChatGPT는 빨라지지 않는다](/posts/llm-inference-1-memory-bound/)
> 2. [ChatGPT를 빠르게 만드는 다섯 가지 방법은 전부 같은 얘기다](/posts/llm-inference-2-five-ways/)
> 3. NVIDIA는 왜 추론을 세 조각으로 쪼개기 시작했나 (이 글)

## GPU 회사가 GPU가 아닌 랙을 내놓았다

2026년 3월 NVIDIA는 Groq 3 LPX라는 랙을 발표했다([The Register](https://www.theregister.com/2026/03/16/nvidia_lpx_groq_3/)). 안에 든 칩은 GPU가 아니다. 2편에서 다룬 Groq의 LPU, 그러니까 HBM 없이 칩 위 SRAM에 weight를 올리는 칩의 3세대다. 석 달 전인 2025년 크리스마스 이브에 NVIDIA는 Groq과 추론 기술 라이선스 계약을 맺었고, Groq 창업자 Jonathan Ross를 비롯한 핵심 인력이 NVIDIA로 옮겼다([Groq 발표](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale)). 금액은 두 회사 모두 밝히지 않았고, 200억 달러라는 숫자는 언론 보도에서 나왔다.

LPX만 있는 게 아니다. 그보다 반년 앞서 NVIDIA는 긴 입력을 처리하는 용도로 Rubin CPX라는 GPU를 따로 발표했고, 2026년 1월에는 KV Cache를 GPU 밖에 보관하는 저장 계층을 발표했다. GPU 한 종류로 학습도 추론도 다 하던 회사가 1년 사이 추론용 하드웨어를 세 갈래로 나눴다.

GPU 회사가 왜 GPU 아닌 칩을 팔까? 1편과 2편을 거치며 쌓아 온 질문으로 보면 답이 보인다.

## 이 글의 주장

> NVIDIA는 추론을 연산, 대역폭, 용량이라는 세 병목으로 나누고, 병목마다 다른 칩을 붙이고 있다.

입력을 한꺼번에 읽는 prefill은 연산이 큰 칩에, 모두가 같이 읽는 weight는 대역폭이 큰 칩에, 사용자마다 쌓이는 KV Cache는 용량이 큰 칩과 저장 계층에 맡긴다. 이 주장은 틀릴 수 있다. 분리한 제품이 시장에서 거의 쓰이지 않거나, 범용 GPU 한 종류가 토큰당 비용에서 분리한 구성을 계속 이긴다면 틀린 것이다. 글 끝에서 이 신호를 다시 본다.

## 먼저 알아야 할 것

### layer 하나 안의 두 부분

LLM은 같은 모양의 layer를 여러 겹 쌓은 구조다. Llama 3.1 70B는 80겹이다. layer 하나 안에는 두 부분이 차례로 있다.

- **attention**: 지금 토큰이 앞의 토큰들 중 어디를 참고할지 정하고, 그 내용을 섞는다
- **FFN**(Feed-Forward Network): attention이 내보낸 벡터 하나를 받아 큰 행렬 몇 개를 곱한다. 앞 토큰은 보지 않는다

decode에서 토큰 하나를 만들려면 이 [attention → FFN] 묶음을 80번 지나고, 마지막에 LM head라는 행렬과 softmax를 거쳐 다음 토큰이 정해진다. LM head는 tokenizer의 어휘 사전에 있는 토큰마다 점수를 하나씩 매기는 행렬이다. [Llama 3.1 70B](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json)는 어휘가 128,256개라서 8,192 × 128,256 크기이고, 이것도 모두가 같이 쓰는 weight다. 여기서 어휘의 단위는 단어보다 잘게 쪼갠 subword 토큰이다.

### weight와 K는 다르다

2편에서 attention은 토큰마다 Q, K, V를 쓰고, 과거 토큰의 K와 V를 저장해 둔 것이 KV Cache라고 했다. 이번 글을 준비하며 나는 이 부분에서 한 번 헷갈렸다. K는 사용자 입력에 따라 달라지는데, 그럼 K를 만드는 weight도 사용자마다 다른 것 아닌가?

배운 적이 있는데 가물가물했던 부분이다. 이름이 비슷한 두 가지를 섞었다.

| | 무엇인가 | 사용자마다 다른가 |
|---|---|---|
| W_K, W_V | 학습으로 정해진 행렬. 모델 파일에 들어 있는 weight | 같다 |
| K, V | 입력 벡터 x에 위 행렬을 곱한 결과. `K = x × W_K` | 다르다 |

레시피와 요리라고 생각하면 된다. 레시피(W_K)는 한 권을 모두가 같이 보고, 손님이 가져온 재료(x)가 다르니 완성된 요리(K)가 다르다. KV Cache에 쌓이는 건 요리 쪽이다.

이 구분이 이 글 전체를 끌고 간다.

- **weight**는 모두가 같이 쓴다. 크기는 고정이다. 70B 모델이면 FP16으로 140GB, 8bit로 70GB다
- **KV Cache**는 사용자마다 따로 쌓인다. 2편에서 계산한 대로 10만 토큰짜리 대화 하나가 약 33GB이고, 사용자 수와 대화 길이에 비례해 늘어난다. 15명이면 약 500GB로 weight보다 커진다

## 조각 1. prefill과 decode를 떼어 놓다

1편에서 본 대로 prefill과 decode는 병목이 다르다. prefill은 프롬프트 전체를 한꺼번에 계산하니 weight를 한 번 읽을 때 수천 토큰분을 계산한다. 연산이 병목이다. decode는 토큰을 하나씩 만드니 weight를 읽어 오는 속도가 병목이다.

그런데 한 GPU가 두 일을 같이 하면 서로 방해한다. 긴 프롬프트의 prefill이 끼어들면 그동안 다른 사용자들의 decode가 멈칫한다. [DistServe(Zhong 등, OSDI 2024)](https://arxiv.org/abs/2401.09670)는 prefill과 decode를 아예 다른 GPU에 맡겼다. 같은 지연 조건에서 처리할 수 있는 요청이 기존 시스템보다 최대 7.4배 많았다.

NVIDIA는 2025년 3월 이 방식을 자사 서빙 소프트웨어 [Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models)에 넣었다. Hopper GPU에서 Llama 70B의 처리량이 2배 넘게, GB200 NVL72에서 DeepSeek-R1이 처리하는 요청 수가 최대 30배까지 늘었다고 밝혔다.

여기까지는 같은 GPU를 역할만 나눠 쓴 것이다. 다음 단계는 역할마다 칩을 다르게 만드는 것이었다. 2025년 9월 발표한 [Rubin CPX](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference)는 긴 입력 처리 전용 GPU다. 연산은 NVFP4 기준 30 PFLOPS인데, 메모리는 HBM 대신 GDDR7 128GB를 쓴다. prefill은 연산이 병목이라 HBM만큼의 대역폭이 없어도 된다. 대역폭이 남아도는 자리에 비싼 메모리를 넣지 않은 것으로 나는 읽었다. 출시는 2026년 말 예정이다.

## 조각 2. decode 안을 한 번 더 쪼개다

### 나는 Groq 3가 decode를 맡는 줄 알았다

1편에서 나는 NVIDIA가 Groq 3 LPU에 "decode 단계를 맡기는 구성"을 내놓았다고 썼다. 2편에서도 같은 식으로 썼다. prefill은 GPU가, decode는 LPU가 하는 그림이었다.

그 그림은 여러 구성 중 하나였다. NVIDIA의 [기술 블로그](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/)(2026년 8월)는 GPU와 LPX를 묶는 방식을 세 가지 소개한다.

1. prefill, decode 분리: GPU가 prefill을 하고 턴마다 KV Cache를 넘기면, LPX가 decode 전체를 맡는다. 내가 생각한 그림이다
2. 추측 디코딩: LPX가 작은 초안 모델을 빠르게 돌리고, GPU의 큰 모델이 검증한다
3. **Attention-FFN Disaggregation(AFD)**: decode 안에서 attention은 GPU가, FFN은 LPX가 맡는다

세 번째가 이 글의 중심이다. 원문은 이렇다. "Vera Rubin NVL72 computes attention and holds the KV cache in DRAM, while Groq 3 LPX executes the FFN layers." 두 랙 사이에는 layer마다 중간 벡터만 오간다.

### 왜 attention이 GPU 쪽인가

처음 이 구성을 봤을 때 나는 attention이 KV를 만들고 FFN이 그걸 읽는다고 생각했다. 그렇지 않다. KV를 쓰는 곳도, 읽는 곳도 attention이다. 새 토큰의 Q를 지금까지 쌓인 모든 K와 비교하고 V를 섞는 일이 attention이고, KV를 저장해 두는 이유가 바로 이 일을 할 때 다시 계산하지 않기 위해서다. FFN은 벡터 하나와 weight만 본다.

그러면 두 부분이 읽는 데이터가 갈린다.

| decode 안의 부분 | 읽는 데이터 | 크기 | 맡는 칩 |
|---|---|---|---|
| attention | 사용자별 KV Cache | 사용자 수 × 대화 길이로 늘어남 | Vera Rubin GPU (HBM 288GB) |
| FFN | 공용 weight | 고정 | Groq 3 LPU (SRAM 500MB, 150 TB/s) |

반대로 놓으면 안 되는 이유는 숫자로 보인다. [IEEE Spectrum](https://spectrum.ieee.org/nvidia-groq-3)에 따르면 Groq 3 LPU 한 장의 SRAM은 500MB이고, LPX 랙 하나에는 256장, 합쳐서 128GB가 들어간다. 10만 토큰 사용자 한 명의 KV가 약 33GB이니 랙 하나를 통째로 써도 네 명이 안 들어간다. 반면 weight는 크기가 고정이라 칩 수백 장에 나눠 담을 수 있다. 8bit로 저장한 70B 모델이면 70GB로 랙 하나에 들어간다.

2편의 비유로 말하면 이렇다. 크고 사람마다 늘어나는 회의록(KV)은 넓은 창고(HBM)에, 작고 모두가 같이 보는 레시피(weight)는 도마 바로 옆 선반(SRAM)에 둔다. 여기서 HBM에 weight를 올려 두면 되지 않느냐는 의문이 들 수 있다. 나도 그랬다. 그런데 HBM은 계산 코어 옆의 별도 칩이라, 올라가 있어도 계산하려면 코어까지 옮겨 와야 한다. decode 한 단계마다 weight 전체를 HBM에서 코어까지 다시 실어 와야 한다. SRAM은 코어와 같은 칩 안에 있다. 대신 칩 면적을 많이 먹어서 한 장에 500MB가 한계라, 선반을 키우는 대신 선반 달린 칩을 수백 장 이어 붙였다.

### batching이 해결하지 못하는 것

여기서 반론이 하나 나올 만하다.

> "FFN은 batching으로 이미 효율이 좋잖아. weight를 한 번 읽고 여러 명이 같이 쓰니까. 굳이 FFN을 다른 칩으로 뺄 이유가 있나?"

batching이 올려 주는 건 처리량이다. 한 사람이 체감하는 속도는 그대로다. 제육볶음 하나를 볶는 데 30분이 걸리는 주방이라면, 한 명이 시켜도 100명이 시켜도 30분이다. 100인분을 한 번에 볶을 수 있게 될 뿐이다.

decode에서는 그 30분이 이렇게 정해진다.

```text
한 단계에 걸리는 시간 ≥ weight 크기 ÷ 메모리 대역폭
```

몇 명을 태우든 한 단계마다 weight 전체를 한 번은 읽어야 하기 때문이다. batching은 이 시간 동안 태우는 사람 수를 늘리고, 대역폭은 이 시간 자체를 줄인다. Rubin GPU의 HBM 대역폭은 22 TB/s, Groq 3의 SRAM 대역폭은 150 TB/s다. NVIDIA 블로그 제목에 들어간 단어도 처리량이 아니라 interactivity, 한 사람이 체감하는 속도다.

2편에서 쓴 계산을 두 칩에 해 보면 성격 차이가 드러난다.

```text
Rubin GPU:  50 PFLOPS (FP4) ÷  22 TB/s ≈ 2,300
Groq 3 LPU: 1.2 PFLOPS (FP8) ÷ 150 TB/s ≈ 8
```

→ 1바이트를 읽을 때 2,300번 계산해야 놀지 않는 칩과 8번이면 되는 칩이 같은 요청을 나눠 처리한다. 정밀도가 달라서(FP4와 FP8) 정확한 비교는 아니지만, 자릿수 차이는 정밀도 차이보다 훨씬 크다. GPU는 수백 명을 태워야 제 성능이 나오는 칩이고, LPU는 몇 명만 태워도 손이 쉬지 않는 칩이다.

성능 수치는 NVIDIA 블로그가 인용한 Artificial Analysis 측정이다. Gemma 4 31B 모델을 10만 토큰 context로 LPX에서 돌렸을 때 출력 속도 중앙값이 초당 3,431토큰이었고, 같은 블로그는 공개 엔드포인트 중 가장 빠른 곳이 초당 870토큰이라고 비교했다. 회사가 자사 블로그에 실은 수치라는 점은 감안해야 한다.

### 처음 나온 생각은 아니다

attention과 FFN을 나누는 생각 자체는 NVIDIA가 처음이 아니다. ByteDance Seed 연구진의 [MegaScale-Infer(2025년 4월)](https://arxiv.org/abs/2504.02263)는 MoE 모델 서빙에서 layer 안의 attention과 FFN을 서로 다른 GPU 묶음에 맡겨 GPU당 처리량을 최대 1.90배 올렸다. attention은 메모리가 넉넉한 GPU에, FFN은 연산이 싼 GPU에 올리는 식으로 GPU 종류를 섞는 것까지 제안했다. StepFun의 [Step-3(2025년 7월)](https://arxiv.org/abs/2507.19427)는 이 방식에 AFD라는 이름을 붙였다. 다만 둘 다 GPU 안에서 고른 것이다. NVIDIA는 FFN 쪽에 메모리 구조부터 다른 칩을 붙였다. 2편의 다섯 가지 방법처럼 소프트웨어에서 먼저 나온 생각이 하드웨어로 내려온 셈이다.

더 큰 모델은 어떨까. OpenAI나 Anthropic 같은 곳의 최신 모델은 크기를 공개하지 않는다. 공개된 모델 중에는 [DeepSeek-V3](https://arxiv.org/abs/2412.19437)가 파라미터 671B로, 70B의 열 배 가까이 된다. 이런 모델은 원래부터 GPU 수십 장에 나눠 올린다. 행렬 하나를 여러 GPU에 쪼개 담기도 하고, layer를 서버마다 나누기도 하고, MoE라면 expert를 GPU마다 나눠 둔다. 그러면 GPU마다 자기 몫만 읽으니 여러 장의 대역폭이 합쳐지는 효과도 난다. Vera Rubin NVL72 랙 하나에는 GPU 72장, HBM으로 약 20TB(72 × 288GB)가 들어간다. 그리고 DeepSeek-V3는 MoE 구조라 토큰 하나를 만들 때 실제로 쓰는 파라미터는 37B뿐이다. NVIDIA 블로그는 GPT-OSS를 2조 파라미터로 키웠을 때 GPU와 LPX 조합에서 어떻게 될지 예측한 그래프를 실었지만, 실측은 아니다.

## 조각 3. KV를 GPU 밖으로

남은 조각은 용량이다. AFD로 GPU의 HBM을 KV 전용으로 쓰더라도, 사용자가 늘고 대화가 길어지면 결국 모자란다. 특히 agent는 도구 결과를 붙여 가며 여러 턴을 이어 간다. 2편의 Prefix Cache처럼 앞 턴의 KV를 다시 쓸 일이 많아지는데, 그걸 GPU에 계속 두면 지금 생성 중인 사용자들의 자리를 뺏는다.

NVIDIA는 2026년 1월 [BlueField-4 기반 저장 계층](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai)을 발표하며 이렇게 썼다. "A KV cache cannot be stored on GPUs long term, as this would create a bottleneck for real-time inference in multi-agent systems." 당장 안 쓰는 KV를 GPU 밖의 저장 계층에 내려 두었다가 다음 턴에 다시 꺼내 쓰는 구조다. 기존 저장 장치와 비교해 초당 토큰 수와 전력 효율이 최대 5배라고 밝혔고, 출시는 2026년 하반기다. 소프트웨어 쪽에서는 Dynamo의 KV Cache 관리자가 오래된 KV를 CPU 메모리, 로컬 저장 장치, 네트워크 저장소로 내려 둔다.

세 조각을 한 장에 모으면 이렇다.

![요청 하나가 prefill 칩, attention을 맡는 GPU와 FFN을 맡는 LPX 사이의 왕복, GPU 밖 KV 저장 계층을 거치는 흐름](/images/posts/llm-inference-3-three-pieces/three-pieces.svg)

NVIDIA가 이 네 가지를 한 구성으로 발표한 것은 아니다. LPX 블로그가 설명한 구성에서는 prefill을 Vera Rubin NVL72가 맡는다. 각 발표에서 맡는 자리를 한 장에 모은 그림으로 보면 된다.

## 다른 쪽에 건 방법들

| 접근 | 병목을 어떻게 봤나 | 해법 | 대가 |
|---|---|---|---|
| 범용 GPU 한 종류 | 모든 단계를 같은 칩으로 감당할 수 있다 | batching, 2편의 기법들 | 단계마다 연산이나 대역폭 중 하나가 논다. 한 사람의 속도는 HBM 대역폭이 정한 천장 그대로 |
| 같은 GPU를 역할별로 분리 (DistServe, Dynamo, MegaScale-Infer) | 성격이 다른 단계가 서로 방해한다 | 단계마다 GPU 묶음을 따로 둔다 | KV나 중간 벡터를 GPU 사이로 옮겨야 한다. 묶음 사이 비율을 맞춰야 한다 |
| 칩 종류까지 분리 (Rubin CPX, Groq 3 LPX, BlueField-4) | 단계마다 모자란 자원이 다르다 | 병목마다 맞는 메모리를 단 칩을 붙인다 | AFD는 토큰 하나에 layer 수만큼 랙 사이를 왕복한다. 칩 종류가 늘어난 만큼 운영이 복잡하다 |
| 칩 하나로 전부 (Groq 단독) | HBM에서 옮겨 오는 길 자체가 느리다 | weight를 SRAM에 상주시킨다 | 칩 수백 장. KV를 둘 자리가 마땅치 않다 |

→ 위에서 아래로 갈수록 나누는 단위가 잘아진다. 마지막 줄의 약점(KV 둘 곳)을 셋째 줄이 GPU로 메웠고, GPU의 약점(한 사람의 속도)을 LPU로 메웠다.

## 왜 지금인가

이유는 두 가지라고 생각한다. 첫째는 오래된 추세다. [AI and Memory Wall(Gholami 등, 2024)](https://arxiv.org/abs/2403.14123)에 따르면 서버 하드웨어의 최대 연산 성능은 2년마다 3.0배 늘었지만 DRAM 대역폭은 1.6배, 칩 사이 연결 대역폭은 1.4배씩만 늘었다. 연산과 메모리의 격차가 벌어질수록, 같은 칩 하나로 둘 다 잘하기는 어려워진다.

둘째는 쓰임새의 변화다. 2편에서 인용한 [OpenRouter의 100조 토큰 분석](https://arxiv.org/abs/2601.10088)에서 요청 하나의 평균 입력은 13개월 동안 약 1,500토큰에서 6,000토큰 이상으로 늘었다. 입력이 길어지면 prefill이 무거워지고 KV가 커진다. 한 쪽에선 연산이, 다른 쪽에선 용량이 동시에 모자라게 된 것이다. 칩 하나로 두 쪽을 다 맞추기 어려워져서 나눠 맞추는 쪽으로 갔다고 본다.

## 다음은

나는 칩을 나누는 흐름이 메모리 쪽에서 더 이어질 것이라고 생각한다. 후보 하나가 HBF(High Bandwidth Flash)다. NAND 플래시를 HBM처럼 쌓아 올린 메모리로, Sandisk와 SK hynix가 2025년 8월 표준화를 [발표](https://www.businesswire.com/news/home/20250806259749/en)했다. 비슷한 비용으로 HBM보다 용량 8~16배를 목표로 하고, 첫 샘플은 2026년 하반기다. 플래시는 쓰기를 반복하면 닳지만, 추론에서 weight는 한 번 올려 두고 읽기만 한다. 이 글의 구분대로라면 HBF는 weight 쪽에 맞는 메모리이고, 계속 새로 쓰는 KV에는 덜 맞을 듯하다.

이 전망이 틀렸다고 볼 신호는 두 가지다.

1. LPX나 CPX 같은 분리 제품이 실제 데이터센터에서 거의 쓰이지 않고, 범용 GPU 한 종류가 토큰당 비용에서 계속 이기는 경우
2. 2편 끝에서 말한 대로 State Space Model처럼 대화가 길어져도 저장할 상태가 커지지 않는 구조가 주류가 되는 경우. KV라는 조각 하나가 작아지면 용량 칩의 자리도 줄어든다

## 내 생각

돌아보면 1편과 2편에서 내린 결론, 데이터를 어디 두고 얼마나 덜 옮기느냐가 추론 속도를 정한다는 문장이 이번에는 칩을 나누는 단위가 됐다. 2편의 GQA와 PagedAttention이 같은 칩 안에서 데이터 자리를 바꿨다면, 이번에는 데이터 종류마다 칩을 따로 만들었다.

> "결국 NVIDIA가 경쟁사를 사들인 것 아닌가. 기술 이야기로 포장한 것 아니야?"

계약의 동기는 내가 알 수 없다. 다만 기술적으로는 두 칩이 서로를 필요로 한다는 점은 확인할 수 있다. Groq 칩만으로는 KV를 둘 자리가 없고(2편에서 Groq이 KV를 어디 두는지 공개 자료를 찾지 못했다), GPU만으로는 한 사람의 속도가 HBM 대역폭에 묶인다. AFD는 서로의 약점이 정확히 맞물리는 자리에서 나왔다.

> "layer마다 랙 사이를 왕복하면 오히려 느려지지 않나?"

오가는 건 토큰 하나당 벡터 하나다. 70B 모델이면 숫자 8,192개, FP16으로 약 16KB다. layer 하나의 weight가 FP16으로 약 1.75GB(140GB ÷ 80)인 것과 비교하면 작다. 그래도 80번 왕복의 지연이 공짜는 아니다. 앞의 3,431토큰은 NVIDIA가 보여 준 결과이고, 다른 모델과 구성에서 같은 이득이 나는지는 앞으로 나올 측정을 봐야 한다.

마지막으로 Groq 쪽 이야기를 덧붙인다. Groq은 칩을 만들었지만, 공개적으로 내건 사업은 그 칩으로 추론을 돌려 파는 것이다. [GroqCloud](https://console.groq.com/docs/models)에서 개발자가 모델을 호출하면 처리한 토큰 수만큼 요금을 받는다. ex) gpt-oss-120b는 100만 토큰에 입력 0.15달러, 출력 0.60달러다(2026-10 기준). 24시간에서 7일 안에만 끝나면 되는 [batch 요청](https://console.groq.com/docs/batch)은 50% 싸다. 다른 축은 국가 단위 계약이다. 2025년 2월 사우디아라비아가 Groq의 LPU 추론 인프라 확대에 15억 달러를 약속했고, Groq은 담맘 데이터센터를 운영하며 현지 AI 기업 HUMAIN의 추론 공급자가 됐다([Groq](https://groq.com/blog/groq-solidifies-status-as-emerging-hyperscaler-with-new-global-deployment)). 매출 규모는 공개되지 않았다. 토큰 단위로 파는 사업에서 이익을 가르는 건 토큰당 원가이고, 이 글의 반증 신호를 토큰당 비용으로 잡은 것도 같은 이유다.

2025년 말 칩 기술은 NVIDIA에 라이선스됐고, 창업자를 비롯한 설계자들은 NVIDIA로 옮겼다. 남은 Groq은 2026년 8월 3억 5천만 달러를 투자받았는데, [보도자료](https://groq.com/newsroom/groq-closes-usd350-million-series-a-building-the-world-s-leading-ai-inference-cloud)의 제목에는 "세계 최고의 AI 추론 클라우드를 만든다"는 문구가 붙었다. 데이터센터 13곳, 개발자 600만 명, 전력 54MW에서 2027년 200MW 이상으로 늘린다는 숫자도 함께 실렸다. 칩을 몇 개 출하하는지보다 데이터센터 전력을 얼마나 늘리는지로 성장을 설명한다는 점에서, 칩 회사보다 클라우드 운영자에 가까운 발표다. 더 눈에 띄는 건 투자금의 용도다. 보도자료는 이 돈이 "NVIDIA accelerated computing" 클러스터를 쓰려는 고객을 지원하는 데 들어간다고 적었다. LPU는 "대규모로 운영해 본 경험"이라는 한 줄로만 나온다.

칩을 데이터 종류에 맞춰 나누는 설계는 칩 회사 안으로 들어갔고, 그 설계를 처음 밀어붙였던 회사는 칩을 운영하는 쪽으로 자리를 옮겼다. 추론의 가치가 칩을 파는 쪽과 추론을 돌려 주는 쪽 중 어디에 더 쌓일지는 아직 모르겠다. 다만 HBM을 만드는 회사, HBF를 준비하는 회사, 추론 전용 칩을 만드는 회사들이 동시에 주목받는 지금의 흐름은 이 시리즈의 결론과 같은 방향을 가리킨다고 생각한다. 그 산업과 공급망 이야기는 따로 써 보려 한다.

시리즈는 여기서 마무리하지만 2편에서 미뤄 둔 질문이 남아 있다. 양자화는 옮길 바이트를 줄이는 방법인가, 계산을 빠르게 하는 방법인가. 다음 글에서 다룬다.

## 출처

- NVIDIA, [How NVIDIA Groq 3 LPX Unlocks Ultrafast Interactivity at Long Context on NVIDIA Vera Rubin](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) (2026-08, AFD, 세 가지 구성, Gemma 4 31B 측정)
- IEEE Spectrum, [Nvidia Groq 3](https://spectrum.ieee.org/nvidia-groq-3) (Groq 3 LPU 500MB SRAM, 1.2 PFLOPS 8bit, 150 TB/s, Rubin GPU 288GB HBM, 50 PFLOPS 4bit, 22 TB/s)
- The Register, [Nvidia slaps $20B Groq tech into massive new LPX racks](https://www.theregister.com/2026/03/16/nvidia_lpx_groq_3/) (2026-03-16)
- Groq, [Groq and NVIDIA Enter Non-Exclusive Inference Technology Licensing Agreement](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale) (2025-12)
- Groq, [GroqCloud Models](https://console.groq.com/docs/models), [Batch API](https://console.groq.com/docs/batch) (토큰당 요금, batch 50% 할인, 2026-10-19 확인)
- Groq, [Groq Solidifies Status as Emerging Hyperscaler with New Global Deployment](https://groq.com/blog/groq-solidifies-status-as-emerging-hyperscaler-with-new-global-deployment) (2025-05-27, 사우디 15억 달러, HUMAIN)
- Groq, [Groq closes $350 million Series A](https://groq.com/newsroom/groq-closes-usd350-million-series-a-building-the-world-s-leading-ai-inference-cloud) (2026-08-17)
- NVIDIA, [NVIDIA Unveils Rubin CPX](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) (2025-09)
- NVIDIA, [BlueField-4 AI-native storage infrastructure](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai) (2026-01)
- NVIDIA, [Introducing NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models) (2025-03)
- Zhong 등, [DistServe (arXiv:2401.09670, OSDI 2024)](https://arxiv.org/abs/2401.09670)
- Zhu 등, [MegaScale-Infer (arXiv:2504.02263)](https://arxiv.org/abs/2504.02263) (ByteDance Seed, GPU당 처리량 최대 1.90배)
- StepFun, [Step-3 is Large yet Affordable (arXiv:2507.19427)](https://arxiv.org/abs/2507.19427)
- DeepSeek-AI, [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) (671B, 토큰당 37B)
- Meta Llama 3.1 70B [config.json](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json) (NousResearch 미러, 80 layer, hidden 8,192, 어휘 128,256)
- Gholami 등, [AI and Memory Wall (arXiv:2403.14123)](https://arxiv.org/abs/2403.14123)
- a16z, OpenRouter, [State of AI: An Empirical 100 Trillion Token Study (arXiv:2601.10088)](https://arxiv.org/abs/2601.10088)
- Sandisk, [Sandisk and SK hynix HBF 표준화 발표](https://www.businesswire.com/news/home/20250806259749/en) (2025-08)
