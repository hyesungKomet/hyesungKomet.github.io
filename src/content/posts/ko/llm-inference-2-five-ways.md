---
title: ChatGPT를 빠르게 만드는 다섯 가지 방법은 전부 같은 얘기다
description: batching, GQA, PagedAttention, Prefix Cache, Groq. 이 중 계산을 빠르게 하는 방법은 하나도 없다. 시리즈 "LLM 추론, 연산에서 메모리로"의 2편.
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

> 시리즈 **LLM 추론, 연산에서 메모리로** (2/3)
> 1. [FLOPS를 2배로 올려도 ChatGPT는 빨라지지 않는다](/posts/llm-inference-1-memory-bound/)
> 2. ChatGPT를 빠르게 만드는 다섯 가지 방법은 전부 같은 얘기다 (이 글)
> 3. NVIDIA는 왜 추론을 세 조각으로 쪼개기 시작했나 (준비 중)

## 요리사는 몇 인분을 만들어야 쉬지 않을까

[1편](/posts/llm-inference-1-memory-bound/)의 결론은 이랬다. LLM이 답을 한 토큰씩 쓰는 decode 단계에서는 토큰 하나를 만들 때마다 weight 전체를 HBM에서 연산 유닛으로 읽어 와야 한다. 70B 모델이면 140GB다. 재료를 실어 오는 트럭(메모리 대역폭)은 쉬지 않고 달리는데, 요리사(연산 유닛)는 트럭이 한 번 올 때마다 1인분만 만들고 다음 트럭을 기다린다.

그렇다면 요리사가 쉬지 않으려면 트럭 한 번에 몇 인분을 만들어야 할까? 1편에서 쓴 H100으로 계산해 보면 이렇다.

```text
H100 연산 약 1,000 TFLOPS (FP16, sparsity 없이) ÷ 대역폭 3.35 TB/s ≈ 300
→ weight를 1바이트 읽을 때 약 300번 계산해야 연산 유닛이 놀지 않는다
```

사용자 한 명분의 decode는 weight 1바이트당 계산이 약 1번이다. 파라미터 하나가 2바이트(FP16)이고, 토큰 하나를 만들 때 파라미터 하나마다 곱셈과 덧셈을 한 번씩 하기 때문이다. 그러니 요리사를 쉬지 않게 하려면 **트럭 한 번에 약 300명분**을 만들어야 한다.

그런데 GPU 8장에 70B 모델을 올리고, 사용자마다 10만 토큰짜리 대화를 한다고 하면 실제로 동시에 받을 수 있는 사람은 **15명**쯤이다. 예전 방식대로 만든 모델이었다면 **1명**이었다. 300명이 필요한데 1명밖에 못 태우던 상황을 업계는 어떻게 바꿔 왔을까?

## 다섯 가지 방법, 하나의 문장

이 글에서 다루는 방법은 다섯 가지다. batching, GQA, PagedAttention, Prefix Cache, 그리고 Groq. 이름도 다르고 나온 곳도 다르다. 하지만 이 글의 주장은 하나다.

> 다섯 가지 방법 중 계산을 빠르게 하는 것은 하나도 없다. 전부 데이터를 어디 두고 얼마나 덜 옮기느냐를 바꾼다.

이 문장에 이르기까지 나는 두 번 틀렸다. 처음에는 공통점을 "대역폭을 넓혀서 로드를 줄이는 것"이라고 생각했다. 그런데 다섯 가지 중 대역폭을 넓힌 방법은 없다. 모두 같은 대역폭으로 덜 옮기게 할 뿐이다. 그다음에는 "많은 사람이 쓸 때 자원을 최대한 쓰면서 속도를 최적으로 내는 것"이라고 했다. 틀린 말은 아니지만 데이터베이스 인덱스에도, 공장 생산 라인에도 똑같이 맞는 말이다. 무엇에나 맞는 문장은 아무것도 설명하지 못한다.

그래서 방법마다 **무엇을 덜 하게 했는지**를 한 줄씩 적어 봤다.

| 방법 | 덜 하게 한 것 |
|---|---|
| batching | weight를 사람마다 따로 옮기기 |
| GQA | KV Cache 저장하기, 읽기 |
| PagedAttention | 쓰지도 않을 자리 잡아 두기 |
| Prefix Cache | 같은 앞부분 다시 계산하기 (대신 저장을 더 한다) |
| Groq | HBM에서 연산 유닛까지 옮기기 |

→ 남는 동사는 **두다**와 **옮기다** 두 개뿐이다. 계산을 빠르게 한 칸은 없다. 이 주장은 틀릴 수 있다. 다섯 가지 중 하나라도 실은 계산이 빨라져서 이긴 것이라면 무너진다. 글 끝에서 반론 두 개를 세워 확인한다.

## 먼저 알아야 할 것

1편에서 다룬 KV Cache를 다시 짚고 간다. attention은 토큰마다 Q(Query), K(Key), V(Value) 세 가지 값을 쓴다. 도서관에 비유하면 Q는 지금 토큰이 던지는 검색어, K는 앞 토큰마다 붙은 색인 카드, V는 그 토큰의 실제 내용이다. 과거 토큰의 K와 V는 이후 모든 토큰이 다시 쓰기 때문에 저장해 둔다. 이것이 KV Cache다.

이 글에서 가장 중요한 차이는 하나다.

- **weight**는 모든 사용자가 같이 쓴다. 누가 질문하든 같은 모델이다
- **KV Cache**는 사용자마다 따로다. 내 대화의 K, V는 내 대화에만 쓸모가 있다

## 방법 1. batching: 트럭 한 번에 여러 명분

가장 먼저 떠올릴 수 있는 방법은 여러 사용자의 요청을 묶어서 처리하는 것이다. weight를 한 번 읽어 올 때 한 명분이 아니라 64명분을 계산하면, 읽는 양은 그대로인데 계산은 64배가 된다. 1바이트당 계산 횟수가 300에 가까워질수록 요리사가 덜 논다.

나는 처음에 batching이 대역폭을 늘리는 방법이라고 생각했다. 아니었다. 트럭의 크기도 속도도 그대로다. 같은 트럭 한 번으로 더 많은 사람분을 만들게 했을 뿐이다. 주방은 원래 100명분을 만들 수 있는 주방이었고, 1명분만 만들고 있던 게 낭비였다.

실제 서빙 시스템은 요청 단위가 아니라 토큰 한 step 단위로 묶는다. 먼저 끝난 요청은 빼고 새 요청을 바로 끼워 넣는 방식인데, [Orca(Yu 등, OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)가 iteration-level scheduling이라는 이름으로 제안했다.

### 그런데 KV가 자리를 먹는다

batching에는 걸림돌이 있다. weight는 같이 쓰지만 KV Cache는 사람마다 따로라서, 태울 사람이 늘수록 KV가 차지하는 메모리도 늘어난다.

[Llama 3.1 70B의 설정 파일](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json)로 토큰 하나의 KV 크기를 계산해 보자. 이 모델은 층이 80개이고, K와 V를 담는 head가 8개, head 하나의 크기가 128이다.

```text
80층 × 2 (K와 V) × 8 head × 128 × 2바이트 (FP16) = 327,680바이트 ≈ 0.33MB
```

도서관으로 치면 80층짜리 건물에서 층마다 카드 2장(K, V)을 쓰고, 카드 한 장에 숫자 1,024개(8 × 128)가 적혀 있고, 숫자 하나가 2바이트인 셈이다. 토큰 하나에 0.33MB면 작아 보이지만 대화가 10만 토큰이면 사용자 한 명이 약 33GB를 차지한다.

H100 8장의 메모리는 640GB다. weight 140GB를 빼면 약 500GB가 남고, 33GB짜리 사용자는 15명 들어간다.

![H100 8장의 메모리에 weight와 사용자별 KV Cache를 담았을 때 MHA는 1명, GQA는 15명이 들어가는 비교](/images/posts/llm-inference-2-five-ways/kv-capacity.svg)

→ 목표는 300명인데 15명이다. 남은 네 가지 방법 중 셋은 이 간극을 메우는 방법이다.

## 방법 2. GQA: K와 V를 나눠 쓴다

위 그림의 윗줄은 같은 모델을 예전 방식으로 만들었을 때다. Llama 3.1 70B는 Q head가 64개인데 K, V head는 8개뿐이다. 원래 Transformer의 Multi-Head Attention(MHA)이라면 K, V head도 64개였을 것이다. 그러면 토큰당 KV가 8배인 약 2.6MB이고, 10만 토큰이면 약 262GB다. 500GB에 한 명이 겨우 들어간다.

64를 8로 줄인 방법이 GQA(Grouped-Query Attention)다. Q head 8개가 한 조가 되어 K, V 한 쌍을 같이 쓴다. 극단적으로 모든 Q head가 K, V 한 쌍을 같이 쓰는 방식은 MQA(Multi-Query Attention)라고 부른다.

![MHA, GQA, MQA에서 Q head와 K, V head가 연결되는 방식 비교](/images/posts/llm-inference-2-five-ways/gqa-groups.svg)

### "관점이 줄어드는 것 아닌가"

처음 GQA를 봤을 때 든 의문이다. multi-head는 head마다 다른 관점으로 문장을 본다고 배웠는데, K와 V를 나눠 쓰면 관점이 64개에서 8개로 줄어드는 것 아닌가?

절반만 맞는 걱정이다. **Q head는 64개 그대로다.** 64가지 검색어는 여전히 던질 수 있다. 줄어드는 건 그 검색어를 받아 주는 색인 카드와 내용의 종류다. 그래서 품질이 조금은 떨어질 수 있는데, 얼마나 떨어지는지가 관건이다.

[GQA 논문(Ainslie 등, 2023)](https://arxiv.org/abs/2305.13245)의 실험(T5-XXL)에서 K, V head를 8개로 줄인 GQA의 평균 점수는 47.1로, MHA의 47.2와 거의 같았다. 샘플 하나를 처리하는 시간은 0.28초로 MQA의 0.24초에 가까웠다. 반면 MQA는 품질이 떨어졌고, 처음부터 MQA로 학습하면 loss가 튀다가 발산하기도 했다. 이미 학습된 MHA 모델을 원래 학습 연산량의 5%만 더 들여 GQA로 바꾸는 방법(uptraining)도 같이 제안돼서, 처음부터 다시 학습하지 않아도 된다.

### 왜 1개까지 줄이지 않고 8개에서 멈췄을까

품질 말고 이유가 하나 더 있다. 70B 모델은 GPU 한 장에 안 들어가서 여러 장에 나눠 올린다. 각 GPU는 자기가 맡은 Q head의 attention을 혼자 계산하니, 필요한 K, V가 자기 메모리에 있어야 한다.

나는 여기서 또 헷갈렸다. MHA의 K, V head 64개도 GPU 8장에 하나씩 복사되니 실제로는 64 × 8개를 쓰는 줄 알았다. 그렇지 않다. 64개는 8장에 **나눠서** 장마다 8개씩 맡는다. 복사는 K, V head가 GPU 수보다 적을 때만 생긴다.

| 방식 | K, V head | GPU 8장일 때 한 장이 드는 것 | 8장 합계 |
|---|---:|---|---:|
| MHA | 64 | 8개 (나눔) | 64 |
| GQA | 8 | 1개 (나눔) | 8 |
| MQA | 1 | 1개 (복사) | 8 |

→ GPU 8장에서는 GQA와 MQA의 메모리 사용량이 같다. MQA는 같은 사본 8벌을, GQA는 서로 다른 8벌을 든다. GQA 논문도 큰 모델을 나눠 올리면 MQA의 K, V head 하나가 그 수만큼 복사된다고 지적한다. Llama 3가 이 이유로 8을 골랐다고 Meta가 밝힌 자료는 찾지 못했다. 다만 8장에 나눠 올리는 한 8 밑으로 내려갈 이유는 약하다.

## 방법 3. PagedAttention: 자리를 미리 잡지 않는다

GQA로 사람 한 명의 KV를 줄였다. 그런데 앞의 15명 계산에는 숨은 가정이 있다. 모든 사용자가 정확히 10만 토큰을 쓴다는 가정이다. 실제로는 세 마디만 하고 끝나는 사람도, 길게 이어 가는 사람도 있고, 답이 얼마나 길어질지는 생성이 끝나기 전까지 모른다.

예전 서빙 시스템은 요청마다 최대 길이만큼 연속된 메모리를 미리 잡아 뒀다. [vLLM 논문(Kwon 등, 2023)](https://arxiv.org/abs/2309.06180)이 측정해 보니 기존 시스템에서 KV Cache용으로 잡은 메모리 중 실제 토큰이 들어 있던 비율은 **20.4~38.2%**였다. 논문은 낭비를 세 가지로 나눈다.

1. 예약해 둔 자리: 앞으로 생성될 토큰 몫으로 미리 잡은 공간. 요청이 끝날 때까지 다른 요청이 못 쓴다
2. 내부 단편화: 최대 길이로 잡았는데 짧게 끝나서 남은 공간
3. 외부 단편화: 요청마다 덩어리 크기가 달라서 생기는 틈. 합치면 충분한데 연속된 공간이 없어서 못 쓴다

처음에 나는 이걸 정적 할당 문제로 봤다. 적게 써도 10만 토큰만큼 비워 둬야 하고, 더 쓰려고 하면 메모리가 부족해진다. 그러니 동적으로 할당하면 되지 않을까? 그런데 동적 할당이라도 KV를 연속된 한 덩어리로 두는 한, 대화가 길어질 때 바로 뒤 공간이 다른 요청 것이면 늘릴 수가 없다. 3번 외부 단편화도 그대로 남는다.

### 운영체제 시험에 나왔던 답

솔직히 고백하면 나는 2학년 때 운영체제 시험을 거의 준비 없이 봤다. 지금 기억나는 건 교착상태를 화장실 칸에 빗댄 비유와 라운드 로빈 스케줄링 정도다. 그런데 이 문제의 답이 바로 그 운영체제 수업에 있었다.

운영체제는 프로세스에게 연속된 메모리를 주는 것처럼 보이게 하면서, 실제 물리 메모리는 고정 크기 페이지로 쪼개 흩어 놓는다. 둘을 잇는 것이 page table이다. PagedAttention은 이걸 KV Cache에 그대로 가져왔다.

- KV Cache를 고정 크기 block으로 자른다. 기본은 block 하나에 토큰 16개
- 요청마다 block table을 둔다. "내 0번 block은 물리 메모리 5번 칸, 1번 block은 1번 칸" 같은 대응표다
- block이 다 차면 그때 빈 칸 하나를 새로 받는다. 크기가 모두 같으니 아무 빈 칸이나 쓸 수 있다

![두 사용자의 block table이 GPU 메모리의 흩어진 물리 block을 가리키고, 공통 앞부분은 같은 block을 공유하는 모습](/images/posts/llm-inference-2-five-ways/paged-blocks.svg)

이렇게 하면 낭비는 요청마다 마지막 block 하나 안으로 줄어든다. vLLM 팀은 [소개 글](https://vllm.ai/blog/2023-06-20-vllm)에서 기존 시스템이 메모리의 60~80%를 낭비했고 PagedAttention은 낭비가 4% 미만이라고 밝혔다. 같은 GPU에 훨씬 많은 요청을 태울 수 있게 되면서 처리량이 같은 지연 시간 기준 2~4배가 됐다.

참고로 나는 vLLM을 ollama 같은 도구라고 생각했었다. 둘 다 모델을 돌리는 추론 엔진이지만 ollama는 내 컴퓨터에서 한 사람이 쓰는 데 맞춰져 있고, vLLM은 서버에서 많은 요청을 동시에 받는 데 맞춰져 있다. 이 글의 주제인 "몇 명을 태우나"는 vLLM 쪽 문제다.

## 방법 4. Prefix Cache: 같은 앞부분은 한 벌만

block table이 포인터라는 점에서 하나가 더 나온다. 회사 서비스라면 수많은 사용자가 똑같은 system prompt로 대화를 시작한다. 그 앞부분의 KV는 누구에게나 같다. 그렇다면 한 번만 계산해서 한 벌만 저장하고, 모든 사용자의 block table이 같은 물리 block을 가리키게 하면 된다. 위 그림에서 0~2번 block이 그렇다.

예전에 나는 이걸 weight를 캐싱하는 것으로 착각했다. Prefix Cache가 재사용하는 건 **프롬프트 앞부분의 KV**이고, weight와는 상관이 없다. 그리고 앞부분이 완전히 같아야 한다. 어떤 토큰의 K, V는 그 앞의 모든 토큰에 영향을 받기 때문에, 중간에 한 글자라도 다르면 그 뒤로는 같은 문장이라도 재사용할 수 없다.

### 반쯤 찬 마지막 block

까다로운 곳은 경계다. system prompt가 5,000토큰이면 block 16토큰 기준으로 이렇게 된다.

```text
5,000 ÷ 16 = 가득 찬 block 312개 + 토큰 8개만 든 block 1개
```

두 사용자가 대화를 이어 가면 둘 다 마지막 block의 빈 8칸에 자기 토큰을 쓰려고 한다. 같은 칸에 서로 다른 내용을 쓸 수는 없다. 그래서 먼저 쓰려는 사람이 그 block만 복사해서 자기 사본에 이어 쓴다. 남은 사람은 원본을 혼자 쓰게 되니 그냥 쓰면 된다. 둘이 같이 보던 공책의 반쯤 찬 마지막 장만 복사해 가는 것이다. 운영체제에서 copy-on-write라고 부르는 방식이고, vLLM 논문도 이 이름을 쓴다. 앞의 312개는 계속 공유한다.

Prefix Cache는 다섯 가지 중 유일하게 **prefill**을 줄인다. 다시 계산하지 않으니 첫 글자가 나오는 시간(TTFT)이 짧아진다. 대신 저장을 더 한다. API에서는 이게 요금으로 드러난다. [Claude의 프롬프트 캐싱](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)은 캐시된 입력을 읽을 때 기본 입력 요금의 0.1배를 받는다.

## 다른 쪽에 건 방법: Groq

여기까지 네 가지는 모두 같은 쪽에 걸었다. HBM에서 weight를 읽어 오는 구조는 그대로 두고, 한 번 읽을 때 더 많은 사람을 태우는 쪽이다. 다섯 번째 방법은 반대쪽에 걸었다.

[Groq](https://groq.com/blog/the-groq-lpu-explained)의 LPU는 weight를 칩 위의 SRAM에 통째로 올리고, HBM은 쓰지 않는다. 처음에 나는 이걸 GPU의 L2 캐시처럼 weight를 더 빨리 로드해 오는 것이라고 이해했다. 같은 SRAM이라는 점은 맞지만 쓰는 방식이 다르다.

| | GPU의 L2 캐시 | Groq의 SRAM |
|---|---|---|
| 역할 | 큰 HBM 앞의 임시 보관함 | weight를 담는 주 메모리 |
| 찾는 데이터가 없으면 | HBM에서 다시 가져온다 | 그럴 일이 없다. 뒤에 HBM이 없다 |
| 무엇을 둘지 | 하드웨어가 실행 중에 정한다 | 컴파일러가 미리 정한다 |

주방으로 치면 GPU는 창고에서 트럭으로 재료를 실어 오고 주방 안에 작은 선반을 둔 구조다. Groq은 창고를 통째로 주방 안에 넣어서 트럭을 없앴다.

이 차이는 앞에서 한 계산으로 보면 선명해진다. [GroqChip 제품 자료](https://groq.com/wp-content/uploads/2024/08/GroqChip™-Processor-Product-Brief-v1.7.pdf)에 따르면 1세대 칩은 연산이 188 TFLOPS(FP16), SRAM 대역폭이 80 TB/s다.

```text
H100:     약 1,000 TFLOPS ÷ 3.35 TB/s ≈ 300
GroqChip:     188 TFLOPS ÷   80 TB/s ≈ 2.4
```

→ GPU는 1바이트당 약 300번 계산해야 연산이 놀지 않으니 수백 명을 태우려고 애쓴다. Groq은 약 2번이면 된다. 트럭이 워낙 빨라서 요리사가 1~2인분만 만들어도 손이 쉬지 않는다.

여기서 볼 숫자는 188이다. Groq 칩의 연산 성능은 H100의 5분의 1도 안 된다. **계산이 더 느린 칩이 한 사람에게는 더 빠르게 답한다.** 이 글의 주장을 이보다 잘 보여 주는 숫자는 없다고 생각한다.

물론 대가가 있다. 칩 하나의 SRAM은 230MB다. 140GB짜리 weight를 담으려면 단순 계산으로 칩 600장 정도가 필요하다. H100이면 2장에 들어가는 양이다. 사용자 한 명의 KV Cache(10만 토큰이면 33GB)도 칩 하나 SRAM의 100배가 넘는다. Groq이 KV를 실제로 어디에 두고 몇 명을 묶는지는 공개 자료에서 찾지 못했다.

### 두 방향, 그리고 하나 더

KV를 줄이는 방식도 GQA만 있는 게 아니다. DeepSeek은 K, V head를 줄이는 대신 K, V를 작은 latent vector로 압축해서 저장하는 MLA(Multi-head Latent Attention)를 썼다. [DeepSeek-V2 논문](https://arxiv.org/abs/2405.04434)에 따르면 이전 모델인 DeepSeek 67B보다 KV Cache를 93.3% 줄였고, 최대 생성 처리량은 5.76배가 됐다.

| 접근 | 병목을 어떻게 봤나 | 해법 | 대가 |
|---|---|---|---|
| 많이 태우기 (batching, GQA, PagedAttention, Prefix Cache) | 한 번 읽을 때 한 명분만 계산한다 | 한 번 읽고 여러 명분을 계산한다. 그 자리를 KV에서 아낀다 | 한 사람의 속도는 대역폭이 정한 천장 그대로 |
| KV 압축 (MLA) | K, V를 그대로 저장하는 게 크다 | 작은 latent로 압축해서 저장한다 | 모델을 MLA 구조로 학습해야 한다 |
| 가까이 두기 (Groq) | HBM에서 옮겨 오는 길 자체가 느리다 | weight를 연산 옆 SRAM에 상주시킨다 | 칩 수백 장, 긴 context와 많은 사용자에 약하다 |

→ 앞의 둘은 처리량과 토큰당 비용을, 마지막은 한 사람이 체감하는 속도(ITL)를 노린다.

## 왜 지금 이 문제가 커졌나

[a16z와 OpenRouter의 100조 토큰 분석](https://arxiv.org/abs/2601.10088)에 따르면 2025년 11월까지 13개월 동안 요청 하나의 평균 입력은 약 1,500토큰에서 6,000토큰 이상으로, 평균 출력은 약 150토큰에서 400토큰으로 늘었다. 같은 기간 reasoning 모델로 가는 토큰의 비중은 거의 0에서 절반 이상이 됐다.

입력이 4배 길어지면 사용자 한 명의 KV Cache도 4배 커진다. 같은 GPU에 태울 수 있는 사람은 그만큼 줄어든다. 출력이 길어지면 그 자리를 더 오래 차지한다. agent처럼 문서와 도구 결과를 계속 붙여 넣는 사용 방식이 늘수록 이 다섯 가지의 무게는 커질 듯하다.

## 다음은

KV Cache가 HBM에 다 안 들어가는 시대가 되면, 다음 수는 KV를 둘 자리를 HBM 밖으로 넓히는 것이라고 생각한다. 실제로 NVIDIA는 2026년 1월 KV Cache를 GPU 밖의 스토리지 계층에 두고 꺼내 쓰는 [BlueField-4 기반 저장 인프라](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai)를 발표했다. Groq의 방식은 NVIDIA 안으로 들어가 decode를 맡는 칩이 됐다. 추론을 단계별로 쪼개 서로 다른 칩과 메모리에 맡기는 흐름은 3편에서 다룬다.

이 전망이 틀렸다고 볼 신호도 있다. 지금의 경쟁은 KV Cache가 대화 길이에 비례해 커진다는 전제 위에 있다. State Space Model이나 linear attention처럼 길이가 늘어도 저장할 상태가 커지지 않는 구조가 주류 모델의 기본이 된다면, KV 자리를 두고 벌이는 이 경쟁은 힘을 잃을 것이다.

## 내 생각

이 주장에 나올 만한 반론 두 개를 세워 본다.

> "batching은 연산을 더 쓰잖아. 그리고 Groq도 결국 계산하는 칩인데, 계산이 빨라서 빠른 거 아니야?"

batching은 연산을 더 쓰지만, 연산 능력을 키운 게 아니라 원래 놀던 연산을 쓰게 한 것이다. Groq은 앞에서 본 대로 연산이 H100의 5분의 1도 안 된다. 계산이 빨라서 빠른 칩이 아니다.

> "FlashAttention은? attention을 몇 배 빠르게 했다고 유명하잖아. 계산을 빠르게 해서 이긴 사례는 있는 거잖아."

[FlashAttention(Dao 등, 2022)](https://arxiv.org/abs/2205.14135)의 원래 제목에는 IO-Awareness라는 단어가 들어 있다. 기존 attention은 계산 중간에 나오는 큰 표(토큰 수 × 토큰 수)를 HBM에 썼다가 다시 읽었다. FlashAttention은 Q, K, V를 SRAM에 들어가는 작은 조각으로 잘라, 중간 결과를 HBM에 보내지 않고 SRAM 안에서 끝낸다. 계산 결과는 같고, 줄어든 건 HBM을 오가는 횟수다. 이 반론은 오히려 주장을 뒷받침하는 사례가 된다.

FlashAttention에는 개인적인 기억도 있다. 다국어 임베딩 모델로 MKQA 벤치마크를 측정하려고 `jinaai/jina-embeddings-v3`를 돌린 적이 있다. 이 모델은 `flash_attn`을 쓰도록 되어 있는데, 연구실 서버 GPU에서는 지원이 안 돼서 없이 돌렸고 측정은 30시간 넘게 걸렸다. 그때는 GPU가 A6000인 줄 알았는데, 이번에 찾아보니 [FlashAttention-2는 Ampere 이후 세대만 지원](https://github.com/Dao-AILab/flash-attention)하고 RTX A6000은 Ampere라서 지원 대상이었다. 서버의 GPU는 이름이 비슷한 Quadro RTX 6000이었고, 이건 한 세대 앞인 Turing이다. 30시간 중 얼마가 FlashAttention 때문이었는지는 재 보지 않았다.

경계에 걸치는 사례도 있다. 양자화는 weight를 4bit로 줄여 옮길 바이트를 줄이기도 하지만, 낮은 정밀도로 계산 자체를 빠르게 하기도 한다(H100은 FP8 연산이 FP16의 2배다). decode에서는 앞의 효과가, prefill에서는 뒤의 효과가 클 듯하다. 이 경계는 다음에 양자화를 따로 다루면서 확인해 보려 한다.

돌아보니 이 시리즈를 처음 시작할 때 세웠던 문장으로 돌아왔다. 추론 속도는 데이터를 어디 두고 얼마나 덜 옮기느냐의 싸움이다. 1편에서는 이 문장이 decode 한 단계에서 맞는다는 걸 봤고, 이번에는 업계가 실제로 쓰는 방법 다섯 가지가 전부 이 문장 안에 들어간다는 걸 확인했다. 운영체제 시험을 대충 본 나에게는 그 답이 운영체제 교과서에 있었다는 게 조금 민망하다.

## 출처

- Ainslie 등, [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245)
- Kwon 등, [Efficient Memory Management for Large Language Model Serving with PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
- vLLM, [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm) (2023-06-20, 낭비 60~80%와 4% 미만)
- Yu 등, [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
- DeepSeek-AI, [DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) (MLA, KV Cache 93.3% 감소)
- Dao 등, [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135), [flash-attention README](https://github.com/Dao-AILab/flash-attention) (지원 GPU)
- Meta Llama 3.1 70B [config.json](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B/blob/main/config.json) (NousResearch 미러)
- NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/) (FP16 1,979 TFLOPS sparsity 기준, FP8 3,958, HBM 3.35 TB/s, 80GB)
- NVIDIA, [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) (FP16 dense 1,000 / sparse 2,000 TFLOPS, 출시 전 예비치)
- Groq, [What is a Language Processing Unit?](https://groq.com/blog/the-groq-lpu-explained), [GroqChip Processor Product Brief v1.7](https://groq.com/wp-content/uploads/2024/08/GroqChip™-Processor-Product-Brief-v1.7.pdf) (188 TFLOPS FP16, SRAM 230MB, 80 TB/s)
- a16z, OpenRouter, [State of AI: An Empirical 100 Trillion Token Study (arXiv:2601.10088)](https://arxiv.org/abs/2601.10088)
- Anthropic, [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- NVIDIA, [BlueField-4 AI-native storage infrastructure](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai) (2026-01)
