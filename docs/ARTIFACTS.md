# 산출물 기록 (ARTIFACTS)

- 마지막 갱신: **2026-08-28**
- 무엇: 학습한 모델, 만든 색인, 학습 자료가 각각 **무엇이고 성능이 어땠는지** 한곳에 적음.
  지운 것도 지우기 전에 여기에 남김.
- 다른 문서: 진행 상황 [PROGRESS.md](PROGRESS.md) · 문제 기록 [ISSUE.md](ISSUE.md) ·
  모듈 구조 [MODULE_SPECIFICATION.md](MODULE_SPECIFICATION.md)

> 2026-08-28 에 이름 규칙을 **모듈 이름 기준**으로 바꿨음. 옛 이름과 새 이름의 짝은
> 4절에 있음. 문서와 코드는 전부 새 이름으로 고쳤으나, **커밋 메시지와 2026-08-28 이전
> 커밋의 파일 내용은 옛 이름 그대로임.** 옛 커밋을 읽을 때 4절 표를 볼 것.

---

## 1. 지금 쓰는 것

### 모델

| 폴더 | 무엇 | 바탕 모델 | 학습 자료 | 상태 |
|---|---|---|---|---|
| `models/query-translator-sft/checkpoint-54` | 쿼리 변환기 1단계(지도 학습) | Qwen3-4B-Instruct-2507 | `train_query_translator_sft.jsonl` | DPO 학습의 출발점. `finetuned.py` 의 기본값 |
| `models/query-translator-dpo` | 쿼리 변환기 2단계(선호 학습) | 위 어댑터 | `train_query_translator_dpo.jsonl` | **서비스가 부르는 것.** arXiv 채널을 켤 때만 올림 |
| `models/retriever-ft` | 미세조정 검색 모델 | BAAI/bge-m3 | `train_retriever.jsonl` 33,000문항 | **2026-08-28 서비스에 채택** |
| `models/reranker-ft` | 미세조정 재정렬기 (1차) | BAAI/bge-reranker-v2-m3 | 옛 자료 33,000문항 | **실패.** 점수가 1 에 몰림 (ISSUE #55) |
| `models/reranker-ft-hardneg` | 미세조정 재정렬기 (2차) | BAAI/bge-reranker-v2-m3 | `train_reranker.jsonl` 33,000문항 | **실패.** 관문 1 불합격, Recall 내려감 (ISSUE #55) |

서비스가 쓰는 검색 모델과 재정렬기는 아직 내려받은 원본임 - `BAAI/bge-m3` 와
`BAAI/bge-reranker-v2-m3`. 위의 미세조정본 둘은 판정을 통과하지 못해 안 걸었음.

**성능 기록**

| 모델 | 잰 것 | 값 |
|---|---|---|
| `query-translator-dpo` | 옛 평가셋 Recall@10 | 0.167 → 0.343 |
| | 새 평가셋(시험용) 기여 | Recall +0.018(판정 불가) · nDCG −0.016(나빠짐) |
| `retriever-ft` | 개발용 Recall@10 | 0.575 → **0.618** (p=0.031, 유의미) |
| | 시험용 Recall@10 | 0.617 → 0.629 (p=0.632, **판정 불가**) |
| `reranker-ft` | 개발용 Recall@10 | 0.618 → 0.629 (p=0.533, 판정 불가) |
| | 개발용 한국어 / 영어 | −0.029 / **+0.052(p=0.033)** |
| | 상위10 점수 중 서로 다른 값 | 2,304개 → **607개** (점수 포화) |

> **다음 재정렬기 학습은 `models/reranker-ft` 를 덮어쓰지 말 것.** 덮어쓰면 ISSUE #55 의
> 값을 다시 잴 수 없음. 손실을 바꿔 다시 학습할 때는 `--output-dir` 을 따로 줄 것.

### 색인 (`data/embeddings/`)

| 이름 | 만든 모델 | 논문 수 | 크기 | 상태 |
|---|---|---|---|---|
| `cs2021` | `BAAI/bge-m3` (원본) | 716,183 | 2.93GB | 되돌릴 때를 위해 남겨 둠 |
| **`cs2021-ft`** | `models/retriever-ft` | 716,183 | 2.93GB | **서비스가 쓰는 것 (2026-08-28 채택)** |

두 색인 모두 `data/corpus/corpus-cs2021.jsonl`(1.03GB, 716,183편)로 만들었음.
색인과 코퍼스의 짝은 불러올 때 자동으로 확인함(ISSUE #28).

### 학습 자료 (`data/training/`)

| 파일 | 무엇 | 크기 | 커밋 |
|---|---|---|---|
| `train_queries.jsonl` | 미세조정용 질문 36,000개 (논문 6,000편 × 난이도 3종 × 언어 2종) | 25MB | 함 |
| `train_retriever.jsonl` / `val_retriever.jsonl` | 검색 모델 학습·검증 쌍 33,000 / 3,000문항 | 298 / 27MB | 안 함 |
| `train_reranker.jsonl` / `val_reranker.jsonl` | 재정렬기 학습·검증 쌍 33,000 / 3,000문항 | 102 / 9.6MB | 안 함 |
| `train_query_translator_sft.jsonl` | 쿼리 변환기 지도 학습 쌍 | 29KB | 함 |
| `train_query_translator_dpo.jsonl` | 쿼리 변환기 선호 학습 쌍 | 54KB | 함 |

`train_*.jsonl` 과 `val_*.jsonl` 은 `train_queries.jsonl` 과 색인에서 다시 만들 수 있어
커밋하지 않음. 다만 재정렬기 쌍은 다시 만드는 데 **4시간 20분** 걸리므로 지우지 말 것.

**평가셋(`data/eval/`)과 헷갈리지 말 것.** `dev.jsonl`(개발용 348문항)과
`test.jsonl`(시험용 342문항)은 성능을 재는 자료이고, 학습에 절대 쓰지 않음(ISSUE #25).
그래서 학습 자료 쪽은 `train_` / `val_` 만 쓰고 `test_` 를 안 씀.

---

## 2. 2026-08-28 에 지운 것

지우기 전에 무엇이었고 성능이 어땠는지 아래에 적었음. 실제로 비운 디스크 **약 17GB**.

### 밀려난 모델

#### `models/bge-m3-papers` (검색 모델 1차 학습, 2.2GB)

- 2026-08-25 학습. 바탕 `BAAI/bge-m3`, LoRA, `MultipleNegativesRankingLoss`, 자료 22,000문항
- **성능: 관문 1(부분집합 등수 확인)에서 easy 층 Recall 0.836 → 0.638 으로 떨어짐**
- 원인: 학습 자료에서 easy 난이도를 뺐더니 그 층을 잊어버림 (ISSUE #51)
- 대체: easy 12,000문항을 넣어 다시 학습한 `models/retriever-ft`(옛 `bge-m3-papers-v2`)

#### `models/qwen3-4b-query-lora-v2` (쿼리 변환기 지도 학습 2차, 6.3GB)

- 바탕 `Qwen/Qwen3-4B-Instruct-2507`, LoRA 지도 학습 8세대(512걸음)
- **성능: 1차본보다 나빴음**

  | | 1차 (`query-translator-sft`) | 2차 (지움) |
  |---|---|---|
  | 검증 손실 (8세대) | **0.7252** | 1.1686 |
  | 검증 낱말 정확도 | **0.8865** | 0.8449 |

- 코드와 문서 어디에서도 부르지 않았음. 서비스가 쓰는 것은 선호 학습까지 끝낸 dpo 쪽임

#### 중간 체크포인트

- `models/query-translator-sft/checkpoint-{18,36,72,90,108,126,144}` (5.4GB)
  - 쓰는 것은 `checkpoint-54`(3세대) 하나뿐임. 세대별 검증 손실을 기록으로 남김

    | 체크포인트 | 18 | 36 | **54** | 72 | 90 | 108 | 126 | 144 |
    |---|---|---|---|---|---|---|---|---|
    | 세대 | 1 | 2 | **3** | 4 | 5 | 6 | 7 | 8 |
    | 검증 손실 | 0.6092 | 0.4873 | **0.5131** | 0.5680 | 0.6343 | 0.6880 | 0.7170 | 0.7252 |
    | 검증 정확도 | 0.8760 | 0.8945 | **0.8929** | 0.8929 | 0.8870 | 0.8883 | 0.8873 | 0.8865 |

- `models/query-translator-dpo/checkpoint-{42,84}` (2.0GB) + `ref/` (253MB)
  - 최종본은 폴더 루트에 있음. `ref/` 는 옛 학습 스크립트가 남긴 것으로 지금 코드가 안 부름
  - 84걸음(2세대) 마지막 값: 검증 손실 0.4062 · 선호 정확도 0.9375 · 보상 차이 0.8842

#### 빈 폴더

- `models/bge-m3-papers-ckpt`, `models/bge-m3-papers-v2-ckpt`, `models/bge-reranker-papers-ckpt`
  - 학습 중간 저장을 안 하도록(`save_strategy="no"`) 두어서 만들어지기만 하고 비어 있었음

### 밀려난 색인

#### `data/embeddings/cs2021-ft` (옛 것, 겉보기 2.9GB)

- **끝까지 만들지 못한 색인임.** 기록된 값이 `done: 245,760 / count: 716,183` 이라
  불러오면 `임베딩이 아직 다 안 됐다` 로 멈춤. 실제로 쓸 수 없었음
- 만든 모델이 위의 결함 모델(`bge-m3-papers`)임
- 지금 남은 미세조정 색인이 `cs2021-ft`(옛 `cs2021-ft-v2`)로 이름을 물려받음
- 파일 크기는 2.9GB 로 잡혀 있었으나 안이 비어 있어 실제로 쓰던 디스크는 그보다 적었음.
  그래서 이번 정리로 실제로 비운 용량은 대부분 모델 쪽에서 나옴

#### `data/embeddings/corpus-v1.BAAI_bge-m3.*` (123MB) · `data/corpus/corpus-v1.jsonl` (44MB)

- 코퍼스 3만 편 시절(2026-07)의 코퍼스와 색인. 지금 코퍼스는 716,183편임
- 이때 잰 값은 전부 **폐기된 평가셋 v1** 에서 나온 것이라 지금 값과 비교할 수 없음
  (난이도 누수 ISSUE #32, 학습 소모 ISSUE #25)

### 그 밖에

- `runs/` 안의 `results/` 완전 중복 5개 (`test_dpo`, `test_dpo_only_arxiv`,
  `test_dpo_only_local_dense`, `test_passthrough`, `test_translate_rerankraw`) - md5 가 같음
- `data/eval/grades_dev.jsonl.bak-20260827` - 등급 정답지를 1,233쌍 넓히기 전 백업.
  깃 기록에 그대로 남아 있음
- `__pycache__/`, `.pytest_cache/`

---

## 3. 안 지운 것과 그 이유

| 파일 | 크기 | 왜 남겼는가 |
|---|---|---|
| `data/corpus/arxiv-metadata-oai-snapshot.json` | 5.4GB | 캐글 원본. 코퍼스를 다시 만들거나 넓힐 때 필요함. 지우면 4GB 를 다시 받아야 함 |
| `data/corpus/corpus-full.jsonl` | 3.7GB | 분야를 넓힐 때(cs 밖) 다시 거르는 출발점 |
| `data/cache/arxiv_search_cache.jsonl` | 148MB | arXiv 검색 캐시. 지우면 평가를 다시 돌릴 때 실제 호출이 다시 일어남 |
| `models/reranker-ft` | 2.2GB | 실패한 학습이지만 ISSUE #55 의 근거 파일임 |
| `data/embeddings/cs2021-ft` | 2.9GB | 채택 보류일 뿐이고 다시 만들면 169.6분 걸림 |
| `runs/` 의 옛 실행 결과 | 130MB | 문서 표의 근거. 지우면 재집계로 검증할 수 없음 |

---

## 4. 이름 바꾼 것 (2026-08-28)

옛 문서와 커밋 메시지는 왼쪽 이름으로 적혀 있음. 두 이름은 같은 것을 가리킴.

### 모델

| 옛 이름 | 새 이름 |
|---|---|
| `models/bge-m3-papers-v2` | `models/retriever-ft` |
| `models/bge-reranker-papers` | `models/reranker-ft` |
| `models/qwen3-4b-query-lora` | `models/query-translator-sft` |
| `models/qwen3-4b-query-dpo` | `models/query-translator-dpo` |

### 색인

| 옛 이름 | 새 이름 |
|---|---|
| `data/embeddings/cs2021-ft-v2` | `data/embeddings/cs2021-ft` |

### 학습 자료

| 옛 이름 | 새 이름 |
|---|---|
| `data/training/ft_queries.jsonl` | `data/training/train_queries.jsonl` |
| `data/training/embed_pairs_train.jsonl` | `data/training/train_retriever.jsonl` |
| `data/training/embed_pairs_val.jsonl` | `data/training/val_retriever.jsonl` |
| `data/training/rerank_pairs_train.jsonl` | `data/training/train_reranker.jsonl` |
| `data/training/rerank_pairs_val.jsonl` | `data/training/val_reranker.jsonl` |
| `data/training/sft_pairs.jsonl` | `data/training/train_query_translator_sft.jsonl` |
| `data/training/dpo_pairs.jsonl` | `data/training/train_query_translator_dpo.jsonl` |

### 코드

| 옛 이름 | 새 이름 |
|---|---|
| `training/build_embed_pairs.py` | `training/build_retrieval_pairs.py` |
| `training/build_training_data.py` | `training/build_translator_pairs.py` |

### 규칙

- **모듈 이름을 앞이 아니라 뒤에 붙임** - `train_reranker.jsonl` 처럼. 앞에 `train_` /
  `val_` 을 두면 같은 쓰임끼리 한 줄로 모여 보기 좋음
- **모델 폴더는 `<모듈>-<학습 방식>`** - `query-translator-sft`, `retriever-ft`
- **`test_` 는 학습 자료에 쓰지 않음.** `data/eval/test.jsonl`(시험용 평가셋)과 헷갈림
- `src/` 안의 꾸러미 이름(`src/rewriter/`, `src/retrieval/`)과 평가 인자 이름
  (`--rewriter`)은 **안 바꿨음.** 확정 결과 파일 `results/*.jsonl` 의 실행 정보 줄에
  `"rewriter"` 로 적혀 있고, `evaluation/report.py` 가 그 이름으로 읽기 때문임
