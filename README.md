# Papers, Please

**한국어나 일상어로 물어도 arXiv 논문을 찾아 주는 검색 서비스.**

논문을 찾을 때 가장 큰 걸림돌은 자기가 읽고 싶은 논문이 실제로 어떤 말로 쓰여 있는지
모른다는 것이다. "사진 보고 글로 설명해주는 AI" 라고 쳐도 논문은 `image captioning`
이라고 쓴다. 한국어로 물으면 arXiv 는 아예 못 찾는다 — 시험용 342문항에서 한국어 질문
171개 중 128개(74.9%)가 결과 0건이었다.

Papers, Please 는 질문을 그대로 키워드 검색에 넣는 대신 **영어로 옮기고, 그 질문에 답할
법한 초록을 지어내고, 논문 71만 편을 뜻으로 찾은 뒤, 사용자 의도와 맞는지 다시 줄
세운다.** 같은 342문항에서 arXiv 키워드 검색의 Recall@10 이 0.190 인데 이 시스템은
**0.658** 이다.

비전문가를 위한 논문 검색 시스템 (졸업작품).

---

## 화면

![Papers, Please 화면](assets/screenshot.png)

*(화면 사진을 `assets/screenshot.png` 에 넣으면 여기 나옵니다. 자세한 것은
[assets/README.md](assets/README.md) 참고)*

### 어떻게 동작하는가

```mermaid
flowchart TD
    Q["질문<br/>사진 보고 글로 설명해주는 AI"]
    T["검색어 1 · 영어로 옮김<br/>Qwen3-4B"]
    H["검색어 2 · 가상 초록 지어내기<br/>Qwen3-4B"]
    I1["논문 71만 편 의미 검색<br/>bge-m3 파인튜닝"]
    I2["논문 71만 편 의미 검색<br/>bge-m3 파인튜닝"]
    F["순위 합치기 → 후보 100편"]
    R["교차 인코더 재정렬<br/>원본 질문과 대조"]
    F2["재정렬 순위 + 검색 순위<br/>3 대 1 로 합침"]
    A["추천 이유 생성<br/>Qwen3-4B"]
    O["논문 10편"]
    X["arXiv 실시간 검색"]
    N["최신 논문 칸<br/>따로 보여 줌"]

    Q --> T --> I1 --> F
    Q --> H --> I2 --> F
    F --> R --> F2 --> A --> O
    Q -.-> X -.-> N
```

**arXiv 실시간 검색 결과는 추천 목록에 섞지 않고 '최신 논문' 칸으로 따로 보여 준다.**
섞으면 비율을 어떻게 잡아도 만족도가 떨어졌다(전부 p<0.001). arXiv 채널의 가치는
정확도가 아니라 색인에 없는 최신 논문이다.

---

## 설치

### 1. 저장소와 파이썬 환경

```bash
git clone https://github.com/jmoongyu/Papers-Please.git
cd Papers-Please

python3 -m venv .venv
source .venv/bin/activate          # 윈도우는 .venv\Scripts\activate
pip install -r requirements.txt
```

GPU 를 쓰려면 torch 를 먼저 CUDA 빌드로 설치한다.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 2. 로컬 언어 모델 (Ollama)

번역, 가상 초록, 추천 이유 생성에 쓴다. [ollama.com](https://ollama.com) 에서 받은 뒤:

```bash
ollama pull qwen3:4b
ollama serve
```

### 3. 논문 코퍼스 만들기

[Kaggle arXiv 데이터셋](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(`arxiv-metadata-oai-snapshot.json`) 을 받아 `data/corpus/` 에 두고:

```bash
python -m src.retrieval.corpus \
    --kaggle data/corpus/arxiv-metadata-oai-snapshot.json \
    --out data/corpus/corpus-cs2021.jsonl \
    --prefixes cs. stat.ML eess. --min-year 2021 --sample 0
```

결과는 716,183편, 약 1GB 다.

### 4. 의미 검색 색인 만들기

서비스는 파인튜닝한 검색 모델로 만든 `cs2021-ft` 색인을 쓴다. 아래
[학습](#학습-선택) 으로 `models/retriever-ft` 를 먼저 만든 뒤 색인을 만든다.

```bash
python -m src.retrieval.local_index \
    --corpus data/corpus/corpus-cs2021.jsonl \
    --model models/retriever-ft \
    --out data/embeddings/cs2021-ft
```

GPU 로 약 3시간, 결과는 2.9GB 다. 중단해도 이어서 한다.

> 학습을 건너뛰고 원본 `BAAI/bge-m3` 로만 색인을 만들려면 `--model` 을 빼고
> `--out data/embeddings/cs2021` 로 만든 뒤, `app.py` 의 `LOCAL_INDEX` 를 `"cs2021"` 로
> `FUSE_RERANK_WEIGHT` 를 `0` 으로 바꾼다. 둘은 한 묶음이다.
> 시험용 342문항 Recall@10 이 0.658 에서 0.617 로 내려간다.

---

## 사용법

### 웹 화면

```bash
streamlit run app.py          # → http://localhost:8501
```

첫 실행은 색인을 올리는 데 1분쯤 걸린다. 왼쪽 설정에서 로컬 의미 검색과 arXiv 실시간
검색을 각각 켜고 끌 수 있다.

### 성능 평가

```bash
# 파이프라인을 돌려 결과를 저장 (중단하면 이어서 함)
python -m evaluation.pipeline_eval --queries data/eval/dev.jsonl \
    --channels local_dense local_hyde --rewriter service \
    --k 100 --rerank cross --rerank-depth 100 --out runs/dev_run.jsonl

# 저장된 결과를 보고 (검색 0회)
python -m evaluation.report --run runs/dev_run.jsonl \
    --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl

# 두 실행을 짝지어 비교 (통계 검정 포함). 기준선을 맨 앞에 둔다
python -m evaluation.report --run results/test_arxiv_only_raw.jsonl \
    results/test_ft_fused.jsonl --queries data/eval/test.jsonl

# 응답 시간 재기
python -m evaluation.pipeline_eval --bench-service --n 5

# 테스트
python -m pytest tests/ -q
```

저장된 검색 결과에 설정만 바꿔 다시 집계할 수 있다. 검색을 다시 하지 않으므로 빠르고
arXiv 도 다시 부르지 않는다.

```bash
python -m evaluation.pipeline_eval --report-only runs/dev_run.jsonl \
    --rerank cross --rerank-depth 100 --fuse-rerank 3.0
```

### 학습 (선택)

```bash
# 쿼리 변환기 - 지도 파인튜닝 다음에 선호 학습
python -m training.train sft --output-dir models/query-translator-sft
python -m training.train dpo --sft-adapter models/query-translator-sft/checkpoint-54 \
    --output-dir models/query-translator-dpo

# 검색 모델 파인튜닝 (서비스가 쓰는 것)
python -m training.build_retrieval_pairs --out data/training/train_retriever.jsonl
python -m training.train embed --output-dir models/retriever-ft
```

### 주요 설정값

전부 `app.py` 맨 위에 있고, 값마다 어떻게 정했는지 주석에 적혀 있다.

| 값 | 기본 | 무엇 |
|---|---|---|
| `LOCAL_INDEX` | `cs2021-ft` | 쓸 색인. 질문 임베딩 모델은 색인 파일에서 읽는다 |
| `DEPTH_LOCAL` | 100 | 검색어마다 가져올 후보 편수 |
| `RERANK_DEPTH` | 100 | 재정렬에 넣을 최대 후보 편수 |
| `FUSE_RERANK_WEIGHT` | 3.0 | 재정렬 순위와 검색 순위를 합칠 때 재정렬 쪽 가중치 |
| `MIN_RERANK_SCORE` | 0.002 | 이 아래는 "관련성이 낮아 접어 둔" 자리로 내린다 |
| `TOP_K` | 10 | 보여 줄 논문 편수 |

---

## 검색 성능

### 어떻게 쟀는가

**평가셋.** 논문에서 질문을 거꾸로 만들었다. 모델에게 **제목을 주지 않고 초록만 주어**,
"이 논문을 아직 못 찾은 사람" 의 자리에서 질문을 쓰게 했다. 제목을 주고 어순만 바꾸게
했더니 정확한 용어를 쓰는 층의 Recall@10 이 1.000 이 나왔는데, 원인은 희귀 용어가 아니라
제목의 낱말 조합을 그대로 재현한 것이었다.

- **시험용 342문항** (`data/eval/test.jsonl`) — 확정 판정에만 쓴다
- **개발용 348문항** (`data/eval/dev.jsonl`) — 설정을 바꿔 보는 탐색은 전부 여기서 한다
- **같은 논문, 같은 난이도를 한국어와 영어로 짝지어** 만들었다. 그래서 언어별 차이가
  논문 차이와 섞이지 않는다
- 난이도 3층 — `easy` 대학원생의 학술어 / `medium` 학부연구생 / `hard` 1~2학년의 일상어

**지표.** 주 지표는 **Recall@10** — 상위 10편 안에 그 질문을 만든 논문이 들어왔는가.
등급 정답지가 필요 없어서 구성을 바꿔도 절대값이 움직이지 않는다.

**판정.** 같은 문항끼리 짝지은 부트스트랩 검정. **p 가 0.05 보다 크면 "차이가 없다" 가
아니라 "있는지 없는지 모른다"** 로 적는다.

### 적용 전후 (시험용 342문항, Recall@10)

| 무리 | 단순 arXiv API 키워드 검색 | **이 프로젝트** | 차이 |
|---|---|---|---|
| **전체** | 0.190 | **0.658** | **+0.468** |
| easy | 0.500 | 0.904 | +0.404 |
| medium | 0.061 | 0.737 | +0.675 |
| hard | 0.009 | 0.333 | +0.325 |
| **한국어** | 0.076 | **0.661** | **+0.585** |
| 영어 | 0.304 | 0.655 | +0.351 |

여섯 무리 전부 **p < 0.001**. 전체 신뢰구간 [+0.415, +0.520].

- 전: `results/test_arxiv_only_raw.jsonl` — 사용자 질문을 그대로 arXiv API 에 넣음
- 후: `results/test_ft_fused.jsonl` — 지금 서비스 구성
- **한국어 질문 171개 중 128개(74.9%)가 arXiv 에서 결과 0건.** 영어는 0건이 없었다

두 파일 다 저장돼 있으므로 직접 다시 계산할 수 있다.

```bash
python -m evaluation.report --run results/test_arxiv_only_raw.jsonl \
    results/test_ft_fused.jsonl --queries data/eval/test.jsonl
```

### 무엇이 성능을 만들었는가

개발용 348문항. 단계마다 하나씩만 더했다.

| 구성 | 전체 | easy | medium | hard | 한국어 | 영어 |
|---|---|---|---|---|---|---|
| ① 단순 arXiv 키워드 검색 | 0.086 | 0.233 | 0.026 | 0.000 | 0.017 | 0.155 |
| ② 쿼리 변환 + arXiv 검색 | 0.284 | 0.586 | 0.241 | 0.026 | 0.270 | 0.299 |
| ③ 로컬 의미 검색만 | 0.428 | 0.724 | 0.474 | 0.086 | 0.368 | 0.489 |
| ④ ③ + 한국어를 영어로 옮김 | 0.468 | 0.784 | 0.500 | 0.121 | 0.448 | 0.489 |
| ⑤ ④ + 가상 초록 검색어 추가 | 0.457 | 0.784 | 0.457 | 0.129 | 0.443 | 0.471 |
| ⑥ ⑤ + 교차 인코더 재정렬 | 0.575 | 0.914 | 0.629 | 0.181 | 0.575 | 0.575 |
| ⑦ ⑥ + 파인튜닝 검색 모델 | 0.618 | 0.914 | 0.672 | 0.267 | 0.615 | 0.621 |
| **⑧ ⑦ + 순위 합치기 3:1** | **0.635** | 0.914 | 0.681 | **0.310** | 0.632 | 0.638 |

**가장 큰 두 몫은 arXiv 키워드 검색을 로컬 의미 검색으로 바꾼 것(+0.144, p<0.001)과
교차 인코더 재정렬(+0.118, p<0.001)이다.** 둘이 전체 상승분의 절반이다.

**한국어와 영어의 격차가 닫혔다.**

```
①  한국어 0.017  영어 0.155   차이 0.138
④  한국어 0.448  영어 0.489   차이 0.041   <- 번역을 넣은 자리
⑧  한국어 0.632  영어 0.638   차이 0.006
```

### 응답 시간 (질문 10개, arXiv 채널 끔)

| 단계 | 중앙값 | 최대 |
|---|---|---|
| 추천 이유 생성 | 11.39초 | 14.82초 |
| 가상 초록 만들기 | 1.05초 | 1.38초 |
| 재정렬 (순위 합치기 포함) | 0.39초 | 0.45초 |
| 한국어를 영어로 옮기기 | 0.37초 | 0.47초 |
| 로컬 의미 검색 | 0.21초 | 0.24초 |
| **합계** | **13.35초** | **16.96초** |

기준 30초를 넘은 질문 0/10개. 시작할 때 부품을 올리는 17.2초는 한 번만 든다.
응답 시간의 85%가 추천 이유 생성이다.

### 함께 적어야 할 것

- **일상어 층(hard)은 0.333 으로 여전히 낮다.** 다만 이 층의 질문 중 상당수에는 정답
  논문이 아니어도 쓸모 있는 논문을 상위 10편에 보여 준다
- **지금 구성을 채택한 근거는 전체 성능이 아니다.** 옛 구성 대비 전체 +0.041 인데
  **p=0.085 로 유의성을 확보하지 못했다**(신뢰구간 [−0.003, +0.085]). 개발용에서는
  +0.060(p=0.003)이었고 시험용에서 방향과 크기가 재현됐으나 342문항으로는 이 크기를
  잡아낼 힘이 모자란다. 채택 근거는 ⓐ 유의미하게 나빠진 무리가 하나도 없고
  ⓑ **한국어가 유의미하게 좋아졌다**는 것이다(+0.070, p=0.045)
- **반증된 것도 그대로 남겼다.** 의도→개념→용어로 단계를 밟는 계층 변환은 세 번 측정해
  세 번 반증됐고, 재정렬기를 우리 자료로 파인튜닝하는 것은 세 번 시도해 세 번 실패했다.
  두 손실 함수 모두 점수의 절대값을 고정하는 항이 없어서, 순서는 가르쳐도 "무관한 것은
  0 에 가깝게" 는 못 가르친다. 대신 **재정렬 순위와 검색 순위를 합치는 것**이 같은
  목표를 학습 0시간으로 달성했다

---

## 폴더 구조

| 위치 | 무엇 |
|---|---|
| `app.py` | 웹 화면 (Streamlit) |
| `src/rewriter/` | 쿼리 변환 — 번역 · 가상 초록 · 학습한 모델 · 특정 논문 지목 |
| `src/retrieval/` | 검색 — 코퍼스 · 로컬 색인 · arXiv · 순위 합치기와 재정렬 |
| `src/recommend_agent/` | 추천 이유 생성 |
| `evaluation/` | 평가셋 제작 · 파이프라인 실행 · 지표 · 보고 |
| `training/` | 학습 — 쿼리 변환기 · 검색 모델 · 재정렬기 |
| `tests/` | 단위 테스트 47건 (모델 없이 돈다) |
| `data/eval/` | 평가셋 4개 (`dev` · `test` · `grades_dev` · `grades_test`) |
| `results/` | 보고서가 인용하는 확정 결과. [설명](results/README.md) |

**디렉터리마다 코드 파일은 4개를 넘지 않는다.** 새 기능은 되도록 기존 파일에 넣는다.

평가 실행 방법은 [evaluation/README.md](evaluation/README.md) 에 자세히 있다.

---

## 쓴 모델

| 자리 | 모델 | 비고 |
|---|---|---|
| 번역 · 가상 초록 · 추천 이유 | `qwen3:4b` (Ollama) | 로컬 실행 |
| 의미 검색 | `BAAI/bge-m3` 를 파인튜닝한 `models/retriever-ft` | 다국어 |
| 재정렬 | `BAAI/bge-reranker-v2-m3` | 원본 그대로 |
| arXiv 검색어 생성 | `Qwen3-4B` + LoRA 학습 | arXiv 채널을 켤 때만 |

**서비스가 도는 동안 유료 인공지능 서비스를 부르지 않는다.** OpenAI 는 평가셋을 미리
만들 때만 쓴다.

모델 가중치와 색인은 크기 때문에 저장소에 없다. 위 [설치](#설치) 와
[학습](#학습-선택) 절차로 다시 만들 수 있다.

---

## arXiv 이용 정책

- 제목, 초록, 논문 번호 같은 메타데이터는 CC0 1.0 으로 배포되어 저장과 재사용이 된다.
  **논문 원문은 이 서비스가 제공하지 않고 arXiv 초록 페이지로 보낸다.**
- arXiv API 는 요청 간 3초 간격, 단일 연결을 지켜야 한다. 코드가 지킨다.
- 이 서비스는 arXiv 와 무관한 프로젝트이며 arXiv 의 후원이나 보증을 받지 않았다.

> Thank you to arXiv for use of its open access interoperability.
> This service was not reviewed or approved by, nor does it necessarily express
> or reflect the policies or opinions of, arXiv.

---

## 라이선스

[MIT](LICENSE)
