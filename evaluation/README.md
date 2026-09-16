# 평가 모듈 사용법

"쿼리 변환과 의미 검색을 하면 결과가 얼마나 좋아지는가"를 숫자로 재는 코드다.
각 파일 맨 위 설명글에 무엇을 왜 그렇게 만들었는지가 있다.

## 파일 네 개

| 파일 | 무엇 |
|---|---|
| `dataset.py` | 평가셋 만들기 (질문 생성 → 분할 → 후보 풀 → 등급 판정)와 누수 검사 |
| `pipeline_eval.py` | 파이프라인을 실제로 돌려 재기 + 응답 시간 실측 |
| `metrics.py` | 지표 계산 (Recall, MRR, nDCG, 부트스트랩 신뢰구간과 검정) |
| `report.py` | 실행 결과를 읽어 보고하기, 여러 실행 짝지어 비교하기 |
| `gates.py` | 재정렬 모델이 고장났는지 빠르게 잡아내는 사전 점검 |

## 평가 데이터 — `data/eval/`에 네 개만 둔다

| 파일 | 무엇 |
|---|---|
| `dev.jsonl` | **개발용.** 설정을 바꿔 가며 탐색하는 것은 전부 여기서 한다 |
| `test.jsonl` | **시험용.** 확정 판정에만 쓴다 |
| `grades_dev.jsonl` | 개발용 등급 정답지 (후보 논문마다 관련도 0~3). nDCG 계산용 |
| `grades_test.jsonl` | 시험용 등급 정답지 |

질문 파일에는 정답 논문이 1편만 들어 있다. 그것만으로는 "만족스러운 논문을 몇 편
찾아줬나"를 못 재기 때문에 등급 정답지가 따로 있다.

중간 산출물(질문 원본, 후보 풀)은 `runs/`에 둔다. 평가 폴더에 중간 파일이 쌓이면
무엇이 진짜 평가셋인지 알 수 없게 된다.

## 1. 평가셋 만들기

네 단계다. 2단계와 3단계는 무료, 1단계와 4단계는 OpenAI를 쓰므로 돈이 든다.

```bash
# (1) 논문에서 질문을 거꾸로 생성 - 제목은 주지 않고, 한국어와 영어를 한 번에
python -m evaluation.dataset generate --n-papers 200 --out runs/queries_raw.jsonl

# (2) 논문 단위로 개발용과 시험용으로 나눔
python -m evaluation.dataset split --queries runs/queries_raw.jsonl --out-dir data/eval

# (3) 등급을 매길 후보 풀 만들기 (로컬 검색만, 비용 0원)
python -m evaluation.dataset pool --queries data/eval/dev.jsonl --out runs/pool_dev.jsonl

# (4) 후보마다 관련도 0~3 판정 (유료, --max-cost로 한도를 걸 수 있다)
python -m evaluation.dataset grade --pool runs/pool_dev.jsonl \
    --out data/eval/grades_dev.jsonl --max-cost 0.80
```

## 2. 평가셋에 정답이 새어 있는지 검사 (비용 0원)

한국어 질문을 로컬 모델로 영어로 옮긴 뒤 제목과 겹치는지 본다.

```bash
python -m evaluation.dataset audit --queries data/eval/test.jsonl \
    --run runs/test_run.jsonl --out runs/leakage_test.jsonl
```

## 3. 파이프라인 돌리기

```bash
# 서비스와 같은 구성으로 개발용 전체
python -m evaluation.pipeline_eval --queries data/eval/dev.jsonl \
    --channels local_dense local_hyde --rewriter service \
    --k 100 --rerank cross --rerank-depth 100 --out runs/dev_run.jsonl

# 저장된 결과만 다시 집계 (검색 0회)
python -m evaluation.pipeline_eval --report-only runs/dev_run.jsonl --rrf-k 30

# 저장된 결과에 재정렬만 다시 적용, 순위 합치기 가중치를 바꿔 가며
python -m evaluation.pipeline_eval --report-only runs/dev_run.jsonl \
    --rerank cross --rerank-depth 100 --fuse-rerank 3.0

# 같은 검색 결과 위에서 채널 조합만 갈라 보기
python -m evaluation.pipeline_eval --report-only runs/dev_run.jsonl \
    --use-channels local_dense --rerank cross --out runs/dev_localonly.jsonl
```

**`--local-query`를 건드리지 말 것 (기본 `raw`).** 학습한 변환기가 내놓는 것은 arXiv
문법 문자열이라 로컬 의미 검색에 넣으면 불리하고, 무엇보다 `app.py`가 로컬 채널에
원본 질문을 넣는다. 기본값이 서비스와 같은 조건이다.

## 4. 보고와 비교

```bash
# 실행 하나 자세히 (단일 정답 + 만족도 + 난이도, 언어별)
python -m evaluation.report --run runs/dev_run.jsonl \
    --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl

# 여러 실행 짝지어 비교 (공통 문항만, 통계 검정 포함). 기준선을 맨 앞에 둔다
python -m evaluation.report --run runs/dev_passthrough.jsonl runs/dev_run.jsonl \
    --queries data/eval/dev.jsonl
```

## 5. 응답 시간 재기

```bash
python -m evaluation.pipeline_eval --bench-service --n 5
python -m evaluation.pipeline_eval --bench-service --n 5 --rerank-depth 50
```

## 6. "못 찾았다" 고 말할 기준선 정하기

로컬 의미 검색은 어떤 질문에도 후보를 채워서 돌려준다. 그대로 뿌리면 무관한 논문을
추천으로 포장하게 된다. 몇 점 아래를 무관으로 볼지 등급 정답지로 실측한다.

```bash
python -m evaluation.pipeline_eval --calibrate-threshold \
    --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl
```

정한 값은 `app.py`의 `MIN_RERANK_SCORE`에 넣는다. **재정렬 모델을 바꾸면 점수 범위가
달라지므로 반드시 다시 재야 한다.**

## 7. 재정렬 모델 사전 점검

성능을 재는 것이 아니라 고장을 잡는 것이다. 떨어지면 Recall을 볼 필요가 없다.

```bash
python -m evaluation.gates --model models/reranker-ft
```

| 점검 | 기준 | 통과선 |
|---|---|---|
| 1. 무작위 논문 600쌍 | 점수 중앙값 | 0.001 미만 |
| | 0.002 이상인 비율 | 0.10 미만 |
| 2. 상위 10편 점수 분포 | 서로 다른 값의 개수 | 1,500개 이상 |
| | 일상어 층 1등−10등 차이 중앙값 | 0.05 이상 |

---

## 고정된 평가 규칙

결과를 보고 바꾸지 않는다. 구성마다 보는 지표가 다르면 "좋아졌다"가 지표를 고른 결과인지
모델이 좋아진 결과인지 가릴 수 없다.

### 주 지표 — 개발용 348문항 Recall@10 하나

`reranked_ids` 상위 10편에 정답 논문이 있는가.

- **등급 정답지가 필요 없다.** 질문을 만든 정답 1편만 보므로, 구성을 바꿔도 절대값이
  움직이지 않고 돈이 들지 않는다
- **판정은 짝지은 부트스트랩의 p 값으로만 한다.** p < 0.05 면 채택 후보, 그 이상이면
  "모름"이고 채택하지 않는다

### 함께 보는 층

| 무리 | 왜 보는가 |
|---|---|
| hard | 이 프로젝트가 돕겠다고 한 사용자층 |
| 한국어 | 확정된 가치 축 (번역 이득 +0.111, p=0.001) |
| 영어 | 한국어 이득이 진짜인지 보는 대조군 |

### nDCG@10은 이렇게만 쓴다

구성을 바꾸면 상위 10편에 판정 없는 논문이 들어와 절대값이 움직인다.
**등급 정답지를 넓히지 않고 다음 규칙으로만 쓴다.**

1. 판정 없는 논문을 0등급으로 놓고 한 번, 3등급으로 놓고 한 번 계산한다
2. 두 값의 부호가 같으면 그 방향을 판정으로 쓴다
3. 부호가 갈리면 "판정 불가"로 적고 넘어간다
4. **주 지표를 대신하지 않는다.** 채택 여부는 Recall@10으로 정하고, nDCG는
   "확실히 나빠지지는 않았는가"를 보는 안전장치로만 쓴다

### 바꾸지 않기로 못박은 값

| 항목 | 값 | 근거 |
|---|---|---|
| 후보 깊이 | 100 | 300이면 만족도가 떨어짐 (p<0.001) |
| 채널 | 로컬 색인 2회 (번역문, 가상 초록) | arXiv는 추천 목록에 안 섞음 |
| 재정렬에 넣는 질문 | 원본 질문 | 번역문으로 바꿔도 이득 없음 |
| 순위 합치기 상수 | 60 | |

---

## 반드시 지킬 것

1. **탐색은 개발용에서만.** 시험용은 확정 판정에만 쓴다. 반복해서 쓰면 소모된다
2. **기준선을 항상 함께 잰다.** 핵심 주장이 "이렇게 하면 좋아진다" 이므로, 아무것도
   하지 않은 값이 없으면 주장이 검증되지 않는다
3. **p 값이 0.05보다 크면 "차이가 없다"가 아니라 "있는지 없는지 모른다".**
4. **무거운 실행 전에 의존성을 확인한다.** 몇 초면 끝난다
