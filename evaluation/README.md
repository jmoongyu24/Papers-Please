# 성능 평가 모듈 사용법

"쿼리 변환을 하면 검색 결과가 변환 전보다 얼마나 더 좋아지는가"를 숫자로 재는 코드다.
설계 설명은 [../docs/MODULE_SPECIFICATION.md](../docs/MODULE_SPECIFICATION.md) 참고.
여기서는 실행 방법만 정리한다.

```
PY=/home/jmoongyu/venvs/paper_py310/bin/python
```

## 파일 네 개가 전부다

| 파일 | 무엇을 하나 |
|---|---|
| `dataset.py` | 평가셋 만들기 (질문 생성 → 분할 → 후보 풀 → 등급 판정) 와 누수 감사 |
| `pipeline_eval.py` | 검색 파이프라인을 실제로 돌려 재기 + 서비스 응답 시간 실측 |
| `metrics.py` | 지표 계산 (Recall, MRR, nDCG, 부트스트랩 신뢰구간·검정) |
| `report.py` | 실행 결과를 읽어 보고하기 · 여러 실행 짝지어 비교하기 |

각 파일 맨 위 설명글에 **왜 그렇게 만들었는지**가 적혀 있다. 고치기 전에 읽을 것.

## 평가 데이터 — `data/eval/` 에 네 개만 둔다

| 파일 | 무엇인가 |
|---|---|
| `dev.jsonl` | **개발용.** 설정을 바꿔 가며 탐색하는 것은 전부 여기서 한다 |
| `test.jsonl` | **시험용.** 확정 판정에만 쓴다. 이걸 보면서 설정을 고르면 시험지가 탄다 |
| `grades_dev.jsonl` | 개발용 등급 정답지 (후보 논문마다 관련도 0~3). 만족도 지표 nDCG 계산용 |
| `grades_test.jsonl` | 시험용 등급 정답지 |

질문 파일에는 **정답 논문이 1편**만 들어 있다. 그것만으로는 "만족스러운 논문을 몇 편
찾아줬나"를 못 재기 때문에 등급 정답지가 따로 있다. 자세한 이유는 `report.py` 설명글 참고.

중간 산출물(질문 원본, 후보 풀)은 `runs/` 에 둔다. 평가 폴더에 중간 파일이 쌓이면
무엇이 진짜 시험지인지 알 수 없게 된다.

## 1. 평가 데이터셋 만들기

네 단계다. 2·3단계는 무료, 1·4단계는 OpenAI 를 쓰므로 돈이 든다.

```bash
# (1) 논문에서 질문을 거꾸로 생성 — 제목은 주지 않는다, 한국어·영어를 한 번에
$PY -m evaluation.dataset generate --n-papers 200 --out runs/queries_raw.jsonl

# (2) 논문 단위로 dev/test 로 나눈다 (같은 논문의 질문이 양쪽에 흩어지지 않게)
$PY -m evaluation.dataset split --queries runs/queries_raw.jsonl --out-dir data/eval

# (3) 등급을 매길 후보 풀을 만든다 (로컬 검색만 — 비용 0원)
$PY -m evaluation.dataset pool --queries data/eval/dev.jsonl --out runs/pool_dev.jsonl

# (4) 후보마다 관련도 0~3 판정 (유료 · --max-cost 로 상한을 걸 수 있다)
$PY -m evaluation.dataset grade --pool runs/pool_dev.jsonl \
    --out data/eval/grades_dev.jsonl --max-cost 0.80
```

## 2. 평가셋에 정답이 새어 있는지 검사

한국어 질문을 로컬 모델로 영어로 옮긴 뒤 제목과 겹치는지 본다. **비용 0원.**

```bash
$PY -m evaluation.dataset audit --queries data/eval/test.jsonl \
    --run runs/test_dpo.jsonl --out runs/leakage_test.jsonl
```

## 3. 검색 파이프라인 돌리기

```bash
# 두 채널로 개발용 전체 (이어하기 켜짐 — 중단해도 이어서 돈다)
$PY -m evaluation.pipeline_eval --queries data/eval/dev.jsonl \
    --channels arxiv local_dense --rewriter dpo --k 100 --out runs/dev_dpo.jsonl

# 저장된 결과만 다시 집계 (검색 0회) — 융합 상수·가중치를 바꿔 가며 몇 번이든
$PY -m evaluation.pipeline_eval --report-only runs/dev_dpo.jsonl --rrf-k 30

# 저장된 결과에 재정렬만 다시 적용 (검색 0회)
$PY -m evaluation.pipeline_eval --report-only runs/dev_dpo.jsonl \
    --rerank cross --rerank-depth 300

# 같은 검색 결과 위에서 채널 조합만 갈라 보기 (검색 0회)
$PY -m evaluation.pipeline_eval --report-only runs/dev_dpo.jsonl \
    --use-channels local_dense --rerank cross --rerank-depth 300 \
    --out runs/dev_dpo_localonly.jsonl
```

**`--local-query` 를 건드리지 말 것 (기본 `raw`).** 학습한 변환기가 내놓는 것은 arXiv 문법
문자열이라 로컬 **의미** 검색에 넣으면 불리하고, 무엇보다 서비스(`app.py`)가 로컬 채널에
원본 질문을 넣는다. 기본값이 서비스와 같은 조건이다. `rewritten` 은 "변환이 의미 검색에도
도움이 되는가"를 따로 물을 때만 쓴다.

## 4. 보고와 비교

```bash
# 실행 하나 자세히 (단일 정답 + 만족도 + 난이도·언어·겹침 층화)
$PY -m evaluation.report --run runs/dev_dpo.jsonl \
    --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl

# 여러 실행 짝지어 비교 (공통 문항만, 통계 검정 포함). 기준선을 맨 앞에 둔다
$PY -m evaluation.report --run runs/dev_passthrough.jsonl runs/dev_dpo.jsonl \
    --queries data/eval/dev.jsonl
```

## 5. 서비스 응답 시간 재기

정확도가 아니라 **시간**을 잰다. 사람은 30초를 넘으면 떠난다.

```bash
$PY -m evaluation.pipeline_eval --bench-service --n 5
$PY -m evaluation.pipeline_eval --bench-service --n 5 --rerank-depth 50   # 깊이를 바꿔 비교
```

## 6. "못 찾았다"고 말할 기준선 정하기

로컬 의미 검색은 **어떤 질문에도 후보를 채워서** 돌려준다. 그대로 뿌리면 무관한 논문을
추천으로 포장하게 된다. 몇 점 아래를 무관으로 볼지 등급 정답지로 실측한다.

```bash
$PY -m evaluation.pipeline_eval --calibrate-threshold \
    --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl
```

정한 값은 `app.py` 의 `MIN_RERANK_SCORE` 에 넣는다. **재정렬 모델을 바꾸면 점수 눈금이
달라지므로 반드시 다시 재야 한다.**

## 반드시 지킬 것

1. **탐색은 `dev` 에서만.** `test` 는 확정 판정에만 쓴다. 옛 평가셋은 이 규칙을 어겨
   16회 실행·5회 설정 선택에 소모됐고, 그래서 폐기해야 했다 (ISSUE #25).
2. **기준선(passthrough)을 항상 함께 잰다.** 이 프로젝트의 핵심 주장이
   "변환하면 좋아진다" 이므로, 변환 안 한 값이 없으면 주장이 검증되지 않는다.
3. **p 값이 0.05 보다 크면 '차이가 없다'가 아니라 '있는지 없는지 모른다'.**
   모르는 차이를 근거로 설정을 고르면 시험지만 소모된다.
4. **무거운 실행 전에 의존성을 확인한다.** 몇 초면 끝난다 (ISSUE #35).
