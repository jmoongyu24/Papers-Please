# Papers-Please

한국어나 일상어로 물어봐도 arXiv 에서 원하는 논문을 찾아주는 검색 서비스.
비전문가를 위한 쿼리 변환 + 의미 검색 논문 검색 프로젝트 (졸업작품).

## 무엇을 푸는가

논문을 찾을 때 **자기가 읽고 싶은 논문이 실제로 어떤 검색어로 쓰여 있는지 모른다**는 것이
문제다. "이미지 설명 생성 모델"이라고 쳐도 논문은 `image captioning` 이라고 쓴다.
한국어로 물으면 arXiv 는 아예 못 찾는다(원본 한국어 질문의 88.2% 가 결과 0건).

```
질문 → 한국어면 영어로 옮김 (Qwen3-4B)
     ├─ 로컬 의미 검색 71만 편        정확도 (성능의 대부분)
     └─ arXiv 실시간 키워드 검색      최신성·범위
          → 후보 합치기 → 교차 인코더 재정렬 → 추천 이유 생성 → 10편
```

> **2026-08-16 주의:** 지금 `app.py` 는 arXiv 채널의 후보를 전부 버리고 있고, 학습한 변환기
> (dpo)도 결과에 영향을 주지 못함(docs/ISSUE.md #39). 구조를 확정하는 것이 현재 1순위임.

## 실행

```bash
PY=/home/jmoongyu/venvs/paper_py310/bin/python

# 웹 화면
$PY -m streamlit run app.py          # → localhost:8501

# 평가 (개발용에서 탐색)
$PY -m evaluation.pipeline_eval --queries data/eval/dev.jsonl \
    --channels arxiv local_dense --rewriter dpo --k 100 --out runs/dev_dpo.jsonl
$PY -m evaluation.report --run runs/dev_dpo.jsonl \
    --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl

# 테스트
$PY -m pytest tests/ -q
```

## 폴더

| 위치 | 무엇 |
|---|---|
| `app.py` | 웹 화면 (Streamlit) |
| `src/rewriter/` | 쿼리 변환 |
| `src/retrieval/` | 검색 (코퍼스 · 로컬 색인 · arXiv · 순위) |
| `src/recommend_agent/` | 추천 이유 생성 |
| `evaluation/` | 평가셋 제작 · 파이프라인 실행 · 지표 · 보고 |
| `training/` | 변환기 학습 (SFT → DPO) |
| `data/eval/` | 평가셋 4개 (`dev` · `test` · `grades_dev` · `grades_test`) |

**디렉터리마다 코드 파일은 4개를 넘지 않는다.** 새 기능은 되도록 기존 파일 안에 넣는다.

## 문서

| 문서 | 무엇 |
|---|---|
| [docs/PROGRESS.md](docs/PROGRESS.md) | 지금 상태 · 모듈별 진행 · 다음 우선순위 · 날짜별 로그 |
| [docs/PLAN.md](docs/PLAN.md) | 큰 그림, 무엇을 왜 하는가 |
| [docs/MODULE_SPECIFICATION.md](docs/MODULE_SPECIFICATION.md) | 모듈별 구조와 인터페이스 |
| [docs/ISSUE.md](docs/ISSUE.md) | 겪은 문제와 해결 과정 (철회한 결론 포함) |
| [docs/ARXIV_API_POLICY.md](docs/ARXIV_API_POLICY.md) | arXiv 이용 정책 준수 사항 |
| [evaluation/README.md](evaluation/README.md) | 평가 실행 방법 |

코드를 고치기 전에 **해당 파일 맨 위 설명글**을 읽을 것. 왜 그렇게 만들었는지가 거기 있고,
"~하면 안 된다"고 적힌 것은 대부분 실제로 겪은 사고다.
