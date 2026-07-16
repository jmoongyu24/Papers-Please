# 성능 평가 모듈 (모듈 ③) 사용법

이 폴더는 "쿼리 변환을 하면 검색 결과가 변환 전보다 얼마나 더 좋아지는가"를 숫자로
재는 코드다. 전체 설계 설명은 [../docs/MODULE_SPECIFICATION.md](../docs/MODULE_SPECIFICATION.md)
3번 모듈을 참고. 여기서는 실행 방법만 정리한다.

파이썬은 프로젝트 가상환경을 쓴다:
```
PY=/home/jmoongyu/venvs/paper_py310/bin/python
```

## 0. 지금 바로 돌려보기 (데모)

추가 설치 없이, 미리 넣어둔 샘플 데이터로 전체 흐름을 한 번에 실행하고 비교표를 본다.
```
$PY -m evaluation.run_experiments --demo
```
'변환 전(passthrough)'과 '임시 변환기(mock_expander)'를 BM25 검색으로 비교한 표가 나온다.
(mock_expander는 진짜 Qwen3-4B 변환기가 준비되기 전까지 쓰는 규칙 기반 임시 대역이다.)

## 1. 평가 데이터셋 만들기

세 단계로 만든다. 각 단계는 JSON Lines 파일 하나를 입력받아 하나를 내보낸다.

```
# (1) 논문에서 일상어 질문을 거꾸로 생성
#     - 진짜 생성: OPENAI_API_KEY 설정 후 --mock 없이
#     - 점검용: --mock (키 없이 규칙 기반 가짜 생성)
$PY -m evaluation.generate_queries \
    --corpus data/corpus/corpus-v1.jsonl \
    --out data/eval/queries.raw.jsonl --n 500

# (2) 품질 나쁜 질문 걸러내기 (용어 베끼기·중복·길이)
$PY -m evaluation.filter_queries \
    --corpus data/corpus/corpus-v1.jsonl \
    --queries data/eval/queries.raw.jsonl \
    --out data/eval/queries.filtered.jsonl

# (3) 개발용/시험용으로 논문 단위 분할 (한 논문의 질문은 한쪽에만)
$PY -m evaluation.build_splits \
    --queries data/eval/queries.filtered.jsonl \
    --out-dir data/eval --test-ratio 0.6
```

> `filter_queries`의 `--max-df`는 코퍼스 크기에 맞춰 조정한다. '드문 전문 용어'를
> 절대 개수와 비율(`--max-df-ratio`) 두 기준으로 함께 판단하므로, 큰 코퍼스에서는
> 기본값으로 충분하다.

## 2. 실험 실행 + 리포트

```
# 여러 (변환 방식 × 검색 방식) 조합을 돌려 결과를 runs/ 에 저장
$PY -m evaluation.run_experiments \
    --corpus data/corpus/corpus-v1.jsonl \
    --queries data/eval/dev.jsonl \
    --rewriters passthrough mock_expander \
    --backends bm25

# 저장된 결과로 비교표 + '변환 전 vs 변환 후' 통계 검정 출력
$PY -m evaluation.report --queries data/eval/dev.jsonl
```

## 지표 설명

- **Recall@K(재현율)**: 정답 논문이 상위 K개 안에 들어왔으면 성공. 그 비율.
- **MRR@10(평균 역순위)**: 정답이 몇 등인지까지 반영 (1등=1점, 2등=0.5점 …).
- **변환 전 vs 후**: 같은 질문끼리 짝지어 비교하고, 그 차이가 우연이 아닌지
  부트스트랩(무작위 재추출)으로 p값과 95% 신뢰구간을 낸다. 자세한 원리는
  `metrics.py`의 주석 참고.

## 앞으로 붙일 것 (아직 임시/미구현)

- 진짜 계층적 변환기(Qwen3-4B) → `src/rewriter/hierarchical.py` (지금은 mock_expander가 대역)
- 의미 기반 검색(FAISS)·혼합 검색 → `--backends` 에 dense, hybrid 추가
- 실제 코퍼스(Kaggle arXiv 스냅샷) → `src/retrieval/corpus.py`의 `build_corpus_from_kaggle`
