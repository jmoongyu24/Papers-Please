# 개발 계획서 — LLM 기반 계층적 쿼리 변환을 활용한 비전문가 학술 논문 검색 시스템

작성일: 2026-07-06. 계획서(제안서)와 사용자 시나리오를 코드·공개 사실 기반으로 구체화한 실행 계획.

---

## 0. 핵심 가설과 검증 대상

> **가설**: 비전문가의 일상어 쿼리를 LLM이 '일상어 → 중간 개념 → 학술 용어' 계층으로
> 변환하면, 키워드 기반 검색 엔진(arXiv API)의 검색 성능이 유의미하게 향상된다.

검증해야 할 것은 정확히 두 가지 비교다.

| 비교 | 질문 |
|---|---|
| 원본 쿼리 vs 변환 쿼리 | 쿼리 변환이 검색 성능을 올리는가? (핵심) |
| 프롬프트만 vs 파인튜닝 | 파인튜닝이 프롬프트 엔지니어링 대비 추가 이득이 있는가? (차별점) |

두 번째 비교가 졸업작품의 기술적 차별점이다. 파인튜닝 없이 프롬프트만으로 되는 일이면
"GPT 프롬프트 데모"에 그치므로, **프롬프트-only 베이스라인을 반드시 먼저 만들고**
파인튜닝 모델과 정량 비교해야 한다. (파인튜닝이 지더라도 그 자체가 보고 가치 있는 결과)

---

## 1. 객관적 사실 정리

### 1.1 arXiv 검색 엔진의 동작 방식 (프로젝트의 존재 근거)

- arXiv API(`http://export.arxiv.org/api/query`)는 **키워드(lexical) 매칭** 검색이다.
  임베딩 기반 의미 검색이 아니므로, 쿼리와 문서의 **용어가 겹쳐야** 검색된다.
  → 일상어 쿼리가 실패하는 이유이자, 학술 용어로의 변환이 유효한 이유.
- 지원 문법 (공식 API User Manual 기준):
  - 필드 프리픽스: `ti:`(제목) `abs:`(초록) `au:`(저자) `cat:`(분류) `all:`(전체)
  - 불리언: `AND`, `OR`, `ANDNOT` + 괄호 그룹핑
  - 구(phrase) 검색: 큰따옴표 `ti:"attention is all you need"`
  - 정렬: `relevance` / `lastUpdatedDate` / `submittedDate`
- 호출 제약: 요청 간 3초 간격 권장. `arxiv` 파이썬 패키지(2.4.1)의 `Client`가
  `delay_seconds=3.0`, `page_size=100`을 기본 처리.

### 1.2 LangChain `ArxivRetriever`의 실제 동작 (코드 확인, `reference/langchain_community/`)

| # | 사실 | 프로젝트에 주는 영향 |
|---|---|---|
| 1 | 결과 개수는 `top_k_results`(기본 3)가 결정. `load_max_docs`는 summary 경로에서 미사용 | 현재 `src/arxiv_search.py`의 `load_max_docs=1`은 효과 없음(3개 반환). **평가 시 `top_k_results=10` 필요** |
| 2 | 쿼리는 300자에서 절단(`ARXIV_MAX_QUERY_LENGTH`) | LLM의 CoT 전체가 아닌 **최종 쿼리만** 검색에 전달 |
| 3 | `get_full_documents=True` 경로는 쿼리에서 `:`, `-`를 제거 | 필드 프리픽스(`ti:`) 쿼리가 깨짐. **`get_full_documents=False` 유지** |
| 4 | arXiv ID 형태 입력은 자동으로 ID 검색으로 전환(`is_arxiv_identifier`) | 시나리오 3번(entry ID 매핑 → 재검색)은 별도 구현 불필요 |
| 5 | metadata `Published`는 갱신일(`updated`)이지 최초 게재일이 아님 | 날짜 필터/표시 시 주의 |

### 1.3 데이터 소스

- **Kaggle arXiv 메타데이터 스냅샷** (`Cornell-University/arxiv`, arXiv 공식 제공):
  전체 논문의 제목·초록·분류·ID를 JSON Lines 하나로 제공, 주기적 갱신.
  → 코퍼스 구축 시 API 크롤링(3초/건)이 불필요. **수집 단계 비용을 사실상 0으로 만든다.**
- 라이브 arXiv API는 데모·실서비스 경로로 사용.

### 1.4 방법론의 학술적 근거 (보고서 인용용)

- 어휘 불일치 문제(vocabulary mismatch): Furnas et al., 1987 — 문제 정의의 고전.
- LLM 쿼리 재작성: Query2doc (Wang et al., 2023), Rewrite-Retrieve-Read (Ma et al., 2023),
  HyDE (Gao et al., 2022) — "LLM으로 쿼리를 확장/재작성하면 lexical·dense 검색이 모두
  개선된다"는 선행 근거.
- 합성 학습 데이터(문서→쿼리 역생성): doc2query (Nogueira et al., 2019),
  InPars (Bonifacio et al., 2022), Promptagator (Dai et al., 2022) —
  '논문에서 일상어 쿼리를 역으로 생성'하는 본 계획의 데이터 구축 방식과 동일 계열.
- Chain-of-Thought: Wei et al., 2022.

---

## 2. 시스템 아키텍처

```
사용자 일상어 쿼리
      │
      ▼
┌─────────────────────┐   CoT 단계별 출력(화면 표시용)
│ query_rewriter      │──► [의도 해석] → [핵심 개념] → [학술 용어 매핑]
│ (HF 소형 LLM+LoRA)  │
└─────────┬───────────┘
          │ 최종 학술 쿼리 (≤300자, 필드 프리픽스 가능)
          ▼
┌─────────────────────┐
│ arxiv_search        │──► arXiv API (relevance 정렬, top_k=10)
│ (ArxivRetriever 래핑)│
└─────────┬───────────┘
          │ [(제목, 초록, entry_id, ...)] × k
          ▼
┌─────────────────────┐
│ relevance_scorer    │──► 원본 쿼리 ↔ 제목+초록 임베딩 유사도로 재정렬(선택)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ pipeline / app      │──► CoT 과정 + 결과 카드(제목/초록 요약) 출력
└─────────────────────┘
```

### 모듈별 책임 (기존 파일명 유지)

| 파일 | 책임 | 핵심 인터페이스 |
|---|---|---|
| `src/config.py` (신규) | 모델명·경로·상수 한곳에 | `REWRITER_MODEL`, `EMBED_MODEL`, `TOP_K` |
| `src/arxiv_search.py` | arXiv 검색 래핑 | `search(query, k=10) -> list[Paper]`, ID 자동 검색 |
| `src/query_rewriter.py` | CoT 계층 변환 | `rewrite(query) -> RewriteResult(steps, final_query)` |
| `src/relevance_scorer.py` | 쿼리↔논문 의미 유사도 | `score(query, papers) -> list[float]` |
| `src/pipeline.py` (신규) | 재작성→검색→(재정렬)→포맷 통합 | `run(query) -> SearchOutput` (터미널 데모 진입점) |
| `data/collect_corpus.py` (신규) | Kaggle 스냅샷에서 cs.* 필터링 | 코퍼스 jsonl 생성 |
| `data/generate_queries.py` (신규) | OpenAI로 일상어 쿼리+CoT trace 합성 | 학습/평가 데이터 jsonl |
| `data/build_splits.py` (신규) | **논문 단위** train/val/test 분할 | 누수 방지 |
| `training/sft_lora.py` (신규) | LoRA/QLoRA SFT (peft+trl) | 어댑터 저장 |
| `experiments/metrics.py` (신규) | Hit@K, MRR, (NDCG) | 순수 함수 |
| `experiments/baseline_search.py` | 원본 쿼리 × 백엔드 실행·결과 캐싱 | jsonl 결과 |
| `experiments/rewritten_search.py` | 변환 쿼리 × 백엔드 실행·결과 캐싱 | jsonl 결과 |
| `experiments/compare_result.py` | 지표 집계·비교표 생성 | 표/그래프 |
| `app.py` | Streamlit 데모 (마지막 단계) | CoT 시각화 + 결과 카드 |

원칙: **재작성기와 검색기는 서로를 모르게** 하고 `pipeline.py`에서만 조립한다.
실험 스크립트는 API 호출 결과를 반드시 디스크에 캐싱한다(재현성 + rate limit 대응).

---

## 3. 작업 패키지 (WP)

### WP1. 검색 모듈 정비 (0.5주)

- `src/arxiv_search.py`를 클래스로 재작성: `top_k_results` 명시(버그 수정),
  `get_full_documents=False` 고정, 결과를 `Paper` dataclass(제목/초록/entry_id/날짜)로 정규화.
- 완료 기준: 동일 쿼리 재호출 시 캐시 사용, k=10 반환 확인 테스트.

### WP2. 코퍼스 + 평가/학습 데이터셋 구축 (1.5주)

1. Kaggle 스냅샷 다운로드 → `cs.CL, cs.CV, cs.LG, cs.IR` 등에서 5천~2만 편 필터링.
2. `generate_queries.py`: 논문(제목+초록)마다 OpenAI API(gpt-4o-mini 등 저가 모델)로
   - 페르소나 프롬프트("도메인 지식이 없는 학부생이 이 논문을 찾고 싶을 때 던질 질문")로
     **일상어 쿼리 1~3개** (한국어/영어 혼합) 생성
   - 동시에 **CoT 변환 trace**(의도→개념→학술 용어→최종 쿼리) 생성 → SFT 정답 라벨
3. 산출 스키마: `{arxiv_id, title, abstract, lay_query, cot_steps, final_query, lang}`
4. `build_splits.py`: **논문 단위** 분할(train 80 / val 10 / test 10). 같은 논문의 쿼리가
   train과 test에 나뉘면 누수.
5. 품질 관리: 무작위 100건 직접 검수 + 동기 10~20명에게 실제 일상어 쿼리를 받아
   **소규모 실사용자 평가셋**(50~100건)을 별도 구축 → 합성 데이터 편향 견제.
- 예상 비용: 5천 쿼리 생성 기준 수 달러 수준.
- 완료 기준: 평가셋(합성 200~300건 + 실사용자 50~100건) 고정 & 버전 태깅.

### WP3. 프롬프트 기반 재작성 베이스라인 (1주) — **첫 번째 성능 숫자가 나오는 지점**

- `query_rewriter.py`: HF 소형 LLM + few-shot CoT 프롬프트로 계층 변환.
  출력 형식을 고정해 파싱 가능하게:
  ```
  [의도] 사용자는 ~을 찾고 있다
  [핵심 개념] ...
  [학술 용어] ...
  [최종 쿼리] all:"..." AND cat:cs.CL   ← 이 줄만 검색에 전달 (≤300자)
  ```
- `experiments/`로 원본 vs 프롬프트 재작성 비교 실행 → 첫 결과표.
- 완료 기준: test셋에서 Hit@10, MRR 리포트 1장.

### WP4. LoRA 파인튜닝 (2주)

- 베이스 모델(4절)에 WP2의 CoT trace로 SFT. peft(LoRA r=16 안팎) + trl `SFTTrainer`.
  GPU 미확보/VRAM 부족 시 QLoRA(4bit) 또는 Colab.
- 완료 기준: 프롬프트-only 대비 지표 비교표. (지면 원인 분석 서술)

### WP5. 고정 코퍼스 Hybrid Retrieval 트랙 (1.5주)

- 라이브 arXiv API는 코퍼스가 매일 변해 재현성이 없으므로, **정량 평가의 주 트랙은
  WP2 코퍼스(고정) 위의 로컬 검색**으로 한다. 라이브 API는 데모 + 보조 평가.
- 구성: BM25(`rank_bm25`, 참고: `reference/.../retrievers/bm25.py`) + FAISS dense
  (bge-m3 임베딩) → RRF(Reciprocal Rank Fusion, k=60)로 융합.
  Elasticsearch는 Docker 단일 노드로 시간이 허락하면 추가(BM25와 결과 유사).
- 완료 기준: 3×4 평가 매트릭스 완성 —
  {원본, 프롬프트 재작성, 파인튜닝 재작성} × {arXiv API, BM25, FAISS, Hybrid}.

### WP6. UI + 마무리 (1주+)

- Streamlit(`app.py`): 입력 → CoT 단계 애니메이션 표시 → 결과 카드.
  (계획서의 React는 시간 남으면; Streamlit으로 심사 데모는 충분)
- 결과 정리, 그래프, 보고서.

**순서 의존성**: WP1 → WP2 → WP3 → (WP4 ∥ WP5) → WP6.
WP3까지 끝나면 어떤 시점에든 "동작하는 시스템 + 비교 숫자"가 존재한다.

---

## 4. LLM 선정

### 4.1 재작성기 (파인튜닝 대상)

요구사항: 로컬 배포 가능한 소형(≤4B), 한국어 일상어 입력 이해, 영어 학술 용어 출력,
파인튜닝 허용 라이선스.

| 후보 | 크기 | 라이선스 | 비고 |
|---|---|---|---|
| **Qwen3-4B (Instruct)** ⭐1순위 | 4B | Apache-2.0 | 한국어 포함 다국어 우수, 제약 없는 라이선스 |
| Qwen3-1.7B | 1.7B | Apache-2.0 | VRAM 빠듯할 때 대안 |
| EXAONE-3.5-2.4B-Instruct | 2.4B | 연구용 라이선스 | 동급 최강 한국어(LG). 비상업 연구용 조건 확인 필요 |
| Llama-3.2-3B-Instruct | 3B | Llama license | 한국어가 상대적으로 약함 |

- Qwen2.5-3B는 별도 연구 라이선스(비 Apache)라 제외. Qwen3 계열이 전부 Apache-2.0.
- 4B 기준 QLoRA(4bit)는 VRAM ~8GB에서 학습 가능, 추론은 4bit 양자화 시 ~4GB.

### 4.2 임베딩 모델 (역할 분리 필수 — 6.3 순환성 참조)

| 용도 | 모델 | 이유 |
|---|---|---|
| dense retrieval (WP5) | `BAAI/bge-m3` | 다국어(한국어↔영문 초록 교차 검색) |
| 만족도 proxy 평가 (6.3) | `intfloat/multilingual-e5-large` | **검색용과 다른 모델**로 순환 평가 방지 |

### 4.3 데이터 생성기

- OpenAI 저가 모델(gpt-4o-mini급)로 충분. 생성물 100건 샘플 검수 후 전체 생성.

---

## 5. 계층적 CoT 변환 설계

시나리오 2번("변환 과정을 보여준다")과 학습 라벨 형식을 일치시킨다. 즉 SFT 라벨 자체가
아래 4단계 구조이므로, 데모에서 모델 출력을 그대로 단계별 표시하면 된다.

```
입력: "요즘 사진 보고 글로 설명해주는 AI는 어떻게 만들어?"

[의도] 이미지를 입력받아 자연어 설명을 생성하는 기술의 원리를 알고 싶음
[핵심 개념] 이미지 이해 + 텍스트 생성의 결합
[학술 용어] image captioning, vision-language model, multimodal learning
[최종 쿼리] abs:"image captioning" AND abs:"vision-language" AND cat:cs.CV
```

- 최종 쿼리 규칙: 300자 이하, 필드 프리픽스·불리언 사용, 콜론 유지가 필요하므로
  검색은 항상 summary 경로(`get_full_documents=False`)로.
- 파싱: `[최종 쿼리]` 라인만 정규식 추출. 파싱 실패 시 마지막 줄 폴백 + 실패율 로깅
  (파싱 실패율 자체가 파인튜닝 효과 지표 중 하나).

---

## 6. 평가 설계

### 6.1 지표 — 계획서 지표의 통계적 문제와 수정안

평가셋이 '쿼리 1개 ↔ 정답 논문 1개' 구조이면:

- Precision@K = Recall@K ÷ K (독립 정보 없음)
- F1@K = 위 둘의 조합 (독립 정보 없음)
- 이진 단일 정답의 NDCG@K = 1/log₂(rank+1) (순위 지표로서 MRR과 중복)

→ **주 지표: Hit@K (K=1,5,10) + MRR@10**. Precision/F1은 보고서에서
"단일 정답 설정에서는 축퇴됨"을 명시하고 제외하거나, 정답을 논문 여러 편으로 확장
(예: 관련도 2단계 라벨)한 경우에만 NDCG를 사용. 이 수정 근거를 보고서에 쓰면
평가 설계의 엄밀성으로 가점 요소가 된다.

### 6.2 2-트랙 평가

| 트랙 | 코퍼스 | 용도 | 재현성 |
|---|---|---|---|
| A. 고정 로컬 코퍼스 | WP2 스냅샷 (5천~2만 편) | **주 정량 평가** (BM25/FAISS/Hybrid) | 완전 재현 가능 |
| B. 라이브 arXiv API | 전체 arXiv | 데모 + 보조 평가 | 날짜 고정·응답 캐싱으로 부분 확보 |

### 6.3 만족도 proxy (사용자 가정의 검증)

가정: "원본 쿼리 ↔ 검색된 논문(제목+초록) 의미 유사도가 높으면 만족도도 높다."

- 측정: multilingual-e5-large 코사인 유사도. **단, dense retrieval에 쓴 모델(bge-m3)과
  분리** — 같은 모델로 검색하고 평가하면 검색기가 잘한 게 아니라 평가자가 자기 자신을
  채점하는 순환이 된다.
- 가정 자체의 검증: 실사용자 평가셋(WP2-5)에서 사람 관련도 평점(5점 척도)과 proxy
  점수의 상관(Spearman)을 1회 측정해 보고. 상관이 낮으면 proxy는 보조 지표로 강등.

### 6.4 최종 결과표 (목표 산출물)

```
                     arXiv API   BM25   FAISS   Hybrid(RRF)
원본 일상어 쿼리        h@10/MRR    ...     ...      ...
프롬프트 CoT 재작성        ...      ...     ...      ...
파인튜닝 CoT 재작성        ...      ...     ...      ...
```

---

## 7. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| GPU 미확보/부족 | QLoRA 4bit(~8GB) → Colab T4 → 최악엔 프롬프트-only로 범위 축소 |
| 합성 쿼리 ≠ 실제 비전문가 쿼리 | 실사용자 쿼리 50~100건 별도 평가셋(WP2-5) |
| 파인튜닝 < 프롬프트 | 그 자체가 유효한 비교 결과. 원인 분석(데이터 품질/규모)으로 서술 |
| arXiv rate limit / 응답 변동 | 3초 딜레이 준수, 모든 응답 디스크 캐싱, 평가 날짜 고정 |
| Elasticsearch 구축 시간 | rank_bm25로 동등 실험 선행, ES는 선택 사항으로 격하 |
| OpenAI 비용 | gpt-4o-mini급 사용, 생성 전 100건 파일럿으로 프롬프트 확정 |

---

## 8. 참고 코드

- `reference/langchain_community/` — ArxivRetriever/Wrapper 원본, BM25/FAISS/ES 참고 구현.
  주의사항 5가지는 해당 폴더 README에 정리됨.
