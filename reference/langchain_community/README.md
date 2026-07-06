# langchain-community 참고 코드 (발췌)

출처: https://github.com/langchain-ai/langchain-community (MIT License, 원문 LICENSE 동봉)

실행용 코드가 아니라 **읽기용 참고 사본**이다. 실제 실행은 pip로 설치된
`langchain-community==0.4.1` 패키지를 사용한다. (다운로드 시점의 main 브랜치
발췌라 설치된 0.4.1과 세부 차이가 있을 수 있음)

## 파일 설명

| 파일 | 용도 |
|---|---|
| `utilities/arxiv.py` | `ArxivAPIWrapper` — arXiv API 호출의 실제 로직. 쿼리 300자 절단(`ARXIV_MAX_QUERY_LENGTH`), arXiv ID 자동 감지(`is_arxiv_identifier`), 결과 수 제어(`top_k_results`) 등 핵심 동작 |
| `retrievers/arxiv.py` | `ArxivRetriever` — 위 wrapper를 감싼 retriever. `get_full_documents=False`면 `get_summaries_as_docs()` 호출 |
| `document_loaders/arxiv.py` | `ArxivLoader` — 논문 수집(코퍼스 구축) 시 참고 |
| `retrievers/bm25.py` | `BM25Retriever` — rank_bm25 기반 순수 파이썬 BM25. Elasticsearch 없이 sparse retrieval 실험 가능 |
| `retrievers/elastic_search_bm25.py` | `ElasticSearchBM25Retriever` — 계획서의 Elasticsearch 트랙 구현 시 참고 |
| `vectorstores/faiss.py` | `FAISS` 벡터스토어 — dense retrieval 트랙 구현 시 참고 |
| `tests/test_arxiv.py` | ArxivRetriever 공식 통합 테스트 = 사용 예제 |

## 코드에서 확인된 주의사항 (프로젝트에 직접 영향)

1. **결과 개수는 `top_k_results`가 결정한다.** `load_max_docs`는 summary 경로에서
   사용되지 않는다 (`_fetch_results()`가 `max_results=self.top_k_results`만 사용).
   Recall@10을 재려면 `top_k_results=10`으로 생성해야 한다.
2. **쿼리는 300자에서 잘린다** (`ARXIV_MAX_QUERY_LENGTH`). CoT 전체가 아니라
   최종 변환 쿼리만 검색에 넘겨야 한다.
3. **`get_full_documents=True`(= `load()`) 경로는 쿼리에서 `:`와 `-`를 제거한다.**
   `ti:"..."` 같은 필드 프리픽스 쿼리가 깨지므로, 필드 검색을 쓰려면 반드시
   `get_full_documents=False`(기본값) 유지.
4. **arXiv ID 형태 쿼리는 자동으로 ID 검색으로 전환된다** (`is_arxiv_identifier`).
   "2202.09741" 같은 입력은 `id_list` 검색이 되므로, 시나리오의
   'entry ID 매핑 → 재검색' 단계를 별도 구현 없이 그대로 쓸 수 있다.
5. metadata의 `Published`는 `result.updated`(갱신일)이지 최초 게재일이 아니다.
