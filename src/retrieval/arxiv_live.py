"""실시간 arXiv 검색기 — 변환된 검색어로 arXiv API에 직접 질의해 논문을 가져온다.

로컬 코퍼스(평가 전용)와 달리, 이건 **실제 서비스용**이다. 데이터베이스가 필요 없고
검색할 때마다 arXiv에서 최신 논문까지 가져온다. 다른 검색기와 같은 인터페이스
(`search(query, k) -> list[ScoredPaper]`)를 따른다.

코드로 확인한 주의사항 반영(reference/langchain_community README):
- 결과 개수는 max_results 로 지정(우리가 k로).
- 검색어는 300자에서 잘림 → 최종 검색어만 넘긴다.
- arXiv API는 관련도 점수를 주지 않고 '순위'만 준다 → score는 1/순위로 채운다.
- 요청 간 3초 간격 권장(arxiv 패키지 Client 기본값). 같은 검색어는 캐싱해 재호출을 줄인다.
"""

from __future__ import annotations

from src.schemas import ScoredPaper

ARXIV_MAX_QUERY_LEN = 300


class ArxivLiveRetriever:
    """arXiv API로 실시간 검색."""

    name = "arxiv_live"

    def __init__(self, delay_seconds: float = 3.0, num_retries: int = 3):
        import arxiv

        self._arxiv = arxiv
        self._client = arxiv.Client(
            page_size=100, delay_seconds=delay_seconds, num_retries=num_retries
        )
        self._cache: dict[tuple[str, int], list[ScoredPaper]] = {}

    def search(self, query: str, k: int) -> list[ScoredPaper]:
        query = (query or "").strip()[:ARXIV_MAX_QUERY_LEN]
        if not query:
            return []
        key = (query, k)
        if key in self._cache:
            return self._cache[key]

        search = self._arxiv.Search(
            query=query, max_results=k,
            sort_by=self._arxiv.SortCriterion.Relevance,
        )
        results: list[ScoredPaper] = []
        try:
            for rank, r in enumerate(self._client.results(search), start=1):
                results.append(ScoredPaper(
                    paper_id=r.get_short_id(),          # 예: "1706.03762v7"
                    score=1.0 / rank,                   # arXiv은 점수 없음 → 순위 기반
                    rank=rank,
                    title=r.title.strip(),
                    abstract=r.summary.strip(),
                ))
                if len(results) >= k:
                    break
        except Exception:
            # 네트워크/파싱 오류 시 빈 결과(서비스가 죽지 않도록)
            results = []

        self._cache[key] = results
        return results
