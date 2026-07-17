"""혼합 검색 (Hybrid): 여러 검색 결과의 순위를 합친다 — RRF.

왜 필요한가: 단어 일치 검색(BM25)과 의미 기반 검색(dense)은 잘하는 질문이 서로 다르다.
- BM25: 정확한 전문 용어가 든 질문(hard)에 강함.
- dense: 일상어·한국어 질문(easy, 한국어)에 강함.
두 결과를 합치면 어느 한쪽만 쓸 때보다 더 안정적인 결과를 낸다.

어떻게 합치나 — RRF(Reciprocal Rank Fusion, 순위 역수 합):
각 검색기가 매긴 '점수'는 단위가 달라(BM25 점수 vs 코사인 유사도) 그대로 더할 수 없다.
그래서 점수는 버리고 '등수'만 쓴다. 한 논문이 어떤 검색에서 r등이면 1/(k+r)점을 받고,
여러 검색에서 받은 점수를 합해 다시 정렬한다. k는 완충 상수(관례상 60).
- 상위권일수록 큰 점수(1등=1/61, 2등=1/62 ...), 하위권 차이는 작아진다.
- 점수 정규화가 필요 없고 파라미터가 k 하나뿐이라 다루기 쉽고 강건하다.
"""

from __future__ import annotations

from src.schemas import ScoredPaper


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredPaper]], k: int = 60, top_k: int = 10
) -> list[ScoredPaper]:
    """여러 검색 결과(각각 순위대로)를 RRF로 합쳐 상위 top_k를 돌려준다.

    Args:
        result_lists: 검색기별 결과 목록. 각 목록은 이미 순위대로(1등부터) 정렬돼 있다고 본다.
        k: RRF 완충 상수.
        top_k: 최종으로 돌려줄 개수.
    """
    fused: dict[str, float] = {}
    meta: dict[str, ScoredPaper] = {}
    for results in result_lists:
        for r in results:
            fused[r.paper_id] = fused.get(r.paper_id, 0.0) + 1.0 / (k + r.rank)
            # 제목·초록 등 표시용 정보는 처음 본 것을 보관
            if r.paper_id not in meta:
                meta[r.paper_id] = r

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out: list[ScoredPaper] = []
    for rank, (pid, score) in enumerate(ordered, start=1):
        m = meta[pid]
        out.append(ScoredPaper(paper_id=pid, score=score, rank=rank,
                               title=m.title, abstract=m.abstract))
    return out


class HybridRetriever:
    """단어 일치 검색기와 의미 기반 검색기를 RRF로 합친 검색기.

    두 검색기는 서로 다른 형태의 검색어를 쓸 수 있으므로(sparse용 / dense용),
    search에 각각의 검색어를 따로 받는다.
    """

    def __init__(self, sparse_retriever, dense_retriever, rrf_k: int = 60):
        self.name = "hybrid"
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(self, sparse_query: str, dense_query: str, k: int,
               pool: int = 50) -> list[ScoredPaper]:
        """각 검색기에서 넉넉히(pool개) 가져와 RRF로 합친 뒤 상위 k편을 돌려준다."""
        sparse_res = self.sparse.search(sparse_query, k=pool)
        dense_res = self.dense.search(dense_query, k=pool)
        return reciprocal_rank_fusion([sparse_res, dense_res], k=self.rrf_k, top_k=k)
