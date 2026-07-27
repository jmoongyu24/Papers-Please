"""검색 결과 의미 재정렬 (Semantic re-ranking) — 1단계 (a).

arXiv는 키워드 텍스트 점수로 순위를 매겨 '사용자 의도'를 모른다(ISSUE #7). 그래서 넓게
많이(예: top-30) 가져온 뒤, **bge-m3 임베딩으로 사용자 쿼리와의 의미 유사도**로 다시 정렬해
상위 k개(예: 10개)를 추린다. "arXiv로 넓게 뽑기(재현율) + 의미로 순서 잡기(정확도)"의 분리.

임베딩 모델은 이미 dense 검색에 쓰던 것(bge-m3, 다국어)을 재사용한다. 다국어라 한국어
쿼리와 영어 초록도 같은 의미 공간에서 비교된다.
"""

from __future__ import annotations

import numpy as np

from src.schemas import ScoredPaper


def rerank_by_similarity(query: str, candidates: list[ScoredPaper], embedder,
                         top_k: int = 10) -> list[ScoredPaper]:
    """후보 논문들을 query와의 의미 유사도로 재정렬해 상위 top_k를 돌려준다.

    Args:
        query: 사용자 의도를 나타내는 문자열(원본 검색어 등).
        candidates: arXiv에서 넓게 가져온 후보들.
        embedder: bge-m3 Embedder (정규화 임베딩 → 내적이 코사인 유사도).
        top_k: 최종 개수.
    """
    if not candidates:
        return []
    q_emb = embedder.encode([query])[0]                       # (d,)
    texts = [f"{c.title}\n{c.abstract}" for c in candidates]
    c_emb = embedder.encode(texts)                            # (n, d)
    sims = c_emb @ q_emb                                       # 정규화돼 있어 코사인 유사도

    order = np.argsort(-sims)[:top_k]
    out: list[ScoredPaper] = []
    for rank, i in enumerate(order, start=1):
        c = candidates[i]
        out.append(ScoredPaper(paper_id=c.paper_id, score=float(sims[i]), rank=rank,
                               title=c.title, abstract=c.abstract))
    return out
