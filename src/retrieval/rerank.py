"""검색 결과 재정렬 — 넓게 가져온 후보를 '사용자 의도와의 관련도'로 다시 줄 세운다.

왜 필요한가 (실측):
arXiv 는 키워드 일치도로 순위를 매기므로 **사용자가 무엇을 원하는지 모른다.** test 300문항에서
정답이 상위 100편 안에 든 비율(Recall@100)은 0.490 인데, 상위 10편 안에 든 비율(Recall@10)은
0.343 이다. 즉 **정답의 14.7%가 이미 후보 안에 있는데 순위가 낮아 사용자 눈에 안 띈다.**
이 격차가 재정렬로 회수할 수 있는 몫이고, 재정렬의 이론적 상한은 0.490 이다.

두 가지 방식을 둔다:

1. **교차 인코더 (CrossEncoderReranker) — 권장**
   질문과 논문을 **함께** 모델에 넣어 관련도를 직접 예측한다. 두 글을 한 번에 보므로
   "이 논문이 이 질문에 답하는가"를 훨씬 정확히 판단한다. 대신 후보 하나마다 모델을
   돌려야 해서 느리다 → 후보를 100편 정도로 좁힌 뒤에만 쓸 수 있는데, 우리가 그 조건이다.
   모델: BAAI/bge-reranker-v2-m3 (568M, 다국어). 한국어 질문 ↔ 영어 초록을 한 모델이 처리한다.

2. **임베딩 유사도 (rerank_by_similarity) — 비교군**
   질문과 논문을 **따로** 숫자로 바꿔 내적을 잰다. 빠르지만, 두 글을 맞대어 보지 않으므로
   교차 인코더보다 정확도가 낮은 것이 정보 검색에서 일반적이다. 어느 쪽이 나은지는
   이 프로젝트 데이터로 직접 비교한다.
"""

from __future__ import annotations

import numpy as np

from src.schemas import ScoredPaper

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """질문과 논문을 함께 읽고 관련도를 매기는 재정렬기."""

    name = "cross_encoder"

    def __init__(self, model_name: str = DEFAULT_RERANKER,
                 device: str | None = None, max_length: int = 512,
                 batch_size: int = 32):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[ScoredPaper],
               top_k: int = 10) -> list[ScoredPaper]:
        """후보를 질문과의 관련도로 다시 정렬해 상위 top_k 를 돌려준다.

        Args:
            query: 사용자가 실제로 입력한 말. **변환된 검색어가 아니라 원본**을 쓴다.
                재정렬의 목적이 '사용자 의도와 맞는가'를 보는 것이기 때문이다.
            candidates: arXiv 등에서 넓게 가져온 후보.
        """
        if not candidates:
            return []
        pairs = [(query, _doc_text(c)) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size,
                                    show_progress_bar=False)
        return _reorder(candidates, np.asarray(scores, dtype=np.float64), top_k)

    def rerank_batch(self, queries: list[str],
                     candidate_lists: list[list[ScoredPaper]],
                     top_k: int = 10) -> list[list[ScoredPaper]]:
        """여러 질문을 한 번에 처리한다 (평가처럼 대량으로 돌릴 때 훨씬 빠르다)."""
        pairs, spans = [], []
        for q, cands in zip(queries, candidate_lists):
            start = len(pairs)
            pairs.extend((q, _doc_text(c)) for c in cands)
            spans.append((start, len(pairs)))
        if not pairs:
            return [[] for _ in queries]
        scores = np.asarray(self.model.predict(pairs, batch_size=self.batch_size,
                                               show_progress_bar=False),
                            dtype=np.float64)
        return [_reorder(cands, scores[s:e], top_k)
                for cands, (s, e) in zip(candidate_lists, spans)]


def _doc_text(c: ScoredPaper) -> str:
    return f"{c.title}\n{c.abstract}".strip()


def _reorder(candidates: list[ScoredPaper], scores: np.ndarray,
             top_k: int) -> list[ScoredPaper]:
    order = np.argsort(-scores)[:top_k]
    out: list[ScoredPaper] = []
    for rank, i in enumerate(order, start=1):
        c = candidates[int(i)]
        out.append(ScoredPaper(paper_id=c.paper_id, score=float(scores[int(i)]),
                               rank=rank, title=c.title, abstract=c.abstract))
    return out


def rerank_by_similarity(query: str, candidates: list[ScoredPaper], embedder,
                         top_k: int = 10) -> list[ScoredPaper]:
    """비교군: 임베딩 유사도로 재정렬 (질문과 논문을 따로 인코딩해 내적).

    교차 인코더보다 빠르지만, 두 글을 맞대어 보지 않아 정확도가 낮은 것이 일반적이다.
    어느 쪽이 이 프로젝트에서 나은지는 실측으로 정한다.
    """
    if not candidates:
        return []
    q_emb = embedder.encode([query])[0]
    c_emb = embedder.encode([_doc_text(c) for c in candidates])
    return _reorder(candidates, c_emb @ q_emb, top_k)
