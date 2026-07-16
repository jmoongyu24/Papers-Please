"""순수 파이썬 BM25 검색기 (단어 일치 기반 검색).

BM25는 "검색어의 단어가 문서에 얼마나 들어 있는가"로 점수를 매기는 고전적 방법이다.
- 흔한 단어(the, model 등)는 여러 문서에 나오므로 점수를 적게 준다.
- 드문 단어는 특정 문서를 잘 가려내므로 점수를 많이 준다.
- 한 문서에 같은 단어가 여러 번 나오면 점수가 올라가되, 무한정 오르지는 않게 조절한다.

실제 서비스(모듈 ②)에서는 Elasticsearch를 쓸 예정이지만, 이 순수 파이썬 구현은
추가 설치 없이 어디서나 돌아가므로 개발·평가·테스트의 기본 검색기로 쓴다.

참고: rank_bm25 패키지(reference/langchain_community/retrievers/bm25.py에서 사용)와
같은 Okapi BM25 공식을 numpy로 직접 구현한 것이다.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from src.retrieval.base import tokenize
from src.schemas import Paper, ScoredPaper


class BM25Retriever:
    """코퍼스(논문 목록) 위에서 BM25 점수로 검색한다."""

    def __init__(self, papers: list[Paper], k1: float = 1.5, b: float = 0.75):
        # k1, b: BM25의 표준 조절값. k1은 같은 단어 반복의 영향, b는 문서 길이 보정 정도.
        self.name = "bm25"
        self.papers = papers
        self.k1 = k1
        self.b = b

        # 각 문서를 단어 목록으로 미리 쪼개 둔다.
        self.doc_tokens: list[list[str]] = [tokenize(p.text) for p in papers]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(papers) else 0.0
        self.N = len(papers)

        # 역색인: 단어 -> {문서번호: 그 문서에서의 등장 횟수}
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        for doc_idx, tokens in enumerate(self.doc_tokens):
            for term, freq in Counter(tokens).items():
                self.postings[term][doc_idx] = freq

        # idf: 단어가 몇 개 문서에 나오는지로 계산 (드물수록 큼)
        self.idf: dict[str, float] = {}
        for term, posting in self.postings.items():
            df = len(posting)  # 이 단어가 나온 문서 수
            # BM25 표준 idf 공식. 음수가 되지 않도록 +1 형태를 쓴다.
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[ScoredPaper]:
        """검색어와 점수가 높은 상위 k개 논문을 순위와 함께 돌려준다."""
        q_terms = tokenize(query)
        scores = np.zeros(self.N, dtype=np.float64)

        for term in q_terms:
            posting = self.postings.get(term)
            if not posting:
                continue  # 코퍼스에 없는 단어는 건너뜀
            idf = self.idf[term]
            for doc_idx, freq in posting.items():
                dl = self.doc_len[doc_idx]
                # BM25 핵심 공식
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * (freq * (self.k1 + 1)) / denom

        # 점수 높은 순으로 상위 k개. 점수가 0이면 (전혀 안 맞으면) 제외.
        order = np.argsort(-scores)[:k]
        results: list[ScoredPaper] = []
        rank = 0
        for doc_idx in order:
            if scores[doc_idx] <= 0:
                break
            rank += 1
            p = self.papers[doc_idx]
            results.append(
                ScoredPaper(
                    paper_id=p.id,
                    score=float(scores[doc_idx]),
                    rank=rank,
                    title=p.title,
                    abstract=p.abstract,
                )
            )
        return results
