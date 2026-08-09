"""로컬 논문 모음 위에서 arXiv 검색을 흉내 내는 채점기.

왜 필요한가:
학습 데이터를 만들 때 "이 검색어가 정답 논문을 실제로 찾아내는가"를 채점해야 한다. 지금은
그 채점을 arXiv API로 하는데, 후보 검색어 하나마다 호출이 1회씩 들어간다(지난 1회 생성에서
실측 1,147회). 데이터를 늘리면 수만 회가 되어 arXiv 정책 취지에 어긋나고, 3초 간격 때문에
느리기까지 하다. arXiv는 이런 용도로 벌크 데이터(Kaggle 논문 정보 모음)를 따로 제공한다.

왜 BM25를 그냥 쓰면 안 되는가:
후보 검색어는 arXiv 문법으로 되어 있다.

    all:"원본 질문" OR abs:"학술용어1" OR abs:"학술용어2"

arXiv에서 `abs:"구"` 는 **그 구가 초록에 그대로 들어 있어야** 걸리고, `OR` 는 하나라도
걸리면 후보에 넣는다는 뜻이다. 그런데 BM25는 구를 단어로 쪼개 점수를 매기므로
`abs:"graph neural network"` 가 graph / neural / network 세 단어로 흩어진다. 즉 arXiv와
전혀 다른 것을 재게 된다.

그래서 두 단계로 나눈다:
  1) **걸림 판정** — 따옴표 구를 하나라도 그대로 포함한 논문만 후보로 남긴다.
     이건 arXiv와 **정확히 같게** 만들 수 있다.
  2) **순위 매기기** — 그 후보들 안에서만 BM25로 정렬한다.
     arXiv의 내부 순위 공식은 공개돼 있지 않으므로 이 부분만 근사다.

이렇게 나눠 두면 나중에 arXiv와 결과가 어긋날 때 **어느 단계가 원인인지** 가릴 수 있다.
"""

from __future__ import annotations

import re

import numpy as np

from src.retrieval.base import tokenize
from src.retrieval.bm25_simple import BM25Retriever
from src.schemas import Paper, ScoredPaper

# arXiv 문법에서 따옴표로 묶인 구를 뽑는다. 예: abs:"graph neural network" -> graph neural network
_PHRASE_RE = re.compile(r'(?:\b(?:all|abs|ti|cat|au)\s*:\s*)?"([^"]+)"')
# 분야 지정 절. 예: cat:cs.CL
_CAT_RE = re.compile(r"\bcat\s*:\s*([A-Za-z\-]+\.[A-Za-z\-]+)")


def extract_phrases(arxiv_query: str) -> list[str]:
    """arXiv 검색어에서 따옴표 구 목록을 뽑는다. 없으면 전체를 하나의 구로 본다."""
    phrases = [p.strip() for p in _PHRASE_RE.findall(arxiv_query or "") if p.strip()]
    if phrases:
        return phrases
    # 따옴표가 없는 자유 검색어(예: passthrough 기준선)는 통째로 하나의 구로 취급하지 않고,
    # 걸림 판정을 건너뛰도록 빈 목록을 준다(호출자가 BM25만 쓰게 된다).
    return []


def extract_categories(arxiv_query: str) -> list[str]:
    return _CAT_RE.findall(arxiv_query or "")


class LocalArxivScorer:
    """arXiv 검색 문법을 로컬 논문 모음에서 흉내 낸다.

    사용법은 다른 검색기와 같다: `search(검색어, k) -> list[ScoredPaper]`
    """

    name = "local_arxiv"

    def __init__(self, papers: list[Paper]):
        self.papers = papers
        self.bm25 = BM25Retriever(papers)
        # 구(句)를 그대로 찾으려면 원문이 필요하다(BM25의 토큰만으로는 어순을 알 수 없다).
        self._texts = [f"{p.title} {p.abstract}".lower() for p in papers]
        self._cats = [set(p.categories) for p in papers]
        self._phrase_cache: dict[str, set[int]] = {}

    # ── 1단계: 걸림 판정 ──────────────────────────────────────────────────
    def _docs_containing(self, phrase: str) -> set[int]:
        """그 구를 그대로 포함한 논문 번호 집합.

        30,000편을 전부 훑으면 느리므로, 먼저 **구를 이루는 단어가 전부 들어 있는 논문**만
        역색인으로 추려낸 뒤(빠름) 그 안에서만 문자열을 확인한다(정확함).
        """
        key = phrase.lower().strip()
        if key in self._phrase_cache:
            return self._phrase_cache[key]

        terms = tokenize(key)
        if not terms:
            self._phrase_cache[key] = set()
            return set()

        # 역색인 교집합으로 후보 축소
        candidate: set[int] | None = None
        for t in terms:
            posting = self.bm25.postings.get(t)
            if not posting:
                candidate = set()
                break
            ids = set(posting.keys())
            candidate = ids if candidate is None else (candidate & ids)
            if not candidate:
                break
        candidate = candidate or set()

        # 단어가 다 있어도 어순이 다를 수 있으므로 원문에서 확인
        hit = {i for i in candidate if key in self._texts[i]}
        self._phrase_cache[key] = hit
        return hit

    # ── 2단계: 순위 매기기 ────────────────────────────────────────────────
    def search(self, query: str, k: int) -> list[ScoredPaper]:
        phrases = extract_phrases(query)
        cats = extract_categories(query)

        if phrases:
            # arXiv의 OR 의미: 어느 구 하나라도 걸리면 후보
            allowed: set[int] = set()
            for p in phrases:
                allowed |= self._docs_containing(p)
        else:
            allowed = set(range(len(self.papers)))     # 따옴표가 없으면 전체 대상

        if cats:                                        # cat: 절이 있으면 분야로 한 번 더 거른다
            catset = set(cats)
            allowed = {i for i in allowed if self._cats[i] & catset}

        if not allowed:
            return []

        # 후보 안에서만 BM25 점수로 정렬한다. 검색어는 구를 이어 붙인 것을 쓴다
        # (arXiv 문법 기호 abs:/OR 자체는 점수에 넣지 않는다).
        text_query = " ".join(phrases) if phrases else query
        scores = self._bm25_scores(text_query, allowed)

        order = sorted(allowed, key=lambda i: -scores[i])[:k]
        out: list[ScoredPaper] = []
        for rank, i in enumerate(order, start=1):
            p = self.papers[i]
            out.append(ScoredPaper(paper_id=p.id, score=float(scores[i]), rank=rank,
                                   title=p.title, abstract=p.abstract))
        return out

    def _bm25_scores(self, text_query: str, allowed: set[int]) -> np.ndarray:
        """BM25 점수를 후보 논문에 대해서만 계산한다."""
        scores = np.zeros(len(self.papers), dtype=np.float64)
        for term in tokenize(text_query):
            posting = self.bm25.postings.get(term)
            if not posting:
                continue
            idf = self.bm25.idf[term]
            for doc_idx, freq in posting.items():
                if doc_idx not in allowed:
                    continue
                dl = self.bm25.doc_len[doc_idx]
                denom = freq + self.bm25.k1 * (
                    1 - self.bm25.b + self.bm25.b * dl / self.bm25.avgdl)
                scores[doc_idx] += idf * (freq * (self.bm25.k1 + 1)) / denom
        return scores
