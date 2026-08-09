"""어휘 조회 — 검색어에 쓸 학술 용어를 '지어내지 않고 실제 논문에서 가져온다'.

왜 필요한가 (전부 실측으로 확인한 것):

1. **성공을 가르는 것은 용어의 구체성이다.** 정답 논문에 걸린 용어의 문서 빈도가
   검색 성공 문항에서는 중앙값 144편, 실패(묻힘) 문항에서는 6,166편이었다(43배 차이).
   `anomaly detection`(6,984편)으로 걸리면 같은 조건의 논문이 수천 편이라 구별이 안 되고,
   `depth separation`(24편)으로 걸리면 경쟁자가 거의 없다.

2. **모델은 넓은 용어는 잘 만들지만 좁은 용어를 못 만든다.** 학습된 변환기가 만든 용어
   1,139개를 정답 논문 기준으로 분류하니, 정확히 걸린 455개(39.9%)의 문서 빈도 중앙값이
   3,191편이었다. 즉 걸리기는 하는데 너무 흔했다.

3. **학습으로 가르치는 데 실패했다.** 정답 논문에서 좁은 용어를 뽑아 라벨로 학습시켰더니,
   학습 논문은 외우지만 **처음 보는 논문에서는 매칭률이 0.79 → 0.22~0.33 으로 떨어졌다.**
   그 논문만의 용어는 질문만 보고 유추할 수 없기 때문이다.

4. **실패 용어의 63%가 '거의 맞음'이었다.** `covid-19 dataset`(논문에는 `covid-19 data`),
   `anomaly detection`(논문에는 `anomaly segmentation`)처럼 의미는 맞는데 문자열이 달랐다.
   arXiv 의 `abs:"구"` 는 정확한 문자열 매칭이라 한 글자만 달라도 걸리지 않는다.

→ 그래서 **모델이 지어내게 하지 말고, 실제 논문에 있는 문자열을 가져다 쓴다.**

동작:
    질문 → 로컬 코퍼스 의미 검색(bge-m3, 다국어라 한국어 질문도 됨)
         → 주제가 비슷한 논문 N편 → 그 논문들의 구를 뽑음
         → 문서 빈도가 목표 구간인 것만 남김 (너무 흔하지도 너무 드물지도 않게)
         → 여러 논문에서 공통으로 쓰이는 구를 우선 (그 주제의 관용 표현일 가능성이 높다)

검증 결과 (test 300문항, arXiv 호출 0회로 측정):
  - 1차 검색을 BM25 로 하면 조회 성공률 19.2% (한국어에서 0건이 나와 무너진다)
  - **1차 검색을 의미 검색으로 하면 35.0%**
  - 현재 실패 문항 중 **13.3%p 에 기존보다 좁은 실재 용어를 제공**한다
    (예: '신경망 깊이 구분' 39,574편 → `depth separation` 24편)

한계 (정직하게):
  - 조회 코퍼스가 3만 편이라 다루는 주제가 제한적이다. 넓히면 더 좋아질 여지가 있다.
  - 1차 의미 검색이 주제를 못 맞히면 어휘도 얻지 못한다(닭과 달걀). 그래서 모델이 만든
    용어를 **대체하지 않고 함께** 쓴다.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

from src import config
from src.retrieval.corpus import load_corpus
from src.retrieval.dense import DenseRetriever, Embedder
from src.retrieval.phrase_stats import PhraseFrequency
from src.schemas import Paper


def _candidate_phrases(text: str):
    """제목·초록에서 학술 용어가 될 만한 구를 뽑는다(문장 조각 제외).

    라벨 생성기와 같은 규칙을 쓴다. 규칙이 갈라지면 학습과 서비스가 어긋난다.
    """
    from training.make_labels import candidate_phrases
    return candidate_phrases(text)


class VocabularyLookup:
    """질문과 주제가 비슷한 논문들에서 '실제로 쓰이는 학술 용어'를 가져온다."""

    def __init__(self,
                 corpus_path: str | Path = None,
                 embeddings_path: str | Path = None,
                 df_corpus: str | Path = None,
                 top_papers: int = 20,
                 df_lo: int = 20,
                 df_hi: int = 1000,
                 device: str | None = None):
        """
        Args:
            top_papers: 어휘를 캐올 논문 수. 많을수록 후보가 늘지만 주제가 흐려진다.
            df_lo/df_hi: 쓸 만한 용어의 문서 빈도 구간. 아래로는 그 논문만의 지문이라
                일반화가 안 되고, 위로는 너무 흔해 정답을 가려내지 못한다.
        """
        corpus_path = corpus_path or (config.CORPUS_DIR / "corpus-v1.jsonl")
        embeddings_path = embeddings_path or (
            config.DATA_DIR / "embeddings" / "corpus-v1.BAAI_bge-m3.npy")
        df_corpus = df_corpus or (config.CORPUS_DIR / "corpus-full.jsonl")

        self.papers: list[Paper] = load_corpus(corpus_path)
        self._by_id = {p.id: p for p in self.papers}
        embeddings = np.load(embeddings_path)
        self.retriever = DenseRetriever(self.papers, embeddings, Embedder(device=device))
        self.stats = PhraseFrequency(df_corpus)
        self.top_papers = top_papers
        self.df_lo, self.df_hi = df_lo, df_hi
        self._cache: dict[str, list[tuple[str, int]]] = {}

    def lookup(self, question: str, k: int = 6,
               exclude_ids: set[str] | None = None) -> list[tuple[str, int]]:
        """질문과 관련된 논문들이 실제로 쓰는 용어를 k개 돌려준다.

        Returns: [(용어, 문서 빈도), ...] — 문서 빈도가 낮은(좁은) 것부터.
        """
        key = f"{question}|{k}"
        if exclude_ids is None and key in self._cache:
            return self._cache[key]

        hits = self.retriever.search(question, k=self.top_papers + 5)
        if exclude_ids:
            hits = [h for h in hits if h.paper_id not in exclude_ids]
        hits = hits[: self.top_papers]

        # 여러 논문에서 공통으로 쓰이는 구일수록 그 주제의 관용 표현일 가능성이 높다.
        counts: collections.Counter = collections.Counter()
        for h in hits:
            paper = self._by_id.get(h.paper_id)
            if paper is None:
                continue
            for phrase in set(_candidate_phrases(f"{paper.title} {paper.abstract}")):
                df = self.stats.df(phrase)
                if df is not None and self.df_lo <= df <= self.df_hi:
                    counts[phrase] += 1

        # 공통 등장 횟수 우선, 같으면 좁은(문서 빈도가 낮은) 것 우선
        ranked = sorted(counts.items(),
                        key=lambda x: (-x[1], self.stats.df(x[0]) or 10 ** 9))
        picked: list[tuple[str, int]] = []
        for phrase, _ in ranked:
            if any(phrase in q or q in phrase for q, _ in picked):   # 겹치는 구는 하나만
                continue
            picked.append((phrase, self.stats.df(phrase) or 0))
            if len(picked) >= k:
                break
        picked.sort(key=lambda x: x[1])       # 좁은 것부터 (검색어 앞자리에 놓기 위함)

        if exclude_ids is None:
            self._cache[key] = picked
        return picked
