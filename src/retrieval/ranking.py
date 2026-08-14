"""순위 매기기 — 채널별 결과를 하나로 합치고(RRF), 그 결과를 다시 줄 세운다(재정렬).

검색은 두 단계로 나뉜다. 이 파일은 그 **두 번째 단계 전부**를 담는다.

    1단계 (찾아 오기)   local_index.py · arxiv_live.py
    2단계 (줄 세우기)   이 파일 — 합치기(rrf_fuse) → 재정렬(CrossEncoderReranker)

두 일을 한 파일에 두는 이유는 **둘이 항상 붙어 다니기 때문**이다. 합치기는 재정렬에 넘길
후보 풀을 만드는 것이 목적이고, 재정렬은 그 풀을 받는 것이 전제다. 따로 두면 "후보를 몇 편
합쳐서 몇 편을 재정렬에 넘기는가"라는 한 덩어리의 결정이 두 파일에 흩어진다. 실제로 그
어긋남이 ISSUE #26 과 #31 로 두 번 났다.


## 1부. 합치기 — 왜 '점수 더하기'가 아니라 '순위 더하기'인가

이 시스템은 성격이 다른 검색 경로를 나란히 둔다(ISSUE #21 의 결론).

- `arxiv` 채널: arXiv 실시간 키워드 검색. **arXiv API 는 관련도 점수를 주지 않아서**
  우리가 `1/순위` 를 점수 자리에 채워 넣는다(`src/retrieval/arxiv_live.py`).
  1등 1.0, 2등 0.5, 10등 0.1 처럼 급격히 떨어지는 값이다.
- `local_dense` 채널: 로컬 의미 검색. 점수는 **코사인 유사도**라 대개 0.4~0.8 사이의
  좁은 범위에 몰려 있고, 1등과 100등의 차이가 0.1도 안 되는 일이 흔하다.

이 둘을 그대로 더하면 `arxiv` 1등(1.0) 하나가 `local_dense` 전체(0.4~0.8)를 눌러 버린다.
즉 **어느 채널이 옳은가가 아니라 점수의 눈금이 큰 쪽이 이긴다.** 점수를 정규화해도
(min-max 등) 채널마다 분포 모양이 달라 기준을 정할 근거가 없다.

그래서 점수는 버리고 **등수만** 쓴다. 어떤 논문이 한 채널에서 r등이면 `1/(k+r)` 점을 받고,
채널별 점수를 합해 다시 줄 세운다(k는 완충 상수, 관례상 60). 눈금 맞추기가 필요 없고
조정할 값이 k 하나뿐이라 채널을 새로 붙일 때마다 다시 튜닝할 필요가 없다.

논문 번호 표기가 채널마다 다른 함정은 `corpus.normalize_paper_id` 가 처리한다. 그 함수의
설명을 반드시 함께 읽을 것 — 오류가 아니라 조용히 성능만 깎는 종류의 문제다.


## 2부. 재정렬 — 넓게 가져온 후보를 '사용자 의도와의 관련도'로 다시 줄 세운다

왜 필요한가 (실측):
arXiv 는 키워드 일치도로 순위를 매기므로 **사용자가 무엇을 원하는지 모른다.** 300문항에서
정답이 상위 100편 안에 든 비율(Recall@100)은 0.490 인데, 상위 10편 안에 든 비율(Recall@10)은
0.343 이었다. 즉 **정답의 14.7%가 이미 후보 안에 있는데 순위가 낮아 사용자 눈에 안 띈다.**
이 격차가 재정렬로 회수할 수 있는 몫이고, 재정렬의 이론적 상한은 0.490 이다.

두 가지 방식을 둔다:

1. **교차 인코더 (CrossEncoderReranker) — 권장**
   질문과 논문을 **함께** 모델에 넣어 관련도를 직접 예측한다. 두 글을 한 번에 보므로
   "이 논문이 이 질문에 답하는가"를 훨씬 정확히 판단한다. 대신 후보 하나마다 모델을
   돌려야 해서 느리다 → 후보를 100편 정도로 좁힌 뒤에만 쓸 수 있는데, 우리가 그 조건이다.
   모델: BAAI/bge-reranker-v2-m3 (568M, 다국어). 한국어 질문 ↔ 영어 초록을 한 모델이 처리한다.

2. **임베딩 유사도 (rerank_by_similarity) — 비교군**
   질문과 논문을 **따로** 숫자로 바꿔 내적을 잰다. 빠르지만, 두 글을 맞대어 보지 않으므로
   교차 인코더보다 정확도가 낮은 것이 정보 검색에서 일반적이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.retrieval.corpus import normalize_paper_id
from src.schemas import ScoredPaper

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


# ══════════════════════════════════════════════════════════════════════════
# 1부. 채널 합치기 (RRF)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FusedPaper(ScoredPaper):
    """합쳐진 결과 한 줄. `ScoredPaper` 에 '어느 채널에서 몇 등이었는지'를 덧붙인다.

    재정렬기·화면 표시는 `ScoredPaper` 만 알면 되므로 그대로 상속하고,
    진단용 정보(channels)만 추가한다. 어떤 채널이 어떤 문항을 건졌는지 남겨 두면
    나중에 "이 논문은 의미 검색만 찾아냈다" 같은 분석을 재검색 없이 할 수 있다.
    """

    channels: dict[str, int] = field(default_factory=dict)   # 채널 이름 -> 그 채널에서의 등수


def rrf_fuse(
    channel_results: dict[str, list[ScoredPaper]],
    k: int = 60,
    top_n: int = 200,
    weights: dict[str, float] | None = None,
) -> list[FusedPaper]:
    """채널별 검색 결과를 RRF로 합쳐 상위 top_n편을 돌려준다.

    Args:
        channel_results: {채널 이름: 결과 목록}. 각 목록은 이미 좋은 순서대로 정렬돼 있다고 본다.
        k: RRF 완충 상수(기본 60). 크게 잡을수록 상위권과 하위권의 점수 차이가 줄어,
            한 채널이 1등으로 올린 논문의 영향력이 작아진다.
        top_n: 돌려줄 개수. 재정렬에 넘길 후보 풀이므로 기본값을 넉넉히(200) 둔다.
        weights: {채널 이름: 가중치}. 안 주면 전부 1.0(동등). 채널마다 신뢰도가 다를 때
            **점수 눈금을 건드리지 않고** 영향력만 조절하는 방법이다. 적지 않은 채널은 1.0.

    등수는 목록에 담긴 **순서**로 매긴다(`ScoredPaper.rank` 값을 믿지 않는다).
    결과를 잘라내거나 걸러낸 뒤에는 rank 필드가 옛 값 그대로 남아 있기 때문이다.

    같은 채널 안에서 번호를 통일한 뒤 중복이 생기면(예: 같은 논문의 다른 버전이 함께 왔다면)
    **가장 앞선 등수 하나만** 인정한다. 한 채널이 같은 논문에 두 번 점수를 주면 그 채널의
    영향력이 부풀려지기 때문이다.
    """
    weights = weights or {}

    scores: dict[str, float] = {}
    provenance: dict[str, dict[str, int]] = {}     # 논문 -> {채널: 등수}
    meta: dict[str, ScoredPaper] = {}              # 논문 -> 제목·초록을 가진 대표 결과
    first_seen: dict[str, int] = {}                # 동점일 때 순서를 고정하기 위한 등장 순서

    for channel, results in channel_results.items():
        weight = float(weights.get(channel, 1.0))
        seen_in_channel: set[str] = set()
        for position, r in enumerate(results or [], start=1):
            pid = normalize_paper_id(r.paper_id)
            if not pid or pid in seen_in_channel:
                continue
            seen_in_channel.add(pid)

            scores[pid] = scores.get(pid, 0.0) + weight / (k + position)
            provenance.setdefault(pid, {})[channel] = position
            first_seen.setdefault(pid, len(first_seen))

            # 제목·초록은 처음 본 것을 쓰되, 그것이 비어 있고 나중 채널이 채워 줄 수 있으면
            # 바꿔 담는다. 재정렬기는 제목·초록으로 판단하므로, 빈 채로 넘어가면
            # 오류 없이 재정렬 품질만 조용히 떨어진다.
            kept = meta.get(pid)
            if kept is None or (not (kept.title or kept.abstract) and (r.title or r.abstract)):
                meta[pid] = r

    # 동점 처리: 점수가 같으면 먼저 등장한 쪽을 앞에 둔다(실행마다 순서가 바뀌지 않게).
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))[:top_n]

    out: list[FusedPaper] = []
    for rank, (pid, score) in enumerate(ordered, start=1):
        m = meta[pid]
        out.append(FusedPaper(
            paper_id=pid, score=score, rank=rank,
            title=m.title, abstract=m.abstract,
            channels=dict(provenance[pid]),
        ))
    return out


def rrf_fuse_ids(
    channel_ids: dict[str, list[str]],
    k: int = 60,
    top_n: int = 200,
    weights: dict[str, float] | None = None,
) -> list[str]:
    """논문 번호 목록만으로 합친다 (제목·초록이 필요 없을 때).

    평가 하네스가 저장해 둔 채널별 논문 번호를 다시 합칠 때 쓴다. 이 경로가 있기 때문에
    **검색을 다시 하지 않고** 융합 방식(k·가중치)을 바꿔 가며 몇 번이든 재계산할 수 있다.
    """
    fake = {
        ch: [ScoredPaper(paper_id=pid, score=0.0, rank=i) for i, pid in enumerate(ids or [], 1)]
        for ch, ids in channel_ids.items()
    }
    return [p.paper_id for p in rrf_fuse(fake, k=k, top_n=top_n, weights=weights)]


# ══════════════════════════════════════════════════════════════════════════
# 2부. 재정렬
# ══════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """질문과 논문을 함께 읽고 관련도를 매기는 재정렬기."""

    name = "cross_encoder"

    def __init__(self, model_name: str = DEFAULT_RERANKER,
                 device: str | None = None, max_length: int = 512,
                 batch_size: int = 32, fp16: bool | None = None):
        import torch
        from sentence_transformers import CrossEncoder

        # 반정밀도(fp16)로 올린다. VRAM 이 3.06GB 에서 절반으로 줄어든다.
        # 이 모델은 순위를 매기는 점수만 내고 그 점수의 절대값은 쓰지 않으므로
        # (상대 순서만 본다) 정밀도를 낮춰도 결과가 사실상 같다.
        # 왜 아껴야 하는지는 local_index.py 의 같은 주석 참고 - 여유가 모자라면
        # 추천 에이전트가 조용히 CPU 로 밀려나 한 번에 229초가 걸린다(실측).
        if fp16 is None:
            fp16 = torch.cuda.is_available()
        kw = {"model_kwargs": {"dtype": torch.float16}} if fp16 else {}
        self.model = CrossEncoder(model_name, max_length=max_length, device=device, **kw)
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
