"""
순위 매기기 - 채널별 결과를 하나로 합치고(rrf_fuse), 그 결과로 순위를 재정렬함.

'local_index.py'와 'arxiv_live.py'가 찾아온 논문들을 질문과의 관련 정도를 바탕으로 이 코드에서 재정렬 진행함.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.retrieval.corpus import normalize_paper_id
from src import config
from src.schemas import ScoredPaper

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

@dataclass
class FusedPaper(ScoredPaper):
    """순위 합쳐진 결과"""

    channels: dict[str, int] = field(default_factory=dict)


def rrf_fuse(
    channel_results: dict[str, list[ScoredPaper]],
    k: int = 60,
    top_n: int = 200,
    weights: dict[str, float] | None = None,
) -> list[FusedPaper]:
    """
    채널별 검색 결과를 하나로 합쳐 상위 top_n편만 리턴

    Args:
        channel_results: {채널 이름: 결과 목록}
        k: RRF 상수. 크게 잡을수록 한 채널이 1등으로 올린 논문의 영향력이 작아짐
        top_n: 재정렬에 넘길 논문 수
        weights: {채널 이름: 가중치}
    """
    weights = weights or {}

    scores: dict[str, float] = {}
    provenance: dict[str, dict[str, int]] = {}     # 논문 -> {채널: 등수}
    meta: dict[str, ScoredPaper] = {}              # 논문 -> 제목, 초록을 가진 대표 결과
    first_seen: dict[str, int] = {}                # 동점일 때 순서를 고정하기 위한 논문 검색 순서

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

            kept = meta.get(pid)
            if kept is None or (not (kept.title or kept.abstract) and (r.title or r.abstract)):
                meta[pid] = r

    # 점수가 같으면 먼저 등장한 쪽을 앞에 둠
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
    """논문 번호 목록만으로 합침"""
    fake = {
        ch: [ScoredPaper(paper_id=pid, score=0.0, rank=i) for i, pid in enumerate(ids or [], 1)]
        for ch, ids in channel_ids.items()
    }
    return [p.paper_id for p in rrf_fuse(fake, k=k, top_n=top_n, weights=weights)]


class CrossEncoderReranker:
    """질문과 논문의 유사도를 계산하고 재정렬함"""

    name = "cross_encoder"

    def __init__(self, model_name: str | None = None,
                 device: str | None = None, max_length: int = 512,
                 batch_size: int = 32, fp16: bool | None = None):
        import torch
        from sentence_transformers import CrossEncoder

        if model_name is None:
            fp16_dir = config.RERANKER_FP16_DIR
            model_name = str(fp16_dir) if fp16_dir.exists() else DEFAULT_RERANKER

        if fp16 is None:
            fp16 = torch.cuda.is_available()
        kw = {"model_kwargs": {"dtype": torch.float16}} if fp16 else {}
        self.model = CrossEncoder(model_name, max_length=max_length, device=device, **kw)
        self.batch_size = batch_size

    def unload(self) -> None:
        """
        그래픽 메모리를 비움
        """
        import torch

        self.model.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rerank(self, query: str, candidates: list[ScoredPaper],
               top_k: int = 10) -> list[ScoredPaper]:
        """
        후보 논문들을 질문과의 관련도로 재정렬해 상위 top_k편을 돌려줌.
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
    order = np.argsort(-scores, kind="stable")[:top_k]
    out: list[ScoredPaper] = []
    for rank, i in enumerate(order, start=1):
        c = candidates[int(i)]
        out.append(ScoredPaper(paper_id=c.paper_id, score=float(scores[int(i)]),
                               rank=rank, title=c.title, abstract=c.abstract))
    return out
