"""순위 매기기 - 채널별 결과를 하나로 합치고(rrf_fuse), 그 결과를 다시 줄 세움(재정렬).

검색 1단계(찾아 오기)는 `local_index.py` 와 `arxiv_live.py` 가 하고, 2단계(줄 세우기)를
이 파일이 전부 맡음.

합치기: 채널마다 점수의 점수 범위가 달라서 점수를 그대로 더할 수 없음. arXiv 는 점수를 안 줘
1/순위 를 채워 넣으므로 1등이 1.0 이고, 로컬 의미 검색은 코사인 유사도라 0.4~0.8 에
몰려 있음. 그대로 더하면 옳은 채널이 아니라 점수 범위가 큰 채널이 이김. 그래서 점수를 버리고
등수만 씀 - r등이면 1/(k+r) 점을 받고 채널별 점수를 합해 다시 줄 세움(순위 합치기).

재정렬: 넓게 가져온 후보를 질문과의 관련도로 다시 줄 세움. 두 가지를 둠.

    CrossEncoderReranker   질문과 논문을 함께 모델에 넣어 관련도를 직접 예측함. 정확한
                           대신 후보 하나마다 모델을 돌려야 해서 느림. 서비스가 쓰는 것
    LLMReranker            언어 모델이 'Yes' 를 낼 값을 점수로 씀. 비교군
    rerank_by_similarity   질문과 논문을 따로 인코딩해 내적을 잼. 비교군

논문 번호 표기 통일은 `corpus.normalize_paper_id` 가 함.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.retrieval.corpus import normalize_paper_id
from src import config
from src.schemas import ScoredPaper

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


# ==========================================================================
# 1부. 채널 합치기 (순위 합치기)
# ==========================================================================

@dataclass
class FusedPaper(ScoredPaper):
    """합쳐진 결과 한 줄. `ScoredPaper` 에 어느 채널에서 몇 등이었는지를 덧붙임."""

    channels: dict[str, int] = field(default_factory=dict)   # 채널 이름 -> 그 채널에서의 등수


def rrf_fuse(
    channel_results: dict[str, list[ScoredPaper]],
    k: int = 60,
    top_n: int = 200,
    weights: dict[str, float] | None = None,
) -> list[FusedPaper]:
    """채널별 검색 결과를 순위로 합쳐 상위 top_n 편을 돌려줌.

    Args:
        channel_results: {채널 이름: 결과 목록}. 각 목록은 이미 좋은 순서라고 봄.
        k: 완충 상수. 크게 잡을수록 한 채널이 1등으로 올린 논문의 영향력이 작아짐.
        top_n: 돌려줄 편수. 재정렬에 넘길 후보 묶음이라 넉넉히 둠.
        weights: {채널 이름: 가중치}. 안 적은 채널은 1.0.

    등수는 `ScoredPaper.rank` 가 아니라 목록에 담긴 순서로 매김. 결과를 잘라내거나
    걸러낸 뒤에는 rank 필드에 옛 값이 남아 있기 때문임. 한 채널 안에서 같은 논문이
    두 번 나오면 앞선 등수 하나만 인정함.
    """
    weights = weights or {}

    scores: dict[str, float] = {}
    provenance: dict[str, dict[str, int]] = {}     # 논문 -> {채널: 등수}
    meta: dict[str, ScoredPaper] = {}              # 논문 -> 제목, 초록을 가진 대표 결과
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

            # 제목, 초록은 처음 본 것을 쓰되 비어 있으면 채워 주는 채널 것으로 바꿈.
            # 재정렬기가 제목과 초록으로 판단하므로 빈 채로 넘어가면 순위가 나빠짐.
            kept = meta.get(pid)
            if kept is None or (not (kept.title or kept.abstract) and (r.title or r.abstract)):
                meta[pid] = r

    # 점수가 같으면 먼저 등장한 쪽을 앞에 둠. 실행마다 순서가 바뀌지 않게 하기 위함
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
    """논문 번호 목록만으로 합침. 저장해 둔 결과를 검색 없이 다시 합칠 때 씀."""
    fake = {
        ch: [ScoredPaper(paper_id=pid, score=0.0, rank=i) for i, pid in enumerate(ids or [], 1)]
        for ch, ids in channel_ids.items()
    }
    return [p.paper_id for p in rrf_fuse(fake, k=k, top_n=top_n, weights=weights)]


# ==========================================================================
# 2부. 재정렬
# ==========================================================================

class CrossEncoderReranker:
    """질문과 논문을 함께 읽고 관련도를 매기는 재정렬기. 서비스가 쓰는 것."""

    name = "cross_encoder"

    def __init__(self, model_name: str | None = None,
                 device: str | None = None, max_length: int = 512,
                 batch_size: int = 32, fp16: bool | None = None):
        import torch
        from sentence_transformers import CrossEncoder

        # 미리 저장해 둔 float16 사본이 있으면 그것을 읽음. 검색마다 올렸다 내리므로
        # 적재 시간이 그대로 응답 시간이 됨. 원본 float32 파일 2.2GB 는 매번 5.4~5.8초,
        # 사본은 첫 검색 4.9초 뒤로 1.8초임. 점수는 128쌍을 대조해 차이가 정확히 0 이고
        # 순위도 같음. 사본은 `training/export.py fp16` 으로 만듦.
        if model_name is None:
            fp16_dir = config.RERANKER_FP16_DIR
            model_name = str(fp16_dir) if fp16_dir.exists() else DEFAULT_RERANKER

        # float16 으로 올려 그래픽 메모리를 3.06GB 에서 절반으로 줄임. 상대 순서만 쓰므로
        # float16 이어도 결과가 사실상 같음.
        if fp16 is None:
            fp16 = torch.cuda.is_available()
        kw = {"model_kwargs": {"dtype": torch.float16}} if fp16 else {}
        self.model = CrossEncoder(model_name, max_length=max_length, device=device, **kw)
        self.batch_size = batch_size

    def unload(self) -> None:
        """그래픽 메모리를 비움. `GpuPool.release` 가 부름.

        참조를 끊는 것만으로는 부족함. 부르는 쪽이 이 객체를 변수에 담아 두면 파이썬이
        객체를 안 없애서 자리가 그대로 남음. 가중치를 CPU 로 옮기면 참조가 남아 있어도
        확실히 돌아감.
        """
        import torch

        self.model.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rerank(self, query: str, candidates: list[ScoredPaper],
               top_k: int = 10) -> list[ScoredPaper]:
        """후보를 질문과의 관련도로 다시 정렬해 상위 top_k 편을 돌려줌.

        query 에는 변환된 검색어가 아니라 사용자가 실제로 입력한 말을 넣음. 재정렬의
        목적이 사용자 의도와 맞는지 보는 것이기 때문임.
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
        """여러 질문을 한 번에 처리함. 평가처럼 대량으로 돌릴 때 훨씬 빠름."""
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
    # 안정 정렬을 씀. 점수가 같을 때 후보 목록에 들어온 순서(= 합친 순위)를 그대로
    # 유지하기 위함임. float16 이라 유효숫자가 세 자리쯤이어서 동점이 실제로 생기는데
    # (후보의 6.0%), 그 자리에서는 1차 검색의 판단을 따르는 것이 근거 없는 순서보다 나음.
    order = np.argsort(-scores, kind="stable")[:top_k]
    out: list[ScoredPaper] = []
    for rank, i in enumerate(order, start=1):
        c = candidates[int(i)]
        out.append(ScoredPaper(paper_id=c.paper_id, score=float(scores[int(i)]),
                               rank=rank, title=c.title, abstract=c.abstract))
    return out


DEFAULT_LLM_RERANKER = "BAAI/bge-reranker-v2-gemma"

# 'Yes' 앞에 붙이는 지시문. 이 모델이 학습된 형식 그대로임 - 형식을 바꾸면 점수가 무너짐.
_LLM_PROMPT = ("Given a query A and a passage B, determine whether the passage contains "
               "an answer to the query by providing a prediction of either 'Yes' or 'No'.")


class LLMReranker:
    """언어 모델을 재정렬기로 씀. 'Yes' 를 낼 값을 관련도 점수로 삼음. 비교군.

    교차 인코더와 바탕 모델이 다른 재정렬기를 견주려고 둔 것임. 서비스가 쓰는
    `bge-reranker-v2-m3` 는 검색에 쓰는 `bge-m3` 와 바탕이 같아서 모르는 것도 같음.

    분류 층(classification head)이 없어 sentence-transformers 의 `CrossEncoder` 로는 못 부름. 점수 범위도
    달라서(대략 -10~+10) `app.py` 의 `MIN_RERANK_SCORE` 를 그대로 쓰면 안 됨 -
    `--calibrate-threshold` 로 다시 재야 함.
    """

    name = "llm_reranker"

    def __init__(self, model_name: str = DEFAULT_LLM_RERANKER,
                 device: str | None = None, max_length: int = 512,
                 batch_size: int = 8, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if load_in_4bit:
            # 4비트로 5.01GB 를 약 1.8GB 로 줄임. 점수가 달라지므로 점수 범위를 다시 재야 함
            from transformers import BitsAndBytesConfig
            kw = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)}
        else:
            kw = {"dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        if device:
            self.model = self.model.to(device)
        elif torch.cuda.is_available() and not load_in_4bit:
            self.model = self.model.cuda()
        self.model.eval()

        # 'Yes' 한 낱말의 번호. 이 자리의 값이 곧 점수임
        self.yes_id = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][-1]
        self.max_length = max_length
        self.batch_size = batch_size

    def _pairs_text(self, query: str, c: ScoredPaper) -> str:
        """이 모델이 학습된 형식 그대로 만듦.

        지시문이 맨 뒤에 와야 함. 다음 낱말을 예측하는 모델이라 지시문을 앞에 두면
        'Yes' 를 낼 자리가 아니게 되어, 오류 없이 점수만 무의미해짐.
        """
        return f"A: {query}\nB: {_doc_text(c)}\n{_LLM_PROMPT}"

    def _score(self, texts: list[str]) -> np.ndarray:
        import torch

        out: list[float] = []
        left_pad = self.tokenizer.padding_side == "left"
        for s in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(
                texts[s:s + self.batch_size],
                return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            # 마지막 진짜 낱말의 자리를 읽어야 함. padding 이 왼쪽이면 맨 끝, 오른쪽이면
            # 실제 길이 - 1 임. 잘못 잡으면 오류 없이 순위만 무너짐.
            if left_pad:
                picked = logits[:, -1, self.yes_id]
            else:
                last = enc["attention_mask"].sum(dim=1) - 1
                picked = logits[torch.arange(logits.size(0)), last, self.yes_id]
            out.extend(picked.float().cpu().tolist())
        return np.asarray(out, dtype=np.float64)

    def rerank(self, query: str, candidates: list[ScoredPaper],
               top_k: int = 10) -> list[ScoredPaper]:
        if not candidates:
            return []
        return _reorder(candidates,
                        self._score([self._pairs_text(query, c) for c in candidates]),
                        top_k)

    def rerank_batch(self, queries: list[str],
                     candidate_lists: list[list[ScoredPaper]],
                     top_k: int = 10) -> list[list[ScoredPaper]]:
        """평가용. 문항 경계를 기억해 두고 한 번에 채점함."""
        texts, spans = [], []
        for q, cands in zip(queries, candidate_lists):
            start = len(texts)
            texts.extend(self._pairs_text(q, c) for c in cands)
            spans.append((start, len(texts)))
        if not texts:
            return [[] for _ in queries]
        scores = self._score(texts)
        return [_reorder(cands, scores[s:e], top_k)
                for cands, (s, e) in zip(candidate_lists, spans)]


def rerank_by_similarity(query: str, candidates: list[ScoredPaper], embedder,
                         top_k: int = 10) -> list[ScoredPaper]:
    """비교군: 질문과 논문을 따로 인코딩해 내적으로 재정렬. 빠르지만 덜 정확함."""
    if not candidates:
        return []
    q_emb = embedder.encode([query])[0]
    c_emb = embedder.encode([_doc_text(c) for c in candidates])
    return _reorder(candidates, c_emb @ q_emb, top_k)
