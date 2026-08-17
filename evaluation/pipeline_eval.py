"""검색 파이프라인을 실제로 돌려 재는 하네스 - 성능(정확도)과 응답 시간 둘 다.

    # 개발용에서 두 채널로 (이어하기 켜짐)
    $PY -m evaluation.pipeline_eval --queries data/eval/dev.jsonl \\
        --channels arxiv local_dense --rewriter dpo --k 100 --out runs/dev_dpo.jsonl

    # 저장된 결과만 다시 집계 (검색 0회). 융합 상수, 가중치를 바꿔 가며 반복 가능
    $PY -m evaluation.pipeline_eval --report-only runs/dev_dpo.jsonl \\
        --rrf-k 30 --weights local_dense=2.0

    # 저장된 결과에 재정렬만 다시 적용 (검색 0회)
    $PY -m evaluation.pipeline_eval --report-only runs/dev_dpo.jsonl \\
        --rerank cross --rerank-depth 100

    # 서비스 응답 시간 실측 (평가셋이 아니라 실제 질문으로)
    $PY -m evaluation.pipeline_eval --bench-service --n 5

만족도(nDCG)까지 포함한 자세한 보고와 실행 간 비교는 `evaluation/report.py` 가 맡음.

## 왜 채널별 결과를 따로 저장하는가

검색 경로가 둘 이상(arXiv 키워드 + 로컬 의미 검색)이면 다음 질문에 답할 수 있어야 함.

  - 각 채널은 단독으로 얼마나 찾아내는가?
  - 두 채널을 합치면 후보 상한이 얼마나 올라가는가? (ISSUE #21: 이 상한을 0.70 위로
    올리지 못하면 목표 달성이 산술적으로 불가능함)
  - 그 상한 중 융합, 재정렬이 실제로 얼마나 회수하는가?

그래서 채널별 검색 결과 논문 번호를 전부 따로 저장함.

    {"channels": {"arxiv": [...100편...], "local_dense": [...100편...]}}

이것이 이 하네스의 핵심 가치임. 융합 상수(k), 채널 가중치, 재정렬 방식을 바꿔 가며 실험할 때
검색을 다시 하지 않음. arXiv 는 요청당 3초 제한이 있어 300문항 재검색이 15분 이상
걸리고 429/503 으로 문항이 유실되기도 하는데, 그 비용을 한 번만 치름.

## 실패를 세는 방식 (ISSUE #23 점검표 3번)

오류, 결과 0건 문항을 제외하지 않고 실패(0점)로 셈. 제외하면 기준선이 부풀려짐
(같은 실행을 그렇게 재면 passthrough 가 0.257 대신 0.467 로 보임). 이 하네스의 모든
비율은 분모가 전체 문항 수임.

## 정답 누수에 대하여 (ISSUE #22)

검색어는 오직 질문 글(`text`) 에서만 만듦. 정답 논문의 제목, 초록은 검색어를 만드는
어느 단계에서도 읽지 않음. 반대로 로컬 색인이 정답 논문을 검색해 후보로 내놓는 것은
정상임 - 실제 서비스도 arXiv 전체를 검색 대상으로 삼음. 금지되는 것은 정답 논문의
글을 보고 검색어를 만드는 것이며, 그 둘은 다른 이야기임.

## 개발용과 시험용을 섞지 않음 (ISSUE #25)

기본값이 `data/eval/dev.jsonl` 인 것은 의도적임. 설정을 이리저리 바꿔 보는 탐색은 전부
개발용에서 하고, 시험용(`test.jsonl`)으로는 확정 판정만 함. 시험용을 보면서 설정을
고르면 그 순간 시험지가 타 버림 - 옛 평가셋이 정확히 그렇게 소모됐음.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluation.metrics import bootstrap_ci
from src import config
from src.retrieval.corpus import normalize_paper_id
from src.retrieval.ranking import rrf_fuse_ids
from src.rewriter.base import build_rewriter
from src.schemas import ScoredPaper
from src.utils import read_jsonl, write_jsonl

CACHE_PATH = config.DATA_DIR / "cache" / "arxiv_search_cache.jsonl"
DEFAULT_K_VALUES = (1, 5, 10, 30, 50, 100)

# 채널이 쓸 검색어 필드 (`RewriteResult.queries` 의 키). 없는 키를 부르면 원본 질문으로
# 자동 폴백되므로, "raw" 를 적으면 변환기가 무엇을 냈든 원본 질문이 들어감.
#
# ## 로컬 의미 검색에 무엇을 넣을지가 왜 선택지인가 (반드시 읽을 것)
#
# 학습한 변환기(dpo)는 arXiv 문법 문자열 하나를 만들어 세 필드에 똑같이 넣음.
# 예: all:"사진 설명 AI" OR abs:"image captioning" OR abs:"vision-language model"
#
# 이것을 로컬 의미 검색에 그대로 넣으면 안 됨. 의미 검색은 문장을 통째로 임베딩해
# 뜻을 견주는데, 위 문자열은 사람이 쓰는 문장이 아니라 검색 문법임. 게다가 서비스
# (`app.py`)는 로컬 채널에 원본 질문을 넣음. 평가가 서비스와 다른 것을 재면 그 숫자는
# 서비스 성능을 예측하지 못함 - 이 프로젝트가 이미 겪은 실패임(ISSUE #10, #13).
#
#   raw       (기본) 로컬 채널에 원본 질문을 넣음. 서비스와 같은 조건
#   rewritten 로컬 채널에도 변환 결과를 넣음. "변환이 의미 검색에도 도움이 되는가"를
#             따로 물을 때만 씀. 지금 변환기 출력 형태로는 불리한 조건임
CHANNEL_QUERY_FIELD = {
    "arxiv": "arxiv",              # 키워드 검색용 (필드 지정, 불리언 연산자가 든 문자열)
    "local_dense": "raw",          # 의미 검색용 - 기본은 원본 질문 (서비스와 같게)
}


def channel_query_fields(local_query: str) -> dict[str, str]:
    """채널별로 어느 검색어를 쓸지 정함. 위 설명 참고."""
    fields = dict(CHANNEL_QUERY_FIELD)
    fields["local_dense"] = "dense" if local_query == "rewritten" else "raw"
    return fields

# 응답 시간을 잴 때 쓰는 질문. 실제 사용자가 칠 법한 것으로, 한국어, 영어와
# 일상어, 전문어를 섞었음. 평가셋이 아님 - 정확도가 아니라 시간만 재는 용도임.
BENCH_QUERIES = [
    "사진 보고 글로 설명해주는 AI",
    "가짜 뉴스 걸러내는 방법",
    "AI가 사람처럼 대화하게 만들기",
    "얼굴 인식이 화장만 바꿔도 속을 수 있는지 궁금해요",
    "graph neural network for molecular property prediction",
    "논문 검색할 때 내가 쓴 말이랑 논문 용어가 달라서 못 찾는 문제",
    "black-box backdoor attack face recognition",
    "로봇이 처음 보는 물건을 집는 방법",
]


def git_commit() -> str:
    """재현을 위해 지금 코드가 어느 커밋인지 기록함."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_weights(spec: str | None) -> dict[str, float]:
    """'arxiv=1.0,local_dense=2.0' 형태를 사전으로 바꿈."""
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition("=")
        out[name.strip()] = float(value)
    return out


# -- 채널 만들기 ------------------------------------------------------------
def build_channel(name: str, args) -> object:
    """이름으로 검색 채널(검색기)을 만듦. 모두 `search(query, k)` 인터페이스를 따름."""
    if name == "arxiv":
        from src.retrieval.arxiv_live import ArxivLiveRetriever
        return ArxivLiveRetriever(cache_path=None if args.no_cache else CACHE_PATH)

    if name == "local_dense":
        from src.retrieval.local_index import LocalDenseRetriever
        return LocalDenseRetriever(args.corpus, args.index, mmap=args.mmap)

    raise ValueError(f"알 수 없는 채널 이름: {name} (쓸 수 있는 것: arxiv, local_dense)")


# -- 질문 하나 평가 ---------------------------------------------------------
def evaluate_one(query_row: dict, rewriter, channels: dict, k: int,
                 query_fields: dict[str, str] | None = None) -> dict:
    """질문 하나를 모든 채널로 검색해 채널별 결과를 따로 기록함.

    한 채널이 실패해도 나머지 채널은 계속 돌림. 채널 오류는 `channel_errors` 에 남기고
    그 채널의 결과는 빈 목록이 됨(집계에서 실패로 셈).
    """
    out = {
        "query_id": query_row["query_id"], "text": query_row["text"],
        "gold_id": normalize_paper_id(query_row["gold_id"]),
        "lang": query_row.get("lang"), "difficulty": query_row.get("difficulty"),
    }

    t0 = time.time()
    try:
        # 주의: 검색어는 질문 글에서만 만듦. 정답 논문의 본문은 여기서 절대 읽지 않음.
        rw = rewriter.rewrite(query_row["text"])
    except Exception as e:
        # 변환 실패면 어느 채널도 검색어를 못 받음 -> 문항 전체를 실패로 기록함.
        out["error"] = f"rewrite_failed: {type(e).__name__}: {e}"
        out["channels"] = {name: [] for name in channels}
        return out
    out["rewrite_ok"] = rw.parse_ok
    out["academic_terms"] = rw.academic_terms
    out["rewrite_sec"] = round(time.time() - t0, 2)

    out["search_queries"], out["channels"] = {}, {}
    out["channel_errors"], out["channel_sec"] = {}, {}
    fields = query_fields or CHANNEL_QUERY_FIELD
    for name, retriever in channels.items():
        query = rw.query_for(fields.get(name, "dense"))
        out["search_queries"][name] = query
        t = time.time()
        try:
            results = retriever.search(query, k=k)
        except Exception as e:
            out["channel_errors"][name] = f"{type(e).__name__}: {e}"
            out["channels"][name] = []
            continue
        # 순위대로 전부 저장 - 융합, 재정렬을 나중에 재검색 없이 다시 계산하기 위함
        out["channels"][name] = [normalize_paper_id(r.paper_id) for r in results]
        out["channel_sec"][name] = round(time.time() - t, 2)
    return out


# -- 저장된 결과에서 순위 계산 ----------------------------------------------
def channel_names_of(rows: list[dict]) -> list[str]:
    """결과 파일에 들어 있는 채널 이름을 등장 순서대로 모음."""
    names: list[str] = []
    for r in rows:
        for name in (r.get("channels") or {}):
            if name not in names:
                names.append(name)
    return names


def rank_in(ids: list[str], gold: str) -> int | None:
    """정답이 몇 등인지. 없으면 None."""
    return (ids.index(gold) + 1) if gold in ids else None


def union_rank(row: dict, depth: int) -> int | None:
    """합집합 안에 정답이 들어는 왔는가 (등수는 의미가 없으므로 1 또는 None).

    각 채널의 상위 `depth` 편을 모은 집합. 재정렬이 도달할 수 있는 상한임.
    재정렬은 후보 집합을 바꾸지 않고 순서만 바꾸므로 이 값을 넘을 수 없음.
    """
    gold = row["gold_id"]
    for ids in (row.get("channels") or {}).values():
        if gold in (ids or [])[:depth]:
            return 1
    return None


def select_channels(rows: list[dict], names: list[str]) -> list[dict]:
    """저장된 결과에서 채널 일부만 남김 - 검색을 다시 하지 않고 채널 조합을 비교함.

    "두 채널을 합치는 것이 단독보다 나은가"를 판정하려면 채널 A 단독, 채널 B 단독 , 
    둘 합침을 같은 검색 결과 위에서 비교해야 함. 조합마다 검색을 다시 하면
    arXiv 결과가 그 사이에 달라져 비교가 성립하지 않음(그리고 3초 제한 때문에 느림).

    남은 행의 `channels` 만 걸러 내면 융합, 상한, 재정렬, 보고가 전부 자동으로 그 조합만
    보게 됨. 원본 행은 건드리지 않고 얕은 사본을 돌려줌.
    """
    keep = set(names)
    have = set(channel_names_of(rows))
    missing = keep - have
    if missing:
        raise ValueError(f"결과 파일에 없는 채널: {sorted(missing)} (있는 것: {sorted(have)})")
    return [dict(r, channels={n: ids for n, ids in (r.get("channels") or {}).items()
                              if n in keep}) for r in rows]


def fused_ids_of(row: dict, rrf_k: int, top_n: int, weights: dict[str, float],
                 depth: int | None = None) -> list[str]:
    """저장된 채널별 논문 번호를 RRF로 합침 (검색 없음)."""
    channels = {name: (ids or [])[:depth] if depth else (ids or [])
                for name, ids in (row.get("channels") or {}).items()}
    if not channels:
        return []
    return rrf_fuse_ids(channels, k=rrf_k, top_n=top_n, weights=weights)


def hits_at(ranks: list[int | None], k: int) -> list[float]:
    """등수 목록을 Recall@k 의 0/1 목록으로. None(못 찾음, 오류, 0건)은 0점임."""
    return [1.0 if (r is not None and r <= k) else 0.0 for r in ranks]


def fmt(hits: list[float], with_ci: bool = True) -> str:
    if not hits:
        return "n/a"
    if not with_ci:
        return f"{float(np.mean(hits)):.3f}"
    m, lo, hi = bootstrap_ci(hits)
    return f"{m:.3f} [{lo:.3f},{hi:.3f}]"


# -- 재정렬 (저장된 결과 위에서, 검색 없이) ---------------------------------
class TextLookup:
    """재정렬 후보의 제목, 초록을 꺼내 옴.

    71만 편을 통째로 메모리에 올리지 않음(제목, 초록만 1GB가 넘음). 대신
    - 로컬 색인: 줄 위치(`offsets.npy`)로 필요한 논문만 파일에서 `seek` 해 읽고,
    - arXiv 결과: 검색 캐시 파일을 한 번만 훑으며 필요한 번호만 골라 담음.
    """

    def __init__(self, index_prefix: str | Path | None, corpus_path: str | Path | None,
                 arxiv_cache: str | Path | None):
        self.corpus_path = Path(corpus_path) if corpus_path else None
        self.arxiv_cache = Path(arxiv_cache) if arxiv_cache else None
        self.pos: dict[str, int] = {}
        self.offsets = None
        if index_prefix and self.corpus_path and self.corpus_path.exists():
            from src.retrieval.local_index import index_paths
            paths = index_paths(index_prefix)
            if paths["ids"].exists() and paths["offsets"].exists():
                # 줄 위치는 그 코퍼스 파일 전용임. 다른 파일에 갖다 쓰면 오류 없이
                # 엉뚱한 논문의 본문을 읽어 재정렬이 조용히 망가짐. 짝이 맞는지 확인함.
                indexed = ""
                if paths["meta"].exists():
                    indexed = Path(json.loads(paths["meta"].read_text()).get("corpus", "")).name
                if indexed and indexed != self.corpus_path.name:
                    print(f"경고: 줄 위치 색인은 '{indexed}' 용인데 코퍼스로 "
                          f"'{self.corpus_path.name}' 를 받았다. 색인을 쓰지 않고 파일을 훑는다.")
                else:
                    ids = paths["ids"].read_text(encoding="utf-8").splitlines()
                    self.pos = {pid: i for i, pid in enumerate(ids)}
                    self.offsets = np.load(paths["offsets"])

    def fetch(self, wanted: set[str]) -> dict[str, tuple[str, str]]:
        """번호 집합 -> {번호: (제목, 초록)}. 못 찾은 번호는 빠짐."""
        out: dict[str, tuple[str, str]] = {}
        if self.offsets is not None:
            todo = sorted((self.pos[p], p) for p in wanted if p in self.pos)
            with open(self.corpus_path, "rb") as f:
                for i, pid in todo:                     # 위치 순서로 읽어 디스크 이동을 줄임
                    f.seek(int(self.offsets[i]))
                    row = json.loads(f.readline())
                    out[pid] = (row.get("title", "").strip(), row.get("abstract", "").strip())
        elif self.corpus_path and self.corpus_path.exists():
            # 줄 위치 색인이 없는 코퍼스는 한 번 훑어 필요한 것만 담음.
            with open(self.corpus_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    pid = normalize_paper_id(str(row.get("id", "")))
                    if pid in wanted:
                        out[pid] = (row.get("title", "").strip(),
                                    row.get("abstract", "").strip())

        missing = wanted - set(out)
        if missing and self.arxiv_cache and self.arxiv_cache.exists():
            with open(self.arxiv_cache, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    for p in json.loads(line).get("results", []):
                        pid = normalize_paper_id(p["paper_id"])
                        if pid in missing and pid not in out:
                            out[pid] = (p.get("title", ""), p.get("abstract", ""))
                    if len(out) >= len(wanted):
                        break
        return out


def rerank_query_of(row: dict, mode: str) -> str:
    """재정렬기에 넣을 질문을 고름.

    raw       (기본) 사용자가 실제로 입력한 말. 재정렬의 목적이 '사용자 의도와 맞는가' 를
              보는 것이므로 원본이 원칙임.
    rewritten 변환기가 만든 검색어. 한국어를 영어로 옮기는 변환기(translate)를 쓸 때만
              의미가 있음. 재정렬 모델이 한국어 질문과 영어 초록을 맞댈 때 점수가 눌리기
              때문임 - 만족 등급 논문의 점수 중앙값이 한국어 0.07, 영어 0.17 이었음
              (2026-08-14 실측). 그 눌림이 순위에도 영향을 주는지 가려 보려는 것임.
    """
    if mode != "rewritten":
        return row["text"]
    sq = row.get("search_queries") or {}
    return sq.get("local_dense") or next(iter(sq.values()), row["text"]) or row["text"]


def rerank_rows(rows: list[dict], method: str, depth: int, lookup: TextLookup,
                rrf_k: int, weights: dict[str, float], batch_size: int = 32,
                query_mode: str = "raw", channel_depth: int | None = None) -> None:
    """저장된 결과를 융합한 뒤 상위 `depth` 편을 재정렬해 `reranked_ids` 로 채움.

    검색은 한 번도 하지 않음. 후보 본문을 못 찾은 논문은 후보에서 빠지는데, 그 논문이
    정답이면 손해이므로 몇 편이나 빠졌는지 반드시 보고함(조용히 성능이 깎이는 자리임).

    channel_depth: 융합 전에 채널마다 몇 편까지만 볼지. 실행 결과 파일에 서비스보다 깊게
        저장돼 있을 때 서비스와 조건을 맞추는 자리임. 안 주면 저장된 것을 다 씀.
        `runs/dev_mq_w1_d100.jsonl` 은 채널당 300편이 저장돼 있는데 서비스는 채널당
        100편만 가져오므로(app.py 의 DEPTH_LOCAL), 안 맞추면 상한이 0.675 대신 0.684 로
        부풀려짐. #10, #13, #39, #40 과 같은 종류의 어긋남임.

    점수를 함께 저장하는 이유:
        교차 인코더는 후보를 하나씩 독립적으로 채점함. 어떤 논문의 점수는 같은 목록에
        무엇이 더 들어 있는지와 무관함. 그래서 깊이 200으로 한 번 채점해 두면, 깊이 100의
        결과는 '융합 상위 100편만 골라 저장된 점수로 다시 줄 세우기' 로 정확히 같은 값이
        나옴. 재정렬을 깊이마다 다시 돌릴 필요가 없음.
    """
    cand_ids = [fused_ids_of(r, rrf_k, top_n=depth, weights=weights,
                             depth=channel_depth) for r in rows]
    wanted = {pid for ids in cand_ids for pid in ids}
    print(f"재정렬 후보 본문 수집: 고유 논문 {len(wanted):,}편 ...", flush=True)
    texts = lookup.fetch(wanted)
    lost = len(wanted) - len(texts)
    if lost:
        print(f"경고: 후보 {lost:,}편의 본문을 못 찾아 재정렬에서 빠진다 "
              f"({lost / max(len(wanted), 1):.1%}). 정답이 여기 섞이면 그만큼 손해다.")
    gold_lost = sum(1 for r, ids in zip(rows, cand_ids)
                    if r["gold_id"] in ids and r["gold_id"] not in texts)
    if gold_lost:
        print(f"경고: 그중 정답 논문이 {gold_lost}문항에서 빠졌다. 재정렬 상한이 그만큼 낮아진다.")

    queries, cand_lists = [], []
    for r, ids in zip(rows, cand_ids):
        kept = [pid for pid in ids if pid in texts]
        cand_lists.append([ScoredPaper(paper_id=pid, score=0.0, rank=i,
                                       title=texts[pid][0], abstract=texts[pid][1])
                           for i, pid in enumerate(kept, start=1)])
        queries.append(rerank_query_of(r, query_mode))   # 기본은 원본 질문 (위 함수 설명 참고)

    if method == "cross":
        from src.retrieval.ranking import CrossEncoderReranker, DEFAULT_RERANKER
        reranker = CrossEncoderReranker(DEFAULT_RERANKER, batch_size=batch_size)
        ranked = reranker.rerank_batch(queries, cand_lists, top_k=depth)
    elif method == "llm":
        # 언어 모델 재정렬기 (bge-reranker-v2-gemma). 점수 눈금이 교차 인코더와 다름 -
        # 시그모이드 0~1 이 아니라 로짓이므로 MIN_RERANK_SCORE 를 그대로 쓰면 안 됨.
        from src.retrieval.ranking import DEFAULT_LLM_RERANKER, LLMReranker
        reranker = LLMReranker(DEFAULT_LLM_RERANKER, batch_size=batch_size)
        ranked = reranker.rerank_batch(queries, cand_lists, top_k=depth)
    elif method == "embedding":
        from sentence_transformers import SentenceTransformer

        from src.retrieval.ranking import rerank_by_similarity
        embedder = SentenceTransformer(config.EMBED_MODEL)
        ranked = [rerank_by_similarity(q, c, embedder, top_k=depth)
                  for q, c in zip(queries, cand_lists)]
    else:
        raise ValueError(f"알 수 없는 재정렬 방식: {method}")

    for r, papers in zip(rows, ranked):
        r["reranked_ids"] = [p.paper_id for p in papers]
        # 점수를 함께 남김. 이게 없으면 '정답과 오답의 점수가 얼마나 벌어졌는가' 를
        # 보려고 재정렬을 통째로 다시 돌려야 함. 깊이를 바꿔 가며 재는 것도 이 값으로 함.
        r["rerank_scores"] = [round(float(p.score), 6) for p in papers]
        # 어느 깊이로 재정렬했는지 문항에 새겨 둠. 이게 없으면 나중에 이 파일을 다시
        # 집계할 때 명령줄 기본값(fuse_top_n)으로 상한을 계산해 버림 - ISSUE #26 과
        # 똑같은 어긋남이 재집계 단계에서 되살아나는 자리임.
        r["rerank_depth"] = depth
        r["rerank_method"] = method
    print(f"재정렬 완료: {len(rows)}문항, 방식 {method}, 깊이 {depth}")


# -- 재정렬 진단 (저장된 점수만 씀, 재실행 없음) -----------------------------
#
# `print_report` 는 "얼마나 맞혔는가"를 셈. 여기서는 "왜 틀렸는가"를 가름. 재정렬이
# 정답을 상위 10편에서 버릴 때 원인이 둘인데 처방이 정반대임.
#
#   가) 정답과 오답의 점수가 거의 같아서 밀림 -> 구분력을 보태야 함
#                                             (1차 검색 순위 함께 쓰기, 동점 처리 고치기)
#   나) 정답에 확신을 갖고 낮은 점수를 줌      -> 모델을 바꿔야 함
#
# 점수를 저장하기 전에는 이 둘을 가를 방법이 없었음 (2026-08-17 부터 저장함).
#
# 자르는 위치(RERANK_DEPTH)도 여기서 함께 잼. 교차 인코더는 후보를 하나씩 독립적으로
# 채점하므로, 어떤 논문의 점수는 같은 목록에 무엇이 더 들어 있는지와 무관함. 그래서
# 깊이 200 으로 한 번 채점해 두면 깊이 100 의 결과는 '융합 상위 100편만 골라 저장된
# 점수로 다시 줄 세우기' 로 정확히 같은 값이 나옴. 깊이마다 재정렬을 돌릴 필요가 없음.

TIERS = ("전체", "easy", "medium", "hard")


def _score_map(row: dict) -> dict[str, float]:
    return dict(zip(row.get("reranked_ids") or [], row.get("rerank_scores") or []))


def _top10_at_depth(row: dict, depth: int, rrf_k: int, weights: dict[str, float],
                    channel_depth: int | None) -> list[str]:
    """깊이 `depth` 로 잘랐을 때의 최종 상위 10편.

    본문을 못 찾아 재정렬에서 빠진 논문은 점수가 없으므로 목록에서도 빠짐. 실제 실행과
    같은 처리임 (`rerank_rows` 가 `kept` 로 거르는 것과 같음).
    """
    scores = _score_map(row)
    fused = fused_ids_of(row, rrf_k, top_n=depth, weights=weights, depth=channel_depth)
    kept = [pid for pid in fused if pid in scores]
    kept.sort(key=lambda p: -scores[p])
    return kept[:10]


def diagnose_rerank(rows: list[dict], depths: list[int], rrf_k: int,
                    weights: dict[str, float], channel_depth: int | None) -> None:
    """자르는 위치, 점수 분포, 동점 세 가지를 한 번에 봄."""
    from evaluation.metrics import paired_bootstrap

    missing = sum(1 for r in rows if not r.get("rerank_scores"))
    if missing:
        raise SystemExit(f"재정렬 점수가 없는 문항이 {missing}개 있다. "
                         f"--rerank cross 로 다시 만들어야 한다 (점수 저장은 2026-08-17 부터임)")

    # (1) 자르는 위치별 Recall@10
    # 짝지은 검정을 하려면 질문 번호를 열쇠로 들고 있어야 함 (paired_bootstrap 규약).
    hits: dict[int, dict] = {}
    for d in depths:
        per: dict[str, dict[str, float]] = {t: {} for t in TIERS}
        for r in rows:
            ok = float(r["gold_id"] in _top10_at_depth(r, d, rrf_k, weights, channel_depth))
            per["전체"][r["query_id"]] = ok
            per.setdefault(r.get("difficulty", "?"), {})[r["query_id"]] = ok
        hits[d] = per

    print("\n" + "=" * 74)
    print("## 자르는 위치(RERANK_DEPTH)별 Recall@10")
    print("   후보가 채널당 100편씩 최대 200편이므로, 깊이 200 은 '한 편도 안 버림' 임.")
    print(f"\n{'깊이':>6}" + "".join(f"{t:>10}" for t in TIERS) + f"{'평균 후보 수':>14}")
    for d in depths:
        n_cand = np.mean([len(fused_ids_of(r, rrf_k, top_n=d, weights=weights,
                                           depth=channel_depth)) for r in rows])
        print(f"{d:>6}" + "".join(
            f"{np.mean(list(hits[d][t].values())):>10.3f}" for t in TIERS)
              + f"{n_cand:>14.1f}")

    base = depths[0]
    print(f"\n## 깊이 {base} 대비 차이 (같은 문항끼리 짝지어 부트스트랩)")
    for d in depths[1:]:
        for t in TIERS:
            a, b = hits[base][t], hits[d][t]
            if not a:
                continue
            res = paired_bootstrap(a, b)
            mark = "유의미" if res["p_value"] < 0.05 else "판정 불가"
            print(f"   깊이 {d:>3}  {t:<8} {res['delta']:+.3f}  "
                  f"[{res['ci_low']:+.3f},{res['ci_high']:+.3f}]  "
                  f"p={res['p_value']:.3f}  {mark}")

    # (2) 정답이 후보에 있었는데 밀려난 문항의 점수
    depth = depths[-1]
    lost_gold, lost_top1, lost_gap = {}, {}, {}
    for r in rows:
        scores, gold = _score_map(r), r["gold_id"]
        fused = fused_ids_of(r, rrf_k, top_n=depth, weights=weights, depth=channel_depth)
        if gold not in scores or gold not in fused:
            continue
        if gold in _top10_at_depth(r, depth, rrf_k, weights, channel_depth):
            continue
        tier = r.get("difficulty", "?")
        pool = sorted((scores[p] for p in fused if p in scores), reverse=True)
        tenth = pool[9] if len(pool) > 10 else pool[-1]
        lost_gold.setdefault(tier, []).append(scores[gold])
        lost_top1.setdefault(tier, []).append(pool[0])
        lost_gap.setdefault(tier, []).append(tenth - scores[gold])

    print("\n" + "=" * 74)
    print(f"## 정답이 후보에 있었는데 상위 10에서 밀린 문항 (깊이 {depth})")
    print(f"\n   {'난이도':<8}{'문항':>5}{'정답 점수(중앙)':>17}"
          f"{'1등 점수(중앙)':>17}{'10등과의 차이(중앙)':>21}")
    for t in TIERS[1:]:
        if not lost_gold.get(t):
            continue
        print(f"   {t:<10}{len(lost_gold[t]):>5}{np.median(lost_gold[t]):>17.4f}"
              f"{np.median(lost_top1[t]):>17.4f}{np.median(lost_gap[t]):>21.4f}")
    print("\n   '10등과의 차이' 가 0 에 가까우면 아슬아슬하게 밀린 것이고(구분력 문제),")
    print("   크면 모델이 확신을 갖고 정답을 낮게 본 것임(모델 문제). 처방이 서로 다름.")

    # (3) 동점 - np.argsort 가 안정 정렬이 아니라 순서가 정해지지 않는 자리
    print("\n" + "=" * 74)
    print("## 동점 (fp16 반올림으로 원래 다른 점수가 같아질 수 있음)")
    per = {t: [0, 0, 0] for t in TIERS}       # [이웃 동점쌍, 후보 총수, 상위 10 안 동점]
    for r in rows:
        s = r.get("rerank_scores") or []
        keys = ("전체", r.get("difficulty", "?"))
        for key in keys:
            per.setdefault(key, [0, 0, 0])[1] += len(s)
        for i in range(len(s) - 1):
            if s[i] == s[i + 1]:
                for key in keys:
                    per[key][0] += 1
                    if i < 10:
                        per[key][2] += 1
    print(f"\n   {'난이도':<8}{'이웃 동점쌍':>13}{'후보 총수':>12}{'비율':>10}{'상위10 안':>12}")
    for t in TIERS:
        ties, total, top = per[t]
        if total:
            print(f"   {t:<10}{ties:>13,}{total:>12,}{ties/total:>10.4f}{top:>12,}")
    print("\n   상위 10 안 동점이 0 이 아니면 그 자리의 순서는 지금 정해져 있지 않음.")
    print("   ranking.py 의 np.argsort 에 kind='stable' 을 주면 1차 검색 순서가 유지됨.")


# -- 보고 ------------------------------------------------------------------
def pool_depth_of(args) -> int:
    """상한을 계산할 후보 깊이. 재정렬을 켰으면 재정렬에 넣은 깊이가 곧 상한의 기준임."""
    return args.rerank_depth if args.rerank != "none" else args.fuse_top_n


def rerank_pool_rank(row: dict, rrf_k: int, depth: int, weights: dict[str, float]) -> int | None:
    """재정렬에 실제로 들어간 후보 안에 정답이 있는가 (1 또는 None).

    `union_rank` 와 헷갈리면 안 됨. 둘은 다른 집합임.

        union_rank      = 채널마다 상위 depth 편을 모은 합집합
        rerank_pool_rank = 그것을 RRF 로 합쳐 상위 depth 편만 남긴 것  <- 재정렬이 보는 것

    융합은 후보를 줄이므로 두 번째가 더 작음. 첫 번째를 '재정렬의 상한'이라 부르면
    융합이 흘린 몫까지 재정렬 탓으로 넘어감. 실측에서 시험용 300문항 기준
    합집합 0.823 대 실제 재정렬 입력 0.797 로 0.026(정답 8편)이 그렇게 넘어가 있었음.
    회수율이 82.6% 로 보였지만 실제로는 85.4% 였음.

    ISSUE #26 에서 고친 것과 같은 종류의 어긋남임. 그때는 '깊이'를 맞췄고, 이번에는
    '집합을 만드는 방식'을 맞춤. 두 곳 다 같은 함수를 부르게 해서 다시 어긋나지 않게 함.
    """
    return 1 if row["gold_id"] in fused_ids_of(row, rrf_k, top_n=depth,
                                               weights=weights) else None

def print_report(rows: list[dict], title: str, k_values=DEFAULT_K_VALUES,
                 rrf_k: int = 60, weights: dict[str, float] | None = None,
                 fuse_top_n: int = 200, pool_depth: int = 100) -> None:
    """채널별, 합집합(상한), 융합 후, 재정렬 후를 한 표에 놓고 비교함."""
    weights = weights or {}
    n = len(rows)
    names = channel_names_of(rows)
    ks = list(k_values)

    # 파일에 새겨진 재정렬 깊이가 있으면 그것을 씀. 재집계할 때 명령줄 기본값으로
    # 상한을 재면 깊이가 어긋나 회수율이 부풀려짐(ISSUE #26).
    stamped = {r["rerank_depth"] for r in rows if r.get("rerank_depth")}
    if len(stamped) == 1:
        depth_from_file = stamped.pop()
        if depth_from_file != pool_depth:
            print(f"\n(참고: 이 파일은 깊이 {depth_from_file} 로 재정렬돼 있다. "
                  f"명령줄 값 {pool_depth} 대신 그 깊이로 상한을 계산한다)")
        pool_depth = depth_from_file
    elif len(stamped) > 1:
        print(f"\n경고: 문항마다 재정렬 깊이가 다르다({sorted(stamped)}). 상한 계산을 믿지 말 것.")

    print("\n" + "=" * 78)
    print(f"## {title}")
    print(f"   문항 {n}개, 융합 k={rrf_k}, 가중치 {weights or '전부 1.0'}")
    print("   * 오류, 결과 0건 문항을 실패(0점)로 세고 분모는 전체 문항 수다.")
    print("      (오류 문항을 빼고 잰 옛 숫자와 직접 비교하면 안 된다)")

    # 문항 상태
    rewrite_err = [r for r in rows if r.get("error")]
    print(f"\n## 문항 상태   변환 실패 {len(rewrite_err)}건")
    for name in names:
        errs = sum(1 for r in rows if name in (r.get("channel_errors") or {}))
        zeros = sum(1 for r in rows if not (r.get("channels") or {}).get(name))
        depth = max((len((r.get("channels") or {}).get(name) or []) for r in rows), default=0)
        print(f"   {name:<18} 검색 오류 {errs:>3}건, 결과 0건 {zeros:>3}건 "
              f",  최대 깊이 {depth}")

    # 채널별 / 합집합
    print(f"\n## 채널별 Recall (분모 {n}문항, [ ]는 부트스트랩 95% 신뢰구간)")
    header = "   " + f"{'':<20}" + "".join(f"{'@'+str(k):>22}" for k in ks)
    print(header)
    for name in names:
        ranks = [rank_in((r.get("channels") or {}).get(name) or [], r["gold_id"]) for r in rows]
        line = "   " + f"{name:<20}" + "".join(f"{fmt(hits_at(ranks, k)):>22}" for k in ks)
        print(line)

    if len(names) > 1:
        line = "   " + f"{'합집합(상한)':<17}" + "".join(
            f"{fmt(hits_at([union_rank(r, k) for r in rows], 1)):>22}" for k in ks)
        print(line)
        print("   > 합집합 = 각 채널 상위 @K 를 모두 모은 것. 재정렬이 도달할 수 있는 상한임")

    # 융합 후
    fused = [fused_ids_of(r, rrf_k, fuse_top_n, weights) for r in rows]
    fused_ranks = [rank_in(ids, r["gold_id"]) for ids, r in zip(fused, rows)]
    print(f"\n## 융합(RRF) 후 Recall")
    print("   " + f"{'fused':<20}" + "".join(f"{fmt(hits_at(fused_ranks, k)):>22}" for k in ks))

    # 재정렬 후
    has_rr = any("reranked_ids" in r for r in rows)
    rr_ranks = None
    if has_rr:
        rr_ranks = [rank_in(r.get("reranked_ids") or [], r["gold_id"]) for r in rows]
        print(f"\n## 재정렬 후 Recall (재정렬 결과가 없는 문항은 실패로 센다)")
        print("   " + f"{'reranked':<20}" + "".join(f"{fmt(hits_at(rr_ranks, k)):>22}" for k in ks))
    else:
        print("\n## 재정렬 후: 아직 재정렬을 돌리지 않았다 (--rerank cross 로 실행)")

    # 상한과 최종값의 차이 - 손실을 융합 몫과 재정렬 몫으로 나눠서 봄.
    #
    # 상한을 하나만 적으면 처방을 잘못 고름. 채널이 후보를 못 물어온 것인지, 융합이
    # 흘린 것인지, 재정렬이 못 끌어올린 것인지가 전부 다른 문제이기 때문임.
    final_ranks = rr_ranks if rr_ranks is not None else fused_ranks
    final_name = "재정렬 후" if rr_ranks is not None else "융합 후"
    final = float(np.mean(hits_at(final_ranks, 10))) if rows else 0.0

    union_ceiling = float(np.mean(hits_at([union_rank(r, pool_depth) for r in rows], 1))) if rows else 0.0
    if has_rr:
        pool_hits = [rerank_pool_rank(r, rrf_k, pool_depth, weights) for r in rows]
        pool_ceiling = float(np.mean(hits_at(pool_hits, 1))) if rows else 0.0
    else:
        pool_ceiling = union_ceiling      # 재정렬을 안 했으면 융합 결과가 곧 최종 후보임

    recovered = (f"상한의 {final / pool_ceiling:.1%} 회수"
                 if pool_ceiling > 0 else "상한이 0이라 계산 불가")
    print(f"\n## 상한과 최종값의 차이")
    print(f"   (1) 채널 합집합 @{pool_depth}            : {union_ceiling:.3f}"
          f"   (융합을 완벽히 하면 도달 가능한 값)")
    print(f"   (2) 재정렬이 실제로 본 후보        : {pool_ceiling:.3f}"
          f"   ((1)을 RRF로 합쳐 상위 {pool_depth}편만 남긴 것)")
    print(f"   (3) 최종 Recall@10 ({final_name})    : {final:.3f}")
    print(f"   융합이 흘린 몫  ((1) -> (2))         : {pool_ceiling - union_ceiling:+.3f}")
    print(f"   재정렬이 못 건진 몫 ((2) -> (3))      : {final - pool_ceiling:+.3f}  ({recovered})")
    print("   해석: (1)이 낮으면 채널을 더 늘린다. (1)->(2) 손실이 크면 융합 방식을 고친다")
    print("         (재정렬기는 후보의 순서를 안 보므로, 후보를 줄이는 융합은 손해만 된다).")
    print("         (2)->(3) 손실이 크면 재정렬을 고친다. 세 처방은 전부 다르다.")

    # 언어별, 난이도별 (어디서 실패하는지)
    for field in ("lang", "difficulty"):
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(str(r.get(field)), []).append(i)
        print(f"\n## {field}별 Recall@10 (최종={final_name}, 상한=재정렬이 실제로 본 후보)")
        for key, idxs in sorted(groups.items()):
            sub_final = float(np.mean([hits_at([final_ranks[i]], 10)[0] for i in idxs]))
            sub_ceiling = float(np.mean([hits_at([pool_hits[i]], 1)[0] for i in idxs])
                                if has_rr else
                                np.mean([hits_at([union_rank(rows[i], pool_depth)], 1)[0]
                                         for i in idxs]))
            print(f"   {key:<10} n={len(idxs):<4} 최종 {sub_final:.3f}, 상한 {sub_ceiling:.3f}")


# -- 결과 파일 입출력 (이어하기) --------------------------------------------
def load_done(out_path: Path) -> dict[str, dict]:
    """이미 평가된 문항을 불러옴.

    어느 채널이든 오류가 난 문항은 다시 시도함. arXiv 의 429/503 은 기다리면 풀리는
    일시적 오류라, 그대로 두면 없어도 될 실패가 지표에 남음.
    """
    if not out_path.exists():
        return {}
    done = {}
    for row in read_jsonl(out_path):
        if row.get("_meta") or row.get("error") or row.get("channel_errors"):
            continue
        done[row["query_id"]] = row
    return done


# -- 서비스 응답 시간 실측 --------------------------------------------------
def bench_service(args) -> None:
    """질문 하나에 몇 초가 걸리는지 단계별로 잼.

    ## 왜 재야 하는가

    성능 숫자는 여러 번 쟀지만 응답 시간은 오래 안 쟀음. 그런데 사람은 30초를 넘으면
    떠남. 정확도를 아무리 올려도 느리면 아무도 안 씀.

    특히 지금 구조는 한 번의 검색에 언어 모델을 세 번 부름.

      1. 논문 지목 확인 (paper_resolver)
      2. 쿼리 변환 (dpo)
      3. 추천 이유 생성 (recommender)

    여기에 arXiv 호출과 교차 인코더 재정렬이 끼어 있음. 재정렬은 후보 수에 비례해 늘어남.

    ## 재기 전에 깊이를 줄이지 않는 이유

    재정렬 깊이를 서비스용으로 줄이고 싶은 유혹이 있지만, 얼마나 느린지 모르는 채로 줄이면
    얻는 것도 잃는 것도 모른 채 바꾸는 것임. 먼저 평가와 같은 설정으로 재고, 그 숫자를
    보고 줄일지 정함.

    ## GPU 를 한 장만 쓴다는 점

    재정렬과 임베딩이 같은 GPU 에서 일어나므로, 사용자가 여러 명이면 줄을 섬. 여기서 재는
    것은 혼자 쓸 때의 시간이라 실제 서비스에서는 더 느릴 수 있음. 그 점을 감안해 읽어야 함.
    """
    import statistics as st

    queries = BENCH_QUERIES[: args.n]

    print("부품 불러오는 중... (이 시간은 서비스 시작 시 한 번만 든다)")
    t0 = time.time()
    from src.recommend_agent.recommender import PaperRecommender
    from src.retrieval.arxiv_live import ArxivLiveRetriever
    from src.retrieval.local_index import LocalDenseRetriever
    from src.retrieval.ranking import CrossEncoderReranker
    from src.rewriter.paper_resolver import PaperResolver, resolve_and_verify

    load: dict[str, float] = {}
    t = time.time(); rewriter = build_rewriter(args.rewriter); load["쿼리 변환기"] = time.time() - t
    t = time.time(); index = LocalDenseRetriever(args.corpus, args.index)
    load["로컬 색인 71만 편"] = time.time() - t
    t = time.time(); reranker = CrossEncoderReranker(); load["재정렬 모델"] = time.time() - t
    # 추천은 변환기가 이미 올린 모델을 빌려 씀 (같은 Qwen3-4B 를 두 벌 올리지 않기 위함)
    t = time.time()
    recommender = (PaperRecommender(client=rewriter)
                   if hasattr(rewriter, "generate_json") else PaperRecommender())
    load["추천 에이전트"] = time.time() - t
    arxiv = None if args.skip_arxiv else ArxivLiveRetriever()
    resolver = None if args.skip_resolver else PaperResolver()

    print("\n## 시작 시 준비 시간 (한 번만)")
    for k, v in load.items():
        print(f"   {k:<20} {v:6.1f}초")
    print(f"   {'합계':<20} {time.time()-t0:6.1f}초")

    stages = ["논문 지목 확인", "쿼리 변환", "로컬 의미 검색", "arXiv 검색", "재정렬", "추천"]
    times: dict[str, list[float]] = {s: [] for s in stages}
    totals: list[float] = []

    print(f"\n## 질문 {len(queries)}개 측정 "
          f"(로컬 {args.k} + arXiv {args.k} -> 재정렬 {args.rerank_depth})\n")

    for i, q in enumerate(queries, 1):
        one: dict[str, float] = {}
        q0 = time.time()

        if resolver and arxiv:
            t = time.time()
            try:
                resolve_and_verify(q, resolver, arxiv)
            except Exception:
                pass
            one["논문 지목 확인"] = time.time() - t

        t = time.time(); rw = rewriter.rewrite(q); one["쿼리 변환"] = time.time() - t

        t = time.time()
        local_hits = index.search(q, k=args.k)
        one["로컬 의미 검색"] = time.time() - t

        arxiv_hits = []
        if arxiv:
            t = time.time()
            try:
                arxiv_hits = arxiv.search(rw.query_for("arxiv"), k=args.k)
            except Exception as e:
                print(f"   (arXiv 오류: {type(e).__name__})")
            one["arXiv 검색"] = time.time() - t

        seen: dict[str, object] = {}
        for p in list(local_hits) + list(arxiv_hits):
            seen.setdefault(normalize_paper_id(p.paper_id), p)
        cands = list(seen.values())[: args.rerank_depth]

        t = time.time()
        results = reranker.rerank(q, cands, top_k=10)
        one["재정렬"] = time.time() - t

        t = time.time(); recommender.recommend(q, results); one["추천"] = time.time() - t

        totals.append(time.time() - q0)
        for k, v in one.items():
            times[k].append(v)
        print(f"   {i}. {totals[-1]:5.1f}초  후보 {len(cands):>3}편  {q[:38]}")

    def p95(xs: list[float]) -> float:
        return sorted(xs)[max(0, int(len(xs) * 0.95) - 1)] if xs else 0.0

    print("\n## 단계별 (질문 하나 기준)\n")
    print(f"   {'단계':<16}{'중앙값':>9}{'95분위':>9}{'비중':>8}")
    med_total = st.median(totals)
    for s in stages:
        xs = times[s]
        if not xs:
            continue
        m = st.median(xs)
        print(f"   {s:<16}{m:>8.1f}초{p95(xs):>8.1f}초{m/med_total:>7.0%}")
    print(f"   {'-'*40}")
    print(f"   {'합계':<16}{med_total:>8.1f}초{p95(totals):>8.1f}초")

    print("\n## 판정")
    verdict = ("사람이 떠나는 30초를 넘는다. 단계를 줄여야 한다." if p95(totals) > 30 else
               "20초를 넘어 체감이 느리다. 줄일 여지를 보는 것이 좋다." if p95(totals) > 20 else
               "30초 기준을 지킨다.")
    print(f"   95분위 {p95(totals):.1f}초 - {verdict}")
    print("   * GPU 한 장에서 잰 값이라 사용자가 여러 명이면 줄을 서서 더 느려진다.")


# -- "못 찾았다"고 말할 기준선 정하기 ---------------------------------------
def calibrate_threshold(args) -> None:
    """재정렬 점수 몇 점 아래를 '무관'으로 볼지 실측으로 정함.

    ## 왜 필요한가

    로컬 의미 검색은 어떤 질문에도 항상 상위 10편을 채워서 돌려줌. 아무리 관련 없는
    질문이어도 유사도가 가장 높은 10편이 나옴. 그대로 화면에 뿌리면 무관한 논문을
    추천으로 포장하는 서비스가 됨. 옛 arXiv 키워드 검색은 못 찾으면 0건을 돌려줘서
    이 문제가 없었는데, 구조를 바꾸면서 새로 생긴 위험임.

    ## 어떻게 정하는가

    기준선을 감으로 정하면 안 됨. 등급 정답지에 관련도 0~3 이 매겨져 있으므로, 그것을
    정답으로 놓고 재정렬 점수가 등급을 얼마나 갈라내는지 잼.

        만족(등급 2~3)  대  무관(등급 0)  을 가장 잘 가르는 점수는 몇인가?

    임계값 후보마다 '무관을 걸러낸 비율'과 '만족을 잘못 버린 비율'을 함께 찍음.
    한쪽만 보고 고르면 안 됨 - 무관을 다 걸러내는 높은 값은 좋은 논문도 같이 버림.

    ## 언어별로 따로 봄

    한국어 질문과 영어 초록을 맞대는 점수와, 영어끼리 맞대는 점수가 같은 눈금이라는 보장이
    없음. 하나의 기준선을 두 언어에 함께 쓰려면 먼저 그래도 되는지 확인해야 함.
    """
    queries = [q for q in read_jsonl(args.queries) if not q.get("_meta")]
    grades = {r["pair_id"]: {normalize_paper_id(k): int(v) for k, v in r["grades"].items()}
              for r in read_jsonl(args.grades)}
    queries = [q for q in queries if q.get("pair_id") in grades]
    if not queries:
        print("등급 정답지와 겹치는 문항이 없다. --queries 와 --grades 가 같은 분할인지 확인할 것.")
        return
    print(f"문항 {len(queries)}개, 등급 정답지 {len(grades)}짝")

    # 후보 본문을 모음 (검색 0회 - 등급 정답지에 이미 논문 번호가 있음)
    wanted = {pid for q in queries for pid in grades[q["pair_id"]]}
    lookup = TextLookup(args.index, args.corpus, None)
    print(f"후보 본문 수집: 고유 논문 {len(wanted):,}편 ...", flush=True)
    texts = lookup.fetch(wanted)
    print(f"  {len(texts):,}편 확보 (못 찾은 {len(wanted)-len(texts):,}편은 뺀다)")

    pairs, langs, gs = [], [], []
    for q in queries:
        for pid, g in grades[q["pair_id"]].items():
            if pid in texts:
                pairs.append((q["text"], f"{texts[pid][0]}\n{texts[pid][1]}".strip()))
                langs.append(q.get("lang", "?"))
                gs.append(g)
    print(f"채점할 (질문, 논문) 쌍 {len(pairs):,}개", flush=True)

    from src.retrieval.ranking import DEFAULT_RERANKER, CrossEncoderReranker
    reranker = CrossEncoderReranker(DEFAULT_RERANKER, batch_size=args.batch_size)
    t0 = time.time()
    scores = np.asarray(reranker.model.predict(pairs, batch_size=args.batch_size,
                                               show_progress_bar=False), dtype=np.float64)
    print(f"채점 완료 ({time.time()-t0:.0f}초)\n")

    langs, gs = np.array(langs), np.array(gs)

    print("=" * 78)
    print("## 등급별 재정렬 점수 분포")
    print(f"\n{'등급':<18}{'n':>6}{'중앙값':>10}{'25분위':>10}{'75분위':>10}")
    labels = {3: "3 정답급", 2: "2 쓸모 있음", 1: "1 주변적", 0: "0 무관"}
    for g in (3, 2, 1, 0):
        s = scores[gs == g]
        if len(s):
            print(f"{labels[g]:<18}{len(s):>6}{np.median(s):>10.2f}"
                  f"{np.percentile(s, 25):>10.2f}{np.percentile(s, 75):>10.2f}")

    print("\n## 언어별로 눈금이 같은가 (하나의 기준선을 함께 써도 되는지 확인)")
    print(f"\n{'언어':<8}{'만족(2~3) 중앙값':>20}{'무관(0) 중앙값':>20}{'차이':>10}")
    for lang in sorted(set(langs.tolist())):
        good = scores[(langs == lang) & (gs >= 2)]
        bad = scores[(langs == lang) & (gs == 0)]
        if len(good) and len(bad):
            print(f"{lang:<8}{np.median(good):>20.2f}{np.median(bad):>20.2f}"
                  f"{np.median(good) - np.median(bad):>10.2f}")

    print("\n## 기준선 후보 - 무관을 얼마나 걸러내고, 좋은 논문을 얼마나 잘못 버리는가")
    print("   (한쪽만 보고 고르면 안 된다. 무관을 다 걸러내는 값은 좋은 논문도 같이 버림)")
    print("   '만족 잘못 버림' 은 언어별로도 봄. 한쪽 언어만 심하게 걸리면 하나의 기준선을")
    print("   두 언어에 함께 쓸 수 없기 때문임.")
    langs_seen = sorted(set(langs.tolist()))
    head = (f"\n{'기준선':>8}{'무관 걸러냄':>14}{'만족 잘못 버림':>16}"
            + "".join(f"{'  (' + lg + ')':>10}" for lg in langs_seen)
            + f"{'남는 것 중 만족':>18}")
    print(head)
    good, bad = scores[gs >= 2], scores[gs == 0]
    for thr in args.thresholds:
        kept = scores >= thr
        precision = float((gs[kept] >= 2).mean()) if kept.any() else float("nan")
        per_lang = ""
        for lg in langs_seen:
            g = scores[(langs == lg) & (gs >= 2)]
            per_lang += f"{float((g < thr).mean()):>10.1%}" if len(g) else f"{'-':>10}"
        print(f"{thr:>8.3f}{float((bad < thr).mean()):>14.1%}"
              f"{float((good < thr).mean()):>16.1%}{per_lang}{precision:>18.1%}")

    print("\n* 이 값은 재정렬 모델을 바꾸면 반드시 다시 재야 한다. 점수 눈금이 모델마다 다르다.")
    print("   정한 값은 app.py 의 MIN_RERANK_SCORE 에 넣고 그 자리에 이 명령을 적어 둔다.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="검색 파이프라인 평가 (채널별 결과를 따로 저장해 재계산) + 응답 시간 실측")
    ap.add_argument("--queries", default="data/eval/dev.jsonl",
                    help="탐색은 dev 로만 한다. test 는 확정 판정 때만 쓴다 (ISSUE #25)")
    ap.add_argument("--channels", nargs="+", default=["arxiv", "local_dense"],
                    help="arxiv, local_dense(71만 편 로컬 의미 검색)")
    ap.add_argument("--local-query", default="raw", choices=["raw", "rewritten"],
                    help="로컬 의미 검색에 넣을 검색어. raw(기본)는 원본 질문으로 **서비스와 "
                         "같은 조건**이다. rewritten 은 변환 결과를 넣는다 (파일 위 설명 참고)")
    ap.add_argument("--rewriter", default="passthrough",
                    help="변환기 이름 (passthrough / hierarchical / single_step / hyde / finetuned / dpo)")
    ap.add_argument("--k", type=int, default=100, help="채널마다 가져올 결과 수")
    ap.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (빠른 점검용)")
    ap.add_argument("--sample", type=int, default=None, help="무작위 N개")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="문항별 상세 결과 저장 경로")
    ap.add_argument("--no-cache", action="store_true", help="arXiv 디스크 캐시 사용 안 함")
    ap.add_argument("--no-resume", action="store_true", help="처음부터 다시 평가")
    ap.add_argument("--report-only", default=None,
                    help="저장된 결과 파일만 다시 집계한다 (검색 0회)")
    ap.add_argument("--use-channels", nargs="+", default=None,
                    help="--report-only 에서 저장된 채널 중 일부만 쓴다. 같은 검색 결과 위에서 "
                         "채널 조합을 갈라 비교할 때 쓴다. --out 과 함께 줄 것")

    ap.add_argument("--rrf-k", type=int, default=config.RRF_K, help="RRF 완충 상수")
    ap.add_argument("--weights", default=None,
                    help="채널 가중치. 예: 'arxiv=1.0,local_dense=2.0'")
    ap.add_argument("--fuse-top-n", type=int, default=200, help="융합 결과를 몇 편까지 볼지")

    ap.add_argument("--rerank", default="none",
                    choices=["none", "cross", "embedding", "llm"],
                    help="재정렬 방식. 저장된 결과 위에서 돌아가므로 검색은 다시 하지 않는다")
    ap.add_argument("--rerank-depth", type=int, default=100, help="재정렬에 넣을 후보 수")
    ap.add_argument("--channel-depth", type=int, default=None,
                    help="융합 전에 채널마다 몇 편까지만 볼지 (서비스와 조건을 맞출 때 씀)")
    ap.add_argument("--diagnose", default=None,
                    help="재정렬 진단: 저장된 재정렬 점수로 자르는 위치, 점수 분포, 동점을 봄")
    ap.add_argument("--diagnose-depths", type=int, nargs="+", default=[100, 150, 200],
                    help="--diagnose 에서 견줄 자르는 위치 목록")
    ap.add_argument("--rerank-query", default="raw", choices=["raw", "rewritten"],
                    help="재정렬기에 넣을 질문. raw(기본)는 원본, rewritten 은 변환 결과. "
                         "한국어를 영어로 옮기는 변환기를 쓸 때만 rewritten 이 의미가 있음")
    ap.add_argument("--batch-size", type=int, default=32)

    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--index", default=str(config.DATA_DIR / "embeddings" / "cs2021"))
    ap.add_argument("--mmap", action="store_true",
                    help="임베딩을 메모리에 올리지 않고 디스크에서 읽는다(메모리 절약, 느림)")

    ap.add_argument("--bench-service", action="store_true",
                    help="정확도가 아니라 응답 시간을 단계별로 잰다 (평가셋 대신 예시 질문)")
    ap.add_argument("--calibrate-threshold", action="store_true",
                    help="'못 찾았다'고 말할 재정렬 점수 기준선을 등급 정답지로 실측한다")
    ap.add_argument("--grades", default=None,
                    help="--calibrate-threshold 에서 쓸 등급 정답지 (grades_dev.jsonl 등)")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
                    help="--calibrate-threshold 에서 시험해 볼 기준선 후보")
    ap.add_argument("--n", type=int, default=5, help="--bench-service 에서 잴 질문 수")
    ap.add_argument("--skip-arxiv", action="store_true",
                    help="--bench-service 에서 arXiv 호출을 뺀다 (요청 제한을 아끼고 싶을 때)")
    ap.add_argument("--skip-resolver", action="store_true",
                    help="--bench-service 에서 논문 지목 확인 단계를 뺀다")
    args = ap.parse_args()

    if args.bench_service:
        bench_service(args)
        return

    if args.calibrate_threshold:
        if not args.grades:
            ap.error("--calibrate-threshold 에는 --grades 가 필요하다")
        calibrate_threshold(args)
        return

    weights = parse_weights(args.weights)

    # -- 재정렬 진단 모드 (검색 0회, 재정렬 0회) ---------------------------
    if args.diagnose:
        rows = [r for r in read_jsonl(Path(args.diagnose))
                if not r.get("_meta") and r.get("gold_id")]
        print(f"문항 {len(rows)}개, 파일 {args.diagnose}")
        diagnose_rerank(rows, args.diagnose_depths, args.rrf_k, weights,
                        args.channel_depth)
        return

    # -- 재집계 전용 모드 (검색 0회) ---------------------------------------
    if args.report_only:
        path = Path(args.report_only)
        all_rows = list(read_jsonl(path))
        rows = [r for r in all_rows if not r.get("_meta")]
        metas = [r for r in all_rows if r.get("_meta")]
        title = path.stem

        if args.use_channels:
            rows = select_channels(rows, args.use_channels)
            title = f"{title}, 채널 {'+'.join(args.use_channels)}"
            for m in metas:
                m["channels"] = list(args.use_channels)

        if args.rerank != "none":
            lookup = TextLookup(args.index, args.corpus,
                                None if args.no_cache else CACHE_PATH)
            rerank_rows(rows, args.rerank, args.rerank_depth, lookup,
                        args.rrf_k, weights, args.batch_size, args.rerank_query,
                        args.channel_depth)
            # 재정렬 결과를 파일에 되써서 재사용함 (실행 정보 _meta 줄은 그대로 보존).
            # --out 을 주면 원본을 건드리지 않고 새 파일로 씀 - 같은 검색 결과에서
            # 채널 조합을 여러 가지로 갈라 볼 때 서로 덮어쓰지 않기 위함임.
            dest = Path(args.out) if args.out else path
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(dest, rows + metas)
            print(f"재정렬 결과를 파일에 반영: {dest}")

        print_report(rows, title, tuple(args.k_values), args.rrf_k, weights,
                     args.fuse_top_n, pool_depth_of(args))
        return

    queries = list(read_jsonl(args.queries))
    if args.sample:
        import random
        random.Random(args.seed).shuffle(queries)
        queries = queries[: args.sample]
    elif args.limit:
        queries = queries[: args.limit]

    out_path = Path(args.out or
                    f"runs/pipeline_{Path(args.queries).stem}_{args.rewriter}.jsonl")

    done = {} if args.no_resume else load_done(out_path)
    todo = [q for q in queries if q["query_id"] not in done]
    if done:
        print(f"이어하기: 이미 평가된 {len(done)}문항 건너뜀 -> 남은 {len(todo)}문항")

    results = [done[q["query_id"]] for q in queries if q["query_id"] in done]
    if todo:
        rewriter = build_rewriter(args.rewriter)
        channels = {name: build_channel(name, args) for name in args.channels}
        print(f"평가 시작: 문항 {len(todo)}개, 변환기={args.rewriter} "
              f",  채널={list(channels)}, k={args.k}")

        fields = channel_query_fields(args.local_query)
        t0 = time.time()
        for i, q in enumerate(todo, 1):
            results.append(evaluate_one(q, rewriter, channels, args.k, fields))
            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(todo) - i)
                print(f"  {i}/{len(todo)}, 경과 {elapsed/60:.1f}분 "
                      f",  남은 예상 {eta/60:.1f}분", flush=True)
                write_jsonl(out_path, results)      # 중간 저장 - 중단돼도 여기까지는 살아남음

    if args.rerank != "none":
        lookup = TextLookup(args.index, args.corpus,
                            None if args.no_cache else CACHE_PATH)
        rerank_rows(results, args.rerank, args.rerank_depth, lookup,
                    args.rrf_k, weights, args.batch_size, args.rerank_query,
                    args.channel_depth)

    meta = {"_meta": True, "rewriter": args.rewriter, "queries": args.queries,
            "channels": args.channels, "k": args.k, "rrf_k": args.rrf_k,
            "local_query": args.local_query,
            "weights": weights, "rerank": args.rerank,
            "n_queries": len(results), "commit": git_commit(),
            "finished_at": datetime.now().isoformat(timespec="seconds")}
    write_jsonl(out_path, results + [meta])

    print_report(results, f"{args.rewriter}, 채널 {'+'.join(args.channels)}",
                 tuple(args.k_values), args.rrf_k, weights, args.fuse_top_n, pool_depth_of(args))
    print(f"\n상세 결과 저장: {out_path}  (커밋 {meta['commit']})")
    print("융합 방식을 바꿔 재계산: "
          f"python -m evaluation.pipeline_eval --report-only {out_path} --rrf-k 30")


if __name__ == "__main__":
    main()
