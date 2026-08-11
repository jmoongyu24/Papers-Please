"""다채널 검색 파이프라인 평가 하네스 — 채널별 결과를 따로 저장해 몇 번이든 재계산한다.

## 왜 하네스를 새로 만드는가

`evaluation/arxiv_eval.py` 는 검색 경로가 **하나(arXiv 실시간)** 라는 전제로 짜여 있어
`retrieved_ids` 한 줄만 저장한다. 이제 경로가 둘 이상(arXiv 키워드 + 로컬 의미 검색)이 되면
그 한 줄로는 다음 질문에 답할 수 없다.

  - 각 채널은 단독으로 얼마나 찾아내는가?
  - **두 채널을 합치면 후보 상한이 얼마나 올라가는가?** (ISSUE #21: 이 상한을 0.70 위로
    올리지 못하면 목표 달성이 산술적으로 불가능하다)
  - 그 상한 중 융합·재정렬이 실제로 얼마나 회수하는가?

그래서 이 하네스는 **채널별 검색 결과 논문 번호를 전부 따로** 저장한다.

    {"channels": {"arxiv": [...100편...], "local_dense": [...100편...]}}

이것이 이 하네스의 핵심 가치다. 융합 상수(k)·채널 가중치·재정렬 방식을 바꿔 가며 실험할 때
**검색을 다시 하지 않는다.** arXiv 는 요청당 3초 제한이 있어 300문항 재검색이 15분 이상
걸리고 429/503 으로 문항이 유실되기도 하는데, 그 비용을 한 번만 치른다.

## 실패를 세는 방식 (ISSUE #23 점검표 3번)

**오류·결과 0건 문항을 제외하지 않고 실패(0점)로 센다.** 제외하면 기준선이 부풀려진다
(같은 실행을 그렇게 재면 passthrough 가 0.257 대신 0.467 로 보인다). 이 하네스의 모든
비율은 분모가 **전체 문항 수**다. `arxiv_eval.py` 는 오류를 제외하는 옛 방식이라 두
하네스의 숫자를 직접 비교하면 안 된다.

## 정답 누수에 대하여 (ISSUE #22)

검색어는 **오직 질문 글(`text`)** 에서만 만든다. 정답 논문의 제목·초록은 검색어를 만드는
어느 단계에서도 읽지 않는다. 반대로 로컬 색인이 정답 논문을 **검색해 후보로 내놓는 것은
정상이다** — 실제 서비스도 arXiv 전체를 검색 대상으로 삼는다. 금지되는 것은 정답 논문의
글을 보고 검색어를 만드는 것이며, 그 둘은 다른 이야기다.

## 실행 예

    # 채널 두 개로 300문항 (이어하기 켜짐)
    python -m evaluation.pipeline_eval --queries data/eval/train.jsonl \\
        --channels arxiv local_dense --rewriter dpo --k 100 \\
        --out runs/train300_multichannel.jsonl

    # 저장된 결과만 다시 집계 (검색 0회). 융합 상수·가중치를 바꿔 가며 반복 가능
    python -m evaluation.pipeline_eval --report-only runs/train300_multichannel.jsonl \\
        --rrf-k 30 --weights local_dense=2.0

    # 저장된 결과에 재정렬만 다시 적용 (검색 0회)
    python -m evaluation.pipeline_eval --report-only runs/train300_multichannel.jsonl \\
        --rerank cross --rerank-depth 100
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
from src.retrieval.fusion import normalize_paper_id, rrf_fuse_ids
from src.rewriter.base import build_rewriter
from src.schemas import ScoredPaper
from src.utils import read_jsonl, write_jsonl

CACHE_PATH = config.DATA_DIR / "cache" / "arxiv_search_cache.jsonl"
DEFAULT_K_VALUES = (1, 5, 10, 30, 50, 100)

# 채널이 쓸 검색어 필드 (`RewriteResult.queries` 의 키). 없는 키를 부르면 원본 질문으로
# 자동 폴백되므로, passthrough 변환기에서는 모든 채널이 원본 질문을 그대로 쓴다.
CHANNEL_QUERY_FIELD = {
    "arxiv": "arxiv",              # 키워드 검색용 (필드 지정·불리언 연산자가 든 문자열)
    "local_dense": "dense",        # 의미 검색용 (자연스러운 문장)
    "local_dense_small": "dense",
}


def git_commit() -> str:
    """재현을 위해 지금 코드가 어느 커밋인지 기록한다."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_weights(spec: str | None) -> dict[str, float]:
    """'arxiv=1.0,local_dense=2.0' 형태를 사전으로 바꾼다."""
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


# ── 채널 만들기 ────────────────────────────────────────────────────────────
def build_channel(name: str, args) -> object:
    """이름으로 검색 채널(검색기)을 만든다. 모두 `search(query, k)` 인터페이스를 따른다."""
    if name == "arxiv":
        from src.retrieval.arxiv_live import ArxivLiveRetriever
        return ArxivLiveRetriever(cache_path=None if args.no_cache else CACHE_PATH)

    if name == "local_dense":
        from src.retrieval.large_index import LargeDenseRetriever
        return LargeDenseRetriever(args.large_corpus, args.large_index, mmap=args.mmap)

    if name == "local_dense_small":
        # 3만 편(corpus-v1) 색인. **코드가 도는지 확인하는 용도로만 쓴다.**
        # 건초더미가 71만 편의 24분의 1이라 여기서 나온 수치는 부풀려진 값이다(ISSUE #10).
        from src.retrieval.dense import DenseRetriever
        r = DenseRetriever.build(config.CORPUS_DIR / "corpus-v1.jsonl")
        r.name = "local_dense_small"
        print("경고: 3만 편 색인은 스모크 테스트용이다. 여기서 나온 수치를 성능으로 읽지 말 것.")
        return r

    raise ValueError(f"알 수 없는 채널 이름: {name}")


# ── 질문 하나 평가 ─────────────────────────────────────────────────────────
def evaluate_one(query_row: dict, rewriter, channels: dict, k: int) -> dict:
    """질문 하나를 모든 채널로 검색해 **채널별 결과를 따로** 기록한다.

    한 채널이 실패해도 나머지 채널은 계속 돌린다. 채널 오류는 `channel_errors` 에 남기고
    그 채널의 결과는 빈 목록이 된다(집계에서 실패로 센다).
    """
    out = {
        "query_id": query_row["query_id"], "text": query_row["text"],
        "gold_id": normalize_paper_id(query_row["gold_id"]),
        "lang": query_row.get("lang"), "difficulty": query_row.get("difficulty"),
    }

    t0 = time.time()
    try:
        # ★ 검색어는 질문 글에서만 만든다. 정답 논문의 본문은 여기서 절대 읽지 않는다.
        rw = rewriter.rewrite(query_row["text"])
    except Exception as e:
        # 변환 실패면 어느 채널도 검색어를 못 받는다 → 문항 전체를 실패로 기록한다.
        out["error"] = f"rewrite_failed: {type(e).__name__}: {e}"
        out["channels"] = {name: [] for name in channels}
        return out
    out["rewrite_ok"] = rw.parse_ok
    out["academic_terms"] = rw.academic_terms
    out["rewrite_sec"] = round(time.time() - t0, 2)

    out["search_queries"], out["channels"] = {}, {}
    out["channel_errors"], out["channel_sec"] = {}, {}
    for name, retriever in channels.items():
        query = rw.query_for(CHANNEL_QUERY_FIELD.get(name, "dense"))
        out["search_queries"][name] = query
        t = time.time()
        try:
            results = retriever.search(query, k=k)
        except Exception as e:
            out["channel_errors"][name] = f"{type(e).__name__}: {e}"
            out["channels"][name] = []
            continue
        # ★ 순위대로 전부 저장 — 융합·재정렬을 나중에 재검색 없이 다시 계산하기 위함
        out["channels"][name] = [normalize_paper_id(r.paper_id) for r in results]
        out["channel_sec"][name] = round(time.time() - t, 2)
    return out


# ── 저장된 결과에서 순위 계산 ──────────────────────────────────────────────
def channel_names_of(rows: list[dict]) -> list[str]:
    """결과 파일에 들어 있는 채널 이름을 등장 순서대로 모은다."""
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

    각 채널의 상위 `depth` 편을 모은 집합. **재정렬이 도달할 수 있는 상한**이다.
    재정렬은 후보 집합을 바꾸지 않고 순서만 바꾸므로 이 값을 넘을 수 없다.
    """
    gold = row["gold_id"]
    for ids in (row.get("channels") or {}).values():
        if gold in (ids or [])[:depth]:
            return 1
    return None


def fused_ids_of(row: dict, rrf_k: int, top_n: int, weights: dict[str, float],
                 depth: int | None = None) -> list[str]:
    """저장된 채널별 논문 번호를 RRF로 합친다 (검색 없음)."""
    channels = {name: (ids or [])[:depth] if depth else (ids or [])
                for name, ids in (row.get("channels") or {}).items()}
    if not channels:
        return []
    return rrf_fuse_ids(channels, k=rrf_k, top_n=top_n, weights=weights)


def hits_at(ranks: list[int | None], k: int) -> list[float]:
    """등수 목록을 Recall@k 의 0/1 목록으로. **None(못 찾음·오류·0건)은 0점**이다."""
    return [1.0 if (r is not None and r <= k) else 0.0 for r in ranks]


def fmt(hits: list[float], with_ci: bool = True) -> str:
    if not hits:
        return "n/a"
    if not with_ci:
        return f"{float(np.mean(hits)):.3f}"
    m, lo, hi = bootstrap_ci(hits)
    return f"{m:.3f} [{lo:.3f},{hi:.3f}]"


# ── 재정렬 (저장된 결과 위에서, 검색 없이) ─────────────────────────────────
class TextLookup:
    """재정렬 후보의 제목·초록을 꺼내 온다.

    71만 편을 통째로 메모리에 올리지 않는다(제목·초록만 1GB가 넘는다). 대신
    - 로컬 색인: 줄 위치(`offsets.npy`)로 필요한 논문만 파일에서 `seek` 해 읽고,
    - arXiv 결과: 검색 캐시 파일을 **한 번만 훑으며** 필요한 번호만 골라 담는다.
    """

    def __init__(self, index_prefix: str | Path | None, corpus_path: str | Path | None,
                 arxiv_cache: str | Path | None):
        self.corpus_path = Path(corpus_path) if corpus_path else None
        self.arxiv_cache = Path(arxiv_cache) if arxiv_cache else None
        self.pos: dict[str, int] = {}
        self.offsets = None
        if index_prefix and self.corpus_path and self.corpus_path.exists():
            from src.retrieval.large_index import index_paths
            paths = index_paths(index_prefix)
            if paths["ids"].exists() and paths["offsets"].exists():
                # 줄 위치는 **그 코퍼스 파일 전용**이다. 다른 파일에 갖다 쓰면 오류 없이
                # 엉뚱한 논문의 본문을 읽어 재정렬이 조용히 망가진다. 짝이 맞는지 확인한다.
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
        """번호 집합 -> {번호: (제목, 초록)}. 못 찾은 번호는 빠진다."""
        out: dict[str, tuple[str, str]] = {}
        if self.offsets is not None:
            todo = sorted((self.pos[p], p) for p in wanted if p in self.pos)
            with open(self.corpus_path, "rb") as f:
                for i, pid in todo:                     # 위치 순서로 읽어 디스크 이동을 줄인다
                    f.seek(int(self.offsets[i]))
                    row = json.loads(f.readline())
                    out[pid] = (row.get("title", "").strip(), row.get("abstract", "").strip())
        elif self.corpus_path and self.corpus_path.exists():
            # 줄 위치 색인이 없는 코퍼스(예: 3만 편 corpus-v1)는 한 번 훑어 필요한 것만 담는다.
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


def rerank_rows(rows: list[dict], method: str, depth: int, lookup: TextLookup,
                rrf_k: int, weights: dict[str, float], batch_size: int = 32) -> None:
    """저장된 결과를 융합한 뒤 상위 `depth` 편을 재정렬해 `reranked_ids` 로 채운다.

    검색은 한 번도 하지 않는다. 후보 본문을 못 찾은 논문은 후보에서 빠지는데, 그 논문이
    정답이면 손해이므로 **몇 편이나 빠졌는지 반드시 보고한다**(조용히 성능이 깎이는 자리다).
    """
    cand_ids = [fused_ids_of(r, rrf_k, top_n=depth, weights=weights) for r in rows]
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
        print(f"경고: 그중 **정답 논문이 {gold_lost}문항**에서 빠졌다. 재정렬 상한이 그만큼 낮아진다.")

    queries, cand_lists = [], []
    for r, ids in zip(rows, cand_ids):
        kept = [pid for pid in ids if pid in texts]
        cand_lists.append([ScoredPaper(paper_id=pid, score=0.0, rank=i,
                                       title=texts[pid][0], abstract=texts[pid][1])
                           for i, pid in enumerate(kept, start=1)])
        queries.append(r["text"])          # ★ 재정렬은 변환된 검색어가 아니라 원본 질문으로 한다

    if method == "cross":
        from src.retrieval.rerank import CrossEncoderReranker, DEFAULT_RERANKER
        reranker = CrossEncoderReranker(DEFAULT_RERANKER, batch_size=batch_size)
        ranked = reranker.rerank_batch(queries, cand_lists, top_k=depth)
    elif method == "embedding":
        from src.retrieval.dense import Embedder
        from src.retrieval.rerank import rerank_by_similarity
        embedder = Embedder()
        ranked = [rerank_by_similarity(q, c, embedder, top_k=depth)
                  for q, c in zip(queries, cand_lists)]
    else:
        raise ValueError(f"알 수 없는 재정렬 방식: {method}")

    for r, papers in zip(rows, ranked):
        r["reranked_ids"] = [p.paper_id for p in papers]
    print(f"재정렬 완료: {len(rows)}문항 · 방식 {method} · 깊이 {depth}")


# ── 보고 ──────────────────────────────────────────────────────────────────
def pool_depth_of(args) -> int:
    """상한을 계산할 후보 깊이. 재정렬을 켰으면 재정렬에 넣은 깊이가 곧 상한의 기준이다."""
    return args.rerank_depth if args.rerank != "none" else args.fuse_top_n

def print_report(rows: list[dict], title: str, k_values=DEFAULT_K_VALUES,
                 rrf_k: int = 60, weights: dict[str, float] | None = None,
                 fuse_top_n: int = 200, pool_depth: int = 100) -> None:
    """채널별 · 합집합(상한) · 융합 후 · 재정렬 후를 한 표에 놓고 비교한다."""
    weights = weights or {}
    n = len(rows)
    names = channel_names_of(rows)
    ks = list(k_values)

    print("\n" + "=" * 78)
    print(f"■ {title}")
    print(f"   문항 {n}개 · 융합 k={rrf_k} · 가중치 {weights or '전부 1.0'}")
    print("   ※ 오류·결과 0건 문항을 **실패(0점)로 세고** 분모는 전체 문항 수다.")
    print("      (오류를 제외하는 arxiv_eval.py 의 숫자와 직접 비교하면 안 된다)")

    # 문항 상태
    rewrite_err = [r for r in rows if r.get("error")]
    print(f"\n■ 문항 상태   변환 실패 {len(rewrite_err)}건")
    for name in names:
        errs = sum(1 for r in rows if name in (r.get("channel_errors") or {}))
        zeros = sum(1 for r in rows if not (r.get("channels") or {}).get(name))
        depth = max((len((r.get("channels") or {}).get(name) or []) for r in rows), default=0)
        print(f"   {name:<18} 검색 오류 {errs:>3}건 · 결과 0건 {zeros:>3}건 "
              f"· 최대 깊이 {depth}")

    # 채널별 / 합집합
    print(f"\n■ 채널별 Recall (분모 {n}문항, [ ]는 부트스트랩 95% 신뢰구간)")
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
        print("   └ 합집합 = 각 채널 상위 @K 를 모두 모은 것. **재정렬이 도달할 수 있는 상한**이다")

    # 융합 후
    fused = [fused_ids_of(r, rrf_k, fuse_top_n, weights) for r in rows]
    fused_ranks = [rank_in(ids, r["gold_id"]) for ids, r in zip(fused, rows)]
    print(f"\n■ 융합(RRF) 후 Recall")
    print("   " + f"{'fused':<20}" + "".join(f"{fmt(hits_at(fused_ranks, k)):>22}" for k in ks))

    # 재정렬 후
    has_rr = any("reranked_ids" in r for r in rows)
    rr_ranks = None
    if has_rr:
        rr_ranks = [rank_in(r.get("reranked_ids") or [], r["gold_id"]) for r in rows]
        print(f"\n■ 재정렬 후 Recall (재정렬 결과가 없는 문항은 실패로 센다)")
        print("   " + f"{'reranked':<20}" + "".join(f"{fmt(hits_at(rr_ranks, k)):>22}" for k in ks))
    else:
        print("\n■ 재정렬 후: 아직 재정렬을 돌리지 않았다 (--rerank cross 로 실행)")

    # 상한과 최종값의 차이.
    # 상한은 **재정렬에 실제로 넣은 깊이**로 재야 한다. 깊이 300으로 재정렬해 놓고 상한을
    # 깊이 100으로 재면 회수율이 부풀려진다.
    ceiling_hits = hits_at([union_rank(r, pool_depth) for r in rows], 1)
    ceiling = float(np.mean(ceiling_hits)) if ceiling_hits else 0.0
    final_ranks = rr_ranks if rr_ranks is not None else fused_ranks
    final_name = "재정렬 후" if rr_ranks is not None else "융합 후"
    final = float(np.mean(hits_at(final_ranks, 10))) if rows else 0.0
    recovered = f"상한의 {final / ceiling:.1%} 회수" if ceiling > 0 else "상한이 0이라 계산 불가"
    print(f"\n■ 상한과 최종값의 차이")
    print(f"   후보 상한 (합집합 @{pool_depth})      : {ceiling:.3f}")
    print(f"   최종 Recall@10 ({final_name})     : {final:.3f}")
    print(f"   못 회수한 몫                 : {final - ceiling:+.3f}  ({recovered})")
    print("   해석: 상한이 낮으면 채널을 더 늘려야 하고, 상한은 높은데 최종이 낮으면")
    print("         융합·재정렬을 고쳐야 한다. 두 문제는 처방이 다르다.")

    # 언어별·난이도별 (어디서 실패하는지)
    for field in ("lang", "difficulty"):
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(str(r.get(field)), []).append(i)
        print(f"\n■ {field}별 Recall@10 (최종={final_name})")
        for key, idxs in sorted(groups.items()):
            sub_final = float(np.mean([hits_at([final_ranks[i]], 10)[0] for i in idxs]))
            sub_ceiling = float(np.mean(
                [hits_at([union_rank(rows[i], pool_depth)], 1)[0] for i in idxs]))
            print(f"   {key:<10} n={len(idxs):<4} 최종 {sub_final:.3f} · 상한 {sub_ceiling:.3f}")


# ── 결과 파일 입출력 (이어하기) ────────────────────────────────────────────
def load_done(out_path: Path) -> dict[str, dict]:
    """이미 평가된 문항을 불러온다.

    **어느 채널이든 오류가 난 문항은 다시 시도한다.** arXiv 의 429/503 은 기다리면 풀리는
    일시적 오류라, 그대로 두면 없어도 될 실패가 지표에 남는다.
    """
    if not out_path.exists():
        return {}
    done = {}
    for row in read_jsonl(out_path):
        if row.get("_meta") or row.get("error") or row.get("channel_errors"):
            continue
        done[row["query_id"]] = row
    return done


def main() -> None:
    ap = argparse.ArgumentParser(
        description="다채널 검색 파이프라인 평가 (채널별 결과를 따로 저장해 재계산)")
    ap.add_argument("--queries", default="data/eval/train.jsonl",
                    help="개발은 train 으로만 한다. test 는 최종 확인 때만 쓴다")
    ap.add_argument("--channels", nargs="+", default=["arxiv", "local_dense"],
                    help="arxiv · local_dense(71만 편) · local_dense_small(3만 편, 스모크용)")
    ap.add_argument("--rewriter", default="passthrough",
                    help="변환기 이름 (passthrough / hierarchical / dpo / grounded:dpo ...)")
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

    ap.add_argument("--rrf-k", type=int, default=config.RRF_K, help="RRF 완충 상수")
    ap.add_argument("--weights", default=None,
                    help="채널 가중치. 예: 'arxiv=1.0,local_dense=2.0'")
    ap.add_argument("--fuse-top-n", type=int, default=200, help="융합 결과를 몇 편까지 볼지")

    ap.add_argument("--rerank", default="none", choices=["none", "cross", "embedding"],
                    help="재정렬 방식. 저장된 결과 위에서 돌아가므로 검색은 다시 하지 않는다")
    ap.add_argument("--rerank-depth", type=int, default=100, help="재정렬에 넣을 후보 수")
    ap.add_argument("--batch-size", type=int, default=32)

    ap.add_argument("--large-corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--large-index", default=str(config.DATA_DIR / "embeddings" / "cs2021"))
    ap.add_argument("--mmap", action="store_true",
                    help="임베딩을 메모리에 올리지 않고 디스크에서 읽는다(메모리 절약, 느림)")
    args = ap.parse_args()

    weights = parse_weights(args.weights)

    # ── 재집계 전용 모드 (검색 0회) ───────────────────────────────────────
    if args.report_only:
        path = Path(args.report_only)
        all_rows = list(read_jsonl(path))
        rows = [r for r in all_rows if not r.get("_meta")]
        if args.rerank != "none":
            lookup = TextLookup(args.large_index, args.large_corpus,
                                None if args.no_cache else CACHE_PATH)
            rerank_rows(rows, args.rerank, args.rerank_depth, lookup,
                        args.rrf_k, weights, args.batch_size)
            # 재정렬 결과를 같은 파일에 되써서 재사용한다 (실행 정보 _meta 줄은 그대로 보존)
            write_jsonl(path, rows + [r for r in all_rows if r.get("_meta")])
            print(f"재정렬 결과를 파일에 반영: {path}")
        print_report(rows, path.stem, tuple(args.k_values), args.rrf_k, weights,
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
        print(f"이어하기: 이미 평가된 {len(done)}문항 건너뜀 → 남은 {len(todo)}문항")

    results = [done[q["query_id"]] for q in queries if q["query_id"] in done]
    if todo:
        rewriter = build_rewriter(args.rewriter)
        channels = {name: build_channel(name, args) for name in args.channels}
        print(f"평가 시작: 문항 {len(todo)}개 · 변환기={args.rewriter} "
              f"· 채널={list(channels)} · k={args.k}")

        t0 = time.time()
        for i, q in enumerate(todo, 1):
            results.append(evaluate_one(q, rewriter, channels, args.k))
            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(todo) - i)
                print(f"  {i}/{len(todo)} · 경과 {elapsed/60:.1f}분 "
                      f"· 남은 예상 {eta/60:.1f}분", flush=True)
                write_jsonl(out_path, results)      # 중간 저장 — 중단돼도 여기까지는 살아남는다

    if args.rerank != "none":
        lookup = TextLookup(args.large_index, args.large_corpus,
                            None if args.no_cache else CACHE_PATH)
        rerank_rows(results, args.rerank, args.rerank_depth, lookup,
                    args.rrf_k, weights, args.batch_size)

    meta = {"_meta": True, "rewriter": args.rewriter, "queries": args.queries,
            "channels": args.channels, "k": args.k, "rrf_k": args.rrf_k,
            "weights": weights, "rerank": args.rerank,
            "n_queries": len(results), "commit": git_commit(),
            "finished_at": datetime.now().isoformat(timespec="seconds")}
    write_jsonl(out_path, results + [meta])

    print_report(results, f"{args.rewriter} · 채널 {'+'.join(args.channels)}",
                 tuple(args.k_values), args.rrf_k, weights, args.fuse_top_n, pool_depth_of(args))
    print(f"\n상세 결과 저장: {out_path}  (커밋 {meta['commit']})")
    print("융합 방식을 바꿔 재계산: "
          f"python -m evaluation.pipeline_eval --report-only {out_path} --rrf-k 30")


if __name__ == "__main__":
    main()
