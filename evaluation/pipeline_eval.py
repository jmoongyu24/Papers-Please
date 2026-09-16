"""
검색 파이프라인을 돌려 성능을 측정하는 평가 코드
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

CHANNEL_QUERY_FIELD = {
    "arxiv": "arxiv",
    "local_dense": "raw",
    "local_hyde": "hyde",
}


def channel_query_fields(local_query: str) -> dict[str, str]:
    """채널별로 어느 검색어를 쓸지 정함"""
    fields = dict(CHANNEL_QUERY_FIELD)
    fields["local_dense"] = "dense" if local_query == "rewritten" else "raw"
    return fields

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
    """재현을 위해 지금 코드가 어느 커밋인지 기록"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_weights(spec: str | None) -> dict[str, float]:
    """'arxiv=1.0,local_dense=2.0' 형태를 딕셔너리로 바꿈"""
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


def build_channels(names: list[str], args) -> dict[str, object]:
    """
    검색 채널을 만듦
    """
    out: dict[str, object] = {}
    local = None
    for name in names:
        if name == "arxiv":
            from src.retrieval.arxiv_live import ArxivLiveRetriever
            out[name] = ArxivLiveRetriever(cache_path=None if args.no_cache else CACHE_PATH)
            continue

        if name in ("local_dense", "local_hyde"):
            if local is None:
                from src.retrieval.local_index import LocalDenseRetriever
                local = LocalDenseRetriever(
                    args.corpus, args.index, mmap=args.mmap,
                    model_name=getattr(args, "embed_model", None) or config.EMBED_MODEL)
            out[name] = local if name == "local_dense" else LocalHydeChannel(local)
            continue

        raise ValueError(f"알 수 없는 채널 이름: {name} "
                         f"(쓸 수 있는 것: arxiv, local_dense, local_hyde)")
    return out


class LocalHydeChannel:
    """
    가상 초록을 검색어로 삼아 로컬 색인에서 찾는 채널
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, query: str, k: int = 100):
        if not query or not query.strip():
            return []
        return self.retriever.search(query, k=k)


class ReplayRewriter:
    """
    이전 실행 결과에 저장된 검색어를 그대로 다시 씀
    """

    name = "replay"

    def __init__(self, run_path: str | Path):
        self.by_text: dict[str, dict[str, str]] = {}
        for r in read_jsonl(Path(run_path)):
            if r.get("_meta") or not r.get("search_queries"):
                continue
            self.by_text[r["text"]] = dict(r["search_queries"])
        self.n_missing = 0
        if not self.by_text:
            raise ValueError(f"{run_path}에 저장된 검색어가 없음")

    def rewrite(self, raw_query: str) -> "RewriteResult":
        from src.schemas import RewriteResult

        saved = self.by_text.get(raw_query)
        if saved is None:
            self.n_missing += 1
            return RewriteResult(raw_query=raw_query, queries={}, intent="(저장된 검색어 없음)",
                                 parse_ok=False)
        return RewriteResult(raw_query=raw_query, queries=saved, intent=saved.get("local_dense", ""),
                             parse_ok=True)

def evaluate_one(query_row: dict, rewriter, channels: dict, k: int,
                 query_fields: dict[str, str] | None = None) -> dict:
    """
    질문 하나를 모든 채널로 검색해 채널별 결과를 따로 기록함
    """
    out = {
        "query_id": query_row["query_id"], "text": query_row["text"],
        "gold_id": normalize_paper_id(query_row["gold_id"]),
        "lang": query_row.get("lang"), "difficulty": query_row.get("difficulty"),
    }

    t0 = time.time()
    try:
        rw = rewriter.rewrite(query_row["text"])
    except Exception as e:
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
        out["channels"][name] = [normalize_paper_id(r.paper_id) for r in results]
        out["channel_sec"][name] = round(time.time() - t, 2)
    return out

def channel_names_of(rows: list[dict]) -> list[str]:
    """결과 파일에 들어 있는 채널 이름을 등장 순서대로 모음"""
    names: list[str] = []
    for r in rows:
        for name in (r.get("channels") or {}):
            if name not in names:
                names.append(name)
    return names


def rank_in(ids: list[str], gold: str) -> int | None:
    """정답이 몇 등인지 리턴, 없으면 None"""
    return (ids.index(gold) + 1) if gold in ids else None


def any_channel_rank(row: dict, depth: int) -> int | None:
    """어느 채널이든 상위 depth편 안에 정답이 있는가 (1 또는 None)"""
    gold = row["gold_id"]
    for ids in (row.get("channels") or {}).values():
        if gold in (ids or [])[:depth]:
            return 1
    return None


def select_channels(rows: list[dict], names: list[str]) -> list[dict]:
    """
    저장된 결과에서 채널 일부만 남김. 검색을 다시 하지 않고 채널끼리 비교함
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
    """저장된 채널별 논문 번호를 순위로 합침. 검색을 다시 하지 않음"""
    channels = {name: (ids or [])[:depth] if depth else (ids or [])
                for name, ids in (row.get("channels") or {}).items()}
    if not channels:
        return []
    return rrf_fuse_ids(channels, k=rrf_k, top_n=top_n, weights=weights)


def hits_at(ranks: list[int | None], k: int) -> list[float]:
    """등수 목록을 Recall@k의 0/1 목록으로. None(못 찾음, 오류, 0건)은 0점임"""
    return [1.0 if (r is not None and r <= k) else 0.0 for r in ranks]


def fmt(hits: list[float], with_ci: bool = True) -> str:
    if not hits:
        return "n/a"
    if not with_ci:
        return f"{float(np.mean(hits)):.3f}"
    m, lo, hi = bootstrap_ci(hits)
    return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

class TextLookup:
    """
    재정렬 후보의 제목, 초록을 가져옴
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
                indexed = ""
                if paths["meta"].exists():
                    indexed = Path(json.loads(paths["meta"].read_text()).get("corpus", "")).name
                if not indexed or indexed == self.corpus_path.name:
                    ids = paths["ids"].read_text(encoding="utf-8").splitlines()
                    self.pos = {pid: i for i, pid in enumerate(ids)}
                    self.offsets = np.load(paths["offsets"])

    def fetch(self, wanted: set[str]) -> dict[str, tuple[str, str]]:
        """번호 집합 -> {번호: (제목, 초록)}. 못 찾은 번호는 빠짐"""
        out: dict[str, tuple[str, str]] = {}
        if self.offsets is not None:
            todo = sorted((self.pos[p], p) for p in wanted if p in self.pos)
            with open(self.corpus_path, "rb") as f:
                for i, pid in todo:
                    f.seek(int(self.offsets[i]))
                    row = json.loads(f.readline())
                    out[pid] = (row.get("title", "").strip(), row.get("abstract", "").strip())
        elif self.corpus_path and self.corpus_path.exists():
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
    """
    재정렬기에 넣을 질문을 고름.

    raw       (기본) 사용자가 실제로 입력한 말. 재정렬의 목적이 '사용자 의도와 맞는가'를 보는 것이므로 기본 설정임
    rewritten 변환기가 만든 검색어. 한국어를 영어로 옮기는 변환기(translate)를 쓸 때만 사용
    """
    if mode != "rewritten":
        return row["text"]
    sq = row.get("search_queries") or {}
    return sq.get("local_dense") or next(iter(sq.values()), row["text"]) or row["text"]


def rerank_rows(rows: list[dict], method: str, depth: int, lookup: TextLookup,
                rrf_k: int, weights: dict[str, float], batch_size: int = 32,
                query_mode: str = "raw", channel_depth: int | None = None,
                model_name: str | None = None, fuse_rerank: float = 0.0) -> None:
    """
    저장된 결과를 합친 뒤 상위 'depth' 편을 재정렬해 'reranked_ids'로 변환

    channel_depth: 합치기 전에 채널마다 몇 편까지만 볼지 결정

    model_name: 사용할 재정렬 모델

    fuse_rerank: 0보다 크면 재정렬 순위를 검색 순위와 한 번 더 합침
        일상어 질문에서 재정렬기가 후보 100편 전부에 '관련 없음'에 해당하는 값을 주는데,
        검색 순위는 같은 문항에서 다른 논문을 맞히고 있어서 둘을 합치면 올라감
    """
    cand_ids = [fused_ids_of(r, rrf_k, top_n=depth, weights=weights,
                             depth=channel_depth) for r in rows]
    wanted = {pid for ids in cand_ids for pid in ids}
    texts = lookup.fetch(wanted)

    queries, cand_lists = [], []
    for r, ids in zip(rows, cand_ids):
        kept = [pid for pid in ids if pid in texts]
        cand_lists.append([ScoredPaper(paper_id=pid, score=0.0, rank=i,
                                       title=texts[pid][0], abstract=texts[pid][1])
                           for i, pid in enumerate(kept, start=1)])
        queries.append(rerank_query_of(r, query_mode))

    if method != "cross":
        raise ValueError(f"알 수 없는 재정렬 방식: {method}")
    from src.retrieval.ranking import CrossEncoderReranker, DEFAULT_RERANKER
    name = model_name or DEFAULT_RERANKER
    reranker = CrossEncoderReranker(name, batch_size=batch_size)
    ranked = reranker.rerank_batch(queries, cand_lists, top_k=depth)

    for r, papers, fused in zip(rows, ranked, cand_ids):
        ids = [p.paper_id for p in papers]
        if fuse_rerank > 0:
            from src.retrieval.ranking import rrf_fuse_ids
            ids = rrf_fuse_ids({"rerank": ids, "search": list(fused)}, k=rrf_k,
                               top_n=depth, weights={"rerank": fuse_rerank, "search": 1.0})
            r["fuse_rerank"] = fuse_rerank
        r["reranked_ids"] = ids
        by_id = {p.paper_id: round(float(p.score), 6) for p in papers}
        r["rerank_scores"] = [by_id.get(pid, 0.0) for pid in ids]
        r["rerank_depth"] = depth
        r["rerank_method"] = method


TIERS = ("전체", "easy", "medium", "hard")


def _score_map(row: dict) -> dict[str, float]:
    return dict(zip(row.get("reranked_ids") or [], row.get("rerank_scores") or []))


def _top10_at_depth(row: dict, depth: int, rrf_k: int, weights: dict[str, float],
                    channel_depth: int | None) -> list[str]:
    """
    'depth'로 잘랐을 때의 최종 상위 10편
    """
    scores = _score_map(row)
    fused = fused_ids_of(row, rrf_k, top_n=depth, weights=weights, depth=channel_depth)
    kept = [pid for pid in fused if pid in scores]
    kept.sort(key=lambda p: -scores[p])
    return kept[:10]


def diagnose_rerank(rows: list[dict], depths: list[int], rrf_k: int,
                    weights: dict[str, float], channel_depth: int | None) -> None:
    """재정렬 깊이별 Recall@10, 밀려난 문항의 점수, 동점 비율을 잼"""
    from evaluation.metrics import paired_bootstrap

    missing = sum(1 for r in rows if not r.get("rerank_scores"))
    if missing:
        raise SystemExit(f"재정렬 점수가 없는 문항이 {missing}개 있음 --rerank cross로 다시 만들어야 함")

    hits: dict[int, dict] = {}
    for d in depths:
        per: dict[str, dict[str, float]] = {t: {} for t in TIERS}
        for r in rows:
            ok = float(r["gold_id"] in _top10_at_depth(r, d, rrf_k, weights, channel_depth))
            per["전체"][r["query_id"]] = ok
            per.setdefault(r.get("difficulty", "?"), {})[r["query_id"]] = ok
        hits[d] = per

    print("\n" + "=" * 74)
    print("RERANK_DEPTH별 Recall@10")
    print(f"\n{'깊이':>6}" + "".join(f"{t:>10}" for t in TIERS) + f"{'평균 후보 수':>14}")
    for d in depths:
        n_cand = np.mean([len(fused_ids_of(r, rrf_k, top_n=d, weights=weights,
                                           depth=channel_depth)) for r in rows])
        print(f"{d:>6}" + "".join(
            f"{np.mean(list(hits[d][t].values())):>10.3f}" for t in TIERS)
              + f"{n_cand:>14.1f}")

    base = depths[0]
    print(f"\n깊이 {base} 대비 차이")
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
    print(f"후보에는 있었으나 상위 10에서 밀린 문항 (깊이 {depth})")
    print(f"\n   {'난이도':<8}{'문항':>5}{'정답 점수(중앙)':>17}"
          f"{'1등 점수(중앙)':>17}{'10등과의 차이(중앙)':>21}")
    for t in TIERS[1:]:
        if not lost_gold.get(t):
            continue
        print(f"   {t:<10}{len(lost_gold[t]):>5}{np.median(lost_gold[t]):>17.4f}"
              f"{np.median(lost_top1[t]):>17.4f}{np.median(lost_gap[t]):>21.4f}")

    print("\n" + "=" * 74)
    per = {t: [0, 0, 0] for t in TIERS}
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


def pool_depth_of(args) -> int:
    """후보에 정답이 있는 비율을 계산할 깊이"""
    return args.rerank_depth if args.rerank != "none" else args.fuse_top_n


def rerank_pool_rank(row: dict, rrf_k: int, depth: int, weights: dict[str, float]) -> int | None:
    """재정렬에 실제로 들어간 후보 안에 정답이 있는가 (1 또는 None)"""
    return 1 if row["gold_id"] in fused_ids_of(row, rrf_k, top_n=depth,
                                               weights=weights) else None

def print_report(rows: list[dict], title: str, k_values=DEFAULT_K_VALUES,
                 rrf_k: int = 60, weights: dict[str, float] | None = None,
                 fuse_top_n: int = 200, pool_depth: int = 100) -> None:
    """채널별, 어느 채널이든, 합친 뒤, 재정렬 뒤의 Recall을 한 표에 놓음"""
    weights = weights or {}
    n = len(rows)
    names = channel_names_of(rows)
    ks = list(k_values)

    stamped = {r["rerank_depth"] for r in rows if r.get("rerank_depth")}
    if len(stamped) == 1:
        pool_depth = stamped.pop()

    print("\n" + "=" * 78)
    print(f"{title}")
    print(f"   문항 {n}개, 합치기 k={rrf_k}, 가중치 {weights or '전부 1.0'}")

    rewrite_err = [r for r in rows if r.get("error")]
    print(f"\n문항 상태   변환 실패 {len(rewrite_err)}건")
    for name in names:
        errs = sum(1 for r in rows if name in (r.get("channel_errors") or {}))
        zeros = sum(1 for r in rows if not (r.get("channels") or {}).get(name))
        depth = max((len((r.get("channels") or {}).get(name) or []) for r in rows), default=0)
        print(f"   {name:<18} 검색 오류 {errs:>3}건, 결과 0건 {zeros:>3}건 "
              f",  최대 깊이 {depth}")

    print(f"\n## 채널별 Recall (분모 {n}문항, [ ]는 95% 신뢰구간)")
    header = "   " + f"{'':<20}" + "".join(f"{'@'+str(k):>22}" for k in ks)
    print(header)
    for name in names:
        ranks = [rank_in((r.get("channels") or {}).get(name) or [], r["gold_id"]) for r in rows]
        line = "   " + f"{name:<20}" + "".join(f"{fmt(hits_at(ranks, k)):>22}" for k in ks)
        print(line)

    if len(names) > 1:
        line = "   " + f"{'어느 채널이든':<14}" + "".join(
            f"{fmt(hits_at([any_channel_rank(r, k) for r in rows], 1)):>22}" for k in ks)
        print(line)

    fused = [fused_ids_of(r, rrf_k, fuse_top_n, weights) for r in rows]
    fused_ranks = [rank_in(ids, r["gold_id"]) for ids, r in zip(fused, rows)]
    print("\n## 순위 합치기 후 Recall")
    print("   " + f"{'fused':<20}" + "".join(f"{fmt(hits_at(fused_ranks, k)):>22}" for k in ks))

    has_rr = any("reranked_ids" in r for r in rows)
    rr_ranks = None
    if has_rr:
        rr_ranks = [rank_in(r.get("reranked_ids") or [], r["gold_id"]) for r in rows]
        print("\n## 재정렬 후 Recall")
        print("   " + f"{'reranked':<20}" + "".join(f"{fmt(hits_at(rr_ranks, k)):>22}" for k in ks))

    final_ranks = rr_ranks if rr_ranks is not None else fused_ranks
    final_name = "재정렬 후" if rr_ranks is not None else "합친 뒤"
    final = float(np.mean(hits_at(final_ranks, 10))) if rows else 0.0

    any_ceiling = float(np.mean(hits_at([any_channel_rank(r, pool_depth) for r in rows], 1))) if rows else 0.0
    if has_rr:
        pool_hits = [rerank_pool_rank(r, rrf_k, pool_depth, weights) for r in rows]
        pool_ceiling = float(np.mean(hits_at(pool_hits, 1))) if rows else 0.0
    else:
        pool_ceiling = any_ceiling

    recovered = (f"{final / pool_ceiling:.1%} 회수"
                 if pool_ceiling > 0 else "후보에 정답이 없어 계산 불가")
    print("\n## 후보에 정답이 있던 비율과 최종값의 차이")
    print(f"   (1) 어느 채널이든 @{pool_depth}          : {any_ceiling:.3f}")
    print(f"   (2) 재정렬이 실제로 본 후보     : {pool_ceiling:.3f}")
    print(f"   (3) 최종 Recall@10 ({final_name})  : {final:.3f}")
    print(f"   합치기에서 생긴 손실 ((1) -> (2))  : {pool_ceiling - any_ceiling:+.3f}")
    print(f"   재정렬에서 생긴 손실 ((2) -> (3))  : {final - pool_ceiling:+.3f}  ({recovered})")

    for field in ("lang", "difficulty"):
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(str(r.get(field)), []).append(i)
        print(f"\n## {field}별 Recall@10")
        for key, idxs in sorted(groups.items()):
            sub_final = float(np.mean([hits_at([final_ranks[i]], 10)[0] for i in idxs]))
            sub_ceiling = float(np.mean([hits_at([pool_hits[i]], 1)[0] for i in idxs])
                                if has_rr else
                                np.mean([hits_at([any_channel_rank(rows[i], pool_depth)], 1)[0]
                                         for i in idxs]))
            print(f"   {key:<10} n={len(idxs):<4} 최종 {sub_final:.3f}, 후보 {sub_ceiling:.3f}")


def bench_service(args) -> None:
    """
    질문 하나에 몇 초가 걸리는지 단계별로 측정함
    """
    import statistics as st

    from src.gpu_pool import GpuPool

    queries = BENCH_QUERIES[: args.n]

    from src.recommend_agent.recommender import PaperRecommender
    from src.retrieval.arxiv_live import ArxivLiveRetriever
    from src.retrieval.local_index import LocalDenseRetriever
    from src.retrieval.ranking import CrossEncoderReranker
    from src.rewriter.paper_resolver import PaperResolver, resolve_and_verify

    index = LocalDenseRetriever(args.corpus, args.index)
    arxiv = None if args.skip_arxiv else ArxivLiveRetriever()

    pool = GpuPool()

    class _Qwen3:
        """논문 지목과 추천 이유를 한 클라이언트로 묶음. 둘 다 qwen3:4b를 부름"""

        def __init__(self):
            from src.rewriter.base import OllamaClient
            self.client = OllamaClient()
            self.resolver = PaperResolver(client=self.client)
            self.recommender = PaperRecommender(client=self.client)

        def unload(self):
            self.client.unload()

    stages = ["논문 지목 확인", "쿼리 변환", "로컬 의미 검색", "arXiv 검색", "재정렬", "추천"]
    times: dict[str, list[float]] = {s: [] for s in stages}
    totals: list[float] = []

    print(f"\n질문 {len(queries)}개 측정 "
          f"(로컬 {args.k} + arXiv {args.k} -> 재정렬 {args.rerank_depth})\n")

    for i, q in enumerate(queries, 1):
        one: dict[str, float] = {}
        q0 = time.time()

        if not args.skip_resolver and arxiv:
            t = time.time()
            try:
                resolve_and_verify(q, pool.get("ollama:qwen3", _Qwen3).resolver, arxiv)
            except Exception:
                pass
            one["논문 지목 확인"] = time.time() - t

        t = time.time()
        pool.release("ollama:qwen3")
        rw = pool.get("ollama:rewriter", lambda: build_rewriter(args.rewriter)).rewrite(q)
        one["쿼리 변환"] = time.time() - t

        pool.release("ollama:qwen3", "ollama:rewriter")
        t = time.time()
        local_hits = index.search(q, k=args.k)
        one["로컬 의미 검색"] = time.time() - t

        arxiv_hits = []
        if arxiv:
            t = time.time()
            try:
                arxiv_hits = arxiv.search(rw.query_for("arxiv"), k=args.k)
            except Exception:
                pass
            one["arXiv 검색"] = time.time() - t

        seen: dict[str, object] = {}
        for p in list(local_hits) + list(arxiv_hits):
            seen.setdefault(normalize_paper_id(p.paper_id), p)
        cands = list(seen.values())[: args.rerank_depth]

        t = time.time()
        results = pool.get("reranker", CrossEncoderReranker).rerank(q, cands, top_k=10)
        one["재정렬"] = time.time() - t

        t = time.time()
        pool.release("reranker")
        pool.get("ollama:qwen3", _Qwen3).recommender.recommend(q, results)
        one["추천"] = time.time() - t

        pool.release_all()

        totals.append(time.time() - q0)
        for k, v in one.items():
            times[k].append(v)
        print(f"   {i}. {totals[-1]:5.1f}초  후보 {len(cands):>3}편  {q[:38]}")

    def p95(xs: list[float]) -> float:
        return sorted(xs)[max(0, int(len(xs) * 0.95) - 1)] if xs else 0.0

    print("\n단계별 (질문 하나 기준)\n")
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

def calibrate_threshold(args) -> None:
    """
    재정렬 점수 몇 점 아래를 '무관'으로 판단할지 측정함
    """
    queries = [q for q in read_jsonl(args.queries) if not q.get("_meta")]
    grades = {r["pair_id"]: {normalize_paper_id(k): int(v) for k, v in r["grades"].items()}
              for r in read_jsonl(args.grades)}
    queries = [q for q in queries if q.get("pair_id") in grades]
    if not queries:
        return
    print(f"문항 {len(queries)}개, 등급 정답지 {len(grades)}짝")

    wanted = {pid for q in queries for pid in grades[q["pair_id"]]}
    lookup = TextLookup(args.index, args.corpus, None)
    texts = lookup.fetch(wanted)

    pairs, langs, gs = [], [], []
    for q in queries:
        for pid, g in grades[q["pair_id"]].items():
            if pid in texts:
                pairs.append((q["text"], f"{texts[pid][0]}\n{texts[pid][1]}".strip()))
                langs.append(q.get("lang", "?"))
                gs.append(g)
    print(f"채점한 (질문, 논문) 쌍 {len(pairs):,}개")

    from src.retrieval.ranking import DEFAULT_RERANKER, CrossEncoderReranker
    reranker = CrossEncoderReranker(DEFAULT_RERANKER, batch_size=args.batch_size)
    scores = np.asarray(reranker.model.predict(pairs, batch_size=args.batch_size,
                                               show_progress_bar=False), dtype=np.float64)

    langs, gs = np.array(langs), np.array(gs)

    print("=" * 78)
    print("등급별 재정렬 점수 분포")
    print(f"\n{'등급':<18}{'n':>6}{'중앙값':>10}{'25분위':>10}{'75분위':>10}")
    labels = {3: "3 정답급", 2: "2 쓸모 있음", 1: "1 주변적", 0: "0 무관"}
    for g in (3, 2, 1, 0):
        s = scores[gs == g]
        if len(s):
            print(f"{labels[g]:<18}{len(s):>6}{np.median(s):>10.2f}"
                  f"{np.percentile(s, 25):>10.2f}{np.percentile(s, 75):>10.2f}")

    print("\n언어별 점수 범위")
    print(f"\n{'언어':<8}{'만족(2~3) 중앙값':>20}{'무관(0) 중앙값':>20}{'차이':>10}")
    for lang in sorted(set(langs.tolist())):
        good = scores[(langs == lang) & (gs >= 2)]
        bad = scores[(langs == lang) & (gs == 0)]
        if len(good) and len(bad):
            print(f"{lang:<8}{np.median(good):>20.2f}{np.median(bad):>20.2f}"
                  f"{np.median(good) - np.median(bad):>10.2f}")

    print("\n기준선 후보")
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



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/eval/dev.jsonl")
    ap.add_argument("--channels", nargs="+", default=["arxiv", "local_dense"])
    ap.add_argument("--local-query", default="raw", choices=["raw", "rewritten"])
    ap.add_argument("--reuse-queries", default=None)
    ap.add_argument("--rewriter", default="passthrough")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--report-only", default=None)
    ap.add_argument("--use-channels", nargs="+", default=None)
    ap.add_argument("--rrf-k", type=int, default=config.RRF_K)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--fuse-top-n", type=int, default=200)
    ap.add_argument("--rerank", default="none", choices=["none", "cross"])
    ap.add_argument("--rerank-depth", type=int, default=100)
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--fuse-rerank", type=float, default=0.0)
    ap.add_argument("--channel-depth", type=int, default=None)
    ap.add_argument("--diagnose", default=None)
    ap.add_argument("--diagnose-depths", type=int, nargs="+", default=[100, 150, 200])
    ap.add_argument("--rerank-query", default="raw", choices=["raw", "rewritten"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--index", default=str(config.DATA_DIR / "embeddings" / "cs2021-ft"))
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--bench-service", action="store_true")
    ap.add_argument("--calibrate-threshold", action="store_true")
    ap.add_argument("--grades", default=None)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--skip-arxiv", action="store_true")
    ap.add_argument("--skip-resolver", action="store_true")
    args = ap.parse_args()

    if args.bench_service:
        bench_service(args)
        return

    if args.calibrate_threshold:
        if not args.grades:
            ap.error("--calibrate-threshold에는 --grades가 필요하다")
        calibrate_threshold(args)
        return

    weights = parse_weights(args.weights)

    if args.diagnose:
        rows = [r for r in read_jsonl(Path(args.diagnose))
                if not r.get("_meta") and r.get("gold_id")]
        diagnose_rerank(rows, args.diagnose_depths, args.rrf_k, weights,
                        args.channel_depth)
        return

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
                        args.channel_depth, args.rerank_model, args.fuse_rerank)
            metas = list(metas)
            redone = {"_meta": True,
                      "replayed_from": str(path),
                      "replayed_at": datetime.now().isoformat(timespec="seconds"),
                      "commit": git_commit(),
                      "queries": args.queries,
                      "use_channels": args.use_channels,
                      "rrf_k": args.rrf_k, "weights": weights,
                      "rerank": args.rerank, "rerank_depth": args.rerank_depth,
                      "rerank_model": args.rerank_model,
                      "rerank_query": args.rerank_query,
                      "channel_depth": args.channel_depth,
                      "fuse_rerank": args.fuse_rerank,
                      "prev_meta": metas[0] if metas else None}
            metas = [redone] + metas[1:]
            dest = Path(args.out) if args.out else path
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(dest, rows + metas)

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

    query_fields = None
    if args.reuse_queries:
        rewriter = ReplayRewriter(args.reuse_queries)
        query_fields = {name: name for name in args.channels}
    else:
        rewriter = build_rewriter(args.rewriter)
    channels = build_channels(args.channels, args)

    results = []
    fields = query_fields or channel_query_fields(args.local_query)
    for i, q in enumerate(queries, 1):
        results.append(evaluate_one(q, rewriter, channels, args.k, fields))
        if i % 10 == 0 or i == len(queries):
            write_jsonl(out_path, results)

    if args.rerank != "none":
        lookup = TextLookup(args.index, args.corpus,
                            None if args.no_cache else CACHE_PATH)
        rerank_rows(results, args.rerank, args.rerank_depth, lookup,
                    args.rrf_k, weights, args.batch_size, args.rerank_query,
                    args.channel_depth, args.rerank_model, args.fuse_rerank)

    meta = {"_meta": True, "rewriter": args.rewriter, "queries": args.queries,
            "reuse_queries": args.reuse_queries,
            "index": args.index, "embed_model": args.embed_model,
            "rerank_model": args.rerank_model,
            "rerank_depth": args.rerank_depth,
            "fuse_rerank": args.fuse_rerank,
            "channels": args.channels, "k": args.k, "rrf_k": args.rrf_k,
            "local_query": args.local_query,
            "weights": weights, "rerank": args.rerank,
            "n_queries": len(results), "commit": git_commit(),
            "finished_at": datetime.now().isoformat(timespec="seconds")}
    write_jsonl(out_path, results + [meta])

    print_report(results, f"{args.rewriter}, 채널 {'+'.join(args.channels)}",
                 tuple(args.k_values), args.rrf_k, weights, args.fuse_top_n, pool_depth_of(args))


if __name__ == "__main__":
    main()
