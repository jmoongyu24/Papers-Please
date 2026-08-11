"""arXiv 실시간 검색 평가 하네스 — 실제 서비스와 '같은 조건'으로 성능을 잰다.

왜 이게 필요한가 (ISSUE #10):
기존 로컬 평가는 고정 코퍼스 3만 편 + 의미 기반 검색을 재서 Recall@10 = 0.74가 나왔지만,
실제 서비스(arXiv 전체 250만 편 + 키워드 전용)에서는 0.20 수준이었다. **서로 다른 시스템을
재고 있었기 때문**이다. 이 하네스는 서비스가 실제로 밟는 경로를 그대로 잰다:

    질문 → 변환기 → arxiv 쿼리 필드 → arXiv 실시간 검색 → 정답 논문이 몇 등인가

무엇을 재는가 (실패를 '유형별로' 나눠야 개선 방향이 보인다):
  - Recall@K (K 여러 개) : 정답이 상위 K에 드는 비율. 여러 K를 함께 봐야
    "못 찾는 것"과 "찾았는데 순위가 낮은 것"을 구분할 수 있다.
  - MRR@10         : 정답이 몇 등인지 (순위 품질 — 재정렬 효과를 볼 때 씀)
  - 결과 0건 비율   : 쿼리가 너무 좁아 아무것도 못 찾음 → '과잉 제약' 실패
  - 오류 비율      : arXiv 호출 실패 → **지표에서 제외**해야 측정이 오염되지 않는다(ISSUE #8)
  - 평균 결과 수    : 쿼리가 얼마나 넓은지 (과잉 제약 vs 과잉 확장 진단)

설계상 중요한 결정 세 가지:

1. **검색 결과 논문 ID를 전부 저장한다.**
   예전에는 `gold_rank`(정답이 몇 등인지)만 남겨서, k=30으로 돌린 결과로는 Recall@100을
   알아낼 방법이 없었다. 다시 재려면 arXiv를 처음부터 다시 불러야 했다. 이제 순위대로
   ID를 저장하므로, **한 번 돌린 결과로 어떤 K든 재검색 없이 다시 계산**할 수 있다.

2. **이어하기(resume)를 지원한다.**
   300문항 × (변환 + arXiv 3초) 는 30분~1시간이 걸린다. 중간에 끊기면 처음부터 다시
   해야 했던 것이 지금까지 40문항만 평가해 온 진짜 이유였다. 결과 파일에 이미 있는
   질문은 건너뛰므로, 몇 번에 나눠 실행해도 결과가 하나로 모인다.

3. **신뢰구간을 함께 낸다.**
   "Recall@10 = 0.35" 만으로는 40문항으로 잰 값인지 300문항으로 잰 값인지 알 수 없다.
   부트스트랩 95% 신뢰구간을 같이 찍어 표본 크기의 무게를 드러낸다.

실행 예:
  # 전체 300문항, 이어하기 켜짐(기본)
  python -m evaluation.arxiv_eval --queries data/eval/test.jsonl \
      --rewriter passthrough --k 100 --out runs/test_passthrough.jsonl

  # 이미 저장된 결과만 다시 집계 (arXiv 호출 없음)
  python -m evaluation.arxiv_eval --report-only runs/test_passthrough.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src import config
from src.retrieval.arxiv_live import ArxivLiveRetriever
from src.retrieval.fusion import normalize_paper_id
from src.rewriter.base import build_rewriter
from src.utils import read_jsonl, write_jsonl

CACHE_PATH = config.DATA_DIR / "cache" / "arxiv_search_cache.jsonl"

# 기본 K 목록. 작은 K는 서비스 체감 성능, 큰 K는 '후보 풀에 정답이 들어는 오는가'를 본다.
# 이 둘의 격차가 재정렬(re-ranking) 계층으로 얻을 수 있는 이득의 상한이다.
DEFAULT_K_VALUES = (1, 5, 10, 30, 50, 100)


def normalize_id(paper_id: str) -> str:
    """arXiv 번호에서 버전 표기를 뗀다. '1706.03762v7' -> '1706.03762'.

    구현은 `fusion.normalize_paper_id` 하나만 쓴다. 예전의 `split("v")[0]` 은
    `solv-int/9611001v1` 을 `sol` 로 잘랐다(ISSUE #27).
    """
    return normalize_paper_id(paper_id)


def git_commit() -> str:
    """재현을 위해 지금 코드가 어느 커밋인지 기록한다."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ── 질문 하나 평가 ─────────────────────────────────────────────────────────
def evaluate_one(query_row: dict, rewriter, retriever, k: int) -> dict:
    """질문 하나를 서비스와 같은 경로로 검색해 결과를 기록한다.

    Returns: 질문별 결과 dict (오류면 error 필드가 채워짐)
    """
    gold = normalize_id(query_row["gold_id"])
    out = {
        "query_id": query_row["query_id"], "text": query_row["text"],
        "gold_id": gold, "lang": query_row.get("lang"),
        "difficulty": query_row.get("difficulty"),
    }

    t0 = time.time()
    try:
        rw = rewriter.rewrite(query_row["text"])
    except Exception as e:
        out["error"] = f"rewrite_failed: {type(e).__name__}: {e}"
        return out
    out["rewrite_ok"] = rw.parse_ok
    out["academic_terms"] = rw.academic_terms
    search_query = rw.query_for("arxiv")      # 서비스가 실제로 쓰는 필드
    out["search_query"] = search_query
    out["rewrite_sec"] = round(time.time() - t0, 2)

    try:
        results = retriever.search(search_query, k=k)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    ids = [normalize_id(r.paper_id) for r in results]
    out["n_results"] = len(ids)
    # ★ 순위대로 전부 저장 — 나중에 어떤 K로든 재검색 없이 다시 계산할 수 있게 한다
    out["retrieved_ids"] = ids
    out["gold_rank"] = (ids.index(gold) + 1) if gold in ids else None
    out["top_titles"] = [r.title for r in results[:3]]
    return out


# ── 집계 ──────────────────────────────────────────────────────────────────
def per_query_hits(rows: list[dict], k: int) -> dict[str, float]:
    """질문별 Recall@K (0 또는 1). 유의성 검정에 그대로 넣을 수 있는 형태."""
    out = {}
    for r in rows:
        if r.get("error"):
            continue
        rank = gold_rank_of(r)
        out[r["query_id"]] = 1.0 if (rank and rank <= k) else 0.0
    return out


def gold_rank_of(row: dict) -> int | None:
    """저장된 검색 결과에서 정답 순위를 얻는다.

    `retrieved_ids`가 있으면 그것으로 계산하고(어떤 K로든 재계산 가능),
    없으면 예전 형식의 `gold_rank`를 그대로 쓴다(구 결과 파일 호환).
    """
    ids = row.get("retrieved_ids")
    if ids is not None:
        gold = row["gold_id"]
        return (ids.index(gold) + 1) if gold in ids else None
    return row.get("gold_rank")


def summarize(rows: list[dict], k_values=DEFAULT_K_VALUES, with_ci: bool = True) -> dict:
    """질문별 결과를 모아 지표를 계산한다. 오류 건은 지표에서 제외한다."""
    from evaluation.metrics import bootstrap_ci

    n_all = len(rows)
    errors = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]

    summary = {
        "질문 수(전체)": n_all,
        "오류 건수": len(errors),
        "오류 비율": round(len(errors) / n_all, 3) if n_all else 0.0,
        "평가에 쓴 질문 수": len(ok),
    }
    if not ok:
        return summary

    depth = max((r.get("n_results", 0) for r in ok), default=0)
    summary["실제 검색 깊이(최대)"] = depth
    zero = [r for r in ok if r.get("n_results", 0) == 0]
    summary["결과 0건 비율"] = round(len(zero) / len(ok), 3)
    summary["평균 결과 수"] = round(float(np.mean([r.get("n_results", 0) for r in ok])), 1)

    ranks = [gold_rank_of(r) for r in ok]
    for k in k_values:
        if k > depth and depth > 0:
            continue        # 그 깊이까지 안 가져왔으면 계산하지 않는다(허위 0 방지)
        hits = [1.0 if (rk and rk <= k) else 0.0 for rk in ranks]
        if with_ci:
            m, lo, hi = bootstrap_ci(hits)
            summary[f"Recall@{k}"] = f"{m:.3f}  [{lo:.3f}, {hi:.3f}]"
        else:
            summary[f"Recall@{k}"] = round(float(np.mean(hits)), 3)

    rr = [1.0 / rk if (rk and rk <= 10) else 0.0 for rk in ranks]
    if with_ci:
        m, lo, hi = bootstrap_ci(rr)
        summary["MRR@10"] = f"{m:.3f}  [{lo:.3f}, {hi:.3f}]"
    else:
        summary["MRR@10"] = round(float(np.mean(rr)), 3)

    summary["변환 실패 비율"] = round(
        float(np.mean([0.0 if r.get("rewrite_ok", True) else 1.0 for r in ok])), 3)
    rs = [r.get("rewrite_sec", 0.0) for r in ok if r.get("rewrite_sec") is not None]
    if rs:
        summary["변환 평균 소요(초)"] = round(float(np.mean(rs)), 2)
    return summary


def summarize_by(rows: list[dict], field: str, k: int = 10) -> dict:
    """언어별·난이도별로 나눠 본다 (어디서 실패하는지 알기 위함)."""
    ok = [r for r in rows if not r.get("error")]
    groups: dict[str, list[dict]] = {}
    for r in ok:
        groups.setdefault(str(r.get(field)), []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        ranks = [gold_rank_of(r) for r in rs]
        hits = [1.0 if (rk and rk <= k) else 0.0 for rk in ranks]
        deep = [1.0 if rk else 0.0 for rk in ranks]      # 가져온 범위 안에 들어는 왔는가
        zero = float(np.mean([1.0 if r.get("n_results", 0) == 0 else 0.0 for r in rs]))
        out[key] = {"n": len(rs), f"R@{k}": round(float(np.mean(hits)), 3),
                    "R@전체깊이": round(float(np.mean(deep)), 3),
                    "0건": round(zero, 3)}
    return out


# ── 결과 파일 입출력 (이어하기) ────────────────────────────────────────────
def load_done(out_path: Path) -> dict[str, dict]:
    """이미 평가된 질문을 불러온다. 오류로 끝난 건은 다시 시도하도록 제외한다."""
    if not out_path.exists():
        return {}
    done = {}
    for row in read_jsonl(out_path):
        if row.get("_meta"):
            continue
        if row.get("error"):
            continue            # 오류 건은 재시도 대상
        done[row["query_id"]] = row
    return done


def print_report(rows: list[dict], title: str, k_values=DEFAULT_K_VALUES) -> None:
    print("\n" + "=" * 72)
    print(f"■ 전체 지표 — {title}   (값 뒤 [ ]는 부트스트랩 95% 신뢰구간)")
    for k, v in summarize(rows, k_values).items():
        print(f"   {k}: {v}")
    print("\n■ 언어별")
    for k, v in summarize_by(rows, "lang").items():
        print(f"   {k}: {v}")
    print("\n■ 난이도별")
    for k, v in summarize_by(rows, "difficulty").items():
        print(f"   {k}: {v}")
    print("\n   ※ 난이도별 해석 주의: easy 질문은 사용자가 만족할 논문이 평균 20편가량 있는데")
    print("      지표는 원출처 1편만 정답으로 센다. 완벽한 검색기라도 easy 의 단일정답 Recall@10")
    print("      상한은 약 0.61 이다(등급 정답지 기준). easy 의 낮은 점수를 시스템 성능으로")
    print("      곧바로 읽으면 안 되고, 난이도 사이 비교가 아니라 같은 난이도 안에서 모델끼리")
    print("      비교해야 한다. 만족도는 별도의 등급 평가(judge_relevance.py)로 재야 한다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="arXiv 실시간 검색 평가 (서비스와 같은 조건)")
    ap.add_argument("--queries", default="data/eval/test.jsonl")
    ap.add_argument("--rewriter", default="hierarchical",
                    help="변환기 이름 (passthrough / hierarchical / single_step / hyde / "
                         "finetuned / dpo). passthrough 는 '변환 안 함' 기준선이므로 "
                         "모든 비교에 반드시 포함할 것")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (빠른 점검용)")
    ap.add_argument("--sample", type=int, default=None, help="무작위 N개 (편향 없는 표본)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=100,
                    help="가져올 결과 수. 100이면 arXiv 호출 1회로 끝난다(page_size=100). "
                         "크게 잡을수록 '후보 풀에 정답이 들어는 오는가'를 볼 수 있다")
    ap.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
                    help="Recall 을 계산할 K 목록")
    ap.add_argument("--out", default=None, help="질문별 상세 결과 저장 경로")
    ap.add_argument("--no-cache", action="store_true", help="디스크 캐시 사용 안 함")
    ap.add_argument("--no-resume", action="store_true",
                    help="이미 저장된 결과를 무시하고 처음부터 다시 평가")
    ap.add_argument("--report-only", default=None,
                    help="저장된 결과 파일을 다시 집계만 한다 (arXiv 호출 없음). "
                         "--k-values 를 바꿔 어떤 깊이로든 재계산할 수 있다")
    args = ap.parse_args()

    # ── 재집계 전용 모드 ──────────────────────────────────────────────────
    if args.report_only:
        rows = [r for r in read_jsonl(args.report_only) if not r.get("_meta")]
        print_report(rows, Path(args.report_only).stem, tuple(args.k_values))
        return

    rows = list(read_jsonl(args.queries))
    if args.sample:
        import random
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]
    elif args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out or
                    f"runs/arxiv_eval_{Path(args.queries).stem}_{args.rewriter}.jsonl")

    done = {} if args.no_resume else load_done(out_path)
    todo = [q for q in rows if q["query_id"] not in done]
    if done:
        print(f"이어하기: 이미 평가된 {len(done)}문항 건너뜀 → 남은 {len(todo)}문항")

    results = list(done.values())
    if todo:
        rewriter = build_rewriter(args.rewriter)
        retriever = ArxivLiveRetriever(cache_path=None if args.no_cache else CACHE_PATH)
        print(f"평가 시작: 질문 {len(todo)}개 · 변환기={args.rewriter} · k={args.k}")

        t0 = time.time()
        for i, q in enumerate(todo, 1):
            results.append(evaluate_one(q, rewriter, retriever, args.k))
            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(todo) - i)
                print(f"  {i}/{len(todo)} · 경과 {elapsed/60:.1f}분 · 남은 예상 {eta/60:.1f}분",
                      flush=True)
                # 중간 저장 — 중단돼도 여기까지는 살아남는다
                write_jsonl(out_path, results)

    meta = {"_meta": True, "rewriter": args.rewriter, "queries": args.queries,
            "k": args.k, "n_queries": len(results), "commit": git_commit(),
            "finished_at": datetime.now().isoformat(timespec="seconds")}
    write_jsonl(out_path, results + [meta])

    print_report(results, f"변환기: {args.rewriter} · 질문 {len(results)}개",
                 tuple(args.k_values))
    print(f"\n상세 결과 저장: {out_path}  (커밋 {meta['commit']})")
    print(f"모델 간 비교: python -m evaluation.compare_runs {out_path} <다른결과.jsonl> ...")


if __name__ == "__main__":
    main()
