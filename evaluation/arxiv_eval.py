"""arXiv 실시간 검색 평가 하네스 — 실제 서비스와 '같은 조건'으로 성능을 잰다.

왜 이게 필요한가 (ISSUE #10):
기존 로컬 평가는 고정 코퍼스 3만 편 + 의미 기반 검색을 재서 Recall@10 = 0.74가 나왔지만,
실제 서비스(arXiv 전체 250만 편 + 키워드 전용)에서는 0.20 수준이었다. **서로 다른 시스템을
재고 있었기 때문**이다. 이 하네스는 서비스가 실제로 밟는 경로를 그대로 잰다:

    질문 → 변환기(Qwen3-4B) → arxiv 쿼리 필드 → arXiv 실시간 검색 → 정답 논문이 몇 등인가

무엇을 재는가 (실패를 '유형별로' 나눠야 개선 방향이 보인다):
  - Recall@10 / @30 : 정답이 상위 10·30에 드는 비율 (서비스 성능의 핵심)
  - MRR@10         : 정답이 몇 등인지 (순위 품질 — 재정렬 효과를 볼 때 씀)
  - 결과 0건 비율   : 쿼리가 너무 좁아 아무것도 못 찾음 → '과잉 제약' 실패
  - 오류 비율      : arXiv 호출 실패 → **지표에서 제외**해야 측정이 오염되지 않는다(ISSUE #8)
  - 평균 결과 수    : 쿼리가 얼마나 넓은지 (과잉 제약 vs 과잉 확장 진단)

강건성 장치:
  - 디스크 캐싱: 같은 검색은 arXiv를 다시 부르지 않는다. 중단돼도 이어서 실행 가능.
  - 오류와 '0건'을 엄격히 구분해 따로 집계.
  - 변환 결과(실제 검색어)를 저장해 실패 원인을 나중에 분석할 수 있게 한다.
  - 변환기를 바꿔 끼울 수 있어 학습 전/후 모델을 같은 자로 비교한다.

실행 예:
  python -m evaluation.arxiv_eval --queries data/eval/test.jsonl --limit 50 \
      --rewriter hierarchical --out runs/arxiv_eval_test_hierarchical.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import config
from src.retrieval.arxiv_live import ArxivLiveRetriever
from src.rewriter.base import build_rewriter
from src.utils import read_jsonl, write_jsonl

CACHE_PATH = config.DATA_DIR / "cache" / "arxiv_search_cache.jsonl"


def normalize_id(paper_id: str) -> str:
    """arXiv 번호에서 버전 표기를 뗀다. '1706.03762v7' -> '1706.03762'."""
    return paper_id.split("v")[0]


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
    rw = rewriter.rewrite(query_row["text"])
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
    out["gold_rank"] = (ids.index(gold) + 1) if gold in ids else None
    out["top_titles"] = [r.title for r in results[:3]]
    return out


def summarize(rows: list[dict], k_values=(10, 30)) -> dict:
    """질문별 결과를 모아 지표를 계산한다. 오류 건은 지표에서 제외한다."""
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

    zero = [r for r in ok if r.get("n_results", 0) == 0]
    summary["결과 0건 비율"] = round(len(zero) / len(ok), 3)
    summary["평균 결과 수"] = round(float(np.mean([r.get("n_results", 0) for r in ok])), 1)

    for k in k_values:
        hits = [1.0 if (r.get("gold_rank") and r["gold_rank"] <= k) else 0.0 for r in ok]
        summary[f"Recall@{k}"] = round(float(np.mean(hits)), 3)

    rr = [1.0 / r["gold_rank"] if (r.get("gold_rank") and r["gold_rank"] <= 10) else 0.0
          for r in ok]
    summary["MRR@10"] = round(float(np.mean(rr)), 3)
    summary["변환 실패 비율"] = round(
        float(np.mean([0.0 if r.get("rewrite_ok", True) else 1.0 for r in ok])), 3)
    return summary


def summarize_by(rows: list[dict], field: str, k: int = 10) -> dict:
    """언어별·난이도별로 나눠 본다 (어디서 실패하는지 알기 위함)."""
    ok = [r for r in rows if not r.get("error")]
    groups: dict[str, list[dict]] = {}
    for r in ok:
        groups.setdefault(str(r.get(field)), []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        hits = [1.0 if (r.get("gold_rank") and r["gold_rank"] <= k) else 0.0 for r in rs]
        zero = float(np.mean([1.0 if r.get("n_results", 0) == 0 else 0.0 for r in rs]))
        out[key] = {"n": len(rs), f"Recall@{k}": round(float(np.mean(hits)), 3),
                    "0건비율": round(zero, 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="arXiv 실시간 검색 평가 (서비스와 같은 조건)")
    ap.add_argument("--queries", default="data/eval/test.jsonl")
    ap.add_argument("--rewriter", default="hierarchical",
                    help="변환기 이름 (passthrough / single_step / hierarchical 등)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (빠른 점검용)")
    ap.add_argument("--sample", type=int, default=None, help="무작위 N개 (편향 없는 표본)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=30, help="가져올 결과 수 (호출 1회로 30개까지 가능)")
    ap.add_argument("--out", default=None, help="질문별 상세 결과 저장 경로")
    ap.add_argument("--no-cache", action="store_true", help="디스크 캐시 사용 안 함")
    args = ap.parse_args()

    rows = list(read_jsonl(args.queries))
    if args.sample:
        import random
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]
    elif args.limit:
        rows = rows[: args.limit]

    rewriter = build_rewriter(args.rewriter)
    retriever = ArxivLiveRetriever(cache_path=None if args.no_cache else CACHE_PATH)

    print(f"평가 시작: 질문 {len(rows)}개 · 변환기={args.rewriter} · k={args.k}")
    results = []
    t0 = time.time()
    for i, q in enumerate(rows, 1):
        results.append(evaluate_one(q, rewriter, retriever, args.k))
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}초)", flush=True)

    out_path = args.out or f"runs/arxiv_eval_{Path(args.queries).stem}_{args.rewriter}.jsonl"
    write_jsonl(out_path, results)

    print("\n" + "=" * 60)
    print(f"■ 전체 지표 (변환기: {args.rewriter})")
    for k, v in summarize(results).items():
        print(f"   {k}: {v}")
    print("\n■ 언어별 Recall@10")
    for k, v in summarize_by(results, "lang").items():
        print(f"   {k}: {v}")
    print("\n■ 난이도별 Recall@10")
    for k, v in summarize_by(results, "difficulty").items():
        print(f"   {k}: {v}")
    print(f"\n상세 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
