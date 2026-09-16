"""
평가 결과 보고용 코드
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.metrics import (bootstrap_ci, ndcg_at_k_single, paired_bootstrap,
                                reciprocal_rank_at_k_single)
from src.retrieval.corpus import normalize_paper_id
from src.utils import read_jsonl

WHO = {"easy": "대학원생", "medium": "학부연구생", "hard": "1~2학년"}
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]


def load_run(path: str) -> tuple[str, dict[str, list[str]]]:
    """실행 결과를 {문항id: 최종 순위}로 읽음"""
    out: dict[str, list[str]] = {}
    name = Path(path).stem
    for r in read_jsonl(path):
        if r.get("_meta"):
            meta = r["_meta"] if isinstance(r["_meta"], dict) else r
            name = meta.get("rewriter") or name
            continue
        if r.get("error"):
            continue
        ids = r.get("reranked_ids") or r.get("fused_ids")
        if not ids:
            ch = r.get("channels") or {}
            ids = next(iter(ch.values()), []) if ch else []
        out[r["query_id"]] = [normalize_paper_id(i) for i in ids]
    return name, out


def score_rows(run: dict[str, list[str]], queries: list[dict],
               grades: dict[str, dict[str, int]], k: int) -> list[dict]:
    """문항별로 단일 정답 성공 여부와 만족도 점수를 측정함"""
    rows = []
    for q in queries:
        ranked = run.get(q["query_id"])
        if ranked is None:
            continue
        gold = normalize_paper_id(q["gold_id"])
        gmap = grades.get(q.get("pair_id", ""), {})
        rows.append({
            "query_id": q["query_id"],
            "lang": q.get("lang"), "difficulty": q.get("difficulty"),
            "hit": 1.0 if gold in ranked[:k] else 0.0,
            "rr": reciprocal_rank_at_k_single(ranked, {gold}, k),
            "ndcg": ndcg_at_k_single(ranked, gmap, k) if gmap else None,
            "overlap": q.get("title_overlap", 0.0),
        })
    return rows


def fmt(xs: list[float]) -> str:
    mean, lo, hi = bootstrap_ci(xs)
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]"


def report_single(name: str, rows: list[dict], k: int) -> None:
    has_ndcg = [r for r in rows if r["ndcg"] is not None]

    print("\n" + "=" * 76)
    print(f"## {name}, 문항 {len(rows)}개")

    print("\n## 전체")
    print(f"   Recall@{k}  {fmt([r['hit'] for r in rows])}")
    print(f"   MRR@{k}     {fmt([r['rr'] for r in rows])}")
    if has_ndcg:
        print(f"   nDCG@{k}    {fmt([r['ndcg'] for r in has_ndcg])}   (n={len(has_ndcg)})")
    else:
        print(f"   nDCG@{k}    -")

    for field, title in (("difficulty", "난이도"), ("lang", "언어")):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r)
        order = ([d for d in WHO if d in groups] if field == "difficulty"
                 else ["ko", "en"])
        print(f"\n## {title}별")
        for key in order:
            g = groups.get(key)
            if not g:
                continue
            label = f"{key} ({WHO[key]})" if field == "difficulty" else key
            line = (f"   {label:<26} n={len(g):<4} Recall {fmt([r['hit'] for r in g])}"
                    f"   MRR {fmt([r['rr'] for r in g])}")
            gn = [r for r in g if r["ndcg"] is not None]
            if gn:
                line += f"   nDCG {fmt([r['ndcg'] for r in gn])}"
            print(line)

    print("\n## 제목 겹침 구간별")
    for lo, hi in BANDS:
        g = [r for r in rows if lo <= r["overlap"] < hi]
        if not g:
            continue
        line = f"   {lo:.1f}~{min(hi,1.0):.1f}   n={len(g):<4} Recall {fmt([r['hit'] for r in g])}"
        gn = [r for r in g if r["ndcg"] is not None]
        if gn:
            line += f"   nDCG {fmt([r['ndcg'] for r in gn])}"
        print(line)

    hits = [r["rr"] for r in rows if r["rr"] > 0]
    if hits:
        top1 = sum(1 for x in hits if x == 1.0)
        print("\n## 맞힌 문항의 순위")
        print(f"   상위 {k}편에 든 {len(hits)}건 중 1등 {top1}건 ({top1 / len(hits):.1%})")
        print(f"   평균 1/등수 {float(np.mean(hits)):.3f}")

    if has_ndcg:
        rec = float(np.mean([r["hit"] for r in rows]))
        nd = float(np.mean([r["ndcg"] for r in has_ndcg]))
        print(f"\n## Recall@{k} {rec:.3f} 대 nDCG@{k} {nd:.3f}")

def report_compare(named_rows: list[tuple[str, list[dict]]], k: int) -> None:
    """공통 문항만 추려 비교하고, 첫 번째를 기준으로 짝지은 검정을 돌림"""
    common = set.intersection(*[{r["query_id"] for r in rows} for _, rows in named_rows])
    print("\n" + "=" * 76)
    print(f"## {len(named_rows)}개 실행 비교, 공통 문항 {len(common)}개")

    scored = {}
    for name, rows in named_rows:
        sel = [r for r in rows if r["query_id"] in common]
        scored[name] = {
            "hit": {r["query_id"]: r["hit"] for r in sel},
            "rr": {r["query_id"]: r["rr"] for r in sel},
            "ndcg": {r["query_id"]: r["ndcg"] for r in sel if r["ndcg"] is not None},
        }

    print(f"\n{'실행':<26}{'Recall@'+str(k):>22}{'MRR@'+str(k):>22}{'nDCG@'+str(k):>22}")
    for name, s in scored.items():
        nd = fmt(list(s["ndcg"].values())) if s["ndcg"] else "-"
        print(f"{name:<26}{fmt(list(s['hit'].values())):>22}"
              f"{fmt(list(s['rr'].values())):>22}{nd:>22}")

    base_name = named_rows[0][0]
    for metric, label in (("hit", f"Recall@{k}"), ("rr", f"MRR@{k}"),
                          ("ndcg", f"nDCG@{k}")):
        if not scored[base_name][metric]:
            continue
        print(f"\n## '{base_name}' 대비 {label} 차이 (짝지은 부트스트랩 검정)")
        print(f"\n{'실행':<26}{'차이':>9}{'95% 신뢰구간':>22}{'p':>9}   판정")
        for name, s in list(scored.items())[1:]:
            st = paired_bootstrap(scored[base_name][metric], s[metric])
            ci = "[{:+.3f}, {:+.3f}]".format(st["ci_low"], st["ci_high"])
            verdict = "유의미" if st["p_value"] < 0.05 else "판정 불가"
            print(f"{name:<26}{st['delta']:>+9.3f}{ci:>22}"
                  f"{st['p_value']:>9.3f}   {verdict}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--grades", default=None)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    queries = [q for q in read_jsonl(args.queries) if not q.get("_meta")]
    grades: dict[str, dict[str, int]] = {}
    if args.grades:
        grades = {r["pair_id"]: {normalize_paper_id(kk): int(v)
                                 for kk, v in r["grades"].items()}
                  for r in read_jsonl(args.grades)}

    named_rows = []
    for path in args.run:
        name, run = load_run(path)
        rows = score_rows(run, queries, grades, args.k)
        if not rows:
            continue
        named_rows.append((name, rows, Path(path).stem))

    names = [n for n, _, _ in named_rows]
    named_rows = [((stem if names.count(n) > 1 else n), rows)
                  for n, rows, stem in named_rows]

    if not named_rows:
        return
    if len(named_rows) == 1:
        report_single(*named_rows[0], k=args.k)
    else:
        report_compare(named_rows, args.k)


if __name__ == "__main__":
    main()
