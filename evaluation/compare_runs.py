"""여러 평가 결과를 공정하게 비교한다 — 공통 문항 정렬 + 짝지은 유의성 검정.

왜 별도 도구가 필요한가:

1. **모델마다 실패한 문항이 다르다.** arXiv 오류로 A 모델은 38문항, B 모델은 40문항이
   평가됐다면, 두 평균을 그냥 비교하는 것은 서로 다른 시험지를 비교하는 것이다.
   실제로 이 프로젝트에서 arXiv HTTPError 7건 때문에 비교가 왜곡된 적이 있다(ISSUE #12).
   → 이 도구는 **모든 결과 파일에서 공통으로 성공한 문항만** 추려 비교한다.

2. **평균 차이만으로는 우연인지 알 수 없다.** 40문항에서 +0.05는 노이즈일 수 있고
   300문항에서 +0.05는 실제일 수 있다. 같은 질문끼리 짝지어 부트스트랩 검정을 돌린다.

3. **기준선이 빠지면 프로젝트의 주장이 검증되지 않는다.** 이 프로젝트의 핵심 주장은
   "쿼리를 변환하면 검색이 좋아진다"이므로, 비교의 기준은 변환을 안 한 passthrough 여야 한다.
   기준선이 목록에 없으면 경고한다.

실행 예:
  python -m evaluation.compare_runs runs/test_passthrough.jsonl \\
      runs/test_hierarchical.jsonl runs/test_dpo.jsonl --k 10

  # 깊이별로 어디서 벌어지는지 보기
  python -m evaluation.compare_runs runs/*.jsonl --k 10 --depth-table
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.arxiv_eval import gold_rank_of
from evaluation.metrics import bootstrap_ci, paired_bootstrap
from src.utils import read_jsonl

BASELINE_HINTS = ("passthrough",)


def load_run(path: str) -> tuple[str, dict[str, dict]]:
    """결과 파일을 {질문id: 행} 으로 읽는다. 메타 줄과 오류 건은 뺀다."""
    rows, name = {}, Path(path).stem
    for r in read_jsonl(path):
        if r.get("_meta"):
            name = r.get("rewriter", name)
            continue
        if r.get("error"):
            continue
        rows[r["query_id"]] = r
    return name, rows


def shorten(names: list[str]) -> list[str]:
    """표가 깨지지 않도록 이름의 공통 앞부분을 떼어낸다.

    'arxiv_eval_test_hierarchical', 'arxiv_eval_test_dpo' → 'hierarchical', 'dpo'
    """
    if len(names) < 2:
        return names
    parts = [n.split("_") for n in names]
    i = 0
    while (i < min(len(p) for p in parts) - 1
           and len({p[i] for p in parts}) == 1):
        i += 1
    out = ["_".join(p[i:]) for p in parts]
    return out if len(set(out)) == len(out) else names


def scores_at(rows: dict[str, dict], qids: list[str], k: int,
              metric: str = "recall") -> dict[str, float]:
    """공통 문항에 대한 질문별 점수. recall = 0/1, mrr = 1/순위."""
    out = {}
    for qid in qids:
        rank = gold_rank_of(rows[qid])
        if metric == "mrr":
            out[qid] = 1.0 / rank if (rank and rank <= k) else 0.0
        else:
            out[qid] = 1.0 if (rank and rank <= k) else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="평가 결과 여러 개를 공통 문항으로 비교")
    ap.add_argument("runs", nargs="+", help="결과 jsonl 경로들 (기준선을 먼저 두면 좋다)")
    ap.add_argument("--k", type=int, default=10, help="Recall@K 의 K")
    ap.add_argument("--baseline", default=None,
                    help="기준선 이름 (기본: 목록에서 passthrough 를 자동 탐색, 없으면 첫 번째)")
    ap.add_argument("--depth-table", action="store_true",
                    help="여러 K에서의 Recall 을 표로 (재정렬 계층의 잠재 이득 진단)")
    ap.add_argument("--by", default=None, choices=["lang", "difficulty"],
                    help="언어별·난이도별로 나눠 비교")
    args = ap.parse_args()

    loaded = [load_run(p) for p in args.runs]
    names = [n for n, _ in loaded]
    if len(set(names)) != len(names):        # 이름이 겹치면 파일명으로 구분
        names = [Path(p).stem for p in args.runs]
    names = shorten(names)
    runs = dict(zip(names, [r for _, r in loaded]))

    # ── 공통 문항 교집합 ──────────────────────────────────────────────────
    common = set.intersection(*(set(r.keys()) for r in runs.values()))
    qids = sorted(common)
    print("=" * 78)
    print("■ 비교 대상")
    for n in names:
        print(f"   {n:<16} 성공 문항 {len(runs[n]):>4}개")
    print(f"\n   → 공통으로 성공한 문항 {len(qids)}개로 비교한다 "
          f"(모델마다 실패 문항이 달라 그냥 평균을 비교하면 불공정하다)")
    if not qids:
        print("   공통 문항이 없다. 같은 질문셋으로 평가했는지 확인할 것.")
        return

    # ── 기준선 결정 ──────────────────────────────────────────────────────
    baseline = args.baseline
    if baseline is None:
        baseline = next((n for n in names if any(h in n for h in BASELINE_HINTS)), names[0])
    if not any(h in baseline for h in BASELINE_HINTS):
        print(f"\n   ⚠ 기준선이 '{baseline}' 이다. 이 프로젝트의 핵심 주장은 "
              f"'변환하면 좋아진다'이므로,\n     변환을 안 한 passthrough 결과를 함께 넣어야 "
              f"그 주장이 검증된다.")

    # ── 절대 성능 표 ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"■ 절대 성능 (공통 {len(qids)}문항, 부트스트랩 95% 신뢰구간)")
    w = max(12, max(len(n) for n in names) + 2)
    print(f"\n   {'모델':<{w}}{f'Recall@{args.k}':>24}{'MRR@10':>24}")
    for n in names:
        rc = scores_at(runs[n], qids, args.k)
        mr = scores_at(runs[n], qids, 10, "mrr")
        m1, l1, h1 = bootstrap_ci(list(rc.values()))
        m2, l2, h2 = bootstrap_ci(list(mr.values()))
        mark = " (기준선)" if n == baseline else ""
        print(f"   {n:<{w}}{m1:>10.3f} [{l1:.3f},{h1:.3f}]{m2:>10.3f} [{l2:.3f},{h2:.3f}]{mark}")

    # ── 기준선 대비 검정 ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"■ '{baseline}' 대비 유의성 검정 (짝지은 부트스트랩, Recall@{args.k})")
    cw = max(24, 2 * max(len(n) for n in names) + 5)
    print(f"\n   {'비교':<{cw}}{'변화':>9}{'95% 신뢰구간':>20}{'p값':>9}  판정")
    base_scores = scores_at(runs[baseline], qids, args.k)
    for n in names:
        if n == baseline:
            continue
        res = paired_bootstrap(base_scores, scores_at(runs[n], qids, args.k))
        verdict = "유의미" if res["p_value"] < 0.05 else "불확실"
        ci = f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]"
        label = f"{baseline} → {n}"
        print(f"   {label:<{cw}}{res['delta']:>+9.3f}{ci:>20}"
              f"{res['p_value']:>9.3f}  {verdict}")

    # ── 인접 모델 간 검정 (SFT → DPO 같은 단계별 이득) ──────────────────
    if len(names) > 2:
        print(f"\n■ 인접 단계 간 추가 이득 (Recall@{args.k})")
        for a, b in zip(names, names[1:]):
            if a == baseline and b == baseline:
                continue
            res = paired_bootstrap(scores_at(runs[a], qids, args.k),
                                   scores_at(runs[b], qids, args.k))
            verdict = "유의미" if res["p_value"] < 0.05 else "불확실"
            print(f"   {a+' → '+b:<{cw}}{res['delta']:>+9.3f}"
                  f"   p={res['p_value']:.3f}  {verdict}")

    # ── 깊이 표 ─────────────────────────────────────────────────────────
    if args.depth_table:
        depths = [1, 5, 10, 30, 50, 100]
        maxd = min(max((len(r.get("retrieved_ids", [])) for r in runs[n].values()), default=0)
                   for n in names)
        depths = [d for d in depths if d <= maxd] or [args.k]
        print("\n" + "=" * 78)
        print("■ 깊이별 Recall — '후보 풀에는 들어왔는데 순위가 낮은' 양을 본다")
        print(f"   최대 저장 깊이 {maxd}\n")
        head = "".join(f"{'R@'+str(d):>9}" for d in depths)
        print(f"   {'모델':<{w}}{head}{'재정렬 여지':>13}")
        for n in names:
            vals = [sum(scores_at(runs[n], qids, d).values()) / len(qids) for d in depths]
            room = vals[-1] - vals[depths.index(10)] if 10 in depths else 0.0
            print(f"   {n:<{w}}" + "".join(f"{v:>9.3f}" for v in vals) + f"{room:>13.3f}")
        print("\n   '재정렬 여지' = R@최대깊이 − R@10. 이만큼은 arXiv가 이미 후보로 돌려줬는데")
        print("   순위가 낮아 놓친 것이므로, 재정렬 계층으로 회수할 수 있는 상한이다.")

    # ── 분해 비교 ────────────────────────────────────────────────────────
    if args.by:
        print("\n" + "=" * 78)
        print(f"■ {args.by} 별 Recall@{args.k} (공통 문항)")
        keys = sorted({str(runs[names[0]][q].get(args.by)) for q in qids})
        print(f"\n   {'모델':<{w}}" + "".join(f"{k:>16}" for k in keys))
        for n in names:
            cells = []
            for key in keys:
                sub = [q for q in qids if str(runs[n][q].get(args.by)) == key]
                sc = scores_at(runs[n], sub, args.k)
                cells.append(f"{sum(sc.values())/len(sub):.3f} (n={len(sub)})" if sub else "-")
            print(f"   {n:<{w}}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
