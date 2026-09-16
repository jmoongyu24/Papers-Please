"""
재정렬 모델 점수가 하락하는지 점검함

점검 1  무작위 논문 600쌍 - 확실히 무관한 논문에 높은 점수를 주는가
점검 2  상위 10편 점수 분포 - 후보 안에서 순서를 만들 신호가 남아 있는가
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter

import numpy as np

from src import config

N_QUESTIONS = 30
N_PAPERS = 20
CORPUS_STRIDE = 35000
GATE1_MEDIAN = 0.001
GATE1_ABOVE = 0.10
GATE2_DISTINCT = 1500
GATE2_HARD_GAP = 0.05

REFERENCE = {
    "BAAI/bge-reranker-v2-m3": {"median": 0.00006, "above": 0.030,
                                     "distinct": 2304, "hard_gap": 0.1168},
    "reranker-ft":            {"median": 0.00172, "above": 0.477,
                                     "distinct": 607, "hard_gap": 0.0249},
}


def load_fixed_sample() -> tuple[list[dict], list[str]]:
    qs = [json.loads(l) for l in open(config.DATA_DIR / "eval" / "dev.jsonl")]
    sel: list[dict] = []
    for d in ("easy", "medium", "hard"):
        sel += [r for r in qs if r["difficulty"] == d][: N_QUESTIONS // 3]
    docs: list[str] = []
    with open(config.CORPUS_DIR / "corpus-cs2021.jsonl") as f:
        for line in itertools.islice(f, 0, CORPUS_STRIDE * N_PAPERS, CORPUS_STRIDE):
            d = json.loads(line)
            docs.append(f"{d['title']}\n{d['abstract']}".strip())
    return sel, docs[:N_PAPERS]


def gate1(model_name: str, batch_size: int = 32) -> dict:
    """무작위 논문에 관련 있다고 답하는지 확인"""
    from src.retrieval.ranking import CrossEncoderReranker

    sel, docs = load_fixed_sample()
    pairs = [(q["text"], t) for q in sel for t in docs]
    rr = CrossEncoderReranker(model_name, batch_size=batch_size)
    s = np.asarray(rr.model.predict(pairs, batch_size=batch_size,
                                    show_progress_bar=False), dtype=float)
    return {"n": len(s), "median": float(np.median(s)), "max": float(s.max()),
            "above": float(np.mean(s >= 0.002)), "p95": float(np.percentile(s, 95))}


def gate2(run_path: str) -> dict:
    """상위 10편 안에 순서를 만들 신호가 남아 있는지 확인함"""
    rows = [r for r in (json.loads(l) for l in open(run_path)) if not r.get("_meta")]
    allv: list[float] = []
    ties = 0
    gaps: dict[str, list[float]] = {}
    for r in rows:
        top = [round(x, 6) for x in r["rerank_scores"][:10]]
        allv += top
        ties += sum(v - 1 for v in Counter(top).values() if v > 1)
        if len(top) == 10:
            gaps.setdefault(r["difficulty"], []).append(top[0] - top[-1])
    return {"n": len(allv), "distinct": len(set(allv)), "ties": ties,
            "hard_gap": float(np.median(gaps.get("hard", [0.0]))),
            "easy_gap": float(np.median(gaps.get("easy", [0.0]))),
            "medium_gap": float(np.median(gaps.get("medium", [0.0])))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not args.model and not args.run:
        ap.error("--model 또는 --run 중 하나를 입력해야 함")

    passed = True
    if args.model:
        g = gate1(args.model, args.batch_size)
        ok = g["median"] < GATE1_MEDIAN and g["above"] < GATE1_ABOVE
        passed &= ok
        print(f"\n[점검 1] 무작위 논문 {g['n']}쌍 · {args.model}")
        print(f"  점수 중앙값   {g['median']:.5f}   통과선 {GATE1_MEDIAN} 미만"
              f"   {'통과' if g['median'] < GATE1_MEDIAN else '불합격'}")
        print(f"  0.002 이상    {g['above']:.3f}     통과선 {GATE1_ABOVE} 미만"
              f"   {'통과' if g['above'] < GATE1_ABOVE else '불합격'}")

        for name, ref in REFERENCE.items():
            print(f"    {name:<28} 중앙 {ref['median']:.5f} · 0.002이상 {ref['above']:.3f}")

    if args.run:
        g = gate2(args.run)
        ok = g["distinct"] >= GATE2_DISTINCT and g["hard_gap"] >= GATE2_HARD_GAP
        passed &= ok
        print(f"\n[점검 2] 상위 10편 점수 분포 · {args.run}")
        print(f"  서로 다른 값  {g['distinct']:,} / {g['n']:,}   통과선 {GATE2_DISTINCT:,} 이상"
              f"   {'통과' if g['distinct'] >= GATE2_DISTINCT else '불합격'}")
        print(f"  hard 1등-10등 {g['hard_gap']:.4f}      통과선 {GATE2_HARD_GAP} 이상"
              f"   {'통과' if g['hard_gap'] >= GATE2_HARD_GAP else '불합격'}")

        for name, ref in REFERENCE.items():
            print(f"    {name:<28} 서로 다른 값 {ref['distinct']:,} · hard 차이 {ref['hard_gap']:.4f}")

    print(f"\n=> {'전부 통과' if passed else '불합격'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
