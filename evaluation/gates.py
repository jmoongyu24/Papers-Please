"""고정된 사전 점검 - 재정렬 모델이 고장났는지 빠르게 잡아내는 것.

성능을 재는 것이 아님. 통과했다고 좋은 모델이라는 뜻이 아니고, 떨어지면 Recall 을 볼
필요가 없다는 뜻임. 기준값은 결과를 보고 바꾸지 않음.

    점검 1  무작위 논문 600쌍 - 확실히 무관한 논문에 높은 점수를 주는가
    점검 2  상위 10편 점수 분포 - 후보 안에서 순서를 만들 신호가 남아 있는가

Recall 만 보면 이 고장이 안 잡힘. 파인튜닝한 재정렬기가 개발용 Recall@10 을 0.618 에서
0.629 로 올렸는데, 실제로는 후보 100편의 99.3% 에 0.002 이상을 주고 아무 상관 없는
논문에도 0.958 을 주는 상태였음.

실행:
    python -m evaluation.gates --model models/reranker-ft
    python -m evaluation.gates --model models/reranker-ft --run runs/dev_rr_new.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter

import numpy as np

from src import config

# 점검 1 이 쓰는 표본. 값을 바꾸면 옛 결과와 견줄 수 없으므로 고정함.
N_QUESTIONS = 30            # 난이도별 10개
N_PAPERS = 20               # 코퍼스를 고르게 훑어 뽑음
CORPUS_STRIDE = 35000       # 71만 편을 20편으로 나누는 간격
GATE1_MEDIAN = 0.001        # 통과선: 중앙값이 이 값 미만
GATE1_ABOVE = 0.10          # 통과선: 0.002 이상인 비율이 이 값 미만
GATE2_DISTINCT = 1500       # 통과선: 상위10 3,480개 중 서로 다른 값이 이 개수 이상
GATE2_HARD_GAP = 0.05       # 통과선: hard 의 1등-10등 차이 중앙값이 이 값 이상

# 견줄 기준값 (실측, 2026-08-28)
REFERENCE = {
    "원래 BAAI/bge-reranker-v2-m3": {"median": 0.00006, "above": 0.030,
                                     "distinct": 2304, "hard_gap": 0.1168},
    "실패한 reranker-ft":            {"median": 0.00172, "above": 0.477,
                                     "distinct": 607, "hard_gap": 0.0249},
}


def load_fixed_sample() -> tuple[list[dict], list[str]]:
    """점검 1 의 질문 30개와 논문 20편. 매번 같은 것이 나와야 함."""
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
    """무작위 논문에 관련 있다고 답하는가."""
    from src.retrieval.ranking import CrossEncoderReranker

    sel, docs = load_fixed_sample()
    pairs = [(q["text"], t) for q in sel for t in docs]
    rr = CrossEncoderReranker(model_name, batch_size=batch_size)
    s = np.asarray(rr.model.predict(pairs, batch_size=batch_size,
                                    show_progress_bar=False), dtype=float)
    return {"n": len(s), "median": float(np.median(s)), "max": float(s.max()),
            "above": float(np.mean(s >= 0.002)), "p95": float(np.percentile(s, 95))}


def gate2(run_path: str) -> dict:
    """상위 10편 안에 순서를 만들 신호가 남아 있는가."""
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
    ap = argparse.ArgumentParser(description="재정렬 모델의 고정 사전 점검")
    ap.add_argument("--model", default=None, help="점검 1 을 돌릴 재정렬 모델")
    ap.add_argument("--run", default=None,
                    help="점검 2 를 돌릴 실행 결과 파일 (rerank_scores 가 있어야 함)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not args.model and not args.run:
        ap.error("--model 또는 --run 중 하나는 줘야 한다")

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
        print(f"  (참고) 95분위 {g['p95']:.5f} · 최대 {g['max']:.5f}")
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
        print(f"  (참고) 동점 {g['ties']:,}건 · easy {g['easy_gap']:.4f}"
              f" · medium {g['medium_gap']:.4f}")
        for name, ref in REFERENCE.items():
            print(f"    {name:<28} 서로 다른 값 {ref['distinct']:,} · hard 차이 {ref['hard_gap']:.4f}")

    print(f"\n=> {'전부 통과. Recall 을 볼 것.' if passed else '불합격. Recall 을 보지 말 것.'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
