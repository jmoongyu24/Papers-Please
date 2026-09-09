"""평가 결과 읽고 보고하기. 실행 하나를 자세히 보거나 여러 실행을 짝지어 비교함.

    # 실행 하나 자세히
    python -m evaluation.report --run runs/dev_dpo.jsonl \\
        --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl

    # 여러 실행 짝지어 비교 (공통 문항만, 통계 검정 포함)
    python -m evaluation.report --run runs/dev_passthrough.jsonl runs/dev_dpo.jsonl \\
        --queries data/eval/dev.jsonl

`--run` 이 하나면 자세한 보고, 둘 이상이면 비교표가 나옴.

지표를 둘 다 보는 이유: Recall@10 은 질문을 만든 그 논문 1편만 셈. 그런데 등급 정답지를
재보니 질문 하나에 정답급 논문이 중앙값 2편이라, 진짜 좋은 논문을 1등에 올려도 그것이
지정된 출처가 아니면 0점을 받음. nDCG@10 은 등급을 이득으로 써서 그것을 잼.

MRR@10 은 Recall 과 독립이 아님. 정답이 문항당 1편이고 K 가 같아서 `MRR = Recall * (1/등수)`
이고, 히트와 미스 판정이 두 지표에서 절대 엇갈리지 않음. 더해 주는 것은 "맞혔을 때 몇 등에
올렸나" 하나임. 시험용 342문항에서 상위 10편에 든 225건 중 141건(62.7%)이 1등이었음.

MRR 로 개선폭을 인용할 때 주의할 것: Recall 보다 작게 나옴(+0.468 대 +0.320). 기준선은
키워드가 정확히 맞는 쉬운 문항만 맞혀서 맞힌 것의 86.2% 가 1등인데, 이 시스템은 3.5배 많이
맞히고 새로 맞힌 어려운 문항이 2~10등에 깔리기 때문임. 순위 품질이 나빠진 것이 아니라
맞히는 문항의 구성이 달라진 선택 효과임.

비교할 때 공통 문항만 쓰는 이유: 모델마다 실패한 문항이 다름. A 는 38문항, B 는 40문항이
평가됐다면 두 평균을 그냥 비교하는 것은 서로 다른 시험지를 비교하는 것임. 그래서 공통
문항만 추려 짝지어 부트스트랩 검정을 돌림.

언어 짝: 평가셋은 같은 논문, 같은 난이도를 한국어와 영어로 만들어 `pair_id` 로 이어
두었음. 그래서 언어별 성능 차이를 논문 차이와 뒤섞이지 않게 볼 수 있음.
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

WHO = {"easy": "대학원생", "medium": "학부연구생", "hard": "1~2학년",
       "known_item": "논문을 아는 사람"}
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]

BASELINE_HINTS = ("passthrough",)


def load_run(path: str) -> tuple[str, dict[str, list[str]]]:
    """실행 결과를 {문항id: 최종 순위} 로 읽음.

    재정렬 결과가 있으면 그것을, 없으면 합치기 결과를, 그것도 없으면 첫 채널을 씀.
    이름은 메타 줄의 변환기 이름을 쓰고, 없으면 파일 이름을 씀.
    """
    out: dict[str, list[str]] = {}
    name = Path(path).stem
    for r in read_jsonl(path):
        if r.get("_meta"):
            # 메타 줄의 모양이 두 가지다: {"_meta": True, "rewriter": ...} (pipeline_eval) 와
            # {"_meta": {...}} (dataset). 둘 다에서 이름을 꺼냄.
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
    """문항별로 단일 정답 성공 여부와 만족도 점수를 매김."""
    rows = []
    for q in queries:
        ranked = run.get(q["query_id"])
        if ranked is None:
            continue
        gold = normalize_paper_id(q["gold_id"])
        # 등급 정답지는 짝(pair) 단위임. 관련도는 언어가 아니라 뜻의 문제라
        # 한국어와 영어 문항이 같은 채점표를 함께 씀.
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


# ==========================================================================
# 실행 하나를 자세히 보기
# ==========================================================================

def report_single(name: str, rows: list[dict], k: int) -> None:
    has_ndcg = [r for r in rows if r["ndcg"] is not None]

    print("\n" + "=" * 76)
    print(f"## {name}, 문항 {len(rows)}개, 단일 정답 Recall@{k} 와 만족도 nDCG@{k}")
    print(f"   Recall 은 '질문을 만든 그 논문 1편'만 정답으로 센다.")
    print(f"   MRR 은 그 1편을 몇 등에 올렸는지까지 센다. 히트 판정은 Recall 과 같다.")
    print(f"   nDCG 는 등급 정답지로 '좋은 논문을 위에 올렸는가'를 잰다.")

    print("\n## 전체")
    print(f"   Recall@{k}  {fmt([r['hit'] for r in rows])}")
    print(f"   MRR@{k}     {fmt([r['rr'] for r in rows])}")
    if has_ndcg:
        print(f"   nDCG@{k}    {fmt([r['ndcg'] for r in has_ndcg])}   (n={len(has_ndcg)})")
    else:
        print(f"   nDCG@{k}    - (등급 정답지를 주지 않았다. --grades 로 주면 함께 낸다)")

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

    # 제목 겹침 구간별 - 누수가 성능을 얼마나 떠받치는지 층으로 봄
    print("\n## 제목 겹침 구간별 (거르지 않고 층으로 본다)")
    for lo, hi in BANDS:
        g = [r for r in rows if lo <= r["overlap"] < hi]
        if not g:
            continue
        line = f"   {lo:.1f}~{min(hi,1.0):.1f}   n={len(g):<4} Recall {fmt([r['hit'] for r in g])}"
        gn = [r for r in g if r["ndcg"] is not None]
        if gn:
            line += f"   nDCG {fmt([r['ndcg'] for r in gn])}"
        print(line)

    # MRR 이 Recall 위에 더해 주는 유일한 정보 - 맞혔을 때 몇 등에 올렸는가
    hits = [r["rr"] for r in rows if r["rr"] > 0]
    if hits:
        top1 = sum(1 for x in hits if x == 1.0)
        print(f"\n## 맞힌 문항의 순위 (MRR 이 Recall 위에 더해 주는 것)")
        print(f"   상위 {k}편에 든 {len(hits)}건 중 1등 {top1}건 ({top1 / len(hits):.1%})")
        print(f"   맞힌 문항의 평균 1/등수 {float(np.mean(hits)):.3f} (1.000 이면 전부 1등)")

    if has_ndcg:
        rec = float(np.mean([r["hit"] for r in rows]))
        nd = float(np.mean([r["ndcg"] for r in has_ndcg]))
        print("\n## 해석")
        print(f"   Recall@{k} {rec:.3f} 대 nDCG@{k} {nd:.3f}")
        if nd > rec + 0.10:
            print("   nDCG 가 눈에 띄게 높다. 시스템이 좋은 논문을 위에 올리는데, 그것이")
            print("   지정된 출처 논문이 아니어서 단일 정답 지표가 못 알아보는 것이다.")
            print("   즉 이 격차는 시스템의 실패가 아니라 지표의 한계다.")
        elif nd < rec:
            print("   nDCG 가 더 낮다. 출처 논문은 찾지만 그 주변의 좋은 논문들을 못 올린다는 뜻이다.")
        else:
            print("   두 지표가 비슷하다. 지정 정답 위주로 잘 찾고 있다.")


# ==========================================================================
# 여러 실행을 짝지어 비교하기
# ==========================================================================

def report_compare(named_rows: list[tuple[str, list[dict]]], k: int) -> None:
    """공통 문항만 추려 비교하고, 첫 번째를 기준으로 짝지은 검정을 돌림."""
    common = set.intersection(*[{r["query_id"] for r in rows} for _, rows in named_rows])
    print("\n" + "=" * 76)
    print(f"## {len(named_rows)}개 실행 비교, 공통 문항 {len(common)}개만 사용")
    for name, rows in named_rows:
        dropped = len(rows) - len(common)
        if dropped:
            print(f"   {name}: {dropped}문항 제외 (다른 실행에 없음)")

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
    if not any(h in base_name for h in BASELINE_HINTS):
        print(f"\n* 기준이 '{base_name}' 이다. 이 프로젝트의 핵심 주장은 '변환하면 좋아진다'이므로,")
        print("   변환하지 않은 passthrough 를 첫 번째로 주는 것이 원칙이다.")

    # 두 지표를 각각 검정함. 하나만 검정하면 "Recall 은 올랐는데 만족도는 떨어졌다"
    # 같은 상황에서 유리한 쪽만 보고 결론을 내리게 됨.
    for metric, label in (("hit", f"Recall@{k}"), ("rr", f"MRR@{k}"),
                          ("ndcg", f"nDCG@{k}")):
        if not scored[base_name][metric]:
            continue
        print(f"\n## '{base_name}' 대비 {label} 차이 (같은 문항끼리 짝지어 부트스트랩 검정)")
        print(f"\n{'실행':<26}{'차이':>9}{'95% 신뢰구간':>22}{'p':>9}   판정")
        for name, s in list(scored.items())[1:]:
            st = paired_bootstrap(scored[base_name][metric], s[metric])
            ci = "[{:+.3f}, {:+.3f}]".format(st["ci_low"], st["ci_high"])
            verdict = "유의미" if st["p_value"] < 0.05 else "판정 불가(잡음과 구분 안 됨)"
            print(f"{name:<26}{st['delta']:>+9.3f}{ci:>22}"
                  f"{st['p_value']:>9.3f}   {verdict}")

    print("\n* p 가 0.05 보다 크면 '차이가 없다'가 아니라 '있는지 없는지 모른다' 임.")
    print("   모르는 차이를 근거로 설정을 고르면 시험용 평가셋만 소모됨.")
    print("* 두 지표가 서로 다른 방향을 가리키면 어느 한쪽만 골라 인용하지 말 것.")
    print("   Recall 은 '질문을 만든 그 논문 1편'만, nDCG 는 '상위 10편 전체의 쓸모'를 잼.")
    print("* MRR 은 Recall 과 독립이 아님. 히트 판정이 같고 등수만 더 봄. 개선폭이 Recall 보다")
    print("   작게 나오는 것이 보통인데, 새로 맞힌 어려운 문항이 아래 등수에 깔리기 때문임.")
    print("   순위 품질이 나빠진 것으로 읽지 말 것.")


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="평가 결과 보고 (--run 하나면 자세히, 둘 이상이면 비교)")
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--queries", required=True, help="평가셋 (dev.jsonl / test.jsonl)")
    ap.add_argument("--grades", default=None,
                    help="등급 정답지 (grades_dev.jsonl / grades_test.jsonl). 주면 nDCG 도 낸다")
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
            print(f"* {path}: 겹치는 문항이 없다. --queries 가 같은 분할인지 확인할 것.")
            continue
        named_rows.append((name, rows, Path(path).stem))

    # 이름이 겹치면 파일 이름으로 구분함. 안 그러면 뒤에서 사전에 담을 때 하나로 뭉개져
    # 비교표가 조용히 한 줄만 나옴 (같은 변환기로 채널 조합만 바꿔 잰 경우가 바로 그렇다).
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
