"""평가 결과 읽고 보고하기 - 실행 하나를 자세히 보거나, 여러 실행을 짝지어 비교함.

    # 실행 하나 자세히 (단일 정답 + 만족도 + 층화)
    $PY -m evaluation.report --run runs/dev_dpo.jsonl \\
        --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl

    # 여러 실행 짝지어 비교 (공통 문항만, 통계 검정 포함)
    $PY -m evaluation.report --run runs/dev_passthrough.jsonl runs/dev_dpo.jsonl \\
        --queries data/eval/dev.jsonl

`--run` 을 하나 주면 자세한 보고, 둘 이상 주면 비교표가 나옴.


## 왜 두 지표를 함께 보는가

단일 정답 지표(Recall@10) 는 "질문을 만들 때 쓴 그 논문 1편"을 찾았는지만 셈.
그런데 등급 정답지를 실제로 재보니 질문 하나에 정답급(3등급) 논문이 중앙값 2편이고,
절반 이상의 질문에서 2편 이상이었음. 즉 시스템이 진짜 좋은 논문을 1등에 올려도 그게
지정된 출처가 아니면 0점을 받음.

만족도 지표(nDCG@10) 는 등급을 이득으로 써서 "좋은 논문을 위에 올렸는가"를 잼.
사용자가 실제로 느끼는 품질에 가까움.

두 숫자가 크게 갈리면 그 차이가 곧 지표의 한계이지 시스템의 실패가 아님.
그래서 하나만 보고 판단하지 않음.

## 비교할 때 공통 문항만 쓰는 이유 (ISSUE #12)

모델마다 실패한 문항이 다름. arXiv 오류로 A 는 38문항, B 는 40문항이 평가됐다면 두 평균을
그냥 비교하는 것은 서로 다른 시험지를 비교하는 것임. 실제로 이 프로젝트에서 arXiv
HTTPError 7건 때문에 비교가 왜곡된 적이 있음. 그래서 모든 결과에서 공통으로 나온 문항만
추려 비교하고, 같은 질문끼리 짝지어 부트스트랩 검정을 돌림.

평균 차이만으로는 우연인지 알 수 없음 - 40문항에서 +0.05 는 잡음일 수 있고 300문항에서
+0.05 는 실제일 수 있음. 실제로 "로컬 단독 0.677 vs 두 채널 0.680" 의 차이는 p=0.889 로
순수한 잡음이었는데, 그 잡음을 보고 설정을 골랐다가 시험지를 태웠음(ISSUE #25).

## 언어 짝을 이용한 비교

평가셋은 같은 논문, 같은 난이도를 한국어와 영어로 만들어 `pair_id` 로 이어 두었음.
그래서 언어별 성능 차이를 논문 차이와 뒤섞이지 않게 볼 수 있음. 옛 평가셋은 한국어
질문과 영어 질문이 서로 다른 논문에 대한 것이라 이 비교가 불가능했음.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.metrics import bootstrap_ci, ndcg_at_k_single, paired_bootstrap
from src.retrieval.corpus import normalize_paper_id
from src.utils import read_jsonl

WHO = {"easy": "대학원생", "medium": "학부연구생", "hard": "1~2학년",
       "known_item": "논문을 아는 사람"}
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]

BASELINE_HINTS = ("passthrough",)


def load_run(path: str) -> tuple[str, dict[str, list[str]]]:
    """실행 결과를 {문항id: 최종 순위} 로 읽음.

    재정렬 결과가 있으면 그것을, 없으면 융합 결과를, 그것도 없으면 첫 채널을 씀.
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
    print(f"   nDCG 는 등급 정답지로 '좋은 논문을 위에 올렸는가'를 잰다.")

    print("\n## 전체")
    print(f"   Recall@{k}  {fmt([r['hit'] for r in rows])}")
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
            line = f"   {label:<26} n={len(g):<4} Recall {fmt([r['hit'] for r in g])}"
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
            "ndcg": {r["query_id"]: r["ndcg"] for r in sel if r["ndcg"] is not None},
        }

    print(f"\n{'실행':<26}{'Recall@'+str(k):>26}{'nDCG@'+str(k):>26}")
    for name, s in scored.items():
        nd = fmt(list(s["ndcg"].values())) if s["ndcg"] else "-"
        print(f"{name:<26}{fmt(list(s['hit'].values())):>26}{nd:>26}")

    base_name = named_rows[0][0]
    if not any(h in base_name for h in BASELINE_HINTS):
        print(f"\n* 기준이 '{base_name}' 이다. 이 프로젝트의 핵심 주장은 '변환하면 좋아진다'이므로,")
        print("   변환하지 않은 passthrough 를 첫 번째로 주는 것이 원칙이다.")

    # 두 지표를 각각 검정함. 하나만 검정하면 "Recall 은 올랐는데 만족도는 떨어졌다"
    # 같은 상황에서 유리한 쪽만 보고 결론을 내리게 됨.
    for metric, label in (("hit", f"Recall@{k}"), ("ndcg", f"nDCG@{k}")):
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
    print("   모르는 차이를 근거로 설정을 고르면 시험지만 소모됨(ISSUE #25).")
    print("* 두 지표가 서로 다른 방향을 가리키면 어느 한쪽만 골라 인용하지 말 것.")
    print("   Recall 은 '질문을 만든 그 논문 1편'만, nDCG 는 '상위 10편 전체의 쓸모'를 잼.")


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
