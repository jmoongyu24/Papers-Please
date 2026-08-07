"""평가 데이터셋 품질 감사 — 이 데이터로 잰 점수를 어디까지 믿어도 되는지 판정한다.

평가 하네스를 아무리 잘 만들어도 데이터가 틀어져 있으면 잘못된 숫자를 정밀하게 뽑을 뿐이다.
그래서 점수를 내기 전에 데이터 자체를 먼저 검사한다. 여기서 재는 것은 네 가지다.

1) **분할 건전성** — train/test 에 같은 논문이 걸쳐 있으면 학습이 시험지를 미리 본 셈이 된다.
2) **정답 누수** — 질문이 논문 제목을 그대로 번역했거나 논문이 제안한 방법 이름(예: SequenceMatch)을
   담고 있으면, 검색이 아니라 '정답 보고 찾기'가 된다. 성능이 실제보다 높게 나온다.
3) **질문 현실성** — 사용자가 실제로 검색창에 칠 법한 형태인가. 자동 생성 질문은 대화체
   ("~는 무엇인가요?")로 쏠리는 경향이 있는데, 실제 검색어와 형태가 다르면 이 데이터로 잰
   성능이 실사용 성능을 예측하지 못한다.
4) **지표 적합성(가장 중요)** — 이 데이터는 질문 하나에 정답 논문이 1편인 known-item 구조다.
   그런데 easy 난이도 질문은 실제로 관련 논문이 수십 편이라, 완벽한 검색기라도 특정 1편을
   상위 10에 올릴 수 없다. 등급 정답지가 있으면 **난이도별 이론적 상한**을 계산해,
   "시스템이 못한 것"과 "과제가 원래 불가능한 것"을 분리한다.

실행:
  python -m evaluation.audit_dataset
  python -m evaluation.audit_dataset --splits data/eval/train.jsonl data/eval/test.jsonl \\
      --corpus data/corpus/corpus-v1.jsonl --qrels data/eval/qrels_graded_train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics

from src.utils import read_jsonl

# 질문/제목 양쪽에서 빼고 볼 흔한 말들 (겹침 비율을 부풀리지 않게)
STOPWORDS = set("""
a an the of for and or in on at with to from using via is are be that this these those
what how can i my we you it its their there here does do done new novel study studies
method methods approach approaches based model models technique techniques
무엇 무엇인가 무엇인가요 어떻게 방법 방법은 위한 대한 있는 어떤 연구 논문 알고 싶어요
있을까요 있나요 궁금해요 그리고 또는 이런 그런 하는 되는
""".split())

LATEX_RE = re.compile(r"\$|\\[a-zA-Z]{2,}|\^\{|_\{")
# 대문자가 2개 이상 섞인 토큰 = 논문이 제안한 방법 이름인 경우가 많다 (SequenceMatch, CODER…)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", s.lower())
            if w not in STOPWORDS and len(w) > 2}


def load_titles(corpus_path: str, wanted: set[str]) -> dict[str, str]:
    titles = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["id"] in wanted:
                titles[p["id"]] = " ".join(p["title"].split())
    return titles


def audit_split_hygiene(splits: dict[str, list[dict]]) -> None:
    print("=" * 78)
    print("■ 1. 분할 건전성 — 학습이 시험지를 미리 보지 않았는가")
    names = list(splits)
    for n in names:
        rows = splits[n]
        print(f"\n   [{n}] 질문 {len(rows)}개 · 정답 논문 {len({r['gold_id'] for r in rows})}편")
        print(f"        난이도 {dict(collections.Counter(r['difficulty'] for r in rows))}")
        print(f"        언어   {dict(collections.Counter(r['lang'] for r in rows))}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa = {r["gold_id"] for r in splits[a]}
            pb = {r["gold_id"] for r in splits[b]}
            overlap = pa & pb
            flag = "정상" if not overlap else f"⚠ 누수 {len(overlap)}편"
            print(f"\n   {a} ∩ {b} 논문 겹침: {len(overlap)}편 → {flag}")
    # 질문 문자열 중복
    allq = [r["text"] for rows in splits.values() for r in rows]
    dup = [t for t, c in collections.Counter(allq).items() if c > 1]
    print(f"   완전히 같은 질문 문자열: {len(dup)}건" + (f" → ⚠ 예: {dup[0][:60]}" if dup else " → 정상"))


def audit_leakage(rows: list[dict], titles: dict[str, str]) -> list[dict]:
    print("\n" + "=" * 78)
    print("■ 2. 정답 누수 — 질문이 논문 제목/방법명을 그대로 담고 있는가")
    print("   (누수가 크면 '검색'이 아니라 '정답 보고 찾기'가 되어 성능이 부풀려진다)")

    tagged = []
    stat = collections.defaultdict(collections.Counter)
    for r in rows:
        d = r["difficulty"]
        title = titles.get(r["gold_id"], "")
        tw, qw = content_words(title), content_words(r["text"])
        overlap = len(tw & qw) / len(tw) if tw else 0.0
        acr = {a for a in ACRONYM_RE.findall(title) if len(a) >= 3}
        leaked = sorted(a for a in acr
                        if re.search(r"\b" + re.escape(a) + r"\b", r["text"], re.I))
        latex = bool(LATEX_RE.search(r["text"]))

        stat[d]["n"] += 1
        stat[d]["ov_sum"] += overlap
        if overlap >= 0.5:
            stat[d]["ov50"] += 1
        if leaked:
            stat[d]["acr"] += 1
        if latex:
            stat[d]["latex"] += 1

        tagged.append({**r, "_title_overlap": round(overlap, 3),
                       "_leaked_names": leaked, "_has_latex": latex})

    print(f"\n   {'난이도':<7}{'n':>5}{'제목겹침 평균':>14}{'겹침≥50%':>12}{'방법명 누수':>13}{'LaTeX 유출':>12}")
    for d in ("easy", "mid", "hard"):
        s = stat[d]
        n = s["n"] or 1
        print(f"   {d:<7}{s['n']:>5}{s['ov_sum']/n:>14.2f}"
              f"{s['ov50']:>7}({s['ov50']/n:>4.0%}){s['acr']:>8}({s['acr']/n:>4.0%})"
              f"{s['latex']:>7}({s['latex']/n:>4.0%})")

    bad = [t for t in tagged if t["_leaked_names"] or t["_title_overlap"] >= 0.6]
    print(f"\n   누수 의심 문항 {len(bad)}개 (방법명 포함 또는 제목겹침 ≥ 0.6)")
    for t in bad[:6]:
        print(f"     [{t['difficulty']}] 겹침={t['_title_overlap']:.2f} 방법명={t['_leaked_names']}")
        print(f"        질문: {t['text'][:78]}")
        print(f"        논문: {titles.get(t['gold_id'],'')[:78]}")
    return tagged


def audit_realism(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("■ 3. 질문 현실성 — 실제 검색창 입력과 형태가 비슷한가")
    for lang in ("ko", "en"):
        sub = [r for r in rows if r["lang"] == lang]
        if not sub:
            continue
        qm = sum(1 for r in sub if r["text"].rstrip().endswith("?"))
        chars = [len(r["text"]) for r in sub]
        words = [len(r["text"].split()) for r in sub]
        print(f"\n   [{lang}] n={len(sub)}")
        print(f"        글자 수 중앙값 {statistics.median(chars):.0f} (최소 {min(chars)}, 최대 {max(chars)})")
        print(f"        어절 수 중앙값 {statistics.median(words):.0f}")
        print(f"        물음표로 끝남 {qm}/{len(sub)} = {qm/len(sub):.0%}")
    print("\n   해석: 실제 검색어는 대개 명사 나열형이고 물음표로 끝나지 않는다.")
    print("   물음표 비율이 높으면 자동 생성 특유의 대화체로 쏠렸다는 뜻이고,")
    print("   그만큼 이 데이터의 점수가 실사용 성능을 덜 대표한다.")


def audit_metric_fit(rows: list[dict], qrels_path: str | None) -> None:
    print("\n" + "=" * 78)
    print("■ 4. 지표 적합성 — 단일정답 Recall 로 재는 것이 타당한가 (가장 중요)")
    if not qrels_path:
        print("   등급 정답지(--qrels)가 없어 건너뜀.")
        return
    try:
        qrels = {r["query_id"]: r["grades"] for r in read_jsonl(qrels_path)}
    except FileNotFoundError:
        print(f"   {qrels_path} 없음 → 건너뜀")
        return

    by_q = {r["query_id"]: r for r in rows}
    print(f"\n   등급 판정된 질문 {len(qrels)}개 기준")
    print(f"\n   {'난이도':<7}{'n':>5}{'정답급(3) 중앙값':>18}{'만족할만함(2+) 중앙값':>22}"
          f"{'단일정답 R@10 상한':>20}")
    for d in ("easy", "mid", "hard"):
        sub = [(qid, g) for qid, g in qrels.items()
               if qid in by_q and by_q[qid]["difficulty"] == d]
        if not sub:
            continue
        n3 = [sum(1 for v in g.values() if v >= 3) for _, g in sub]
        n2 = [sum(1 for v in g.values() if v >= 2) for _, g in sub]
        # 완벽한 검색기가 '만족할 만한' 논문을 전부 상위에 올려도 그 안 순서는 모른다고 보면
        # 원출처가 상위 10에 들 확률은 min(1, 10/관련논문수) 이다.
        ceiling = statistics.mean(min(1.0, 10 / max(x, 1)) for x in n2)
        print(f"   {d:<7}{len(sub):>5}{statistics.median(n3):>18.0f}"
              f"{statistics.median(n2):>22.0f}{ceiling:>20.2f}")

    print("\n   해석: easy 질문은 사용자가 만족할 논문이 여러 편이라, 원출처 1편만 정답으로 세는")
    print("   단일정답 Recall 로는 완벽한 검색기도 높은 점수를 받을 수 없다. 위 '상한'보다 낮게")
    print("   나온 부분만 시스템 책임이고, 상한 자체는 과제 설계의 한계다.")
    print("   → easy·mid 는 등급 평가(NDCG)로 함께 재고, 단일정답 Recall 은 난이도 사이가 아니라")
    print("     같은 난이도 안에서 모델끼리 비교할 때만 쓸 것.")
    print("   ※ 위 상한은 3만 편 코퍼스의 상위 후보만 판정한 값이므로, arXiv 전체(250만 편)에서는")
    print("     경쟁 논문이 더 많아 실제 상한은 이보다 낮다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="평가 데이터셋 품질 감사")
    ap.add_argument("--splits", nargs="+",
                    default=["data/eval/train.jsonl", "data/eval/test.jsonl"])
    ap.add_argument("--corpus", default="data/corpus/corpus-v1.jsonl")
    ap.add_argument("--qrels", default="data/eval/qrels_graded_train.jsonl")
    ap.add_argument("--out", default=None,
                    help="누수 태그를 붙인 문항을 저장할 경로 (선택)")
    args = ap.parse_args()

    splits = {p.split("/")[-1].replace(".jsonl", ""): list(read_jsonl(p))
              for p in args.splits}
    all_rows = [r for rows in splits.values() for r in rows]

    audit_split_hygiene(splits)
    titles = load_titles(args.corpus, {r["gold_id"] for r in all_rows})
    print(f"\n   (코퍼스에서 정답 논문 제목 {len(titles)}/{len({r['gold_id'] for r in all_rows})}편 확인)")
    tagged = audit_leakage(all_rows, titles)
    audit_realism(all_rows)
    audit_metric_fit(all_rows, args.qrels)

    if args.out:
        from src.utils import write_jsonl
        write_jsonl(args.out, tagged)
        print(f"\n누수 태그 포함 문항 저장: {args.out}")


if __name__ == "__main__":
    main()
