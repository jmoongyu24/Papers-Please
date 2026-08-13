"""평가셋 v2 를 개발용(dev2)과 시험용(test2)으로 나눈다.

## 왜 논문 단위로 나누는가

질문이 아니라 **정답 논문**을 기준으로 나눈다. 같은 논문에서 나온 질문이 양쪽에 흩어지면,
개발하면서 시험 논문을 미리 보는 셈이 되어 성적이 부풀려진다. v1 도 같은 원칙을 썼다
(`build_splits.py`). v2 는 한 논문이 질문 6개를 갖는다(난이도 3 × 언어 2)는 점만 다르다.

논문 단위로 나누면 난이도와 언어는 **저절로 균형이 맞는다.** 어느 논문이든 난이도별 2개
(한국어·영어), 언어별 3개를 정확히 기여하기 때문이다. 따로 맞출 필요가 없다.

## 왜 옛 시험용 300문항을 은퇴시키는가

`data/eval/test.jsonl` 은 이미 **16회 실행, 5회 설정 선택**에 쓰였다(변환기 선택, 후보 깊이
선택, 재정렬 방식 선택, 재정렬 깊이 선택, 채널 수 선택). 시험지의 정의는 "설정을 고르는 데
쓰지 않은 데이터"이므로, 지금의 0.680 은 시험 점수가 아니라 개발 점수다(ISSUE #25).

특히 마지막 선택이 교과서적이다 — 로컬 단독 0.677 과 두 채널 0.680 중 **시험 점수가 높은
쪽을 골랐는데**, 그 차이는 p=0.889 로 순수한 잡음이었다. 시험지를 태워 얻은 것이 잡음이었다.

    → v2 가 준비되면 옛 test.jsonl 은 **개발용으로 강등**하고, 최종 보고는 test2 로 한다.

## 이 스크립트가 거르는 것과 거르지 않는 것

**거른다** (명백히 사람이 안 하는 행동):
  - 초록을 5낱말 이상 그대로 옮긴 문항
  - 한국어나 영어가 비어 있거나 짝이 깨진 문항

**거르지 않고 기록만 한다** (사람마다 기준이 다를 수 있는 것):
  - 제목 겹침. 대학원생이 정확한 용어로 물으면 제목과 겹치는 것이 **정상**이다.
    `image captioning`(71만 편 중 1,136편)을 썼다고 부정행위가 아니다.
    임계값을 여기서 정해 버리면, 나중에 유리한 값을 고르고 싶은 유혹이 생긴다.
    대신 분포를 그대로 남겨 두고 **성능을 겹침 구간별로 층화 보고**한다.

실행:
  python -m evaluation.build_splits_v2 --queries data/eval/v2_raw.jsonl --out-dir data/eval
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.utils import read_jsonl, write_jsonl

MAX_ABSTRACT_COPY = 5      # 초록을 이만큼 연속으로 옮겼으면 사람이 쓴 검색어가 아니다


def drop_reasons(rows_of_paper: list[dict]) -> dict[str, str]:
    """문항별 탈락 사유. 짝(pair)이 깨지면 그 짝 전체를 뺀다."""
    bad: dict[str, str] = {}
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows_of_paper:
        by_pair[r["pair_id"]].append(r)

    for pair, rs in by_pair.items():
        langs = {r["lang"] for r in rs}
        why = None
        if langs != {"ko", "en"}:
            why = "짝 없음(한 언어만 생성됨)"
        elif any(not r["text"].strip() for r in rs):
            why = "빈 검색어"
        elif any(r.get("abstract_copy_words", 0) >= MAX_ABSTRACT_COPY for r in rs):
            why = f"초록 {MAX_ABSTRACT_COPY}낱말 이상 그대로 복사"
        if why:
            for r in rs:
                bad[r["query_id"]] = why
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="평가셋 v2 를 논문 단위로 dev2/test2 로 나눈다")
    ap.add_argument("--queries", default="data/eval/v2_raw.jsonl")
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--test-ratio", type=float, default=0.5)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_paper[r["gold_id"]].append(r)
    print(f"불러온 문항 {len(rows)}개 · 정답 논문 {len(by_paper)}편")

    # ── 거르기 ──
    dropped: dict[str, str] = {}
    for pid, rs in by_paper.items():
        dropped.update(drop_reasons(rs))
    if dropped:
        print(f"\n■ 탈락 {len(dropped)}문항")
        for why, n in Counter(dropped.values()).most_common():
            print(f"   {why}: {n}건")
    kept = [r for r in rows if r["query_id"] not in dropped]

    by_paper = defaultdict(list)
    for r in kept:
        by_paper[r["gold_id"]].append(r)
    # 문항 6개(난이도 3 × 언어 2)가 온전한 논문만 쓴다 — 층 균형이 저절로 맞는다
    full = {p: rs for p, rs in by_paper.items() if len(rs) == 6}
    partial = len(by_paper) - len(full)
    if partial:
        print(f"   문항이 6개가 안 되는 논문 {partial}편은 제외(층 균형 유지)")

    # ── 논문 단위로 나누기 ──
    papers = sorted(full)
    random.Random(args.seed).shuffle(papers)
    n_test = int(len(papers) * args.test_ratio)
    test_papers, dev_papers = set(papers[:n_test]), set(papers[n_test:])

    def collect(ps: set[str]) -> list[dict]:
        out = [r for p in sorted(ps) for r in sorted(full[p], key=lambda x: x["query_id"])]
        return out

    dev, test = collect(dev_papers), collect(test_papers)
    assert not (dev_papers & test_papers), "논문이 양쪽에 들어갔다"

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "dev2.jsonl", dev)
    write_jsonl(out_dir / "test2.jsonl", test)

    # ── 보고 ──
    for name, rs, ps in (("dev2", dev, dev_papers), ("test2", test, test_papers)):
        print(f"\n■ {name}: 문항 {len(rs)}개 · 논문 {len(ps)}편")
        print(f"   난이도 {dict(Counter(r['difficulty'] for r in rs))}")
        print(f"   언어   {dict(Counter(r['lang'] for r in rs))}")
        bands = Counter()
        for r in rs:
            ov = r.get("title_overlap", 0.0)
            bands["0.0~0.2" if ov < 0.2 else "0.2~0.4" if ov < 0.4
                  else "0.4~0.6" if ov < 0.6 else "0.6~1.0"] += 1
        print(f"   제목 겹침 분포 (거르지 않고 기록) {dict(sorted(bands.items()))}")

    print(f"\n→ {out_dir/'dev2.jsonl'} · {out_dir/'test2.jsonl'}")
    print("\n※ 옛 test.jsonl 은 이미 16회 실행·5회 설정 선택에 쓰였다(ISSUE #25).")
    print("   test2 가 검증되면 옛 test.jsonl 은 개발용으로 강등하고 최종 보고는 test2 로 한다.")


if __name__ == "__main__":
    main()
