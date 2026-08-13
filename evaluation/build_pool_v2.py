"""등급 정답지를 매길 후보 풀을 만든다 (검색은 전부 로컬 — API 비용 0원).

## 왜 풀이 필요한가

지금 지표는 "질문 하나에 정답 논문 딱 1편"을 센다. 그런데 실제로는 한 질문에 맞는 논문이
여러 편 있어서, 시스템이 다른 좋은 논문을 1등에 올려도 **0점**을 받는다. 이것이 이론적
상한(#14)의 정체다. 그 한계를 풀려면 **후보마다 관련도를 매긴 등급 정답지**가 필요하다.

## 한 짝(한국어·영어)당 한 번만 매긴다

v2 는 같은 논문·같은 난이도를 두 언어로 만든다(`pair_id` 로 이어져 있다). 두 문항은 **같은
정보 요구**를 다른 언어로 표현한 것이므로, "이 논문이 그 요구를 만족시키는가"의 답은 같다.
관련도는 언어가 아니라 뜻의 문제이기 때문이다.

    → 판정은 **영어판으로 한 번만** 하고 두 문항이 그 결과를 함께 쓴다. 비용이 정확히 절반이 된다.

다만 **후보 풀은 두 언어의 검색 결과를 합친다.** 한국어로만 걸리는 논문과 영어로만 걸리는
논문이 다를 수 있는데, 한쪽만 쓰면 그 언어에 유리한 풀이 된다.

## 왜 정답 논문을 풀에 넣고도 편향이 아닌가

원 출처 논문은 자동 3등급으로 두고 판정하지 않는다(비용 절약). 정답 논문이 후보에 있는
것은 편향이 아니다 — 실제 서비스도 그 논문을 결과로 내놓는다. #22 에서 문제였던 것은
정답 논문의 **글을 읽고 검색어를 만든 것**이지, 검색 결과로 내놓은 것이 아니다.

실행:
  python -m evaluation.build_pool_v2 --queries data/eval/test2.jsonl \\
      --out data/eval/_pool_test2.jsonl --depth 20
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src import config
from src.retrieval.fusion import normalize_paper_id
from src.utils import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="등급 판정용 후보 풀 만들기 (로컬 검색만)")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--index", default=str(config.DATA_DIR / "embeddings" / "cs2021"))
    ap.add_argument("--depth", type=int, default=20,
                    help="짝당 후보 수. 깊을수록 이론적 상한 추정이 정확해지지만 판정비가 는다")
    ap.add_argument("--per-lang", type=int, default=15,
                    help="언어별로 몇 편까지 가져와 합칠지")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    by_pair: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_pair[r["pair_id"]][r["lang"]] = r
    print(f"문항 {len(rows)}개 · 짝 {len(by_pair)}개")

    from src.retrieval.large_index import LargeDenseRetriever
    print("색인 불러오는 중… (짝 확인 포함)", flush=True)
    t0 = time.time()
    ret = LargeDenseRetriever(args.corpus, args.index)
    print(f"색인 준비 완료 ({time.time()-t0:.0f}초)", flush=True)

    out, t0 = [], time.time()
    for i, (pair, langs) in enumerate(sorted(by_pair.items()), 1):
        gold = normalize_paper_id(next(iter(langs.values()))["gold_id"])
        # 두 언어의 검색 결과를 합친다 — 한쪽 언어에만 유리한 풀이 되지 않게
        seen: dict[str, dict] = {}
        for lang, r in sorted(langs.items()):
            for p in ret.search(r["text"], k=args.per_lang):
                pid = normalize_paper_id(p.paper_id)
                if pid not in seen:
                    seen[pid] = {"paper_id": pid, "title": p.title,
                                 "abstract": p.abstract, "found_by": [lang]}
                elif lang not in seen[pid]["found_by"]:
                    seen[pid]["found_by"].append(lang)

        cands = list(seen.values())[: args.depth]
        if gold not in seen:                       # 정답이 안 걸렸으면 반드시 넣는다
            cands.append({"paper_id": gold, "title": "", "abstract": "", "found_by": []})

        out.append({
            "pair_id": pair, "gold_id": gold,
            "query_en": langs.get("en", {}).get("text", ""),
            "query_ko": langs.get("ko", {}).get("text", ""),
            "difficulty": next(iter(langs.values()))["difficulty"],
            "candidates": cands,
        })
        if i % 50 == 0:
            print(f"  {i}/{len(by_pair)} ({time.time()-t0:.0f}초)", flush=True)

    write_jsonl(args.out, out)
    n_pairs = sum(len(r["candidates"]) for r in out)
    n_judge = sum(1 for r in out for c in r["candidates"] if c["paper_id"] != r["gold_id"])
    print(f"\n풀 {len(out)}짝 · 후보 {n_pairs:,}편")
    print(f"판정이 필요한 (질문,논문) 쌍: {n_judge:,}개 (정답 논문은 자동 3등급이라 제외)")
    print(f"예상 비용 (gpt-4.1-mini, 쌍당 약 $0.0002): 약 ${n_judge*0.0002:.2f}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
