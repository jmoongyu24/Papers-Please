"""한 변환기로 평가셋 전체를 변환해 결과를 저장하는 배치 스크립트 (내부용).

언어 모델 호출은 느리므로, 변환 결과(검색 방식별 검색어)를 미리 파일에 저장해 둔다.
그러면 이후 검색 실험에서 언어 모델을 다시 부르지 않고 검색만 반복할 수 있다.

실행 예:
  python -m evaluation._run_rewrite_batch --rewriter single_step \
      --queries data/eval/train.jsonl --out data/eval/_rw_single_step_train.jsonl
"""

from __future__ import annotations

import argparse
import time

from src.rewriter.base import build_rewriter
from src.utils import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewriter", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rw = build_rewriter(args.rewriter)
    queries = list(read_jsonl(args.queries))
    print(f"[{args.rewriter}] {len(queries)}개 변환 시작", flush=True)

    rows = []
    t0 = time.time()
    n_fail = 0
    for i, q in enumerate(queries, 1):
        r = rw.rewrite(q["text"])
        n_fail += (not r.parse_ok)
        rows.append({
            "query_id": q["query_id"], "raw": q["text"],
            "lang": q["lang"], "difficulty": q["difficulty"],
            "parse_ok": r.parse_ok,
            # 계단식 평가용 단계별 산출물 (계층 변환기만 채움, 기준선은 비어 있음)
            "intent": r.intent,
            "concepts": r.concepts,
            "academic_terms": r.academic_terms,
            "sparse": r.queries.get("sparse", ""),
            "dense": r.queries.get("dense", ""),
            "arxiv": r.queries.get("arxiv", ""),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(queries)} ({time.time()-t0:.0f}s)", flush=True)

    write_jsonl(args.out, rows)
    print(f"완료: {len(rows)}개, 실패(parse_ok=False) {n_fail}개, "
          f"{time.time()-t0:.0f}초 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
