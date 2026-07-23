"""판정 풀(pool) 구성 — 관련도 등급을 매길 후보 논문을 모은다.

만족도 평가를 하려면 "질문당 관련 논문 여러 편 + 등급" 정답지가 필요하다. 그 후보를
어떻게 고르나? 비교하는 **모든 시스템**의 상위 결과를 한데 모은다(풀링). 이렇게 해야
어떤 시스템의 결과가 판정에서 빠져 부당하게 '무관' 취급되는 일이 없다(풀링의 핵심 규칙).

입력: runs/ 의 검색 결과들(변환 4종 × 검색 3종) + 평가셋
출력: 질문마다 후보 논문 id 목록 (원 출처 정답 논문은 항상 포함)

실행:
  python -m evaluation.build_pool --queries data/eval/train.jsonl \
      --runs-glob 'runs/train__*.json' --depth 10 --out data/eval/_pool_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils import read_json, read_jsonl, write_jsonl

# 임시 대역(mock)이나 placeholder 변환기는 풀에서 제외 (진짜 시스템만)
EXCLUDE = ("mock_expander",)


def build_pool(queries: list[dict], run_paths: list[str], depth: int) -> list[dict]:
    """질문마다 모든 시스템 top-`depth`의 합집합 + 정답 논문을 후보로 모은다."""
    # 시스템별 run 적재: cond 이름 -> {qid: [doc_ids]}
    runs = {}
    for p in run_paths:
        cond = Path(p).stem
        if any(x in cond for x in EXCLUDE):
            continue
        runs[cond] = read_json(p)["run"]

    rows = []
    for q in queries:
        qid = q["query_id"]
        pool: list[str] = []
        seen = set()
        # 정답(원 출처) 논문을 먼저 넣는다 (등급 3으로 고정될 것)
        gold = q["gold_id"]
        pool.append(gold)
        seen.add(gold)
        # 각 시스템의 상위 depth개를 합친다 (등장 순서 유지)
        for cond, run in runs.items():
            for doc_id in run.get(qid, [])[:depth]:
                if doc_id not in seen:
                    seen.add(doc_id)
                    pool.append(doc_id)
        rows.append({
            "query_id": qid, "raw": q["text"], "gold_id": gold,
            "lang": q["lang"], "difficulty": q["difficulty"],
            "candidates": pool,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="관련도 판정용 후보 풀 구성")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--runs-glob", required=True, help="예: 'runs/train__*.json'")
    ap.add_argument("--depth", type=int, default=10, help="각 시스템에서 상위 몇 개를 풀에 넣을지")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    queries = list(read_jsonl(args.queries))
    run_paths = sorted(str(p) for p in Path().glob(args.runs_glob))
    rows = build_pool(queries, run_paths, args.depth)

    sizes = [len(r["candidates"]) for r in rows]
    write_jsonl(args.out, rows)
    print(f"풀 구성 완료: 질문 {len(rows)}개, 사용한 시스템 run {len(run_paths)}개")
    print(f"  질문당 후보 수: 평균 {sum(sizes)/len(sizes):.1f}, 최소 {min(sizes)}, 최대 {max(sizes)}")
    print(f"  총 판정 대상(정답 제외): {sum(s-1 for s in sizes)}건 → {args.out}")


if __name__ == "__main__":
    main()
