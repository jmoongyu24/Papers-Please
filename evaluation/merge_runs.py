"""옛 형식(arxiv_eval) 실행 결과를 새 형식(pipeline_eval)에 채널 하나로 합친다.

arXiv 채널은 이미 시험용 300문항을 돌려 논문 번호를 저장해 뒀다(`results/test300_dpo.jsonl`).
그걸 로컬 의미 검색 결과 옆에 붙이면 **arXiv 를 다시 부르지 않고** 두 채널 합집합·융합·
재정렬을 계산할 수 있다.

실행:
  python -m evaluation.merge_runs --base runs/test300_localdense_k1000.jsonl \
      --add results/test300_dpo.jsonl --name arxiv --out runs/test300_two_channel.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.retrieval.fusion import normalize_paper_id
from src.utils import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="옛 실행 결과를 채널로 합치기")
    ap.add_argument("--base", required=True, help="pipeline_eval 형식 결과")
    ap.add_argument("--add", required=True, help="arxiv_eval 형식 결과 (retrieved_ids 사용)")
    ap.add_argument("--name", default="arxiv", help="붙일 채널 이름")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    extra = {}
    for r in read_jsonl(args.add):
        if r.get("_meta") or r.get("error"):
            continue
        extra[r["query_id"]] = [normalize_paper_id(i) for i in (r.get("retrieved_ids") or [])]

    rows, matched = [], 0
    for r in read_jsonl(args.base):
        if r.get("_meta"):
            continue
        ids = extra.get(r["query_id"])
        # 없는 문항은 빈 목록으로 둔다 — 그 채널의 실패로 세어야 기준선이 안 부풀려진다
        r.setdefault("channels", {})[args.name] = ids or []
        matched += bool(ids)
        rows.append(r)

    write_jsonl(args.out, rows)
    print(f"{len(rows)}문항 중 {matched}문항에 '{args.name}' 채널을 붙였다 → {args.out}")


if __name__ == "__main__":
    main()
