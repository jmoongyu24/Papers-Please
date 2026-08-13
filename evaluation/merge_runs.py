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
import subprocess
import time
from pathlib import Path

from src.retrieval.fusion import normalize_paper_id
from src.utils import read_jsonl, write_jsonl


def git_commit() -> str:
    """지금 코드가 어느 커밋인지. 결과 파일만 보고 재현할 수 있어야 한다."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="옛 실행 결과를 채널로 합치기")
    ap.add_argument("--base", required=True, help="pipeline_eval 형식 결과")
    ap.add_argument("--add", required=True, help="arxiv_eval 형식 결과 (retrieved_ids 사용)")
    ap.add_argument("--name", default="arxiv", help="붙일 채널 이름")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 원본들의 출처 정보. 예전에는 이걸 건너뛰기만 하고 다시 안 써서, 합친 파일에는
    # 커밋 해시도 설정도 실행 시각도 남지 않았다. 나중에 "이 숫자가 어떤 조건에서 나왔나"를
    # 물으면 답할 수 없게 된다 — runs/test300_two_channel.jsonl 이 실제로 그렇게 됐다.
    base_meta, add_meta = None, None

    extra = {}
    for r in read_jsonl(args.add):
        if r.get("_meta"):
            add_meta = r["_meta"]
            continue
        if r.get("error"):
            continue
        extra[r["query_id"]] = [normalize_paper_id(i) for i in (r.get("retrieved_ids") or [])]

    rows, matched = [], 0
    for r in read_jsonl(args.base):
        if r.get("_meta"):
            base_meta = r["_meta"]
            continue
        ids = extra.get(r["query_id"])
        # 없는 문항은 빈 목록으로 둔다 — 그 채널의 실패로 세어야 기준선이 안 부풀려진다
        r.setdefault("channels", {})[args.name] = ids or []
        matched += bool(ids)
        rows.append(r)

    meta = {"_meta": {
        "produced_by": "evaluation.merge_runs",
        "commit": git_commit(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base": args.base, "base_meta": base_meta,
        "add": args.add, "add_meta": add_meta,
        "added_channel": args.name, "matched": matched, "n_rows": len(rows),
        # 두 실행은 서로 다른 시점에 돌았다. arXiv 색인은 그 사이에도 바뀌므로
        # "같은 날 같은 조건"이 아니라는 점을 결과 파일 자체에 남긴다.
        "warning": "채널별 실행 시점이 다르다. 동시 측정이 아님을 감안해 해석할 것.",
    }}
    write_jsonl(args.out, [meta] + rows)
    print(f"{len(rows)}문항 중 {matched}문항에 '{args.name}' 채널을 붙였다 → {args.out}")


if __name__ == "__main__":
    main()
