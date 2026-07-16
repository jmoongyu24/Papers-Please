"""걸러낸 질문을 개발용(dev)과 시험용(test)으로 나누고 고정하는 스크립트.

- 개발용 성능 평가 데이터셋(dev): 변환기·검색기를 다듬는 동안 계속 보면서 성능을 확인한다.
- 최종 성능 평가 데이터셋(test): 마지막에 최종 성적을 잴 때 딱 한 번만 본다.

— 같은 정답 논문에서 나온 질문들이 dev와 test에 흩어지면, 개발하면서 사실상 시험 논문을 미리 보는 셈이 되어 성적이 부풀려진다(정보 누수). 
그래서 질문이 아니라 '정답 논문'을 기준으로 나눠, 한 논문의 질문은 전부 같은 쪽에 몰아 둔다.

실행 예:
  python -m evaluation.build_splits --queries data/eval/queries.filtered.jsonl \
      --out-dir data/eval --test-ratio 0.6 --seed 42
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from src.schemas import EvalQuery
from src.utils import read_jsonl, write_jsonl


def split_by_paper(
    queries: list[EvalQuery], test_ratio: float, seed: int = 42
) -> tuple[list[EvalQuery], list[EvalQuery]]:
    """정답 논문 단위로 dev/test를 나눈다."""
    by_paper: dict[str, list[EvalQuery]] = defaultdict(list)
    for q in queries:
        by_paper[q.gold_id].append(q)

    paper_ids = sorted(by_paper.keys())
    rng = random.Random(seed)
    rng.shuffle(paper_ids)

    n_test_papers = int(round(len(paper_ids) * test_ratio))
    test_papers = set(paper_ids[:n_test_papers])

    dev, test = [], []
    for pid, qs in by_paper.items():
        (test if pid in test_papers else dev).extend(qs)

    dev.sort(key=lambda q: q.query_id)
    test.sort(key=lambda q: q.query_id)
    return dev, test


def main() -> None:
    ap = argparse.ArgumentParser(description="질문을 dev/test로 논문 단위 분할")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-ratio", type=float, default=0.6,
                    help="시험용으로 뺄 논문 비율 (기본 0.6 → 시험 300 : 개발 200 느낌)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    queries = [EvalQuery.from_dict(r) for r in read_jsonl(args.queries)]
    dev, test = split_by_paper(queries, args.test_ratio, args.seed)

    dev_path = f"{args.out_dir}/dev.jsonl"
    test_path = f"{args.out_dir}/test.jsonl"
    write_jsonl(dev_path, (q.to_dict() for q in dev))
    write_jsonl(test_path, (q.to_dict() for q in test))
    print(f"개발용 {len(dev)}개 → {dev_path}")
    print(f"시험용 {len(test)}개 → {test_path}")
    print("주의: test.jsonl 은 최종 측정 때만 열어보세요.")


if __name__ == "__main__":
    main()
