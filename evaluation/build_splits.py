"""걸러낸 질문을 개발용(dev)과 시험용(test)으로 나누고 고정하는 스크립트.

- 개발용 성능 평가 데이터셋(dev): 변환기·검색기를 다듬는 동안 계속 보면서 성능을 확인한다.
- 최종 성능 평가 데이터셋(test): 마지막에 최종 성적을 잴 때 딱 한 번만 본다.

— 같은 정답 논문에서 나온 질문들이 dev와 test에 흩어지면, 개발하면서 사실상 시험 논문을 미리 보는 셈이 되어 성적이 부풀려진다(정보 누수). 
그래서 질문이 아니라 '정답 논문'을 기준으로 나눠, 한 논문의 질문은 전부 같은 쪽에 몰아 둔다.

실행 예:
  python -m evaluation.build_splits --queries data/eval/queries.filtered.jsonl \
      --out-dir data/eval --train-size 200 --test-size 300 --seed 42
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from itertools import zip_longest

from src.schemas import EvalQuery
from src.utils import read_jsonl, write_jsonl


def _group_by_paper(queries: list[EvalQuery]) -> dict[str, list[EvalQuery]]:
    by_paper: dict[str, list[EvalQuery]] = defaultdict(list)
    for q in queries:
        by_paper[q.gold_id].append(q)
    return by_paper


def _fill_to_size(
    size3: list[str], size2: list[str], target: int
) -> tuple[list[str], list[str], list[str]]:
    """질문 3개짜리·2개짜리 논문 풀에서, 질문 수 합이 정확히 target이 되도록 논문을 고른다.

    난이도 균형을 위해 3개짜리 논문(easy/mid/hard가 다 있는 논문)을 우선 쓰고, 3의 배수로
    안 떨어지는 나머지(2 또는 4)만 2개짜리 논문으로 메운다. 예: target=300 → 3개짜리 100편.
    target=200 → 3개짜리 66편(198) + 2개짜리 1편(2).

    Returns: (고른 논문 id들, 남은 size3, 남은 size2).
    """
    for t2 in (0, 1, 2):                 # 2개짜리 논문을 몇 편 쓸지 (나머지 보정용)
        rem = target - 2 * t2
        if rem < 0 or rem % 3 != 0:
            continue
        t3 = rem // 3
        if t3 <= len(size3) and t2 <= len(size2):
            chosen = size3[:t3] + size2[:t2]
            return chosen, size3[t3:], size2[t2:]
    raise ValueError(
        f"목표 질문 수 {target}개를 논문 단위로 정확히 맞출 수 없습니다 "
        f"(3개짜리 {len(size3)}편, 2개짜리 {len(size2)}편으로는 부족)."
    )


def split_sized(
    queries: list[EvalQuery], train_size: int, test_size: int, seed: int = 42
) -> tuple[list[EvalQuery], list[EvalQuery]]:
    """정답 논문 단위로 나누되, train/test의 '질문 개수'를 정확히 맞춘다.

    한 논문의 질문들은 절대 train과 test로 흩어지지 않는다(정보 누수 방지). test를 먼저
    채우고, 남은 논문으로 train을 채운다. 남는 논문은 버린다.
    """
    by_paper = _group_by_paper(queries)
    size3 = sorted(pid for pid, qs in by_paper.items() if len(qs) == 3)
    size2 = sorted(pid for pid, qs in by_paper.items() if len(qs) == 2)
    rng = random.Random(seed)
    rng.shuffle(size3)
    rng.shuffle(size2)

    test_papers, size3, size2 = _fill_to_size(size3, size2, test_size)
    train_papers, size3, size2 = _fill_to_size(size3, size2, train_size)

    def collect(pids: list[str]) -> list[EvalQuery]:
        out = [q for pid in pids for q in by_paper[pid]]
        out.sort(key=lambda q: q.query_id)
        return out

    return collect(train_papers), collect(test_papers)


def _lang_interleave(pids: list[str], by_paper, rng) -> list[str]:
    """논문 목록을 한국어/영어가 번갈아 나오도록 섞는다 (언어 균형 보조)."""
    rng.shuffle(pids)
    ko = [p for p in pids if by_paper[p][0].lang == "ko"]
    en = [p for p in pids if by_paper[p][0].lang != "ko"]
    out = []
    for a, b in zip_longest(ko, en):
        if a is not None:
            out.append(a)
        if b is not None:
            out.append(b)
    return out


def split_stratified(
    queries: list[EvalQuery],
    cat_of: dict[str, str],
    categories: list[str],
    train_size: int,
    test_size: int,
    seed: int = 42,
) -> tuple[list[EvalQuery], list[EvalQuery]]:
    """분야가 train/test **양쪽 모두** 고르게 퍼지도록 나눈다.

    방식: ① 분야별 논문 더미에서 하나씩 번갈아 꺼내 '분야가 골고루 섞인 한 줄'을 만든다.
    ② 그 줄을 훑으며 각 논문을, 아직 목표에 덜 찬 쪽(test/train 중 남은 필요량이 큰 쪽)에
    번갈아 넣는다. 이렇게 양쪽을 동시에 채우면 두 데이터셋 모두 분야가 균형을 이룬다.
    (예전처럼 test를 먼저 다 채우면, train은 남은 논문만 받아 분야가 쏠렸다.)

    난이도 균형을 위해 질문 3개짜리 논문을 우선 쓰고, 개수가 3의 배수로 안 떨어지는
    나머지에만 2개짜리 논문을 쓴다. 각 분야 더미 안에서는 한국어/영어를 번갈아 배치한다.
    """
    by_paper = _group_by_paper(queries)
    rng = random.Random(seed)

    buckets = list(categories) + ["other"]
    size3_by_cat: dict[str, list[str]] = {c: [] for c in buckets}
    size2_all: list[str] = []
    for pid, qs in by_paper.items():
        c = cat_of.get(pid, "other")
        if c not in size3_by_cat:
            c = "other"
        if len(qs) == 3:
            size3_by_cat[c].append(pid)
        elif len(qs) == 2:
            size2_all.append(pid)

    for c in buckets:
        size3_by_cat[c] = _lang_interleave(size3_by_cat[c], by_paper, rng)
    rng.shuffle(size2_all)

    # ① 분야가 골고루 섞인 한 줄 (round-robin)
    ordered: list[str] = []
    while any(size3_by_cat[c] for c in buckets):
        for c in buckets:
            if size3_by_cat[c]:
                ordered.append(size3_by_cat[c].pop(0))

    # 각 split이 필요로 하는 3개짜리 논문 수 (나머지는 2개짜리로 보정)
    def size3_need(target: int) -> tuple[int, int]:
        b = next(x for x in (0, 1, 2) if (target - 2 * x) % 3 == 0)
        return (target - 2 * b) // 3, b  # (3개짜리 수, 2개짜리 수)

    need3 = {"train": size3_need(train_size)[0], "test": size3_need(test_size)[0]}
    need2 = {"train": size3_need(train_size)[1], "test": size3_need(test_size)[1]}
    picked: dict[str, list[str]] = {"train": [], "test": []}
    cat_count: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "test": 0})

    # ② 각 논문을, "그 분야를 아직 덜 받은 split"에 배정한다.
    #    - 1순위: 그 분야를 더 적게 받은 split (분야가 양쪽에 반씩 가게)
    #    - 2순위: 아직 채울 양이 더 많이 남은 split (전체 개수 맞추기)
    #    - 3순위: test 먼저 (동점 처리)
    # 분야 수가 짝수여도 특정 분야가 한쪽으로 쏠리지 않는다.
    for pid in ordered:
        cat = cat_of.get(pid, "other")
        cand = [s for s in ("test", "train") if need3[s] > 0]
        if not cand:
            break
        s = min(cand, key=lambda s: (cat_count[cat][s], -need3[s], 0 if s == "test" else 1))
        picked[s].append(pid)
        need3[s] -= 1
        cat_count[cat][s] += 1

    # 나머지(2개짜리) 보정
    for s in ("test", "train"):
        take = size2_all[: need2[s]]
        size2_all = size2_all[need2[s]:]
        picked[s].extend(take)

    train_papers, test_papers = picked["train"], picked["test"]

    def collect(pids: list[str]) -> list[EvalQuery]:
        out = [q for pid in pids for q in by_paper[pid]]
        out.sort(key=lambda q: q.query_id)
        return out

    return collect(train_papers), collect(test_papers)


def main() -> None:
    ap = argparse.ArgumentParser(description="질문을 train/test로 논문 단위 분할")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-size", type=int, default=200, help="개발용(train) 질문 개수")
    ap.add_argument("--test-size", type=int, default=300, help="시험용(test) 질문 개수")
    ap.add_argument("--corpus", default=None,
                    help="코퍼스 경로를 주면 분야가 train/test에 고르게 퍼지도록 나눈다 "
                         "(안 주면 순수 무작위 분할)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    queries = [EvalQuery.from_dict(r) for r in read_jsonl(args.queries)]

    if args.corpus:
        from src import config
        from src.retrieval.corpus import load_corpus, primary_bucket
        cats = list(config.CORPUS_CATEGORIES)
        cat_of = {p.id: primary_bucket(p, cats) for p in load_corpus(args.corpus)}
        train, test = split_stratified(
            queries, cat_of, cats, args.train_size, args.test_size, args.seed
        )
        print("분야 균형 분할(round-robin) 사용")
    else:
        train, test = split_sized(queries, args.train_size, args.test_size, args.seed)

    train_path = f"{args.out_dir}/train.jsonl"
    test_path = f"{args.out_dir}/test.jsonl"
    write_jsonl(train_path, (q.to_dict() for q in train))
    write_jsonl(test_path, (q.to_dict() for q in test))
    dropped = len(queries) - len(train) - len(test)
    print(f"개발용(train) {len(train)}개 → {train_path}")
    print(f"시험용(test) {len(test)}개 → {test_path}")
    print(f"(목표 개수를 맞추느라 논문 단위로 {dropped}개 질문은 제외됨)")
    print("주의: test.jsonl 은 최종 측정 때만 열어보세요.")


if __name__ == "__main__":
    main()
