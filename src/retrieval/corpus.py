"""논문 모음(코퍼스)을 불러오고 거르는 부분.

코퍼스는 JSON Lines 파일이다 (줄마다 논문 하나). 캐글에서 받은 arXiv 스냅샷을
우리 형식으로 정리해 저장하거나, 데모용 작은 샘플 파일을 불러올 때 쓴다.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Iterator, Optional

from src.schemas import Paper
from src.utils import read_jsonl, write_jsonl


def load_corpus(path: str | Path) -> list[Paper]:
    """JSON Lines 코퍼스 파일을 읽어 Paper 목록으로 돌려준다."""
    return [Paper.from_dict(row) for row in read_jsonl(path)]


def save_corpus(path: str | Path, papers: Iterable[Paper]) -> int:
    """Paper 목록을 JSON Lines 코퍼스 파일로 저장한다."""
    return write_jsonl(path, (p.to_dict() for p in papers))


def filter_by_category(
    papers: Iterable[Paper], keep: Iterable[str]
) -> list[Paper]:
    """지정한 분야에 속하는 논문만 남긴다.

    keep 에 들어 있는 분야(cs.CL 등)를 하나라도 가진 논문을 남긴다.
    """
    keep_set = set(keep)
    out = []
    for p in papers:
        if keep_set.intersection(p.categories):
            out.append(p)
    return out


def primary_bucket(paper: Paper, categories: list[str]) -> str:
    """논문의 '대표 분야'를 정한다.

    categories(우리가 다루는 분야 목록) 중, 그 논문의 categories에 실제로 있는 첫 번째
    것을 대표 분야로 삼는다. 하나도 없으면 "other"로 묶는다.
    """
    for c in categories:
        if c in paper.categories:
            return c
    return "other"


def stratified_sample(
    papers: list[Paper],
    n: int,
    categories: Iterable[str],
    seed: int = 42,
    prior_counts: Optional[dict[str, int]] = None,
) -> list[Paper]:
    """분야별로 최대한 고르게 논문을 뽑는다 ('물 채우기' 방식).

    그냥 무작위로 뽑으면 코퍼스에 원래 많은 분야(예: cs.LG)가 결과에도 많이 뽑히고,
    적은 분야(예: cs.IR)는 거의 안 뽑힌다. 이 함수는 매 순간 "지금까지 가장 적게 뽑힌
    분야"에서 하나씩 뽑기를 반복해서, 최종적으로 분야별 개수가 최대한 비슷해지게 한다.
    (양동이에 물을 부을 때 가장 낮은 곳부터 채워져 수평을 이루는 것과 같은 원리다.)

    Args:
        papers: 뽑을 후보 논문들 (이미 사용한 논문은 미리 제외하고 넣는다).
        n: 뽑을 논문 수.
        categories: 균형을 맞출 분야 목록.
        prior_counts: 이전에 이미 뽑아 둔 분야별 개수. 배치를 나눠 이어서 뽑을 때,
            "이 분야는 이미 이만큼 뽑았다"고 알려주면 이번 배치에서 그 분야를
            상대적으로 덜 뽑아 전체 균형을 맞춘다.
    """
    cats = list(categories)
    rng = random.Random(seed)

    pool: dict[str, list[Paper]] = {c: [] for c in cats}
    pool["other"] = []
    for p in papers:
        pool[primary_bucket(p, cats)].append(p)
    for bucket in pool.values():
        rng.shuffle(bucket)

    counts: dict[str, int] = {c: 0 for c in pool}
    if prior_counts:
        for c, v in prior_counts.items():
            if c in counts:
                counts[c] = v

    idx = {c: 0 for c in pool}
    selected: list[Paper] = []
    bucket_names = list(pool.keys())
    while len(selected) < n:
        # 아직 후보가 남은 분야 중, 지금까지 가장 적게 뽑힌 분야를 고른다.
        available = [c for c in bucket_names if idx[c] < len(pool[c])]
        if not available:
            break  # 모든 분야의 후보를 다 썼다
        available.sort(key=lambda c: counts[c])
        chosen_bucket = available[0]
        selected.append(pool[chosen_bucket][idx[chosen_bucket]])
        idx[chosen_bucket] += 1
        counts[chosen_bucket] += 1
    return selected


def _iter_kaggle_matches(
    kaggle_jsonl: str | Path,
    keep: set[str],
    min_year: Optional[int],
    prefixes: Optional[tuple[str, ...]] = None,
) -> Iterator[Paper]:
    """캐글 원본을 한 줄씩 읽어, 분야·연도 조건에 맞는 논문만 Paper로 내보낸다.

    캐글 원본 한 줄의 주요 필드: id, title, abstract, categories(공백으로 이은 문자열),
    update_date("YYYY-MM-DD").

    분야를 고르는 방법 세 가지:
    - `keep`에 정확한 분야명을 넣으면 그것만 (예: {"cs.CL", "cs.CV"})
    - `prefixes`에 앞글자를 넣으면 그 계열 전체 (예: ("cs.", "stat.") → cs 하위 40여 개 전부)
    - 둘 다 비우면 **분야 제한 없이 전부**. arXiv 검색은 물리·수학까지 전 분야를 대상으로
      하므로, 어떤 표현이 얼마나 흔한지(문서 빈도)를 arXiv와 비슷하게 재려면 전체가 필요하다.
    """
    for row in read_jsonl(kaggle_jsonl):
        cats = row.get("categories", "")
        cat_list = cats.split() if isinstance(cats, str) else list(cats)
        if keep and not keep.intersection(cat_list):
            continue
        if prefixes and not any(c.startswith(prefixes) for c in cat_list):
            continue
        updated = row.get("update_date", row.get("updated", ""))
        if min_year and updated:
            try:
                if int(updated[:4]) < min_year:
                    continue
            except ValueError:
                pass
        yield Paper(
            id=str(row["id"]),
            title=(row.get("title") or "").strip(),
            abstract=(row.get("abstract") or "").strip(),
            categories=cat_list,
            updated=updated,
        )


def build_corpus_from_kaggle(
    kaggle_jsonl: str | Path,
    out_path: str | Path,
    categories: Iterable[str],
    max_papers: Optional[int] = None,
    min_year: Optional[int] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    prefixes: Optional[tuple[str, ...]] = None,
) -> int:
    """캐글 arXiv 스냅샷(원본)에서 원하는 분야만 걸러 우리 코퍼스 파일로 저장한다.

    논문을 고르는 방식은 두 가지다.
    - sample_size 지정 시: 조건에 맞는 논문 전체에서 무작위로 sample_size개를 고르게 뽑는다
      (reservoir sampling). 원본이 오래된 논문부터 정렬돼 있어, 앞에서 자르면 시기가
      한쪽으로 치우치므로 무작위 추출이 더 대표성 있다. 파일 전체를 한 번 훑는다.
    - max_papers 지정 시(또는 둘 다 없을 때): 조건에 맞는 논문을 앞에서부터 최대 max_papers개
      취한다 (빠르지만 시기 편향 가능).

    Args:
        categories: 남길 분야 목록 (예: cs.CL, cs.CV ...).
        max_papers: 앞에서부터 최대 개수 (sample_size가 없을 때만 사용).
        min_year: 이 연도 이후(갱신일 기준) 논문만.
        sample_size: 무작위로 뽑을 논문 수.
        seed: 무작위 추출 재현용 시드.

    Returns:
        저장한 논문 수.
    """
    keep = set(categories)
    matches = _iter_kaggle_matches(kaggle_jsonl, keep, min_year, prefixes)

    if sample_size:
        rng = random.Random(seed)
        reservoir: list[Paper] = []
        for i, paper in enumerate(matches):
            if len(reservoir) < sample_size:
                reservoir.append(paper)
            else:
                j = rng.randint(0, i)
                if j < sample_size:
                    reservoir[j] = paper
        reservoir.sort(key=lambda p: p.id)
        return save_corpus(out_path, reservoir)

    def _capped() -> Iterator[Paper]:
        for count, paper in enumerate(matches, start=1):
            yield paper
            if max_papers and count >= max_papers:
                break

    return save_corpus(out_path, _capped())


def main() -> None:
    from src import config

    ap = argparse.ArgumentParser(
        description="캐글 arXiv 스냅샷에서 분야를 걸러 코퍼스를 만든다"
    )
    ap.add_argument("--kaggle", required=True, help="캐글 원본 JSON 경로")
    ap.add_argument("--out", required=True, help="만들 코퍼스 저장 경로")
    ap.add_argument("--categories", nargs="+", default=list(config.CORPUS_CATEGORIES))
    ap.add_argument("--min-year", type=int, default=2021)
    ap.add_argument("--sample", type=int, default=30000,
                    help="무작위로 뽑을 논문 수 (0이면 앞에서부터 --max-papers개)")
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = build_corpus_from_kaggle(
        kaggle_jsonl=args.kaggle,
        out_path=args.out,
        categories=args.categories,
        max_papers=args.max_papers,
        min_year=args.min_year,
        sample_size=(args.sample or None),
        seed=args.seed,
    )
    print(f"코퍼스 {n}편 저장 → {args.out}")
    print(f"  분야: {args.categories}")
    print(f"  {args.min_year}년 이후, "
          f"{'무작위 ' + str(args.sample) + '편' if args.sample else '앞에서부터'}")


if __name__ == "__main__":
    main()
