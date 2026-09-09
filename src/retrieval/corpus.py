"""논문 코퍼스 만들기와 논문 번호 표기 통일.

코퍼스는 JSON Lines 파일임 (줄마다 논문 하나). 캐글 arXiv 스냅샷에서 원하는 분야만
걸러 만듦.

`normalize_paper_id` 도 여기 있음. 검색, 평가, 화면 어디서든 "같은 논문인가" 를
판단할 때 반드시 거쳐야 하는 함수임.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

from src.schemas import Paper
from src.utils import read_jsonl, write_jsonl

# 끝에 붙은 버전 표기(v + 숫자)만 떼어냄. 중간의 v 는 건드리지 않음.
_VERSION_SUFFIX = re.compile(r"v\d+$")


def normalize_paper_id(paper_id: str) -> str:
    """논문 번호를 어디서든 비교 가능한 한 형태로 만듦.

    '2103.00020v2' -> '2103.00020',  'solv-int/9611001v1' -> 'solv-int/9611001'

    arXiv 실시간 결과에는 버전 표기가 붙어 오고 로컬 색인에는 안 붙어 있음. 그대로
    합치면 같은 논문이 둘로 갈려 점수가 쪼개짐 - 오류 없이 성능만 깎임.

    반드시 정규식으로 뗄 것. `paper_id.split("v")[0]` 은 옛 형식 번호
    `solv-int/9611001v1` 을 `sol` 로 자름.
    """
    return _VERSION_SUFFIX.sub("", (paper_id or "").strip())


def save_corpus(path: str | Path, papers: Iterable[Paper]) -> int:
    """Paper 목록을 JSON Lines 코퍼스 파일로 저장함."""
    return write_jsonl(path, (p.to_dict() for p in papers))


def _iter_kaggle_matches(
    kaggle_jsonl: str | Path,
    keep: set[str],
    min_year: Optional[int],
    prefixes: Optional[tuple[str, ...]] = None,
) -> Iterator[Paper]:
    """캐글 원본을 한 줄씩 읽어 분야, 연도 조건에 맞는 논문만 Paper 로 내보냄.

    분야를 고르는 방법 세 가지.
    - `keep` 에 정확한 분야명 (예: {"cs.CL", "cs.CV"})
    - `prefixes` 에 앞글자를 주면 그 계열 전체 (예: ("cs.", "stat."))
    - 둘 다 비우면 분야 제한 없이 전부
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
    """캐글 arXiv 스냅샷에서 조건에 맞는 논문만 걸러 코퍼스 파일로 저장하고 편수를 돌려줌.

    `sample_size` 를 주면 조건에 맞는 논문 전체에서 무작위로 그만큼 뽑음. 원본이 오래된
    논문부터 정렬돼 있어 앞에서 자르면 시기가 한쪽으로 치우치기 때문임. 안 주면
    앞에서부터 `max_papers` 편까지 취함.
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
    ap.add_argument("--prefixes", nargs="+", default=None,
                    help="분야 앞글자로 고르기 (예: cs. stat.ML eess.). 주면 --categories 대신 "
                         "이 조건을 쓴다")
    ap.add_argument("--min-year", type=int, default=2021)
    ap.add_argument("--sample", type=int, default=30000,
                    help="무작위로 뽑을 논문 수 (0이면 조건에 맞는 논문 전부)")
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = build_corpus_from_kaggle(
        kaggle_jsonl=args.kaggle,
        out_path=args.out,
        categories=[] if args.prefixes else args.categories,
        max_papers=args.max_papers,
        min_year=args.min_year,
        sample_size=(args.sample or None),
        seed=args.seed,
        prefixes=tuple(args.prefixes) if args.prefixes else None,
    )
    print(f"코퍼스 {n}편 저장 -> {args.out}")
    print(f"  분야: {args.prefixes or args.categories}")
    print(f"  {args.min_year}년 이후, "
          f"{'무작위 ' + str(args.sample) + '편' if args.sample else '앞에서부터'}")


if __name__ == "__main__":
    main()
