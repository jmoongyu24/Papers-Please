"""논문 코퍼스 만들기"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

from src.schemas import Paper
from src.utils import read_jsonl, write_jsonl

# 끝에 붙은 버전 표기(v+숫자) 제거
_VERSION_SUFFIX = re.compile(r"v\d+$")


def normalize_paper_id(paper_id: str) -> str:
    """
    논문 번호를 비교 가능한 형태로 가공
    '2103.00020v2' -> '2103.00020',  'solv-int/9611001v1' -> 'solv-int/9611001'
    """
    return _VERSION_SUFFIX.sub("", (paper_id or "").strip())


def save_corpus(path: str | Path, papers: Iterable[Paper]) -> int:
    """Paper 목록을 JSON 코퍼스 파일로 저장"""
    return write_jsonl(path, (p.to_dict() for p in papers))


def _iter_kaggle_matches(
    kaggle_jsonl: str | Path,
    keep: set[str],
    min_year: Optional[int],
    prefixes: Optional[tuple[str, ...]] = None
) -> Iterator[Paper]:
    """캐글 논문 데이터를 읽어 조건에 맞는 논문만 Paper로 저장

    분야를 고르는 방법 세 가지.
    - 'keep'에 있는 분야명 (ex: {"cs.CL", "cs.CV"})
    - 'prefixes'로 있는 분야 전체 (예: ("cs.", "stat."))
    - 두 조건 다 없으면 분야 구분 없이 전부
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
            updated=updated
        )


def build_corpus_from_kaggle(
    kaggle_jsonl: str | Path,
    out_path: str | Path,
    categories: Iterable[str],
    max_papers: Optional[int] = None,
    min_year: Optional[int] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    prefixes: Optional[tuple[str, ...]] = None
) -> int:
    """
    캐글 arXiv 데이터셋에서 조건에 맞는 논문만 걸러 코퍼스 파일로 저장함

    'sample_size'를 주면 조건에 맞는 논문을 전체에서 무작위로 그만큼 뽑음
    값을 안 주면 앞에서부터 'max_papers'만큼 뽑음
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", nargs="+", default=list(config.CORPUS_CATEGORIES))
    ap.add_argument("--prefixes", nargs="+", default=None)
    ap.add_argument("--min-year", type=int, default=2021)
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build_corpus_from_kaggle(
        kaggle_jsonl=args.kaggle,
        out_path=args.out,
        categories=[] if args.prefixes else args.categories,
        max_papers=args.max_papers,
        min_year=args.min_year,
        sample_size=(args.sample or None),
        seed=args.seed,
        prefixes=tuple(args.prefixes) if args.prefixes else None
    )


if __name__ == "__main__":
    main()
