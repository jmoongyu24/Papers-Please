"""구(句)가 arXiv 논문 몇 편에 등장하는지 세는 도구 — '문서 빈도' 계산.

왜 필요한가:
검색어가 정답 논문을 찾아내느냐는 **구가 얼마나 흔한가**에 크게 좌우된다. 실측(ISSUE #16)에
따르면 검색에 성공한 검색어가 쓴 구는 3만 편 중 중앙값 5편에만 등장했고, 정답이 묻힌
검색어의 구는 226편에 등장했다 — 45배 차이다. 넓은 구는 합집합에 수천 편을 끌어와
정답을 밀어낸다.

왜 작은 코퍼스로는 안 되는가 (실패에서 배운 것):
처음에는 3만 편 코퍼스로 검색어를 채점했는데, "로컬에서 정답 1등"인 검색어 30개를 실제
arXiv로 확인하니 **29개가 100등 밖**이었다. 원인은 채점 방식이 아니라 코퍼스 크기였다.
3만 편에서는 구 대부분이 0~5편에만 걸려 합집합이 너무 작고, 그래서 정답이 자동으로 1등이
된다. arXiv 310만 편에서는 같은 구가 수백~수천 편에 걸린다. 즉 **문서 빈도를 재려면
arXiv와 비슷한 규모의 코퍼스가 필요하다.**

어떻게 세는가:
310만 편에서 임의의 구를 세는 것은 느리다. 그래서 세 가지를 쓴다.
  1) **묶음 처리** — 필요한 구 수천 개를 모아 코퍼스를 딱 한 번만 훑는다.
  2) **첫 단어로 미리 거르기** — 문서마다 수천 개를 전부 검사하지 않고, 그 문서에 있는
     단어로 시작하는 구만 실제로 확인한다.
  3) **디스크 저장** — 한 번 센 구는 다시 세지 않는다.

사용 예:
    stats = PhraseFrequency("data/corpus/corpus-full.jsonl")
    stats.ensure(["graph neural network", "age of information"])   # 없는 것만 센다
    stats.df("graph neural network")     # -> 등장 논문 수
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from src import config

# 구를 단어로 쪼갤 때 쓰는 규칙. 소문자화 + 영숫자/하이픈만.
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")

DEFAULT_CACHE = config.DATA_DIR / "cache" / "phrase_df.json"


def normalize(phrase: str) -> str:
    """구를 비교용 형태로 통일한다(소문자 + 공백 정리)."""
    return " ".join(phrase.lower().split())


def first_word(phrase: str) -> str:
    m = _WORD_RE.search(phrase)
    return m.group(0) if m else ""


class PhraseFrequency:
    """구별 문서 빈도를 세고 보관한다."""

    def __init__(self, corpus_path: str | Path,
                 cache_path: str | Path | None = DEFAULT_CACHE):
        self.corpus_path = Path(corpus_path)
        self.cache_path = Path(cache_path) if cache_path else None
        self.n_docs: int = 0
        self._df: dict[str, int] = {}
        if self.cache_path and self.cache_path.exists():
            saved = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._df = saved.get("df", {})
            self.n_docs = saved.get("n_docs", 0)

    # ── 조회 ─────────────────────────────────────────────────────────────
    def df(self, phrase: str) -> int | None:
        """그 구가 등장하는 논문 수. 아직 안 세었으면 None."""
        return self._df.get(normalize(phrase))

    def known(self) -> int:
        return len(self._df)

    # ── 계산 ─────────────────────────────────────────────────────────────
    def ensure(self, phrases: list[str], verbose: bool = True) -> int:
        """아직 안 센 구만 골라 코퍼스를 한 번 훑으며 센다.

        Returns: 이번에 새로 센 구의 개수.
        """
        wanted = {normalize(p) for p in phrases if normalize(p)}
        todo = sorted(wanted - self._df.keys())
        if not todo:
            if verbose:
                print(f"  전부 이미 세어 둠 (보관 중인 구 {len(self._df):,}개)")
            return 0

        # 첫 단어 -> 그 단어로 시작하는 구 목록
        by_first: dict[str, list[str]] = {}
        for p in todo:
            by_first.setdefault(first_word(p), []).append(p)

        counts = dict.fromkeys(todo, 0)
        if verbose:
            print(f"  새로 셀 구 {len(todo):,}개 · 코퍼스를 한 번 훑는다", flush=True)

        t0 = time.time()
        n_docs = 0
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                n_docs += 1
                text = f"{row.get('title','')} {row.get('abstract','')}".lower()
                # 이 문서에 등장하는 단어로 시작하는 구만 실제로 확인한다
                seen_first = set(_WORD_RE.findall(text))
                for w in seen_first:
                    for p in by_first.get(w, ()):
                        if p in text:
                            counts[p] += 1
                if verbose and n_docs % 500000 == 0:
                    print(f"    {n_docs:,}편 ({time.time()-t0:.0f}초)", flush=True)

        self._df.update(counts)
        self.n_docs = n_docs
        if verbose:
            print(f"  완료: 논문 {n_docs:,}편 · {time.time()-t0:.0f}초", flush=True)
        self.save()
        return len(todo)

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps({"n_docs": self.n_docs, "df": self._df}, ensure_ascii=False),
            encoding="utf-8")
