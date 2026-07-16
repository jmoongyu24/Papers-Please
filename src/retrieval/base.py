"""검색기들이 공통으로 따르는 약속(인터페이스)과 공용 도구.

모든 검색기는 `search(query, k)` 함수를 갖는다. 이렇게 모양을 맞춰야
평가 모듈이 검색기를 갈아 끼우며 비교할 수 있다.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from src.schemas import ScoredPaper

# 토큰(단어) 나누기: 소문자로 바꾼 뒤, 글자/숫자 덩어리만 뽑는다.
# 한글은 한글대로, 영어는 영어대로 하나의 토큰이 된다.
_TOKEN_RE = re.compile(r"[0-9a-z가-힣]+")


def tokenize(text: str) -> list[str]:
    """검색용으로 문장을 단어 목록으로 쪼갠다."""
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Retriever(Protocol):
    """검색기 인터페이스. 검색어를 받아 점수순 논문 목록을 돌려준다."""

    name: str

    def search(self, query: str, k: int) -> list[ScoredPaper]:
        ...
