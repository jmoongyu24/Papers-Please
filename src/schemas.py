"""모듈 사이에 주고 받는 자료 구조"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Paper:
    """논문 한 편. 코퍼스의 기본 단위이자 검색 결과의 단위"""

    id: str                       # arXiv 번호. 예) "2103.00020"
    title: str
    abstract: str
    categories: list[str] = field(default_factory=list)
    updated: str = ""             # 갱신일. 예) "2021-03-01"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RewriteResult:
    """쿼리 변환 결과. 단계별 내용과 검색 방식별 최종 검색어를 담음.

    검색 방식 이름: "dense"(의미 검색) · "arxiv"(arXiv 실시간 검색).
    """

    raw_query: str                                           # 원래 입력
    queries: dict[str, str]                                  # 검색 방식 -> 최종 검색어
    intent: str = ""                                         # 의도
    concepts: list[str] = field(default_factory=list)        # 핵심 개념
    academic_terms: list[str] = field(default_factory=list)  # 영어 학술 용어
    categories: list[str] = field(default_factory=list)      # 추정 분야
    parse_ok: bool = True                                    # 출력 형식이 정상이었는지

    def query_for(self, backend: str) -> str:
        """해당 검색 방식의 검색어. 없으면 원본 질문"""
        return self.queries.get(backend) or self.raw_query

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredPaper:
    """검색 결과 한 줄: 논문과 그 점수, 등수"""

    paper_id: str
    score: float
    rank: int                     # 1등이 1
    title: str = ""
    abstract: str = ""
