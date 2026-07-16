"""프로젝트에서 주고받는 데이터의 '모양'을 정의하는 파일.

여기서 정의한 형태(논문, 평가용 질문, 검색 결과 등)를 모든 모듈이 공유한다.
파이썬 표준 dataclass만 써서 가볍게 유지한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Paper:
    """논문 한 편. 코퍼스의 기본 단위이자 검색 결과의 단위."""

    id: str                       # arXiv 번호. 예: "2103.00020"
    title: str
    abstract: str
    categories: list[str] = field(default_factory=list)
    updated: str = ""             # 갱신일 (문자열, 예: "2021-03-01")

    @property
    def text(self) -> str:
        """검색 색인 대상 텍스트 = 제목 + 초록."""
        return f"{self.title}\n{self.abstract}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        cats = d.get("categories", [])
        if isinstance(cats, str):
            cats = cats.split()
        return cls(
            id=str(d["id"]),
            title=d.get("title", "").strip(),
            abstract=d.get("abstract", "").strip(),
            categories=list(cats),
            updated=d.get("updated", ""),
        )


@dataclass
class EvalQuery:
    """평가용 질문 한 개. '이 질문의 정답은 이 논문'이라는 정답지의 한 줄."""

    query_id: str
    gold_id: str                  # 정답 논문의 arXiv 번호
    text: str                     # 사용자가 입력하는 검색어 (일상어/어설픈 전문어 등)
    difficulty: str = "easy"      # easy(일상어) / mid(어설픈 전문어) / hard(정확한 전문어)
    lang: str = "ko"              # ko / en
    source: str = "synthetic"     # synthetic(자동 생성) / user(실사용자)
    gen_model: str = ""           # 어떤 모델로 생성했는지 (자동 생성일 때)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQuery":
        return cls(
            query_id=str(d["query_id"]),
            gold_id=str(d["gold_id"]),
            text=d["text"],
            difficulty=d.get("difficulty", "easy"),
            lang=d.get("lang", "ko"),
            source=d.get("source", "synthetic"),
            gen_model=d.get("gen_model", ""),
        )


@dataclass
class RewriteResult:
    """쿼리 변환 결과.

    단계별 내용(steps)과, 검색 방식별로 다르게 만든 최종 검색어(queries)를 담는다.
    검색 방식 이름: "sparse"(단어 일치), "dense"(의미 기반), "arxiv"(실시간 arXiv).
    """

    raw_query: str                                   # 원래 입력
    queries: dict[str, str]                          # 검색 방식 -> 최종 검색어
    intent: str = ""                                 # 1단계: 의도
    concepts: list[str] = field(default_factory=list)  # 2단계: 핵심 개념
    academic_terms: list[str] = field(default_factory=list)  # 3단계: 전문 용어
    categories: list[str] = field(default_factory=list)      # 추정 분야
    parse_ok: bool = True                            # 출력 형식이 정상이었는지

    def query_for(self, backend: str) -> str:
        """해당 검색 방식에 맞는 최종 검색어를 준다. 없으면 원본으로 폴백."""
        return self.queries.get(backend) or self.raw_query

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredPaper:
    """검색 결과 한 줄: 논문과 그 점수/순위."""

    paper_id: str
    score: float
    rank: int                     # 1등이 1
    title: str = ""
    abstract: str = ""
