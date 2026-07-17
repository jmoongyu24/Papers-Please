"""비교용(baseline) 변환기 — 성능 평가에서 우리 계층 변환과 비교하는 대조군.

- SingleStepRewriter: 계층 없이 "학술 검색어로 바꿔줘" 한 번만 시킨다.
  → 우리의 '계층 구조' 자체가 성능에 얼마나 기여하는지 가려낸다.
- HydeRewriter: 질문에 답할 법한 가상의 초록을 지어내, 그 글로 의미 검색을 한다(HyDE).
  → 이미 알려진 다른 방법과 비교한다.

둘 다 hierarchical.py와 같은 OllamaClient를 재사용하고, 같은 rewrite 인터페이스를 따른다.
"""

from __future__ import annotations

from src.rewriter.base import BACKENDS
from src.rewriter.llm_client import OllamaClient
from src.rewriter.prompts import (
    HYDE_SCHEMA, SINGLE_STEP_SCHEMA, hyde_prompt, single_step_prompt,
)
from src.schemas import RewriteResult


class SingleStepRewriter:
    """한 번에 변환: 원본 → 영어 학술 검색어 (계층 없음)."""

    name = "single_step"

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def rewrite(self, raw_query: str) -> RewriteResult:
        try:
            data = self.client.generate_json(
                single_step_prompt(raw_query), SINGLE_STEP_SCHEMA, temperature=0.0
            )
            text = str(data.get("query", "")).strip()
            if not text:
                raise ValueError("빈 출력")
            # 세 검색 방식에 같은 결과를 넣는다(계층 변환처럼 방식별로 나누지 않음)
            return RewriteResult(
                raw_query=raw_query,
                queries={b: text for b in BACKENDS},
                intent=text, academic_terms=text.split(), parse_ok=True,
            )
        except Exception as e:
            return RewriteResult(
                raw_query=raw_query, queries={b: raw_query for b in BACKENDS},
                intent=f"(실패, 원본 사용) {e}", parse_ok=False,
            )


class HydeRewriter:
    """HyDE: 가상 초록을 생성해 의미(dense) 검색어로 쓴다.

    단어 일치(sparse)·arxiv 검색어에는 가상 초록이 너무 길어 부적합하므로 원본을 쓴다.
    HyDE의 효과는 주로 의미 기반 검색에서 나타난다.
    """

    name = "hyde"

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def rewrite(self, raw_query: str) -> RewriteResult:
        try:
            data = self.client.generate_json(
                hyde_prompt(raw_query), HYDE_SCHEMA, temperature=0.7, max_tokens=400
            )
            abstract = str(data.get("abstract", "")).strip()
            if not abstract.strip():
                raise ValueError("빈 출력")
            return RewriteResult(
                raw_query=raw_query,
                queries={"sparse": raw_query, "dense": abstract, "arxiv": raw_query},
                intent="(HyDE 가상 초록 생성)", parse_ok=True,
            )
        except Exception as e:
            return RewriteResult(
                raw_query=raw_query, queries={b: raw_query for b in BACKENDS},
                intent=f"(실패, 원본 사용) {e}", parse_ok=False,
            )
