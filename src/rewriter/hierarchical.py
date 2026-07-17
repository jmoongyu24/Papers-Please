"""계층적(단계별) 쿼리 변환 — 이 프로젝트의 핵심 시스템.

사용자 검색어를 Qwen3-4B에게 주고, intent(의도) → concepts(개념) → academic_terms(전문 용어)
→ queries(검색 방식별 최종 검색어) 순서로 한 번에 생성하게 한다. JSON 스키마를 강제하므로
출력이 항상 파싱 가능하다.

실패 대비: 혹시 파싱/검증이 실패하면 한 번 재시도하고, 그래도 안 되면 원본 검색어를 그대로
쓰는 폴백으로 넘어간다(parse_ok=False로 기록). 실패율 자체가 모델·프롬프트 품질 지표다.
"""

from __future__ import annotations

from src.rewriter.base import BACKENDS
from src.rewriter.llm_client import OllamaClient
from src.rewriter.prompts import OUTPUT_SCHEMA, SYSTEM, build_messages
from src.schemas import RewriteResult


def _to_result(raw_query: str, data: dict) -> RewriteResult:
    """모델이 준 JSON(dict)을 RewriteResult로 바꾼다. 필수 항목이 없으면 예외."""
    queries = data["queries"]
    # 세 검색어가 모두 비지 않았는지 확인 (비었으면 폴백하도록 예외)
    for b in BACKENDS:
        if not str(queries.get(b, "")).strip():
            raise ValueError(f"queries.{b} 가 비어 있음")
    return RewriteResult(
        raw_query=raw_query,
        queries={b: str(queries[b]).strip() for b in BACKENDS},
        intent=str(data.get("intent", "")),
        concepts=list(data.get("concepts", [])),
        academic_terms=list(data.get("academic_terms", [])),
        categories=list(data.get("categories", [])),
        parse_ok=True,
    )


class HierarchicalRewriter:
    """Qwen3-4B로 4단계 계층 변환을 수행한다."""

    name = "hierarchical"

    def __init__(self, client: OllamaClient | None = None, retries: int = 1):
        self.client = client or OllamaClient()
        self.retries = retries

    def rewrite(self, raw_query: str) -> RewriteResult:
        messages = build_messages(raw_query)
        last_err = None
        for _ in range(self.retries + 1):
            try:
                data = self.client.generate_json(
                    messages, OUTPUT_SCHEMA, system=SYSTEM, temperature=0.0
                )
                return _to_result(raw_query, data)
            except Exception as e:  # 파싱/검증/통신 실패 → 재시도
                last_err = e
                continue

        # 폴백: 원본 검색어를 그대로 쓴다 (검색이 아예 멈추지 않도록)
        return RewriteResult(
            raw_query=raw_query,
            queries={b: raw_query for b in BACKENDS},
            intent=f"(변환 실패, 원본 사용) {last_err}",
            parse_ok=False,
        )
