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


def build_arxiv_query(raw_query: str, academic_terms: list[str],
                      max_len: int = 300) -> str:
    """arXiv 실시간 검색용 쿼리를 '코드로' 만든다.

    형태: all:"원본" OR abs:"학술용어1" OR abs:"학술용어2" OR ...  (300자 이내)

    왜 이렇게(실측으로 확정):
    - arXiv는 따옴표 구(句) 검색이라야 정확 매칭이 되고, 따옴표 없이 넓은 단어를 나열하면
      수천 편에 묻혀 원하는 논문이 사라진다. 그래서 각 용어를 따옴표 구로 만든다.
    - 원본을 따옴표로 넣으면, 사용자가 정확한 제목·구절을 쳤을 때 그게 잡힌다
      (예: "Attention is all you need"). 원본이 일상어·한국어라 안 걸리면 그 절은 무해하게 빈다.
    - **학술 용어는 하나만 쓰면 재현율이 떨어진다**(좁은 구는 arXiv에 드물어 0건이 되기도 함).
      그래서 변환기가 만든 용어를 **전부 OR로** 넣어, 어느 하나라도 맞으면 나오게 한다.
      각 용어가 따옴표 구라 넓은 단어 나열 같은 희석은 생기지 않는다. (특정 논문 콕 집기는
      paper_resolver가 따로 처리하므로 본 검색은 넓게 잡는 게 맞다.)
    - arXiv 문법 구성을 불안정한 LLM에 맡기지 않고 코드가 만들어 재현성을 확보한다.
    """
    def _clean(s: str) -> str:
        return s.replace('"', " ").strip()

    parts = []
    orig = _clean(raw_query)
    if orig:
        parts.append(f'all:"{orig}"')
    for t in academic_terms:
        term = _clean(str(t))
        if term:
            parts.append(f'abs:"{term}"')

    # 300자 제한 내에서 OR로 잇는다 (넘치면 뒤 용어는 버림)
    query = ""
    for p in parts:
        candidate = p if not query else f"{query} OR {p}"
        if len(candidate) > max_len:
            break
        query = candidate
    return query


def _to_result(raw_query: str, data: dict) -> RewriteResult:
    """모델이 준 JSON(dict)을 RewriteResult로 바꾼다. 필수 항목이 없으면 예외."""
    queries = data["queries"]
    # sparse·dense 검색어는 모델이 준 것을 쓴다(로컬 검색기용). 비었으면 폴백하도록 예외.
    for b in ("sparse", "dense"):
        if not str(queries.get(b, "")).strip():
            raise ValueError(f"queries.{b} 가 비어 있음")
    academic_terms = list(data.get("academic_terms", []))
    return RewriteResult(
        raw_query=raw_query,
        queries={
            "sparse": str(queries["sparse"]).strip(),
            "dense": str(queries["dense"]).strip(),
            # arXiv 검색어는 LLM 대신 코드로 구성(위 함수) — arXiv 로직에 맞춤
            "arxiv": build_arxiv_query(raw_query, academic_terms),
        },
        intent=str(data.get("intent", "")),
        concepts=list(data.get("concepts", [])),
        academic_terms=academic_terms,
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
