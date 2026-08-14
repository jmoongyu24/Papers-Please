"""논문 추천 에이전트 — 검색된 논문을 사용자 의도와 대조해 '추천 + 이유'를 낸다.

동작(한 번의 로컬 Qwen3-4B 호출):
  입력: 사용자 원본 검색어 + 검색된 논문 목록(제목 + '전체 초록')
  판단: 각 논문이 사용자 의도에 얼마나 맞는지(high/medium/low)와 그 이유
  출력(JSON 강제): [{index, relevance, reason}, ...] + 전체 요약(summary)

설계 이유:
- **로컬 Qwen3-4B**: 서비스 런타임은 로컬 실행이 제약(오프라인 평가에만 유료 모델 사용). 그래서
  이 에이전트도 로컬 모델을 쓴다.
- **JSON 스키마 강제**: 사고 과정 유출(ISSUE #3)을 막고 논문별 판단을 안정적으로 파싱한다.
- **그라운딩**: 주어진 초록 내용에만 근거하고, 없는 사실을 지어내거나 목록 밖 번호를 만들지
  않도록 프롬프트로 강제 + 코드로 밖 번호를 필터한다(환각 방지).
- **arXiv 키워드 검색의 한계 보완**: arXiv는 키워드 순위라 의도와 어긋난 논문이 위에 올 수 있는데
  (ISSUE #7), 이 에이전트가 '의미로' 다시 골라줘 사용자가 볼 최종 추천의 정확도를 높인다.
"""

from __future__ import annotations

from src.rewriter.base import OllamaClient
from src.schemas import ScoredPaper

RECOMMEND_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["index", "relevance", "reason"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["recommendations", "summary"],
}

RECOMMEND_SYSTEM = (
    "너는 학술 논문 추천 도우미다. 사용자의 검색 의도와, 검색된 논문 목록(제목+초록)을 보고 "
    "각 논문이 사용자 의도에 얼마나 맞는지 판단해 추천한다.\n"
    "규칙:\n"
    "- 반드시 **주어진 논문의 초록 내용에만** 근거하라. 초록에 없는 내용을 지어내지 마라.\n"
    "- 각 논문에 relevance(high=핵심적으로 맞음 / medium=관련 있음 / low=거의 무관)와 "
    "reason(한국어 1~2문장, 초록의 어떤 점이 의도에 맞고 안 맞는지)을 매겨라.\n"
    "- index는 입력에 준 번호를 **그대로** 쓴다. 목록에 없는 번호를 만들지 마라.\n"
    "- 의도에 맞는 논문이 하나도 없으면 전부 low로 하고 summary에 '맞는 논문이 없다'고 밝혀라.\n"
    "- summary: 어떤 논문(번호)을 왜 추천하는지 사용자에게 한국어로 2~4문장 종합한다.\n"
    "- 영어 학술 용어를 제외한 모든 설명은 한국어로."
)

# 초록이 너무 길면 자른다(입력 토큰 관리). 대부분 arXiv 초록은 이 안에 들어온다.
_ABSTRACT_CHARS = 3000


def build_prompt(query: str, papers: list[ScoredPaper]) -> str:
    lines = [f"[사용자 검색 의도] {query}", "", "[검색된 논문 목록]"]
    for i, p in enumerate(papers, start=1):
        lines.append(f"\n{i}. 제목: {p.title}")
        lines.append(f"   초록: {p.abstract[:_ABSTRACT_CHARS]}")
    lines.append("\n각 논문의 relevance와 이유, 그리고 종합 summary를 JSON으로 답하라.")
    return "\n".join(lines)


class PaperRecommender:
    """검색된 논문을 사용자 의도로 분석해 추천 목록과 이유를 만든다."""

    _ORDER = {"high": 0, "medium": 1, "low": 2}

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def recommend(self, query: str, papers: list[ScoredPaper]) -> dict:
        """Returns: {"recommendations": [{index, relevance, reason}...], "summary": str}.

        recommendations는 관련도 높은 순으로 정렬되며, 각 항목의 index는 papers의 1-기반 번호다.
        """
        if not papers:
            return {"recommendations": [], "summary": "검색 결과가 없습니다."}
        try:
            data = self.client.generate_json(
                build_prompt(query, papers), RECOMMEND_SCHEMA,
                system=RECOMMEND_SYSTEM, temperature=0.0, max_tokens=2000,
            )
        except Exception as e:
            return {"recommendations": [], "summary": f"(추천 생성 실패: {e})"}

        # 목록 밖 번호 제거 + 관련도 순 정렬 (환각 방지)
        recs = []
        seen = set()
        for r in data.get("recommendations", []):
            idx = r.get("index")
            if isinstance(idx, int) and 1 <= idx <= len(papers) and idx not in seen:
                seen.add(idx)
                recs.append({
                    "index": idx,
                    "relevance": r.get("relevance", "low"),
                    "reason": str(r.get("reason", "")).strip(),
                })
        recs.sort(key=lambda r: self._ORDER.get(r["relevance"], 3))
        return {"recommendations": recs, "summary": str(data.get("summary", "")).strip()}
