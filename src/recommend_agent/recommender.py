"""논문 추천 에이전트 - 검색된 논문을 사용자 의도와 대조해 추천 이유를 붙임.

로컬 Qwen3-4B 를 한 번 불러서 처리함.

    입력  사용자 원본 질문 + 논문 목록(제목 + 전체 초록)
    판단  각 논문이 의도에 얼마나 맞는지(high/medium/low)와 그 이유
    출력  [{index, relevance, reason}, ...] 와 전체 요약

JSON 스키마를 강제해 형식을 고정하고, 주어진 초록 내용에만 근거하도록 프롬프트로
지시한 뒤 목록 밖 번호는 코드로 걸러 냄.

summary 에는 번호를 쓰지 말고 제목을 쓰라고 지시함. 화면은 관련도 순으로 다시 정렬하고
점수가 낮은 것을 걸러낸 뒤 번호를 새로 매기므로, 여기서 붙인 번호가 화면 번호와 어긋남.
`index` 필드는 그대로 번호를 쓰는데, 그쪽은 부르는 쪽이 원래 순번으로 짝지어 처리함.
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
    "- summary: 어떤 논문을 왜 추천하는지 사용자에게 한국어로 2~4문장 종합한다. "
    "논문을 가리킬 때는 **제목**을 쓰고 번호는 절대 쓰지 마라.\n"
    "- 영어 학술 용어를 제외한 모든 설명은 한국어로."
)

# 초록이 길면 자름. 대부분의 arXiv 초록은 이 안에 들어옴
_ABSTRACT_CHARS = 3000


def build_prompt(query: str, papers: list[ScoredPaper]) -> str:
    lines = [f"[사용자 검색 의도] {query}", "", "[검색된 논문 목록]"]
    for i, p in enumerate(papers, start=1):
        lines.append(f"\n{i}. 제목: {p.title}")
        lines.append(f"   초록: {p.abstract[:_ABSTRACT_CHARS]}")
    lines.append("\n각 논문의 relevance와 이유, 그리고 종합 summary를 JSON으로 답하라.")
    return "\n".join(lines)


class PaperRecommender:
    """검색된 논문을 사용자 의도와 대조해 추천 목록과 이유를 만듦."""

    _ORDER = {"high": 0, "medium": 1, "low": 2}

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def recommend(self, query: str, papers: list[ScoredPaper]) -> dict:
        """{"recommendations": [{index, relevance, reason}...], "summary": str} 를 돌려줌.

        관련도가 높은 순으로 정렬되고, index 는 papers 의 1부터 시작하는 번호임.
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

        # 목록에 없는 번호를 지어냈으면 버리고, 관련도 순으로 정렬함
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
