"""특정 논문을 설명으로 가리키는 질문 처리.

"트랜스포머를 최초로 소개한 논문" 같은 질문은 키워드 검색으로 못 알아냄. 언어 모델의
지식으로 제목을 추정하고, arXiv 에서 그 제목의 논문이 실제로 있는지 확인한 뒤에만
사용자에게 보여 줌. 확인이 안 되면 지어낸 제목일 수 있으므로 버림.
"""

from __future__ import annotations

import re

from src.rewriter.base import OllamaClient
from src.schemas import ScoredPaper

RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "names_specific_paper": {"type": "boolean"},
        "title": {"type": "string"},
    },
    "required": ["names_specific_paper", "title"],
}

RESOLVE_SYSTEM = (
    "사용자의 검색어가 '특정한 하나의 잘 알려진 논문'을 가리키는지 판단하라.\n"
    "예: '트랜스포머를 처음 소개한 논문', 'BERT 원 논문', 'ResNet을 제안한 논문' → 특정 논문.\n"
    "일반적인 주제 검색(예: '이미지 분류 방법', '가짜 뉴스 걸러내기')이면 특정 논문이 아니다.\n\n"
    "특정 논문이면 names_specific_paper=true, 그 논문의 **정확한 영어 제목**을 title에 넣어라.\n"
    "아니거나 제목이 확실치 않으면 names_specific_paper=false, title은 빈 문자열로 하라.\n"
    "확실히 아는 제목만 답하라. 모르면 false."
)


def _norm_words(s: str) -> set[str]:
    return set(re.findall(r"[0-9a-z]+", s.lower()))


def title_match(resolved: str, candidate: str, threshold: float = 0.8) -> bool:
    """추정 제목과 arXiv 결과 제목이 낱말 겹침 비율로 충분히 맞는지."""
    rw, cw = _norm_words(resolved), _norm_words(candidate)
    if not rw:
        return False
    return len(rw & cw) / len(rw) >= threshold


class PaperResolver:
    """질문이 특정 논문을 가리키면 그 제목을 추정함. 확인은 아래 함수가 함."""

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def resolve(self, query: str) -> str | None:
        try:
            data = self.client.generate_json(
                f"검색어: {query}", RESOLVE_SCHEMA,
                system=RESOLVE_SYSTEM, temperature=0.0,
            )
            if data.get("names_specific_paper") and str(data.get("title", "")).strip():
                return str(data["title"]).strip()
        except Exception:
            pass
        return None


def resolve_and_verify(query: str, resolver: PaperResolver, arxiv_retriever,
                       k: int = 3) -> tuple[ScoredPaper | None, str | None]:
    """제목을 추정한 뒤 arXiv 에서 실제로 있는지 확인함.

    Returns:
        (논문, 제목)   제목이 arXiv 에서 확인됨. 사용자에게 보여도 됨
        (None, 제목)   제목을 추정했으나 확인 안 됨. 지어낸 것일 수 있어 안 보여 줌
        (None, None)   특정 논문을 가리키는 질문이 아님
    """
    title = resolver.resolve(query)
    if not title:
        return None, None
    # 제목으로 arXiv 를 찾아 실제로 있는지 확인
    for r in arxiv_retriever.search(f'ti:"{title}"', k=k):
        if title_match(title, r.title):
            return r, title
    return None, title
