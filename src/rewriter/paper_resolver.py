"""특정 논문 지목(known-item) 해결기 — 방안 A.

'트랜스포머를 최초로 소개한 논문' 처럼 사용자가 특정 유명 논문을 '설명'으로 가리킬 때,
키워드 검색은 그게 어떤 논문인지 못 알아낸다. 대신 LLM의 지식으로 그 논문의 실제 제목을
추정하고, **arXiv에서 그 제목의 논문이 실제로 있는지 검증**한 뒤에만 제시한다.

검증이 핵심이다: LLM이 없는 제목을 그럴싸하게 지어낼 수 있으므로(할루시네이션), arXiv에서
제목이 실제로 확인될 때만 "이 논문을 찾으시는 것 같습니다"로 내놓는다. 확인 안 되면 버린다.
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
    """추정 제목과 arXiv 결과 제목이 충분히 일치하는지(단어 겹침 비율)."""
    rw, cw = _norm_words(resolved), _norm_words(candidate)
    if not rw:
        return False
    return len(rw & cw) / len(rw) >= threshold


class PaperResolver:
    """질문이 특정 논문을 가리키면 그 제목을 추정한다 (검증 전)."""

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
    """설명→제목 추정 후 arXiv에서 검증한다.

    Returns:
        (검증된 논문 or None, 추정 제목 or None)
        - (논문, 제목): 특정 논문으로 판단 + arXiv에서 제목 확인됨 → 제시 가능
        - (None, 제목): 제목은 추정했으나 arXiv에서 확인 안 됨 → 할루시네이션 의심, 제시 안 함
        - (None, None): 특정 논문을 가리키는 질문이 아님
    """
    title = resolver.resolve(query)
    if not title:
        return None, None
    # 제목 구문으로 arXiv 검색해 실제 존재를 확인
    for r in arxiv_retriever.search(f'ti:"{title}"', k=k):
        if title_match(title, r.title):
            return r, title
    return None, title
