"""쿼리 변환기들이 공통으로 따르는 약속(인터페이스)과, 평가용 기본 변환기들.

모든 변환기는 `rewrite(raw_query) -> RewriteResult` 함수를 갖는다.
평가 모듈은 이 인터페이스만 알고, 변환기를 갈아 끼우며 성능을 비교한다.

여기 담긴 변환기:
- PassthroughRewriter : 변환을 아예 안 함. "변환 전" 기준점(baseline).
- MockExpanderRewriter: 규칙 기반의 아주 단순한 확장기. 진짜 언어 모델(Qwen3-4B)이
  준비되기 전까지 파이프라인을 돌려보기 위한 '임시 대역'이다. 성능 숫자를 위한 것이
  아니라, 평가 파이프라인이 변환 전/후 차이를 제대로 잡아내는지 확인하는 용도다.

진짜 계층적 변환기(hierarchical.py)는 모듈 ①에서 Qwen3-4B로 구현하며,
이 파일과 똑같은 인터페이스를 따르므로 그대로 평가에 꽂을 수 있다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.schemas import RewriteResult

# 검색 방식 이름 (schemas.RewriteResult.queries 의 키)
BACKENDS = ("sparse", "dense", "arxiv")


@runtime_checkable
class Rewriter(Protocol):
    name: str

    def rewrite(self, raw_query: str) -> RewriteResult:
        ...


class PassthroughRewriter:
    """변환하지 않고 원본 검색어를 그대로 쓴다 (변환 전 기준점)."""

    name = "passthrough"

    def rewrite(self, raw_query: str) -> RewriteResult:
        return RewriteResult(
            raw_query=raw_query,
            queries={b: raw_query for b in BACKENDS},
            intent=raw_query,
        )


# 데모용 아주 작은 '개념 → 전문 용어' 사전.
# 진짜 시스템에서는 언어 모델이 이 역할을 하지만, 여기서는 파이프라인 점검을 위해
# 몇 가지 예시만 규칙으로 넣어 둔다. (한국어/영어 일상어 표현 -> 영어 학술 용어)
_MOCK_DICT: dict[str, list[str]] = {
    "사진": ["image"],
    "이미지": ["image"],
    "설명": ["captioning", "description"],
    "글로 설명": ["captioning"],
    "캡션": ["captioning"],
    "describe": ["captioning"],
    "photo": ["image"],
    "번역": ["machine translation"],
    "말을 글로": ["speech recognition"],
    "음성": ["speech recognition"],
    "추천": ["recommendation"],
    "가짜 뉴스": ["fake news detection", "misinformation"],
    "그래프": ["graph neural network"],
    "강화학습": ["reinforcement learning"],
    "물체 찾기": ["object detection"],
    "물체 인식": ["object detection"],
    "요약": ["summarization"],
    "챗봇": ["dialogue system"],
}


class MockExpanderRewriter:
    """규칙 기반 임시 변환기 (진짜 언어 모델의 대역).

    입력에 위 사전의 표현이 들어 있으면, 대응하는 영어 학술 용어를 검색어에 덧붙인다.
    """

    name = "mock_expander"

    def rewrite(self, raw_query: str) -> RewriteResult:
        low = raw_query.lower()
        terms: list[str] = []
        for key, expansions in _MOCK_DICT.items():
            if key.lower() in low:
                for e in expansions:
                    if e not in terms:
                        terms.append(e)

        # 단어 일치 검색용: 원본 + 전문 용어 나열
        sparse_q = (raw_query + " " + " ".join(terms)).strip()
        # 의미 기반 검색용: 자연문 형태
        dense_q = raw_query if not terms else f"{raw_query} ({', '.join(terms)})"

        return RewriteResult(
            raw_query=raw_query,
            queries={"sparse": sparse_q, "dense": dense_q, "arxiv": sparse_q},
            intent=raw_query,
            academic_terms=terms,
            parse_ok=True,
        )


# 이름 -> 변환기 인스턴스를 만들어주는 등록소.
# 진짜 계층적 변환기가 준비되면 여기에 "hierarchical" 을 추가한다.
def build_rewriter(name: str) -> Rewriter:
    if name == "passthrough":
        return PassthroughRewriter()
    if name == "mock_expander":
        return MockExpanderRewriter()
    raise ValueError(f"알 수 없는 변환기 이름: {name}")
