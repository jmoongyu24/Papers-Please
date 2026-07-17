"""쿼리 변환에 쓰는 안내문(프롬프트), 예시, 출력 형식(JSON 스키마).

로직(hierarchical.py)과 분리해 둔다. 프롬프트는 가장 자주 고칠 부분이라, 데이터로 빼두면
버전 관리와 실험 비교가 쉽다. 프롬프트를 바꾸면 아래 PROMPT_VERSION을 올린다.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

# 모델이 반드시 이 형태로만 답하도록 강제하는 JSON 스키마.
# 필드 순서(intent → concepts → academic_terms → queries)가 곧 '단계적 사고' 순서다.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "concepts": {"type": "array", "items": {"type": "string"}},
        "academic_terms": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
        "queries": {
            "type": "object",
            "properties": {
                "sparse": {"type": "string"},
                "dense": {"type": "string"},
                "arxiv": {"type": "string"},
            },
            "required": ["sparse", "dense", "arxiv"],
        },
    },
    "required": ["intent", "concepts", "academic_terms", "queries"],
}

# 예시는 few-shot '대화 기록'으로 넣지 않고 시스템 프롬프트 안에 1개만 둔다.
# (대화 기록으로 넣으면 특정 입력이 예시 주제로 붕괴해 그대로 복사되는 문제가 있었다.)
# 예시 주제는 실제 검색 주제와 겹치지 않도록 speech recognition 하나만 쓴다.
_EXAMPLE = (
    '예시 — 입력이 "음성을 텍스트로 바꾸는 기술"이라면 출력은:\n'
    '{"intent":"음성 신호를 텍스트로 변환하는 기술","concepts":["음성 인식","음향 모델",'
    '"시퀀스 변환"],"academic_terms":["speech recognition","automatic speech recognition",'
    '"acoustic model"],"categories":["cs.CL"],"queries":{"sparse":"speech recognition '
    'automatic speech recognition acoustic model","dense":"automatic speech recognition '
    'systems that convert speech audio into text","arxiv":"abs:\\"speech recognition\\" '
    'AND cat:cs.CL"}}'
)

SYSTEM = (
    "너는 학술 논문 검색을 돕는 전문가다. 사용자는 그 분야를 잘 모르는 비전문가라, "
    "일상어를 쓰거나 부정확한 전문 용어를 쓴다. 너의 임무는 사용자가 방금 입력한 검색어를 "
    "단계적으로 다듬어, 실제 영어 논문이 쓰는 정확한 학술 용어와 검색어로 바꿔 JSON으로 "
    "출력하는 것이다.\n\n"
    "각 필드(이 순서로 생각한다):\n"
    "- intent: 사용자가 진짜 찾고 싶은 것 한 문장 (입력 언어로 써도 됨).\n"
    "- concepts: 핵심 개념 2~4개.\n"
    "- academic_terms: 그 주제의 정확한 영어 학술 용어 3~5개 (반드시 영어).\n"
    "- categories: 관련 arXiv 분야 (cs.CL, cs.CV, cs.LG, cs.IR, cs.AI, stat.ML 중에서).\n"
    "- queries.sparse: 위 영어 용어를 공백으로 나열 (단어 일치 검색용).\n"
    "- queries.dense: 찾는 내용을 설명하는 영어 문장 하나 (의미 검색용).\n"
    "- queries.arxiv: arXiv 문법 검색어 (abs:/ti:/cat: 와 AND/OR, 큰따옴표 구). 300자 이내.\n\n"
    f"{_EXAMPLE}\n\n"
    "중요: 반드시 사용자가 방금 준 검색어의 내용을 분석해서 답하라. 위 예시 주제(음성 인식)나 "
    "이 설명 문구를 그대로 베끼지 마라. academic_terms 와 queries 는 항상 영어로 쓴다."
)


def build_messages(raw_query: str) -> list[dict]:
    """사용자 검색어 하나만 담은 메시지 목록 (예시는 시스템 프롬프트에 있음)."""
    return [{"role": "user", "content": f"검색어: {raw_query}"}]


# ── 비교용(baseline) 변환기가 쓰는 프롬프트 + 출력 스키마 ──────────────────
# Qwen3는 추론 모델이라 자유 텍스트로 받으면 사고 과정이 답에 새어나온다(ISSUE #3).
# 계층 변환과 똑같이 JSON 스키마로 출력을 필드에 가둬 이를 막는다.
# 단, 기준선의 성격을 지키려고 필드는 결과 1개만 둔다(계층 없음: intent/concepts 등 없음).

SINGLE_STEP_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
HYDE_SCHEMA = {
    "type": "object",
    "properties": {"abstract": {"type": "string"}},
    "required": ["abstract"],
}


def single_step_prompt(raw_query: str) -> str:
    """한 번에 변환(계층 없이): 그냥 학술 검색어로 바꿔달라고만 시킨다.

    계층 변환과 달리 의도·개념·용어 단계를 밟지 않는다. 오직 최종 영어 검색어 하나만
    query 필드에 담는다. '계층 구조의 유무'만 차이나는 공정한 대조군을 만들기 위함.
    """
    return (
        "다음 검색어를, 영어 학술 논문을 찾기 좋은 정확한 영어 검색어로 바꿔라. "
        "결과 검색어만 query 필드에 담아라 (설명·과정 없이).\n\n"
        f"검색어: {raw_query}"
    )


def hyde_prompt(raw_query: str) -> str:
    """HyDE: 이 질문에 답할 법한 가상의 논문 초록을 짧게 지어내게 한다."""
    return (
        "다음 검색 의도에 딱 맞는 학술 논문이 있다고 상상하고, 그 논문의 초록을 "
        "영어로 3~4문장 써서 abstract 필드에 담아라. 실제 존재 여부는 상관없다.\n\n"
        f"검색 의도: {raw_query}"
    )
