"""언어 모델에게 시켜서 변환하는 변환기 세 개 - 지금은 모두 비교 대상(기준선) 임.

## 왜 이 셋이 한 파일에 있는가

셋 다 "Ollama 로 Qwen3-4B 를 부르고, JSON 스키마로 출력을 가둔다"는 같은 뼈대를 씀.
다른 것은 무엇을 시키느냐(프롬프트) 뿐이라, 프롬프트까지 같은 파일에 두면 셋을 나란히
놓고 비교하며 고칠 수 있음.

| 변환기 | 무엇을 시키나 | 무엇을 가려내는 비교군인가 |
|---|---|---|
| `HierarchicalRewriter` | 의도 -> 개념 -> 학술용어 -> 검색어, 4단계를 한 번에 | 계층 구조 자체의 기여 |
| `SingleStepRewriter` | "학술 검색어로 바꿔줘" 한 번만 | 계층이 없을 때의 값 |
| `HydeRewriter` | 답이 될 법한 가상 초록을 지어내게 함 | 이미 알려진 다른 방법(HyDE) |

지금 서비스가 쓰는 변환기는 여기 없음. 학습한 모델(`finetuned.py` 의 dpo)이 쓰임.
`HierarchicalRewriter` 는 그 모델의 학습 데이터를 만드는 데도 쓰임
(`training/build_training_data.py`). 그래서 지우지 않고 남겨 둠.

## 프롬프트를 고칠 때

프롬프트는 가장 자주 고치는 부분임. 고치면 아래 `PROMPT_VERSION` 을 올려, 어느 프롬프트로
잰 결과인지 나중에 구분할 수 있게 함.
"""

from __future__ import annotations

from src.rewriter.base import BACKENDS, OllamaClient
from src.schemas import RewriteResult

PROMPT_VERSION = "v1"


# ==========================================================================
# 1. 프롬프트와 출력 형식
# ==========================================================================

# 모델이 반드시 이 형태로만 답하도록 강제하는 JSON 스키마.
# 필드 순서(intent -> concepts -> academic_terms -> queries)가 곧 '단계적 사고' 순서임.
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

# 예시는 few-shot '대화 기록'으로 넣지 않고 시스템 프롬프트 안에 1개만 둠.
# (대화 기록으로 넣으면 특정 입력이 예시 주제로 붕괴해 그대로 복사되는 문제가 있었음 - ISSUE #1)
# 예시 주제는 실제 검색 주제와 겹치지 않도록 speech recognition 하나만 씀.
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


# 기준선 변환기용 프롬프트 + 스키마.
# Qwen3는 추론 모델이라 자유 텍스트로 받으면 사고 과정이 답에 새어나옴(ISSUE #3).
# 계층 변환과 똑같이 JSON 스키마로 출력을 필드에 가둬 이를 막음.
# 단, 기준선의 성격을 지키려고 필드는 결과 1개만 둠(계층 없음: intent/concepts 등 없음).

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
    """한 번에 변환(계층 없이): 그냥 학술 검색어로 바꿔달라고만 시킴.

    계층 변환과 달리 의도, 개념, 용어 단계를 밟지 않음. 오직 최종 영어 검색어 하나만
    query 필드에 담음. '계층 구조의 유무'만 차이나는 공정한 대조군을 만들기 위함.
    """
    return (
        "다음 검색어를, 영어 학술 논문을 찾기 좋은 정확한 영어 검색어로 바꿔라. "
        "결과 검색어만 query 필드에 담아라 (설명·과정 없이).\n\n"
        f"검색어: {raw_query}"
    )


def hyde_prompt(raw_query: str) -> str:
    """HyDE: 이 질문에 답할 법한 가상의 논문 초록을 짧게 지어내게 함."""
    return (
        "다음 검색 의도에 딱 맞는 학술 논문이 있다고 상상하고, 그 논문의 초록을 "
        "영어로 3~4문장 써서 abstract 필드에 담아라. 실제 존재 여부는 상관없다.\n\n"
        f"검색 의도: {raw_query}"
    )


# ==========================================================================
# 2. arXiv 검색어 만들기 (코드로 - 모델에게 맡기지 않음)
# ==========================================================================

def build_arxiv_query(raw_query: str, academic_terms: list[str],
                      max_len: int = 300) -> str:
    """arXiv 실시간 검색용 쿼리를 '코드로' 만듦.

    형태: all:"원본" OR abs:"학술용어1" OR abs:"학술용어2" OR ...  (300자 이내)

    왜 이렇게(실측으로 확정):
    - arXiv는 따옴표 구(句) 검색이라야 정확 매칭이 되고, 따옴표 없이 넓은 단어를 나열하면
      수천 편에 묻혀 원하는 논문이 사라짐. 그래서 각 용어를 따옴표 구로 만듦.
    - 원본을 따옴표로 넣으면, 사용자가 정확한 제목, 구절을 쳤을 때 그게 잡힘
      (예: "Attention is all you need"). 원본이 일상어, 한국어라 안 걸리면 그 절은 무해하게 빔.
    - 학술 용어는 하나만 쓰면 재현율이 떨어짐(좁은 구는 arXiv에 드물어 0건이 되기도 함).
      그래서 변환기가 만든 용어를 전부 OR로 넣어, 어느 하나라도 맞으면 나오게 함.
      각 용어가 따옴표 구라 넓은 단어 나열 같은 희석은 생기지 않음. (특정 논문 콕 집기는
      paper_resolver가 따로 처리하므로 본 검색은 넓게 잡는 게 맞음.)
    - arXiv 문법 구성을 불안정한 LLM에 맡기지 않고 코드가 만들어 재현성을 확보함.
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

    # 300자 제한 내에서 OR로 잇음 (넘치면 뒤 용어는 버림)
    query = ""
    for p in parts:
        candidate = p if not query else f"{query} OR {p}"
        if len(candidate) > max_len:
            break
        query = candidate
    return query


# ==========================================================================
# 3. 변환기 셋
# ==========================================================================

def _to_result(raw_query: str, data: dict) -> RewriteResult:
    """모델이 준 JSON(dict)을 RewriteResult로 바꿈. 필수 항목이 없으면 예외."""
    queries = data["queries"]
    # sparse, dense 검색어는 모델이 준 것을 씀(로컬 검색기용). 비었으면 폴백하도록 예외.
    for b in ("sparse", "dense"):
        if not str(queries.get(b, "")).strip():
            raise ValueError(f"queries.{b} 가 비어 있음")
    academic_terms = list(data.get("academic_terms", []))
    return RewriteResult(
        raw_query=raw_query,
        queries={
            "sparse": str(queries["sparse"]).strip(),
            "dense": str(queries["dense"]).strip(),
            # arXiv 검색어는 LLM 대신 코드로 구성(위 함수) - arXiv 로직에 맞춤
            "arxiv": build_arxiv_query(raw_query, academic_terms),
        },
        intent=str(data.get("intent", "")),
        concepts=list(data.get("concepts", [])),
        academic_terms=academic_terms,
        categories=list(data.get("categories", [])),
        parse_ok=True,
    )


class HierarchicalRewriter:
    """Qwen3-4B로 4단계 계층 변환을 수행함.

    intent(의도) -> concepts(개념) -> academic_terms(전문 용어) -> queries(검색어) 순서로
    한 번에 생성하게 함. JSON 스키마를 강제하므로 출력이 항상 파싱 가능함.

    실패 대비: 파싱, 검증이 실패하면 한 번 재시도하고, 그래도 안 되면 원본 검색어를 그대로
    쓰는 폴백으로 넘어감(parse_ok=False로 기록). 실패율 자체가 모델, 프롬프트 품질 지표임.
    """

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
            except Exception as e:  # 파싱/검증/통신 실패 -> 재시도
                last_err = e
                continue

        # 폴백: 원본 검색어를 그대로 씀 (검색이 아예 멈추지 않도록)
        return RewriteResult(
            raw_query=raw_query,
            queries={b: raw_query for b in BACKENDS},
            intent=f"(변환 실패, 원본 사용) {last_err}",
            parse_ok=False,
        )


class SingleStepRewriter:
    """한 번에 변환: 원본 -> 영어 학술 검색어 (계층 없음)."""

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
            # 세 검색 방식에 같은 결과를 넣음(계층 변환처럼 방식별로 나누지 않음)
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
    """HyDE: 가상 초록을 생성해 의미(dense) 검색어로 씀.

    단어 일치(sparse), arxiv 검색어에는 가상 초록이 너무 길어 부적합하므로 원본을 씀.
    HyDE의 효과는 주로 의미 기반 검색에서 나타남.
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
