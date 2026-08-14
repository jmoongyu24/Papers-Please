"""쿼리 변환기의 공통 약속(인터페이스), 기준점 변환기, 그리고 로컬 언어 모델 호출부.

## 이 파일에 무엇이 있고, 왜 한군데 있는가

    1. Rewriter 규약        - 모든 변환기가 따르는 `rewrite(raw_query) -> RewriteResult`
    2. PassthroughRewriter  - 변환을 아예 안 하는 기준점
    3. build_rewriter       - 이름으로 변환기를 만들어 주는 등록소
    4. OllamaClient         - 로컬 언어 모델(Qwen3-4B)을 부르는 창구

4번(모델 호출)이 여기 있는 이유: 이것은 '모델과 대화하는 법'만 아는 얇은 층이라 변환 로직과
섞일 여지가 없고, 어차피 변환기를 만드는 모든 경로가 이것을 거침. 파일을 나눠 두면
"변환기 하나 고치려는데 파일 세 개를 연다"가 됨.

모델을 바꾸거나(qwen3:4b -> 1.7b), 서빙 방식을 바꾸거나, 재시도, 지연시간 기록 같은 공통 처리를
넣을 때는 `OllamaClient` 만 고치면 됨.

## 실제 변환기들이 있는 곳

    baselines.py  - hierarchical(계층 변환), single_step, hyde  <- 비교 대상
    finetuned.py  - 학습한 모델(sft/dpo)                        <- 지금 서비스가 쓰는 것
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol, runtime_checkable

from src import config
from src.schemas import RewriteResult

# 검색 방식 이름 (schemas.RewriteResult.queries 의 키)
BACKENDS = ("sparse", "dense", "arxiv")


# ==========================================================================
# 1. 변환기 규약과 기준점
# ==========================================================================

@runtime_checkable
class Rewriter(Protocol):
    name: str

    def rewrite(self, raw_query: str) -> RewriteResult:
        ...


class PassthroughRewriter:
    """변환하지 않고 원본 검색어를 그대로 씀 (변환 전 기준점).

    이 프로젝트의 핵심 주장은 "질문을 변환하면 검색이 좋아진다"임. 그 주장을 검증하려면
    변환하지 않았을 때의 값이 반드시 있어야 하므로, 모든 평가에 이것을 포함함.
    """

    name = "passthrough"

    def rewrite(self, raw_query: str) -> RewriteResult:
        return RewriteResult(
            raw_query=raw_query,
            queries={b: raw_query for b in BACKENDS},
            intent=raw_query,
        )


def build_rewriter(name: str) -> Rewriter:
    """이름으로 변환기를 만듦.

    언어 모델을 쓰는 변환기는 무거운 패키지(ollama, transformers, peft)를 필요로 하므로,
    그 이름을 부를 때만 늦게(lazy) 가져옴. passthrough 는 의존성이 없음.
    """
    if name == "passthrough":
        return PassthroughRewriter()
    if name == "hierarchical":
        from src.rewriter.baselines import HierarchicalRewriter
        return HierarchicalRewriter()
    if name == "single_step":
        from src.rewriter.baselines import SingleStepRewriter
        return SingleStepRewriter()
    if name == "hyde":
        from src.rewriter.baselines import HydeRewriter
        return HydeRewriter()
    if name == "finetuned":
        # SFT(지도 미세조정)로 학습한 모델. 학습 전/후 비교용
        from src.rewriter.finetuned import FinetunedRewriter
        return FinetunedRewriter()
    if name == "dpo":
        # SFT 위에 DPO(선호 학습)까지 얹은 모델 - 서비스가 쓰는 것
        from src.rewriter.finetuned import FinetunedRewriter
        return FinetunedRewriter(adapter_path="models/qwen3-4b-query-dpo")
    raise ValueError(
        f"알 수 없는 변환기 이름: {name} "
        f"(쓸 수 있는 것: passthrough, hierarchical, single_step, hyde, finetuned, dpo)")


# ==========================================================================
# 2. 로컬 언어 모델 호출 (Ollama)
# ==========================================================================

class OllamaClient:
    """로컬 Ollama 서버에 붙어 모델을 호출함.

    핵심 기능은 두 가짐.
    - generate_json: JSON 스키마를 강제해서, 모델이 스키마를 벗어난 답을 아예 못 내게 함.
      ("특정 줄을 정규식으로 뽑기"는 형식이 조금만 어긋나도 깨지는데, 그 문제를 원천 차단함.)
    - generate_text: 자유 형식 텍스트 답 (HyDE 같은 단순 생성용).
    """

    def __init__(self, model: str = config.REWRITER_MODEL,
                 host: Optional[str] = None, think: bool = False):
        import ollama

        self.model = model
        # think=False: Qwen3의 내부 '사고 모드'를 끔. 우리는 4단계 구조(JSON 필드 순서)로
        # 이미 단계적 사고를 유도하므로, 내부 사고를 끄면 더 빠르고 출력이 깔끔함.
        self.think = think
        self._client = ollama.Client(host=host) if host else ollama.Client()

    @staticmethod
    def _as_messages(prompt_or_messages) -> list[dict]:
        """문자열이면 user 메시지 하나로, 리스트면 그대로 씀.

        few-shot 예시를 넣을 때는 [user예시, assistant예시, ..., 실제 user] 형태의
        '대화 기록'을 넘김. 예시를 프롬프트 텍스트에 이어붙이면 모델이 예시를 그대로
        복사하는 문제가 있어, 이렇게 실제 주고받은 대화처럼 분리해서 줌.
        """
        if isinstance(prompt_or_messages, str):
            return [{"role": "user", "content": prompt_or_messages}]
        return list(prompt_or_messages)

    def _chat(self, prompt_or_messages, system: Optional[str], fmt: Any,
              temperature: float, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self._as_messages(prompt_or_messages))
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            format=fmt,                       # dict(JSON 스키마)이면 그 형식을 강제
            think=self.think,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        # ollama 응답은 dict 유사 객체. 내용은 message.content 에 들어 있음.
        return resp["message"]["content"]

    def generate_json(self, prompt_or_messages, schema: dict,
                      system: Optional[str] = None,
                      temperature: float = 0.0,
                      max_tokens: int = 512) -> dict:
        """JSON 스키마를 강제해 답을 받아 파싱함. 파싱 실패 시 예외를 던짐."""
        raw = self._chat(prompt_or_messages, system, schema, temperature, max_tokens)
        return json.loads(raw)

    def generate_text(self, prompt_or_messages, system: Optional[str] = None,
                      temperature: float = 0.7, max_tokens: int = 256) -> str:
        """자유 형식 텍스트를 생성함 (스키마 강제 없음)."""
        return self._chat(prompt_or_messages, system, None, temperature, max_tokens).strip()
