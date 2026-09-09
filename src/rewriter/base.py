"""쿼리 변환기의 공통 규약, 변환하지 않는 기준점, 로컬 언어 모델 호출부.

    Rewriter            모든 변환기가 따르는 `rewrite(raw_query) -> RewriteResult`
    PassthroughRewriter 변환을 하지 않는 기준점
    build_rewriter      이름으로 변환기를 만들어 주는 함수
    OllamaClient        로컬 언어 모델(Qwen3-4B)을 부르는 호출부

실제 변환기들은 `baselines.py`(번역 · 가상 초록 · 계층 · 한번에)와
`finetuned.py`(학습한 모델)에 있음.
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
    """변환하지 않고 원본 검색어를 그대로 씀. 변환 전 기준점이라 모든 평가에 넣음."""

    name = "passthrough"

    def rewrite(self, raw_query: str) -> RewriteResult:
        return RewriteResult(
            raw_query=raw_query,
            queries={b: raw_query for b in BACKENDS},
            intent=raw_query,
        )


def build_rewriter(name: str) -> Rewriter:
    """이름으로 변환기를 만듦. 무거운 패키지는 그 이름을 부를 때만 가져옴."""
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
    if name == "translate":
        # 한국어를 영어로 옮기기만 함. 로컬 의미 검색 채널용
        from src.rewriter.baselines import TranslateRewriter
        return TranslateRewriter()
    if name == "service":
        # app.py 가 실제로 쓰는 조합: 번역문과 가상 초록을 한 번에 만듦
        from src.rewriter.baselines import ServiceRewriter
        return ServiceRewriter()
    if name == "dpo":
        # 지도 파인튜닝 위에 선호 학습까지 얹은 모델. arXiv 채널이 쓰는 것
        from src.rewriter.finetuned import FinetunedRewriter
        return FinetunedRewriter()
    raise ValueError(
        f"알 수 없는 변환기 이름: {name} "
        f"(쓸 수 있는 것: passthrough, translate, service, hierarchical, single_step, "
        f"hyde, dpo)")


# ==========================================================================
# 2. 로컬 언어 모델 호출 (Ollama)
# ==========================================================================

class OllamaClient:
    """로컬 Ollama 서버에 붙어 모델을 호출함.

    generate_json 은 JSON 스키마를 강제해 모델이 형식을 벗어난 답을 못 내게 하고,
    generate_text 는 자유 형식 텍스트를 받음.
    """

    def __init__(self, model: str = config.REWRITER_MODEL,
                 host: Optional[str] = None, think: bool = False):
        import ollama

        self.model = model
        # Qwen3 의 내부 사고 모드를 끔. JSON 필드 순서로 이미 단계를 유도하므로
        # 끄는 편이 빠르고 출력이 깔끔함
        self.think = think
        self._client = ollama.Client(host=host) if host else ollama.Client()

    @staticmethod
    def _as_messages(prompt_or_messages) -> list[dict]:
        """문자열이면 user 메시지 하나로, 목록이면 그대로 씀."""
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
            format=fmt,                       # JSON 스키마를 주면 그 형식을 강제함
            think=self.think,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return resp["message"]["content"]

    def generate_json(self, prompt_or_messages, schema: dict,
                      system: Optional[str] = None,
                      temperature: float = 0.0,
                      max_tokens: int = 512) -> dict:
        """JSON 스키마를 강제해 답을 받아 파싱함. 실패하면 예외를 던짐."""
        raw = self._chat(prompt_or_messages, system, schema, temperature, max_tokens)
        return json.loads(raw)

    def generate_text(self, prompt_or_messages, system: Optional[str] = None,
                      temperature: float = 0.7, max_tokens: int = 256) -> str:
        """자유 형식 텍스트를 생성함."""
        return self._chat(prompt_or_messages, system, None, temperature, max_tokens).strip()
