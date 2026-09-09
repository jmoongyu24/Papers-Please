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
    if name in ("dpo", "dpo-q8", "dpo-f16"):
        # 지도 파인튜닝 위에 선호 학습까지 얹은 모델. 셋 다 같은 가중치이고 정밀도만 다름.
        # 서비스는 4비트(dpo)를 씀. 나머지 둘은 정밀도를 견주는 평가용임
        from src.rewriter.finetuned import OllamaFinetunedRewriter
        model = {"dpo": config.ARXIV_REWRITER_MODEL,
                 "dpo-q8": "papers-rewriter-q8",
                 "dpo-f16": "papers-rewriter-f16"}[name]
        return OllamaFinetunedRewriter(model=model)
    if name == "dpo-hf":
        # 같은 모델을 transformers 로 올림. 4비트 변환 전후를 비교할 때만 씀
        from src.rewriter.finetuned import FinetunedRewriter
        return FinetunedRewriter()
    raise ValueError(
        f"알 수 없는 변환기 이름: {name} "
        f"(쓸 수 있는 것: passthrough, translate, service, hierarchical, single_step, "
        f"hyde, dpo, dpo-q8, dpo-f16, dpo-hf)")


# ==========================================================================
# 2. 로컬 언어 모델 호출 (Ollama)
# ==========================================================================

class OllamaClient:
    """로컬 Ollama 서버에 붙어 모델을 호출함.

    generate_json 은 JSON 스키마를 강제해 모델이 형식을 벗어난 답을 못 내게 하고,
    generate_text 는 자유 형식 텍스트를 받음.

    `keep_alive` 는 호출이 끝난 뒤 모델을 그래픽 메모리에 얼마나 둘지임. Ollama 기본값은
    5분이고, 0 을 주면 호출이 끝나는 즉시 내림. 언어 모델 하나가 3.25GB 라 재정렬 모델과
    자리를 나눠 쓰려면 다 쓴 시점에 내려야 함. 다시 올리며 호출하는 데 2.4초 걸림.
    """

    def __init__(self, model: str = config.REWRITER_MODEL,
                 host: Optional[str] = None, think: bool = False,
                 keep_alive: Any = None):
        import ollama

        self.model = model
        # Qwen3 의 내부 사고 모드를 끔. JSON 필드 순서로 이미 단계를 유도하므로
        # 끄는 편이 빠르고 출력이 깔끔함
        self.think = think
        self.keep_alive = keep_alive
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def unload(self) -> None:
        """이 클라이언트가 쓰는 모델을 그래픽 메모리에서 내림.

        내리는 전용 명령이 없어서, 아무것도 생성하지 않는 호출에 keep_alive=0 을 붙여
        보냄. 모델이 안 올라와 있으면 아무 일도 일어나지 않음. 서버가 죽어 있는 등의
        오류는 삼킴 - 자리를 비우려던 것뿐이라 검색을 멈출 이유가 없음.
        """
        try:
            self._client.chat(model=self.model, messages=[{"role": "user", "content": ""}],
                              think=False, options={"num_predict": 1}, keep_alive=0)
        except Exception:
            pass

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
        kw = {} if self.keep_alive is None else {"keep_alive": self.keep_alive}
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            format=fmt,                       # JSON 스키마를 주면 그 형식을 강제함
            think=self.think,
            options={"temperature": temperature, "num_predict": max_tokens},
            **kw,
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
