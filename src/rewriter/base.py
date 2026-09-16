"""
쿼리 변환기 코드
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol

from src import config
from src.schemas import RewriteResult

BACKENDS = ("dense", "arxiv")

class Rewriter(Protocol):
    name: str

    def rewrite(self, raw_query: str) -> RewriteResult:
        ...


class PassthroughRewriter:
    """변환하지 않고 원본 검색어를 그대로 씀"""

    name = "passthrough"

    def unload(self) -> None:
        """`GpuPool.release`가 부름. 모델을 안 쓰므로 내릴 것이 없음"""

    def rewrite(self, raw_query: str) -> RewriteResult:
        return RewriteResult(
            raw_query=raw_query,
            queries={b: raw_query for b in BACKENDS},
            intent=raw_query,
        )


def build_rewriter(name: str) -> Rewriter:
    """이름으로 변환기를 만듦"""
    if name == "passthrough":
        return PassthroughRewriter()
    if name == "hierarchical":
        from src.rewriter.baselines import HierarchicalRewriter
        return HierarchicalRewriter()
    if name == "hyde":
        from src.rewriter.baselines import HydeRewriter
        return HydeRewriter()
    if name == "translate":
        # 한국어를 영어로 번역하기만 함. 로컬 의미 검색 채널용
        from src.rewriter.baselines import TranslateRewriter
        return TranslateRewriter()
    if name == "service":
        # app.py가 쓰는 조합: 번역문과 가상 초록을 한 번에 만듦
        from src.rewriter.baselines import ServiceRewriter
        return ServiceRewriter()
    if name in ("dpo", "dpo-q8", "dpo-f16"):
        # 지도 파인튜닝 위에 선호 학습까지 얹은 모델. 셋 다 같은 가중치이고 정밀도만 다름.
        from src.rewriter.finetuned import OllamaFinetunedRewriter
        model = {"dpo": config.ARXIV_REWRITER_MODEL,
                 "dpo-q8": "papers-rewriter-q8",
                 "dpo-f16": "papers-rewriter-f16"}[name]
        return OllamaFinetunedRewriter(model=model)
    if name == "dpo-hf":
        # 같은 모델을 transformers로 올림. 4비트 변환 전후를 비교할 때만 씀
        from src.rewriter.finetuned import FinetunedRewriter
        return FinetunedRewriter()
    raise ValueError(
        f"알 수 없는 변환기 이름: {name} "
        f"(쓸 수 있는 것: passthrough, translate, service, hierarchical, hyde, "
        f"dpo, dpo-q8, dpo-f16, dpo-hf)")

class OllamaClient:
    """로컬 Ollama 서버에 붙어 모델을 호출함"""

    def __init__(self, model: str = config.REWRITER_MODEL,
                 host: Optional[str] = None, think: bool = False,
                 keep_alive: Any = None):
        import ollama

        self.model = model
        self.think = think
        self.keep_alive = keep_alive
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def unload(self) -> None:
        """이 클라이언트가 쓰는 모델을 그래픽 메모리에서 내림"""
        try:
            self._client.chat(model=self.model, messages=[{"role": "user", "content": ""}],
                              think=False, options={"num_predict": 1}, keep_alive=0)
        except Exception:
            pass

    @staticmethod
    def _as_messages(prompt_or_messages) -> list[dict]:
        """문자열이면 user 메시지 하나로, 목록이면 그대로 씀"""
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
            format=fmt,
            think=self.think,
            options={"temperature": temperature, "num_predict": max_tokens},
            **kw,
        )
        return resp["message"]["content"]

    def generate_json(self, prompt_or_messages, schema: dict,
                      system: Optional[str] = None,
                      temperature: float = 0.0,
                      max_tokens: int = 512) -> dict:
        """JSON 스키마로 답을 받아 파싱함. 실패하면 예외 던짐"""
        raw = self._chat(prompt_or_messages, system, schema, temperature, max_tokens)
        return json.loads(raw)

    def generate_text(self, prompt_or_messages, system: Optional[str] = None,
                      temperature: float = 0.7, max_tokens: int = 256) -> str:
        return self._chat(prompt_or_messages, system, None, temperature, max_tokens).strip()
