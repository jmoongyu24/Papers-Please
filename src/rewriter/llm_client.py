"""Ollama로 로컬 언어 모델(Qwen3-4B)을 호출하는 부분.

이 파일은 '모델과 대화하는 법'만 안다. 쿼리 변환이 무슨 단계로 이뤄지는지는 모른다.
그래서 나중에 모델을 바꾸거나(qwen3:4b → 1.7b), 서빙 방식을 바꾸거나, 재시도·지연시간
기록 같은 공통 처리를 넣을 때 이 파일만 고치면 된다.

핵심 기능은 두 가지다.
- generate_json: JSON 스키마를 강제해서, 모델이 스키마를 벗어난 답을 아예 못 내게 한다.
  (예전 계획의 "특정 줄을 정규식으로 뽑기"는 형식이 조금만 어긋나도 깨졌는데, 그 문제를
   원천 차단한다.)
- generate_text: 자유 형식 텍스트 답 (HyDE 같은 단순 생성용).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from src import config


class OllamaClient:
    """로컬 Ollama 서버에 붙어 모델을 호출한다."""

    def __init__(self, model: str = config.REWRITER_MODEL,
                 host: Optional[str] = None, think: bool = False):
        import ollama

        self.model = model
        # think=False: Qwen3의 내부 '사고 모드'를 끈다. 우리는 4단계 구조(JSON 필드 순서)로
        # 이미 단계적 사고를 유도하므로, 내부 사고를 끄면 더 빠르고 출력이 깔끔하다.
        self.think = think
        self._client = ollama.Client(host=host) if host else ollama.Client()

    @staticmethod
    def _as_messages(prompt_or_messages) -> list[dict]:
        """문자열이면 user 메시지 하나로, 리스트면 그대로 쓴다.

        few-shot 예시를 넣을 때는 [user예시, assistant예시, ..., 실제 user] 형태의
        '대화 기록'을 넘긴다. 예시를 프롬프트 텍스트에 이어붙이면 모델이 예시를 그대로
        복사하는 문제가 있어, 이렇게 실제 주고받은 대화처럼 분리해서 준다.
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
        # ollama 응답은 dict 유사 객체. 내용은 message.content 에 들어 있다.
        return resp["message"]["content"]

    def generate_json(self, prompt_or_messages, schema: dict,
                      system: Optional[str] = None,
                      temperature: float = 0.0,
                      max_tokens: int = 512) -> dict:
        """JSON 스키마를 강제해 답을 받아 파싱한다. 파싱 실패 시 예외를 던진다."""
        raw = self._chat(prompt_or_messages, system, schema, temperature, max_tokens)
        return json.loads(raw)

    def generate_text(self, prompt_or_messages, system: Optional[str] = None,
                      temperature: float = 0.7, max_tokens: int = 256) -> str:
        """자유 형식 텍스트를 생성한다 (스키마 강제 없음)."""
        return self._chat(prompt_or_messages, system, None, temperature, max_tokens).strip()
