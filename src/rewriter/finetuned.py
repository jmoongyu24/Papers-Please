"""파인튜닝(LoRA)한 쿼리 변환기를 평가와 서비스에 꽂는 어댑터.

학습 결과물은 transformers 형식이라 Ollama 가 바로 못 읽음. 그래서 여기서는
transformers 로 기본 모델을 올리고 그 위에 어댑터를 얹어 직접 생성함.

다른 변환기와 같은 인터페이스를 따르므로 `--rewriter dpo` 로 바꿔 끼우면 학습 전후를
같은 기준으로 비교할 수 있음. 학습 때 쓴 지시문과 대화 형식을 그대로 써야 함.
"""

from __future__ import annotations

import json
import re

from src.rewriter.base import BACKENDS
from src.schemas import RewriteResult

# training/train.py 의 INSTRUCTION 과 반드시 같아야 함
INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER = "models/query-translator-dpo"


class FinetunedRewriter:
    """파인튜닝한 모델로 arXiv 검색어를 생성함."""

    name = "dpo"

    def __init__(self, base_model: str = DEFAULT_BASE,
                 adapter_path: str = DEFAULT_ADAPTER,
                 max_new_tokens: int = 150):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _generate(self, question: str) -> str:
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # -- 같은 모델을 추천 이유 생성에도 빌려줌 -------------------------
    #
    # Qwen3-4B 를 검색어 변환과 추천 이유 생성 두 곳에서 씀. 따로 올리면 같은 모델이
    # 두 개(8.64GB + 3.54GB) 메모리에 있게 되고, 자리가 모자라면 추천 쪽 모델이 오류
    # 없이 CPU 로 밀려나 한 번에 229초가 걸림. 그래서 이미 올라온 이 모델을 빌려줌.
    # 인터페이스를 OllamaClient 와 맞춰 두어 추천 쪽 코드를 안 고치고 갈아 끼움.

    _JSON_RE = re.compile(r"\{.*\}", re.S)

    def generate_json(self, prompt: str, schema: dict | None = None,
                      system: str | None = None, temperature: float = 0.0,
                      max_tokens: int = 2000) -> dict:
        """JSON 을 받아 냄. `OllamaClient.generate_json` 과 같은 인터페이스.

        Ollama 와 달리 transformers 에는 형식을 강제하는 장치가 없어서, JSON 만 내라고
        지시하고 출력에서 가장 바깥 중괄호 덩어리만 뽑아 파싱함. 실패하면 한 번 더 시도함.

        최상위 키 이름을 반드시 못박아야 함. 안 그러면 4B 모델이 내용은 제대로 채우면서
        키 이름을 마음대로 바꿔서(`recommendations` 를 `relevance` 로), 부르는 쪽이 빈
        목록으로 읽음. 스키마 원문을 통째로 붙이면 오히려 빈 껍데기를 흉내 내므로,
        필수 키 이름만 적어 줌.
        """
        required = list((schema or {}).get("required") or [])
        guide = ""
        if schema:
            keys = ", ".join(f'"{k}"' for k in required) or "스키마의 키"
            item = (schema.get("properties", {}).get(required[0], {})
                    if required else {}).get("items", {})
            item_keys = ", ".join(f'"{k}"' for k in (item.get("required") or []))
            guide = (f"\n\nJSON **하나만** 출력하라. 설명이나 머리말을 붙이지 마라.\n"
                     f"- 최상위 키는 정확히 {keys} 여야 한다. 다른 이름을 쓰지 마라.\n")
            if item_keys:
                guide += f"- 목록의 각 항목은 정확히 {item_keys} 키를 가져야 한다.\n"
            guide += "- 빈 목록으로 답하지 말고 위 논문들을 실제로 판단해 채워라."
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt + guide})

        for attempt in range(2):
            text = self._chat(messages, max_new_tokens=max_tokens,
                              temperature=temperature if attempt == 0 else 0.0)
            # 코드 블록으로 감싸 나오는 일이 잦음
            text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
            m = self._JSON_RE.search(text)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    missing = [k for k in required if k not in data]
                    if not missing:
                        return data
                    # 키 이름만 틀린 경우가 대부분이라 무엇이 빠졌는지 짚어 다시 시킴
                    hint = f"직전 출력에 {', '.join(missing)} 키가 없었다. 그 이름을 그대로 써라."
                    messages = messages[:-1] + [{"role": "user",
                                                 "content": prompt + guide + f"\n\n({hint})"}]
                    continue
            messages = messages[:-1] + [{
                "role": "user",
                "content": prompt + guide + "\n\n(직전 출력이 올바른 JSON 이 아니었다. "
                                            "JSON 만 다시 출력하라.)"}]
        raise ValueError("JSON 파싱 실패 또는 필수 키 누락")

    def _chat(self, messages: list[dict], max_new_tokens: int,
              temperature: float = 0.0) -> str:
        """어댑터를 끄고 기본 모델로 답함.

        어댑터는 검색어 변환 한 가지만 하도록 학습됐음. 다른 일에 그대로 쓰면 엉뚱한
        짧은 검색어를 뱉을 수 있어, 가중치는 같은 것을 쓰되 어댑터만 잠깐 끔.
        """
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kw = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            gen_kw.update(do_sample=True, temperature=temperature)
        else:
            gen_kw.update(do_sample=False)

        with self._torch.no_grad(), self.model.disable_adapter():
            out = self.model.generate(**enc, **gen_kw)
        gen = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def rewrite(self, raw_query: str) -> RewriteResult:
        try:
            query = self._generate(raw_query).splitlines()[0].strip()
            if not query:
                raise ValueError("빈 출력")
            # 학습 모델은 arXiv 문법 문자열 하나만 냄. 세 필드에 같은 값을 넣는 것은
            # 인터페이스를 맞추기 위한 것임. dense 필드에 든 값도 문법 문자열이라 의미
            # 검색에 넣으면 불리하므로, 서비스도 평가도 로컬 채널에는 원본 질문을 넣음.
            return RewriteResult(
                raw_query=raw_query,
                queries={b: query for b in BACKENDS},
                intent=raw_query,
                parse_ok=True,
            )
        except Exception as e:
            return RewriteResult(
                raw_query=raw_query,
                queries={b: raw_query for b in BACKENDS},
                intent=f"(변환 실패, 원본 사용) {e}",
                parse_ok=False,
            )
