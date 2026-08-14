"""미세조정(LoRA)된 쿼리 변환기 — 학습 결과를 평가·서비스에 꽂기 위한 어댑터.

기존 계층 변환기(hierarchical)는 Ollama로 Qwen3-4B를 부르지만, 학습 결과물(LoRA 어댑터)은
transformers 형식이라 Ollama가 바로 못 읽는다. 그래서 이 변환기는 transformers로 기본 모델을
올리고 그 위에 LoRA 어댑터를 얹어 직접 생성한다.

다른 변환기와 **같은 인터페이스**(`rewrite(질문) -> RewriteResult`)를 따르므로,
평가 하네스에서 `--rewriter finetuned` 로 바꿔 끼우기만 하면 학습 전/후를 같은 자로 비교할 수 있다.

학습 때 쓴 것과 **똑같은 지시문·대화 형식**을 써야 한다(형식이 다르면 학습 효과가 사라짐).
"""

from __future__ import annotations

import json
import re

from src.rewriter.base import BACKENDS
from src.schemas import RewriteResult

# training/train_lora.py 의 INSTRUCTION 과 반드시 동일해야 한다
INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER = "models/qwen3-4b-query-lora/checkpoint-54"


class FinetunedRewriter:
    """LoRA로 미세조정된 모델로 arXiv 검색 쿼리를 생성한다."""

    name = "finetuned"

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

    # ── 같은 모델을 다른 용도로도 빌려준다 (VRAM 절약) ──────────────────
    #
    # 왜 이 기능이 여기 있는가:
    # 이 프로젝트는 Qwen3-4B 를 두 곳에서 쓴다. 검색어 변환(여기, transformers + LoRA)과
    # 추천 이유 생성(recommend_agent, Ollama)이다. 그런데 둘을 따로 올리면 같은 모델이
    # **두 벌** 메모리에 있게 된다 - 8.64GB + 3.54GB = 12.2GB.
    #
    # GPU 가 16GB 라 임베더와 재정렬까지 올리면 자리가 모자라고, 그러면 추천 쪽 모델이
    # **오류 없이 조용히 CPU 로 밀려난다.** 그 결과 추천 한 번이 10.5초에서 229.6초가
    # 됐다(실측). 응답 시간의 91% 가 이 한 단계였다.
    #
    # 그래서 이미 올라와 있는 이 모델을 추천에도 빌려준다. 인터페이스는 OllamaClient 의
    # generate_json 과 맞춰 두어, 추천 쪽 코드를 고치지 않고 갈아 끼울 수 있게 했다.

    _JSON_RE = re.compile(r"\{.*\}", re.S)

    def generate_json(self, prompt: str, schema: dict | None = None,
                      system: str | None = None, temperature: float = 0.0,
                      max_tokens: int = 2000) -> dict:
        """JSON 을 받아 낸다. OllamaClient.generate_json 과 같은 인터페이스.

        Ollama 는 문법을 강제해 스키마 밖 출력을 못 내게 막지만, transformers 에는 그런
        장치가 없다. 대신 두 가지로 막는다.

        1) JSON 만 내라고 짧게 지시한다.
        2) 출력에서 가장 바깥 중괄호 덩어리만 뽑아 파싱하고, 실패하면 한 번 더 시도한다.

        **최상위 키 이름을 반드시 못박아야 한다 (실측으로 확인한 함정):**
        지시 없이 두면 4B 모델이 내용은 제대로 채우면서 **키 이름을 자기 마음대로 바꾼다.**
        실제로 `recommendations` 대신 `relevance` 라는 키로 답했고, 항목 안에서도
        `relevance` 를 `level` 로 바꿔 썼다. 내용은 멀쩡한데 이름이 달라서 호출하는 쪽이
        빈 목록으로 읽었다. Ollama 는 문법 강제로 이걸 원천 차단하지만 여기서는 못 한다.
        그래서 스키마에서 필수 키를 뽑아 그대로 적어 주고, 빠지면 다시 시도한다.

        스키마 원문(`json.dumps(schema)`)을 통째로 붙이는 것은 오히려 해로웠다. 모델이
        스키마를 보고 내용을 채우는 대신 빈 껍데기를 흉내 내는 경우가 있었다. 필요한 것은
        구조 설명이 아니라 **키 이름**이다.

        기본 모델(Qwen3-4B-Instruct-2507)은 사고 과정을 따로 뱉지 않는 지시 모델이라
        ISSUE 3 의 사고 유출 문제는 해당하지 않는다. 다만 마크다운 코드블록으로 감싸는
        일이 잦아 그것도 벗겨 낸다.
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
            # 마크다운 코드블록으로 감싸 나오는 일이 잦다
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
                    # 키 이름만 틀린 경우가 대부분이라, 무엇이 빠졌는지 짚어 다시 시킨다
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
        """LoRA 어댑터를 **끄고** 기본 모델로 답한다.

        어댑터는 '검색어 변환' 한 가지 일만 하도록 학습됐다. 추천 이유를 쓰는 것 같은
        다른 일에 그대로 쓰면 학습된 편향이 끼어들어 엉뚱한 짧은 검색어를 뱉을 수 있다.
        가중치는 같은 것을 쓰되 어댑터만 잠깐 꺼서, 메모리를 더 쓰지 않고 기본 모델의
        일반 능력을 그대로 얻는다.
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
            # 학습 모델은 arXiv 쿼리 문자열 하나를 낸다. 로컬 검색용 필드에는 같은 값을 넣어
            # 인터페이스를 맞춘다(평가는 arxiv 필드만 사용).
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
