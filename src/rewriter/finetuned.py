"""미세조정(LoRA)된 쿼리 변환기 — 학습 결과를 평가·서비스에 꽂기 위한 어댑터.

기존 계층 변환기(hierarchical)는 Ollama로 Qwen3-4B를 부르지만, 학습 결과물(LoRA 어댑터)은
transformers 형식이라 Ollama가 바로 못 읽는다. 그래서 이 변환기는 transformers로 기본 모델을
올리고 그 위에 LoRA 어댑터를 얹어 직접 생성한다.

다른 변환기와 **같은 인터페이스**(`rewrite(질문) -> RewriteResult`)를 따르므로,
평가 하네스에서 `--rewriter finetuned` 로 바꿔 끼우기만 하면 학습 전/후를 같은 자로 비교할 수 있다.

학습 때 쓴 것과 **똑같은 지시문·대화 형식**을 써야 한다(형식이 다르면 학습 효과가 사라짐).
"""

from __future__ import annotations

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
