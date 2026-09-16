"""
파인튜닝한 쿼리 변환기를 서비스에 연결함
"""

from __future__ import annotations


from src import config
from src.rewriter.base import BACKENDS, OllamaClient
from src.schemas import RewriteResult

INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER = "models/query-translator-dpo"


class FinetunedRewriter:
    """파인튜닝한 모델로 arXiv 검색어를 생성함"""

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

    def rewrite(self, raw_query: str) -> RewriteResult:
        try:
            query = self._generate(raw_query).splitlines()[0].strip()
            if not query:
                raise ValueError("빈 출력")

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


class OllamaFinetunedRewriter:
    """
    변환기를 Ollama 4비트로 부름
    """

    name = "dpo"

    def __init__(self, model: str = config.ARXIV_REWRITER_MODEL,
                 client: OllamaClient | None = None, max_tokens: int = 150):
        self.client = client or OllamaClient(model=model)
        self.max_tokens = max_tokens

    def unload(self) -> None:
        """`GpuPool.release`가 부름. 다 쓴 뒤 그래픽 메모리를 비워 재정렬 모델에 넘김."""
        self.client.unload()

    def rewrite(self, raw_query: str) -> RewriteResult:
        try:
            text = self.client.generate_text(
                [{"role": "system", "content": INSTRUCTION},
                 {"role": "user", "content": raw_query}],
                temperature=0.0, max_tokens=self.max_tokens)
            query = text.strip().splitlines()[0].strip()
            if not query:
                raise ValueError("빈 출력")

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
