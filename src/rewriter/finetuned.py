"""파인튜닝(LoRA)한 쿼리 변환기를 평가와 서비스에 꽂는 어댑터.

학습 결과물은 transformers 형식이라 Ollama 가 바로 못 읽음. 그래서 여기서는
transformers 로 기본 모델을 올리고 그 위에 어댑터를 얹어 직접 생성함.

다른 변환기와 같은 인터페이스를 따르므로 `--rewriter dpo` 로 바꿔 끼우면 학습 전후를
같은 기준으로 비교할 수 있음. 학습 때 쓴 지시문과 대화 형식을 그대로 써야 함.
"""

from __future__ import annotations


from src import config
from src.rewriter.base import BACKENDS, OllamaClient
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


class OllamaFinetunedRewriter:
    """같은 변환기를 Ollama 4비트로 부름. 서비스가 쓰는 것.

    `FinetunedRewriter` 와 가중치가 같고 정밀도만 다름. transformers 로 올리면 8.27GB 를
    쓰는데 Ollama 4비트는 3.25GB 라, 재정렬 모델과 자리를 나눠 쓸 수 있음.

    모델은 `training/export_ollama.py` 로 미리 만들어 둬야 함. LoRA 를 기본 모델에 합쳐
    GGUF 로 바꾼 뒤 `ollama create -q q4_K_M` 로 등록하는 절차임.

    지시문과 대화 형식은 `FinetunedRewriter` 와 똑같이 맞춰야 함. 어긋나면 오류 없이
    엉뚱한 문자열만 나옴.
    """

    name = "dpo"

    def __init__(self, model: str = config.ARXIV_REWRITER_MODEL,
                 client: OllamaClient | None = None, max_tokens: int = 150):
        self.client = client or OllamaClient(model=model)
        self.max_tokens = max_tokens

    def unload(self) -> None:
        """`GpuPool.release` 가 부름. 다 쓴 뒤 이 모델 자리를 비워 재정렬 모델에 넘김."""
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
            # 세 필드에 같은 값을 넣는 것은 인터페이스를 맞추기 위한 것임. dense 필드에 든
            # 값도 arXiv 문법 문자열이라 의미 검색에 넣으면 불리하므로, 서비스도 평가도
            # 로컬 채널에는 원본 질문을 넣음.
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
