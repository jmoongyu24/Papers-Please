"""쿼리 변환기 DPO 학습 — '잘 찾는 쿼리 vs 못 찾는 쿼리'의 차이를 배우게 한다.

지도 미세조정(SFT)과의 차이:
- SFT는 "이게 정답이다"라는 예시만 보여준다. 무엇이 나쁜지는 안 가르친다.
- DPO(Direct Preference Optimization)는 **좋은 답과 나쁜 답을 쌍으로** 보여주고,
  좋은 쪽의 확률은 올리고 나쁜 쪽은 내리도록 학습한다. 즉 "왜 이게 더 나은가"의 경계를 배운다.

우리 데이터가 이 방식에 잘 맞는 이유:
  같은 질문에 대해 후보 쿼리를 여러 개 만들고 실제 arXiv 검색으로 채점했으므로,
  **정답 논문을 찾아낸 쿼리(chosen) vs 못 찾은 쿼리(rejected)** 쌍이 자연스럽게 생겼다.
  둘 다 "학술적으로 그럴싸한" 쿼리지만 검색 성공 여부가 갈렸으므로, 모델이 배워야 할
  것은 정확히 '실제로 통하는 어휘'의 미묘한 차이다.

이미 SFT로 학습한 어댑터 위에 이어서 학습한다(SFT → DPO 순서가 표준).

실행 예:
  python -m training.train_dpo --data data/training/dpo_pairs.jsonl \
      --sft-adapter models/qwen3-4b-query-lora/checkpoint-54 \
      --output-dir models/qwen3-4b-query-dpo
"""

from __future__ import annotations

import argparse
import json

# SFT와 동일한 지시문을 써야 한다(형식이 다르면 앞서 배운 것이 흐트러짐)
INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)


def load_preference_dataset(path: str):
    """DPO 형식으로 불러온다: prompt / chosen / rejected."""
    from datasets import Dataset

    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append({
            "prompt": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": r["input"]},
            ],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        })
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="쿼리 변환기 DPO 학습")
    ap.add_argument("--data", default="data/training/dpo_pairs.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--sft-adapter", default="models/qwen3-4b-query-lora/checkpoint-54",
                    help="SFT로 학습한 LoRA 어댑터. 그 위에 이어서 DPO 학습한다")
    ap.add_argument("--output-dir", default="models/qwen3-4b-query-dpo")
    ap.add_argument("--epochs", type=int, default=2,
                    help="DPO는 SFT보다 적은 에폭으로도 충분(과하면 성능이 무너짐)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="DPO는 SFT보다 훨씬 낮은 학습률을 쓴다(1e-6~1e-5). 크면 모델이 붕괴")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="원본 모델에서 얼마나 벗어날지 조절. 작을수록 자유롭게 변함")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    print(f"기본 모델: {args.base_model}")
    print(f"SFT 어댑터: {args.sft_adapter}")
    print("양자화 없음 — bf16 전체 정밀도")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto"
    )
    # SFT 어댑터를 얹고, 그 위에서 이어서 학습할 수 있도록 학습 가능 상태로 연다
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)

    dataset = load_preference_dataset(args.data)
    split = dataset.train_test_split(test_size=args.val_ratio, seed=42)
    print(f"학습 쌍 {len(split['train'])}개 · 검증 쌍 {len(split['test'])}개")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_len,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        warmup_ratio=0.1,
        bf16=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nDPO 학습 완료. 어댑터 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
