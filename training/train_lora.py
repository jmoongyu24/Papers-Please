"""쿼리 변환기 LoRA 미세조정 — '실제로 찾아내는 쿼리'를 생성하도록 학습한다.

무엇을 학습하나:
  입력  = 사용자의 일상어 질문
  출력  = 그 질문의 정답 논문을 arXiv에서 실제로 찾아낸 검색 쿼리
  (라벨은 training/build_training_data.py 가 검색 성공을 기준으로 뽑아 놓은 것)

왜 LoRA인가:
  Qwen3-4B는 값이 40억 개라 전부 학습시키려면 메모리가 매우 많이 필요하다. LoRA는 원래 모델은
  얼려두고 **작은 보조 행렬(전체의 1% 미만)만 새로 학습**해 끼우는 방식이라, 우리 그래픽카드
  (16기가바이트)로 충분히 돌아간다. 학습 결과물도 수십 메가바이트로 작아 관리가 쉽다.

왜 4비트 양자화(QLoRA)를 쓰나:
  모델 가중치를 4비트로 압축해 메모리를 더 줄인다. 성능 손실은 작고, 남는 메모리를 배치 크기와
  문맥 길이에 쓸 수 있다.

전제:
  pip install transformers peft trl bitsandbytes accelerate datasets

실행 예:
  python -m training.train_lora --data data/training/sft_pairs.jsonl \
      --output-dir models/qwen3-4b-query-lora --epochs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 학습에 쓰는 지시문. 실제 서비스에서 쓰는 프롬프트와 형식을 맞춰야 학습 효과가 산다.
INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)


def format_example(row: dict) -> dict:
    """학습 예시 한 건을 대화 형식으로 만든다(모델이 실제로 쓰이는 방식과 동일하게)."""
    return {
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def load_dataset_from_jsonl(path: str):
    from datasets import Dataset

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return Dataset.from_list([format_example(r) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description="쿼리 변환기 LoRA 미세조정")
    ap.add_argument("--data", default="data/training/sft_pairs.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="허깅페이스 기본 모델 이름")
    ap.add_argument("--output-dir", default="models/qwen3-4b-query-lora")
    ap.add_argument("--epochs", type=int, default=8,
                    help="데이터가 84건으로 적어, 에폭을 늘려 학습 스텝 수를 확보한다")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="양자화를 안 쓰면 메모리 여유가 있어 4까지 가능")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="배치 누적(실질 배치 = batch_size × grad_accum). 데이터가 적을 때 "
                         "누적을 크게 하면 스텝이 너무 적어져 학습이 거의 안 되므로 1로 둔다")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="데이터가 적을 때는 낮게(과적합 억제). LoRA 통상 1e-4~3e-4")
    ap.add_argument("--lora-r", type=int, default=32,
                    help="LoRA 보조 행렬 크기. 메모리 여유가 있으니 32로 표현력 확보")
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="실제 데이터가 최대 424글자라 512로 충분(길면 메모리만 낭비)")
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="검증용으로 떼어둘 비율(과적합 감시용)")
    ap.add_argument("--use-4bit", action="store_true",
                    help="4비트 양자화 켜기. **기본은 끔** — 16GB VRAM에서 bf16(약 9.7GB)이 "
                         "충분히 들어가고, 양자화는 성능을 깎기 때문. 메모리가 부족할 때만 사용")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    print(f"기본 모델: {args.base_model}")
    print(f"학습 데이터: {args.data}")

    # 기본은 양자화 없이 bf16으로 학습한다.
    # 이유: Qwen3-4B를 bf16으로 올리면 가중치 약 8GB + 옵티마이저·활성화 약 1.7GB = 약 9.7GB로,
    # 16GB 그래픽카드에 충분히 들어간다. 4비트 양자화는 메모리가 모자랄 때 쓰는 타협책이며
    # 가중치를 압축하는 만큼 품질이 떨어지므로, 여유가 있으면 쓰지 않는 것이 성능에 유리하다.
    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("4비트 양자화 사용 (메모리 절약, 품질 손실 감수)")
    else:
        print("양자화 없음 — bf16 전체 정밀도로 학습 (품질 우선)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # LoRA: 주의(attention)와 피드포워드 층에만 작은 보조 행렬을 붙여 학습한다
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_dataset_from_jsonl(args.data)
    # 데이터가 적으므로 일부를 검증용으로 떼어 과적합(외워버리기)을 감시한다.
    # 학습 손실만 계속 떨어지고 검증 손실이 오르기 시작하면 과적합 신호다.
    split = dataset.train_test_split(test_size=args.val_ratio, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"학습 예시 {len(train_ds)}개 · 검증 예시 {len(eval_ds)}개")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_len,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",          # 에폭마다 검증 손실 확인
        warmup_ratio=0.1,               # 초반에 학습률을 서서히 올려 안정화
        lr_scheduler_type="cosine",     # 후반에 학습률을 낮춰 과적합 억제
        weight_decay=0.01,              # 가중치가 과하게 커지지 않도록
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n학습 완료. LoRA 어댑터 저장: {args.output_dir}")
    print("다음 단계: 어댑터를 병합해 Ollama 형식(GGUF)으로 변환하거나, "
          "transformers로 직접 불러 평가한다.")


if __name__ == "__main__":
    main()
