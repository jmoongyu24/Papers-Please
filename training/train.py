"""쿼리 변환기 학습 — 지도 미세조정(SFT) 과 선호 학습(DPO) 을 한 파일에서.

    # 1단계 SFT — "이런 검색어를 만들어라" 를 흉내 내게 한다
    $PY -m training.train sft --data data/training/sft_pairs.jsonl \\
        --output-dir models/qwen3-4b-query-lora --epochs 8

    # 2단계 DPO — SFT 어댑터 위에 "좋은 것과 나쁜 것의 차이" 를 얹는다
    $PY -m training.train dpo --data data/training/dpo_pairs.jsonl \\
        --sft-adapter models/qwen3-4b-query-lora/checkpoint-54 \\
        --output-dir models/qwen3-4b-query-dpo

**순서가 중요하다.** SFT → DPO 가 표준이고, DPO 는 SFT 어댑터 위에 이어서 학습한다.
서비스가 쓰는 것은 2단계까지 끝낸 `models/qwen3-4b-query-dpo` 다.

## 무엇을 학습하나

    입력  = 사용자의 일상어 질문
    출력  = 그 질문의 정답 논문을 arXiv 에서 **실제로 찾아낸** 검색어

라벨은 사람이 고른 것이 아니라 `training/build_training_data.py` 가 **검색 성공 여부로**
뽑아 놓은 것이다. 즉 "학술적으로 그럴싸한 말" 이 아니라 "실제로 통하는 말" 을 배운다.

## SFT 와 DPO 의 차이

- **SFT** 는 "이게 정답이다" 라는 예시만 보여준다. 무엇이 나쁜지는 안 가르친다.
- **DPO** 는 좋은 답과 나쁜 답을 **쌍으로** 보여주고, 좋은 쪽의 확률은 올리고 나쁜 쪽은
  내린다. 즉 "왜 이게 더 나은가" 의 경계를 배운다.

우리 데이터가 DPO 에 잘 맞는 이유: 같은 질문에 후보 검색어를 여러 개 만들고 실제 arXiv
검색으로 채점했으므로, **정답을 찾아낸 검색어(chosen) vs 못 찾은 검색어(rejected)** 쌍이
자연스럽게 생겼다. 둘 다 그럴싸한데 결과가 갈렸으므로, 모델이 배워야 할 것은 정확히
'실제로 통하는 어휘' 의 미묘한 차이다.

## 왜 LoRA 인가

Qwen3-4B 는 값이 40억 개라 전부 학습시키려면 메모리가 매우 많이 필요하다. LoRA 는 원래
모델은 얼려두고 **작은 보조 행렬(전체의 1% 미만)만 새로 학습**해 끼우는 방식이라, 16GB
그래픽카드로 충분히 돌아간다. 결과물도 수십 메가바이트로 작아 관리가 쉽다.

전제: `pip install transformers peft trl bitsandbytes accelerate datasets`
"""

from __future__ import annotations

import argparse
import json
import random

# 학습에 쓰는 지시문. **SFT 와 DPO 가 반드시 같아야 한다** — 형식이 다르면 앞서 배운 것이
# 흐트러진다. 실제 서비스에서 쓰는 프롬프트와도 형식을 맞춰야 학습 효과가 산다.
INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)


def read_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ══════════════════════════════════════════════════════════════════════════
# 1단계. SFT (지도 미세조정)
# ══════════════════════════════════════════════════════════════════════════

def format_example(row: dict) -> dict:
    """학습 예시 한 건을 대화 형식으로 만든다(모델이 실제로 쓰이는 방식과 동일하게)."""
    return {
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def split_by_paper(path: str, val_ratio: float, seed: int = 42):
    """**논문 단위**로 학습/검증을 나눈다.

    왜 질문 단위로 나누면 안 되는가:
    학습 데이터는 논문 한 편당 질문 여러 개로 만들어졌고, **같은 논문에서 나온 질문들은
    정답 라벨이 거의 같다**(라벨을 그 논문에서 뽑았으므로). 질문 단위로 무작위 분할하면
    같은 논문이 학습과 검증 양쪽에 들어가, 검증 손실이 실제보다 좋게 나온다. 그러면
    과적합이 시작되는 지점을 놓쳐 잘못된 체크포인트를 고르게 된다.
    (평가셋을 논문 단위로 나눈 것과 같은 이유다.)

    Returns: (학습용 Dataset, 검증용 Dataset)
    """
    from datasets import Dataset

    rows = read_rows(path)
    papers = sorted({r.get("gold_id", r["input"]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(papers)
    n_val = max(1, int(len(papers) * val_ratio))
    val_papers = set(papers[:n_val])

    train_rows = [r for r in rows if r.get("gold_id", r["input"]) not in val_papers]
    val_rows = [r for r in rows if r.get("gold_id", r["input"]) in val_papers]
    print(f"논문 단위 분할: 학습 논문 {len(papers)-n_val}편 / 검증 논문 {n_val}편 (겹침 0)")
    return (Dataset.from_list([format_example(r) for r in train_rows]),
            Dataset.from_list([format_example(r) for r in val_rows]))


def cmd_sft(args) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    print(f"기본 모델: {args.base_model}")
    print(f"학습 데이터: {args.data}")

    # 기본은 양자화 없이 bf16 으로 학습한다.
    # 이유: Qwen3-4B 를 bf16 으로 올리면 가중치 약 8GB + 옵티마이저·활성화 약 1.7GB = 약 9.7GB
    # 로, 16GB 그래픽카드에 충분히 들어간다. 4비트 양자화는 메모리가 모자랄 때 쓰는 타협책이며
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

    # 데이터가 적으므로 일부를 검증용으로 떼어 과적합(외워버리기)을 감시한다.
    # 학습 손실만 계속 떨어지고 검증 손실이 오르기 시작하면 과적합 신호다.
    train_ds, eval_ds = split_by_paper(args.data, args.val_ratio)
    print(f"학습 예시 {len(train_ds)}개 · 검증 예시 {len(eval_ds)}개")

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_length=args.max_len,
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch",          # 에폭마다 검증 손실 확인
            warmup_ratio=0.1,               # 초반에 학습률을 서서히 올려 안정화
            lr_scheduler_type="cosine",     # 후반에 학습률을 낮춰 과적합 억제
            weight_decay=0.01,              # 가중치가 과하게 커지지 않도록
            bf16=True,
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nSFT 학습 완료. LoRA 어댑터 저장: {args.output_dir}")
    print(f"다음 단계: 검증 손실이 가장 낮은 체크포인트를 골라 DPO 로 넘긴다.\n"
          f"  $PY -m training.train dpo --sft-adapter {args.output_dir}/checkpoint-<번호>")


# ══════════════════════════════════════════════════════════════════════════
# 2단계. DPO (선호 학습)
# ══════════════════════════════════════════════════════════════════════════

def load_preference_dataset(path: str):
    """DPO 형식으로 불러온다: prompt / chosen / rejected."""
    from datasets import Dataset

    return Dataset.from_list([{
        "prompt": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": r["input"]},
        ],
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}],
    } for r in read_rows(path)])


def cmd_dpo(args) -> None:
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

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
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
        ),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nDPO 학습 완료. 어댑터 저장: {args.output_dir}")
    print("서비스와 평가에서 쓰려면 변환기 이름을 'dpo' 로 부르면 된다.")


# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="쿼리 변환기 학습 (sft → dpo 순서로 쓴다)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sft", help="1단계: 지도 미세조정")
    s.add_argument("--data", default="data/training/sft_pairs.jsonl")
    s.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    s.add_argument("--output-dir", default="models/qwen3-4b-query-lora")
    s.add_argument("--epochs", type=int, default=8,
                   help="데이터가 적어 에폭을 늘려 학습 스텝 수를 확보한다")
    s.add_argument("--batch-size", type=int, default=4,
                   help="양자화를 안 쓰면 메모리 여유가 있어 4까지 가능")
    s.add_argument("--grad-accum", type=int, default=1,
                   help="배치 누적(실질 배치 = batch_size × grad_accum). 데이터가 적을 때 "
                        "누적을 크게 하면 스텝이 너무 적어져 학습이 거의 안 되므로 1로 둔다")
    s.add_argument("--lr", type=float, default=1e-4,
                   help="데이터가 적을 때는 낮게(과적합 억제). LoRA 통상 1e-4~3e-4")
    s.add_argument("--lora-r", type=int, default=32,
                   help="LoRA 보조 행렬 크기. 메모리 여유가 있으니 32로 표현력 확보")
    s.add_argument("--max-len", type=int, default=512,
                   help="실제 데이터가 최대 424글자라 512로 충분(길면 메모리만 낭비)")
    s.add_argument("--val-ratio", type=float, default=0.15, help="검증용으로 뗄 비율")
    s.add_argument("--use-4bit", action="store_true",
                   help="4비트 양자화 켜기. **기본은 끔** — 16GB VRAM 에서 bf16(약 9.7GB)이 "
                        "충분히 들어가고, 양자화는 성능을 깎기 때문. 메모리가 부족할 때만")
    s.set_defaults(func=cmd_sft)

    d = sub.add_parser("dpo", help="2단계: 선호 학습 (SFT 어댑터 위에)")
    d.add_argument("--data", default="data/training/dpo_pairs.jsonl")
    d.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    d.add_argument("--sft-adapter", default="models/qwen3-4b-query-lora/checkpoint-54",
                   help="SFT로 학습한 LoRA 어댑터. 그 위에 이어서 DPO 학습한다")
    d.add_argument("--output-dir", default="models/qwen3-4b-query-dpo")
    d.add_argument("--epochs", type=int, default=2,
                   help="DPO는 SFT보다 적은 에폭으로도 충분(과하면 성능이 무너짐)")
    d.add_argument("--batch-size", type=int, default=2)
    d.add_argument("--lr", type=float, default=5e-6,
                   help="DPO는 SFT보다 훨씬 낮은 학습률을 쓴다(1e-6~1e-5). 크면 모델이 붕괴")
    d.add_argument("--beta", type=float, default=0.1,
                   help="원본 모델에서 얼마나 벗어날지 조절. 작을수록 자유롭게 변함")
    d.add_argument("--max-len", type=int, default=512)
    d.add_argument("--val-ratio", type=float, default=0.15)
    d.set_defaults(func=cmd_dpo)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
