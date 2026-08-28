"""쿼리 변환기 학습 - 지도 미세조정(SFT) 과 선호 학습(DPO) 을 한 파일에서.

    # 1단계 SFT - "이런 검색어를 만들어라" 를 흉내 내게 함
    $PY -m training.train sft --data data/training/train_query_translator_sft.jsonl \\
        --output-dir models/query-translator-sft --epochs 8

    # 2단계 DPO - SFT 어댑터 위에 "좋은 것과 나쁜 것의 차이" 를 얹음
    $PY -m training.train dpo --data data/training/train_query_translator_dpo.jsonl \\
        --sft-adapter models/query-translator-sft/checkpoint-54 \\
        --output-dir models/query-translator-dpo

순서가 중요함. SFT -> DPO 가 표준이고, DPO 는 SFT 어댑터 위에 이어서 학습함.
서비스가 쓰는 것은 2단계까지 끝낸 `models/query-translator-dpo` 다.

## 무엇을 학습하나

    입력  = 사용자의 일상어 질문
    출력  = 그 질문의 정답 논문을 arXiv 에서 실제로 찾아낸 검색어

라벨은 사람이 고른 것이 아니라 `training/build_translator_pairs.py` 가 검색 성공 여부로
뽑아 놓은 것임. 즉 "학술적으로 그럴싸한 말" 이 아니라 "실제로 통하는 말" 을 배움.

## SFT 와 DPO 의 차이

- SFT 는 "이게 정답이다" 라는 예시만 보여줌. 무엇이 나쁜지는 안 가르침.
- DPO 는 좋은 답과 나쁜 답을 쌍으로 보여주고, 좋은 쪽의 확률은 올리고 나쁜 쪽은
  내림. 즉 "왜 이게 더 나은가" 의 경계를 배움.

우리 데이터가 DPO 에 잘 맞는 이유: 같은 질문에 후보 검색어를 여러 개 만들고 실제 arXiv
검색으로 채점했으므로, 정답을 찾아낸 검색어(chosen) vs 못 찾은 검색어(rejected) 쌍이
자연스럽게 생겼음. 둘 다 그럴싸한데 결과가 갈렸으므로, 모델이 배워야 할 것은 정확히
'실제로 통하는 어휘' 의 미묘한 차이임.

## 왜 LoRA 인가

Qwen3-4B 는 값이 40억 개라 전부 학습시키려면 메모리가 매우 많이 필요함. LoRA 는 원래
모델은 얼려두고 작은 보조 행렬(전체의 1% 미만)만 새로 학습해 끼우는 방식이라, 16GB
그래픽카드로 충분히 돌아감. 결과물도 수십 메가바이트로 작아 관리가 쉬움.

전제: `pip install transformers peft trl bitsandbytes accelerate datasets`
"""

from __future__ import annotations

import argparse
import json
import random
import time

# 학습에 쓰는 지시문. SFT 와 DPO 가 반드시 같아야 함 - 형식이 다르면 앞서 배운 것이
# 흐트러짐. 실제 서비스에서 쓰는 프롬프트와도 형식을 맞춰야 학습 효과가 삶.
INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)


def read_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ==========================================================================
# 1단계. SFT (지도 미세조정)
# ==========================================================================

def format_example(row: dict) -> dict:
    """학습 예시 한 건을 대화 형식으로 만듦(모델이 실제로 쓰이는 방식과 동일하게)."""
    return {
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def split_by_paper(path: str, val_ratio: float, seed: int = 42):
    """논문 단위로 학습/검증을 나눔.

    왜 질문 단위로 나누면 안 되는가:
    학습 데이터는 논문 한 편당 질문 여러 개로 만들어졌고, 같은 논문에서 나온 질문들은
    정답 라벨이 거의 같음(라벨을 그 논문에서 뽑았으므로). 질문 단위로 무작위 분할하면
    같은 논문이 학습과 검증 양쪽에 들어가, 검증 손실이 실제보다 좋게 나옴. 그러면
    과적합이 시작되는 지점을 놓쳐 잘못된 체크포인트를 고르게 됨.
    (평가셋을 논문 단위로 나눈 것과 같은 이유임.)

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

    # 기본은 양자화 없이 bf16 으로 학습함.
    # 이유: Qwen3-4B 를 bf16 으로 올리면 가중치 약 8GB + 옵티마이저, 활성화 약 1.7GB = 약 9.7GB
    # 로, 16GB 그래픽카드에 충분히 들어감. 4비트 양자화는 메모리가 모자랄 때 쓰는 타협책이며
    # 가중치를 압축하는 만큼 품질이 떨어지므로, 여유가 있으면 쓰지 않는 것이 성능에 유리함.
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
        print("양자화 없음 - bf16 전체 정밀도로 학습 (품질 우선)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # LoRA: 주의(attention)와 피드포워드 층에만 작은 보조 행렬을 붙여 학습함
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # 데이터가 적으므로 일부를 검증용으로 떼어 과적합(외워버리기)을 감시함.
    # 학습 손실만 계속 떨어지고 검증 손실이 오르기 시작하면 과적합 신호임.
    train_ds, eval_ds = split_by_paper(args.data, args.val_ratio)
    print(f"학습 예시 {len(train_ds)}개, 검증 예시 {len(eval_ds)}개")

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


# ==========================================================================
# 2단계. DPO (선호 학습)
# ==========================================================================

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
    print("양자화 없음 - bf16 전체 정밀도")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto"
    )
    # SFT 어댑터를 얹고, 그 위에서 이어서 학습할 수 있도록 학습 가능 상태로 열어 둠
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)

    dataset = load_preference_dataset(args.data)
    split = dataset.train_test_split(test_size=args.val_ratio, seed=42)
    print(f"학습 쌍 {len(split['train'])}개, 검증 쌍 {len(split['test'])}개")

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



# ==========================================================================
# 검색 모델(임베딩) 미세조정
# ==========================================================================
#
# ## 무엇을 왜 하는가
#
# 지금 막힌 곳은 재정렬이 아니라 1차 검색임. 시험용 342문항에서 후보 100편 안에 정답이
# 아예 없는 문항이 98개(일상어 층 69개)이고, 후보에 있는데 재정렬이 버린 문항은 33개임.
# 후보 상한이 0.713 이고 회수율이 0.865 라 지금 값이 0.617 인데, 목표 0.700 에 닿으려면
# 회수율을 0.982 까지 올리거나(사실상 불가능) 후보 상한을 0.809 로 올려야 함.
#
# 후보를 깊게 가져오는 길은 이미 막혔음 - 깊이 300 에서 상한은 0.721 로 오르지만 만족도가
# 확실히 떨어짐(-0.036, p<0.001). 쿼리 변환 쪽으로도 다섯 번 시험해 다섯 번 실패했음.
# 남은 것이 **검색 모델 자체를 학습시키는 것**임.
#
# ## 무엇을 학습하나
#
#     질문        "조건이 많은 문제를 아주 적은 메모리로 대충 잘 푸는 방법이 있나"
#     정답 논문    그 질문을 만들어 낸 논문의 제목 + 초록      <- 가깝게
#     오답 논문 6편  재정렬기가 무관하다고 한 논문             <- 멀게
#
# 학습 쌍은 `training/build_retrieval_pairs.py` 가 만듦. 오답을 어떻게 골랐는지와 왜 그렇게
# 골랐는지는 그 파일 설명글에 실측표와 함께 있음. **오답 고르는 규칙이 이 학습의 성패를
# 가르므로 반드시 읽을 것.**
#
# ## 반드시 지킬 것
#
# 1. **바탕 모델은 지금 색인을 만든 것과 같아야 함** (`BAAI/bge-m3`). 다르면 학습한 것과
#    색인이 어긋남.
# 2. **최대 길이 512** - `local_index.build_embeddings` 가 512 로 색인을 만들었음.
#    학습을 다른 길이로 하면 학습할 때 본 글과 색인에 들어간 글이 달라짐.
# 3. **저장은 합쳐서 함** - `LocalDenseRetriever` 는 `SentenceTransformer(경로)` 로 모델을
#    올리므로, 보조 행렬만 저장하면 못 읽음. 합친 모델을 통째로 저장함.
# 4. 학습이 끝나면 **색인을 다른 이름으로 새로 만들 것.** 같은 이름을 주면
#    `build_embeddings` 가 기존 임베딩 2.93GB 를 덮어씀 (되돌릴 수 없음).


def merge_lora_into(model) -> int:
    """보조 행렬을 본체 가중치에 더하고 껍데기를 벗겨냄. 합친 층 수를 돌려줌.

    ## 왜 손으로 합치는가

    `get_peft_model` 은 보조 행렬을 **본체 안에 직접 끼워 넣고** 껍데기 객체를 따로
    돌려줌. 그런데 `SentenceTransformer` 의 `auto_model` 자리에는 껍데기가 남지 않아서
    (본체가 `XLMRobertaModel` 그대로임) 껍데기의 `merge_and_unload` 를 부를 수가 없음.

    학습은 껍데기 없이도 제대로 됨 - 끼워 넣은 층을 그대로 통과하기 때문임. 문제는
    저장뿐이고, 합치지 않은 채로 저장하면 `LocalDenseRetriever` 가 못 읽어 색인을
    만들 수 없음. 그래서 층을 직접 찾아 합치고 원래 층으로 되돌려 놓음.

    합친 층 수가 0 이면 학습이 안 된 것이므로 부르는 쪽에서 멈출 것.
    """
    from peft.tuners.lora import LoraLayer

    merged = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoraLayer):
                child.merge()
                setattr(parent, name, child.get_base_layer())
                merged += 1
    return merged


def cmd_embed(args) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    rows = read_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    n_neg = min(len(r["negatives"]) for r in rows)
    if n_neg < args.negatives:
        print(f"경고: 오답이 {n_neg}편뿐인 문항이 있어 전부 {n_neg}편으로 맞춘다")
    n_neg = min(n_neg, args.negatives)
    print(f"학습 문항 {len(rows):,}개 · 오답 {n_neg}편씩")

    cols = {"anchor": [r["query"] for r in rows],
            "positive": [r["positive"] for r in rows]}
    for j in range(n_neg):
        cols[f"negative_{j + 1}"] = [r["negatives"][j] for r in rows]
    train_ds = Dataset.from_dict(cols)

    print(f"바탕 모델: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_len
    print(f"최대 길이 {model.max_seq_length} (색인을 만든 값과 같아야 함)")

    # 큰 모델은 얼려 두고 작은 보조 행렬만 학습함. 주의 층과 완전연결 층에 붙임.
    #
    # `SentenceTransformer.add_adapter` 가 아니라 `get_peft_model` 로 감싸는 이유:
    # 앞의 것은 transformers 의 자체 연동을 써서 본체가 `XLMRobertaModel` 그대로 남는데,
    # 그러면 학습이 끝난 뒤 보조 행렬을 본체에 합칠 방법(`merge_and_unload`)이 없음.
    # 합치지 않은 모델은 `LocalDenseRetriever` 가 못 읽으므로 색인을 만들 수 없음.
    from peft import get_peft_model
    model[0].auto_model = get_peft_model(model[0].auto_model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"], bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"학습하는 값 {trainable:,}개 / 전체 {total:,}개 ({trainable / total:.2%})")

    # 대조 학습. 한 묶음 안의 다른 문항들이 자동으로 추가 오답이 됨 - 그래서 묶음이 클수록
    # 오답이 많아져 학습이 세짐. 여기서는 문항 하나가 글 (1 + 1 + 오답 수) 개를 통과하므로
    # 묶음 크기를 크게 잡으면 그래픽카드 메모리를 금방 넘김.
    loss = MultipleNegativesRankingLoss(model)

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir + "-ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=args.grad_checkpoint,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs,
                                         train_dataset=train_ds, loss=loss)

    t0 = time.time()
    trainer.train()
    print(f"\n학습 시간 {(time.time() - t0) / 60:.1f}분")

    # 보조 행렬을 본체에 합쳐서 저장함. 합치지 않으면 LocalDenseRetriever 가 못 읽음.
    n_merged = merge_lora_into(model)
    if not n_merged:
        raise RuntimeError("합칠 보조 행렬 층을 하나도 못 찾았다 - 학습이 안 붙은 것이다")
    print(f"보조 행렬 {n_merged}개 층을 본체에 합침")
    model.save(args.output_dir)
    print(f"모델 저장: {args.output_dir}")
    print("다음 단계 - 색인을 **다른 이름으로** 새로 만들 것:")
    print(f"  $PY -m src.retrieval.local_index --model {args.output_dir} "
          f"--out data/embeddings/cs2021-ft")



# ==========================================================================
# 재정렬기 미세조정 - 2단계(줄 세우기)를 고치는 것
# ==========================================================================
#
# ## 무엇을 고치려는 것인가
#
# ISSUE #41 이 재정렬 점수를 직접 열어 확인한 것임. 정답이 후보 안에 있었는데 상위 10편에서
# 밀린 문항의 점수임(개발용 348문항, 미세조정 색인).
#
#     난이도   밀린 문항   정답 점수(중앙)   10등 점수(중앙)   못 알아봄(0.002 미만)
#     easy        2         0.5432          0.5865          0.000
#     medium     12         0.0470          0.1758          0.083
#     hard       24         0.0035          0.0152          0.250
#
# **hard 층은 정답도 10등도 전부 무관 구간에 있음.** 정답을 오답보다 낮게 매긴 것이 아니라
# 일상어 질문에 대해 쓸 만한 점수 눈금 자체를 못 만듦. 고칠 것은 순서가 아니라 눈금임.
#
# 상한: 밀린 문항이 38개임(easy 2 + medium 12 + hard 24). 전부 살리면 개발용 Recall@10 이
# 0.618 에서 0.727, hard 가 0.267 에서 0.474 가 됨. 후보 상한과 같은 값임.
#
# ## 왜 이분 라벨(관련 1 / 무관 0)을 쓰지 않는가
#
# 정답을 밀어낸 논문의 **77.1% 가 등급 2 이상, 즉 실제로 쓸모 있는 논문임**(개발용 등급
# 정답지로 실측, `build_retrieval_pairs.py` 설명글 참고). 그것들에 "무관" 라벨을 붙이면
# **좋은 논문을 내리라고 가르치게 됨.** ISSUE #49 가 검색 모델에서 막았던 것과 같은 함정임.
#
# 그래서 순서 손실(`CachedMultipleNegativesRankingLoss`)을 씀. 이 손실은 "정답이 이
# 논문들보다 위" 까지만 가르치고 "이 논문들은 무관" 이라고는 말하지 않음. 라벨이 없음.
#
# ## 오답 편수가 문항마다 다른 것을 어떻게 다루는가
#
# 정답이 이미 1등이면 밀어낸 논문이 없어서 오답이 0편임. 억지로 채우면 재정렬기가 이미
# 확실히 버리는 논문을 도로 넣게 됨(그것이 검색 모델용 자료였고, 이 자리에서는 신호가 없음).
#
# `CrossEncoderTrainer` 는 `DatasetDict` 를 받으므로 **오답 편수로 갈라 담음.**
#
#     pairs           (질문, 정답)                     오답이 모자란 문항
#     hard_negatives  (질문, 정답, 오답1, ..., 오답N)    오답이 N편 이상인 문항
#
# 양쪽 다 같은 손실을 씀. `pairs` 쪽도 묶음 안 다른 질문의 논문이 자동으로 오답이 되므로
# 학습이 됨 - 그쪽이 이번 학습의 주된 신호임(점수 눈금 만들기).
#
# ## LoRA 를 붙이는 자리
#
# `["query", "key", "value", "dense"]` 로 145개 층에 붙음. 여기에 `classifier.dense` 가
# 들어가서 점수를 내는 머리의 앞쪽도 함께 학습됨. 마지막 `classifier.out_proj` 는 얼어 있음.
#
# ## 껍데기를 씌우지 않고 층만 끼워 넣는 이유 (2026-08-27 실패에서 배운 것)
#
# `cmd_embed` 처럼 `get_peft_model` 로 껍데기를 씌우면 **학습이 시작되자마자 멈춤.**
#
#     AttributeError  (transformers/models/xlm_roberta/... input_ids.ne(padding_idx))
#
# 껍데기의 `forward` 인자가 `(*args, **kwargs)` 로 바뀌기 때문임. 실측함.
#
#     본체 forward 인자    ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']
#     껍데기 forward 인자   ['args', 'kwargs']
#
# `CrossEncoder` 는 모델의 인자 이름을 보고 어떻게 부를지 정하는데, 이름이 사라지면
# 토큰 묶음을 통째로 첫 번째 자리에 넣어 버림. `SentenceTransformer` 는 부르는 방식이
# 달라서 `cmd_embed` 에서는 이 문제가 안 났음.
#
# 그래서 `inject_adapter_in_model` 로 **층만 제자리에 끼워 넣음.** 본체가
# `XLMRobertaForSequenceClassification` 그대로 남아 인자 이름이 유지됨.
# 대신 이 함수는 원래 값을 얼려 주지 않으므로 손으로 얼려야 함.
#
# ## 병합해서 저장하는 이유
#
# `cmd_embed` 와 같음. 합치지 않은 채로 저장하면 `CrossEncoderReranker` 가 못 읽음.
#
# ## 학습이 끝난 뒤 반드시 할 것
#
# `app.py` 의 `MIN_RERANK_SCORE = 0.002` 를 **다시 재야 함.** 그 값은 재정렬 모델의 점수
# 눈금에 딸린 것이고, 이번 학습의 목적이 바로 그 눈금을 바꾸는 것임
# (`evaluation/README.md` 6절).


def cmd_rerank(args) -> None:
    import torch
    from datasets import Dataset, DatasetDict
    from peft import LoraConfig, inject_adapter_in_model
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.losses import \
        CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    rows = read_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    # 자료는 `query` / `docs`(정답 맨 앞) / `labels` 로 저장돼 있음. 두 손실이 요구하는
    # 열 구성이 다르므로 여기서 갈라 만듦.
    #   ranknet  (query, [글...]) + 라벨 목록      - 그 질문에 딸린 글끼리만 견줌
    #   mnrl     anchor / positive / negative_N   - 묶음 안 다른 질문의 글도 오답으로 씀
    for r in rows:
        if "docs" not in r:
            raise SystemExit(
                f"{args.data} 에 `docs` 열이 없다. 2026-08-28 이전 형식으로 보인다.\n"
                f"  build_retrieval_pairs.py --for-rerank 로 다시 만들 것.")
    n_neg = min(len(r["labels"]) - 1 for r in rows)
    if n_neg < args.negatives:
        print(f"경고: 오답이 {n_neg}편뿐인 문항이 있어 전부 {n_neg}편으로 맞춘다")
    n_neg = min(n_neg, args.negatives)
    print(f"학습 문항 {len(rows):,}개 · 문항당 어려운 오답 {n_neg}편 · 손실 {args.loss}")
    if n_neg == 0:
        raise SystemExit("오답이 0편인 문항이 있다. 이 자료로는 순서를 가르칠 수 없다 (#55).")

    def pos_of(r):
        return r["docs"][r["labels"].index(1)]

    def negs_of(r):
        return [d for d, l in zip(r["docs"], r["labels"]) if l == 0][:n_neg]

    subsets = {}
    if args.loss == "ranknet":
        subsets["listwise"] = Dataset.from_dict(
            {"query": [r["query"] for r in rows],
             "docs": [[pos_of(r)] + negs_of(r) for r in rows],
             "labels": [[1] + [0] * n_neg for _ in rows]})
    else:
        cols = {"anchor": [r["query"] for r in rows],
                "positive": [pos_of(r) for r in rows]}
        for j in range(n_neg):
            cols[f"negative_{j + 1}"] = [negs_of(r)[j] for r in rows]
        subsets["hard_negatives"] = Dataset.from_dict(cols)

    for name, ds in subsets.items():
        print(f"  {name:<16}{len(ds):>8,}문항  열 {list(ds.column_names)}")

    # 난이도가 고르게 들어갔는지 확인함. 한 층을 빼면 그 층이 그대로 있는 것이 아니라
    # 나빠짐 - 검색 모델에서 easy 를 빼고 겪었음(ISSUE #51).
    from collections import Counter
    print(f"  난이도 {dict(Counter(r.get('difficulty') for r in rows))}")
    print(f"  언어   {dict(Counter(r.get('lang') for r in rows))}")

    # -- 모델 --------------------------------------------------------------
    print(f"바탕 모델: {args.base_model}")
    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_len)
    print(f"최대 길이 {args.max_len} (서비스가 쓰는 값과 같아야 함)")

    # 껍데기를 씌우지 않고 층만 끼워 넣음 (위 설명글 참고). 본체가 그대로 남아야
    # CrossEncoder 가 인자 이름을 보고 제대로 부를 수 있음.
    inject_adapter_in_model(LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"], bias="none"), model.model)
    # inject_adapter_in_model 은 원래 값을 얼려 주지 않음. 손으로 얼림.
    for name, prm in model.model.named_parameters():
        prm.requires_grad = "lora_" in name
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"학습하는 값 {trainable:,}개 / 전체 {total:,}개 ({trainable / total:.2%})")

    if args.loss == "ranknet":
        # 그 질문에 딸린 글끼리만 견줌. 묶음 안 다른 질문의 글(쉬운 오답)이 안 들어감.
        from sentence_transformers.cross_encoder.losses import RankNetLoss
        loss = RankNetLoss(model, mini_batch_size=args.mini_batch_size)
    else:
        loss = CachedMultipleNegativesRankingLoss(
            model, num_negatives=args.in_batch_negatives,
            mini_batch_size=args.mini_batch_size)

    targs = CrossEncoderTrainingArguments(
        output_dir=args.output_dir + "-ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=args.grad_checkpoint,
        # 묶음 안에 같은 글이 두 번 들어가면 그것이 서로의 오답이 되어 버림.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )
    trainer = CrossEncoderTrainer(
        model=model, args=targs,
        train_dataset=DatasetDict(subsets) if len(subsets) > 1 else next(iter(subsets.values())),
        loss=loss)

    t0 = time.time()
    trainer.train()
    print(f"\n학습 시간 {(time.time() - t0) / 60:.1f}분")

    n_merged = merge_lora_into(model.model)
    if not n_merged:
        raise RuntimeError("합칠 보조 행렬 층을 하나도 못 찾았다 - 학습이 안 붙은 것이다")
    print(f"보조 행렬 {n_merged}개 층을 본체에 합침")
    model.save_pretrained(args.output_dir)
    print(f"모델 저장: {args.output_dir}")
    print("다음 단계 - 개발용에서 견줄 것 (색인은 다시 안 만들어도 됨):")
    print(f"  $PY -m evaluation.pipeline_eval --queries data/eval/dev.jsonl \\")
    print(f"      --channels local_dense local_hyde \\")
    print(f"      --reuse-queries runs/dev_service_repro.jsonl \\")
    print(f"      --index data/embeddings/cs2021-ft \\")
    print(f"      --embed-model models/retriever-ft \\")
    print(f"      --k 100 --rerank cross --rerank-depth 100 --out runs/dev_rr_ft.jsonl")
    print("그리고 app.py 의 MIN_RERANK_SCORE 를 다시 잴 것 (점수 눈금이 바뀌었음)")


# ==========================================================================
# 관문 1 - 미세조정이 정답 등수를 끌어올렸는가 (부분집합에서 값싸게 확인)
# ==========================================================================
#
# ## 왜 부분집합인가
#
# 71만 편을 다시 임베딩하는 데 3시간이 걸림. 그 전에 "오르긴 하는가"를 값싸게 걸러냄.
# 부분집합은 **정답 논문 + 지금 색인이 데려온 상위 후보** 로 만듦. 그 후보들이 정답과
# 실제로 경쟁하는 논문이므로, 정답이 그것들 위로 올라가는지가 곧 우리가 알고 싶은 것임.
#
# ## 이 값을 어떻게 읽어야 하는가 (반드시 지킬 것)
#
# **이 관문은 "접는 판정"에만 씀.** 부분집합은 지금 모델이 고른 후보로 만들어졌으므로,
# 미세조정한 모델이 **새로 끌어올릴 엉뚱한 논문**은 이 안에 없음. 그래서 여기 값은
# 실제보다 좋게 나옴. 한 방향으로만 믿을 수 있음.
#
#     여기서 안 오름  ->  71만 편에서도 안 오름. 재색인하지 말고 접을 것
#     여기서 오름     ->  아직 모름. 재색인해서 관문 2 에서 판정할 것
#
# ISSUE #26 · #31 이 "상한은 재정렬이 실제로 본 후보로 잰다"고 정한 것과 같은 정신임.
#
# ## 두 무리를 나눠서 보는 이유
#
#     검증용   학습에서 뺀 논문 500편의 문항. 학습 자료와 **같은 분야 분포**
#     개발용   data/eval/dev.jsonl. 평가셋 분포 (분야가 크게 다름)
#
# 학습 자료는 코퍼스 비율대로 뽑아 cs.CV·cs.LG·cs.CL·cs.AI 가 45.9% 인데, 개발용
# 평가셋은 분야를 고르게 뽑아 그 넷이 2.9%(348문항 중 10개) 뿐임. 그래서 검증용에서는
# 오르는데 개발용에서만 안 오르면 그것은 **방법이 안 되는 것이 아니라 분야가 안 맞는 것**임.
# 두 무리를 나눠 재야 그 둘을 가를 수 있음.


def _rank_of(sub_emb, q_vec, gold_row: int) -> int:
    """부분집합 안에서 정답이 몇 등인지 (1등이 1)."""
    scores = sub_emb @ q_vec
    return int((scores > scores[gold_row]).sum()) + 1


def _report_ranks(name: str, ranks_before: list, ranks_after: list, groups: list) -> None:
    import numpy as np

    print(f"\n[{name}]")
    print(f"  {'무리':<12}{'문항':>7}{'등수 중앙값':>22}{'10등 안':>18}{'100등 안':>18}")
    print(f"  {'':<12}{'':>7}{'전':>10}{'후':>10}{'전':>8}{'후':>8}{'전':>9}{'후':>8}")
    keys = sorted(set(groups)) + ["전체"]
    for g in keys:
        sel = [i for i in range(len(groups)) if g == "전체" or groups[i] == g]
        if not sel:
            continue
        b = np.array([ranks_before[i] for i in sel], dtype=float)
        a = np.array([ranks_after[i] for i in sel], dtype=float)
        print(f"  {g:<12}{len(sel):>7,}{np.median(b):>10.0f}{np.median(a):>10.0f}"
              f"{(b <= 10).mean():>8.3f}{(a <= 10).mean():>8.3f}"
              f"{(b <= 100).mean():>9.3f}{(a <= 100).mean():>8.3f}")
        moved_up = (a < b).mean()
        moved_dn = (a > b).mean()
        print(f"  {'':<12}       올라감 {moved_up:.3f} · 내려감 {moved_dn:.3f} · "
              f"그대로 {1 - moved_up - moved_dn:.3f}")


def cmd_embed_check(args) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from src import config
    from src.retrieval.corpus import normalize_paper_id
    from src.retrieval.local_index import LocalDenseRetriever
    from src.utils import read_jsonl
    from training.build_retrieval_pairs import doc_text, score_batch

    # -- 잴 문항 모으기 ----------------------------------------------------
    val = [r for r in read_rows(args.pairs)]
    if args.val_sample and args.val_sample < len(val):
        val = random.Random(args.seed).sample(val, args.val_sample)
    dev = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    items = ([{"text": r["query"], "gold_id": r["gold_id"],
               "group": "검증 " + r["difficulty"], "set": "val"} for r in val]
             + [{"text": r["text"], "gold_id": r["gold_id"],
                 "group": "개발 " + r["difficulty"], "set": "dev"} for r in dev])
    print(f"검증용 {len(val):,}문항 · 개발용 {len(dev):,}문항")

    # -- 지금 색인으로 후보 모으기 (이미 계산된 임베딩을 그대로 씀, 비용 0) ----
    print("지금 색인 불러오는 중...", flush=True)
    ret = LocalDenseRetriever(args.corpus, args.index)
    items = [it for it in items if normalize_paper_id(it["gold_id"]) in ret._pos]
    gold_pos = [ret._pos[normalize_paper_id(it["gold_id"])] for it in items]

    print(f"질문 {len(items):,}개로 후보 {args.depth}편씩 모으는 중...", flush=True)
    top_idx, _, _ = score_batch(ret, [it["text"] for it in items], args.depth, gold_pos,
                                batch_size=args.batch_size)

    subset = sorted(set(gold_pos) | {int(x) for row in top_idx for x in row})
    row_of = {p: i for i, p in enumerate(subset)}
    print(f"부분집합 논문 {len(subset):,}편 (71만 편 중 {len(subset) / len(ret.ids):.1%})")

    # -- 미세조정 전 등수: 이미 있는 임베딩을 그대로 쓰므로 정확하고 공짜 ------
    sub_before = np.asarray(ret.emb[subset], dtype=np.float32)
    qb = ret.embedder.encode([it["text"] for it in items], normalize_embeddings=True,
                             convert_to_numpy=True, batch_size=64).astype(np.float32)
    ranks_before = [_rank_of(sub_before, qb[i], row_of[gold_pos[i]])
                    for i in range(len(items))]
    del sub_before, qb

    # -- 부분집합 본문은 한 번만 읽음 (모델마다 다시 읽을 필요 없음) ----------
    texts = [doc_text(r) for r in ret.read_rows(subset)]
    del ret.emb                                   # 2.93GB 를 놓아 줌

    # -- 모델마다: 부분집합을 다시 임베딩해 등수를 잼 -------------------------
    import torch

    summary = []
    for model_path in args.model:
        print(f"\n{'=' * 74}\n미세조정 모델 불러오는 중: {model_path}", flush=True)
        kw = {"model_kwargs": {"dtype": torch.float16}} if torch.cuda.is_available() else {}
        ft = SentenceTransformer(model_path, **kw)
        ft.max_seq_length = args.max_len

        print(f"부분집합 {len(subset):,}편을 다시 임베딩하는 중...", flush=True)
        t0 = time.time()
        sub_after = ft.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                              batch_size=64, show_progress_bar=False).astype(np.float32)
        qa = ft.encode([it["text"] for it in items], normalize_embeddings=True,
                       convert_to_numpy=True, batch_size=64).astype(np.float32)
        print(f"다시 임베딩 완료 ({(time.time() - t0) / 60:.1f}분)", flush=True)
        ranks_after = [_rank_of(sub_after, qa[i], row_of[gold_pos[i]])
                       for i in range(len(items))]
        del sub_after, qa, ft
        torch.cuda.empty_cache()

        print(f"\n### {model_path}")
        for tag, which in (("검증용 - 학습 자료와 같은 분야 분포", "val"),
                           ("개발용 - 평가셋 분포", "dev")):
            sel = [i for i in range(len(items)) if items[i]["set"] == which]
            if sel:
                _report_ranks(tag, [ranks_before[i] for i in sel],
                              [ranks_after[i] for i in sel],
                              [items[i]["group"] for i in sel])
        summary.append((model_path, ranks_after))

    # -- 모델끼리 한 표로 견줌 ------------------------------------------------
    #
    # 고를 때 봐야 하는 것은 한 층의 값이 아니라 **층별 맞바꿈**임. 일상어 층이 올라도
    # 학술어 층이 그만큼 내려가면 전체는 제자리임. 그래서 세 층을 나란히 놓음.
    print(f"\n{'=' * 74}\n[모델 견주기] 개발용 348문항, 100등 안에 정답이 든 비율")
    groups = sorted({it["group"] for it in items if it["set"] == "dev"})
    head = f"  {'모델':<34}" + "".join(f"{g.replace('개발 ', ''):>10}" for g in groups) + f"{'전체':>10}"
    print(head)

    def _row(name: str, ranks: list) -> None:
        sel_all = [i for i in range(len(items)) if items[i]["set"] == "dev"]
        line = f"  {name[-33:]:<34}"
        for g in groups:
            sel = [i for i in sel_all if items[i]["group"] == g]
            line += f"{np.mean([ranks[i] <= 100 for i in sel]):>10.3f}"
        line += f"{np.mean([ranks[i] <= 100 for i in sel_all]):>10.3f}"
        print(line)

    _row("(미세조정 전)", ranks_before)
    for name, ranks in summary:
        _row(name, ranks)

    print("\n[관문 1 판정] 부분집합 안에서 잰 값이라 접는 판정에만 씀")
    print("  기준: 검증용 등수 중앙값이 내려가면 재색인으로 넘어가고, "
          "안 내려가면 여기서 접음")



# ==========================================================================
# 가중치 섞기 - 학습한 정도를 배율로 조절 (다시 학습하지 않음)
# ==========================================================================
#
# ## 왜 필요한가 (2026-08-25, 관문 1 에서 드러난 문제)
#
# 미세조정이 일상어 층(hard)은 크게 올렸는데 **정확한 학술어 층(easy)을 떨어뜨렸음.**
# 개발용 348문항 부분집합에서 잰 값임.
#
#     무리       100등 안 (전 -> 후)
#     easy       0.836 -> 0.638      <- 잊어버림
#     medium     0.655 -> 0.888
#     hard       0.250 -> 0.776
#
# 원인은 학습 자료에 easy 를 안 넣은 것임. 당시 근거는 "후보 상한이 0.931 이라 올릴 자리가
# 없다" 였는데, **올릴 자리가 없는 것과 잃을 자리가 없는 것은 다른 이야기였음.**
# 개발용 easy 한국어 문항의 99.1%가 낱말 나열인데, 모델이 문장형 질문만 보고 학습해
# 낱말 나열을 잊었음.
#
# ## 어떻게 고치는가
#
# 보조 행렬 학습은 원래 가중치에 변화량을 **더하는** 방식이라, 그 변화량에 배율을 곱하면
# 원래 모델과 학습한 모델 사이의 중간 지점이 나옴.
#
#     섞은 모델 = 원래 모델 + 배율 x (학습한 모델 - 원래 모델)
#
#     배율 0.0  원래 모델 그대로       easy 안 잃음, hard 안 얻음
#     배율 1.0  학습한 모델 그대로     easy 많이 잃음, hard 많이 얻음
#
# **다시 학습하지 않음.** 두 모델이 디스크에 있으면 가중치를 섞기만 하면 되고, 배율마다
# 관문 1(`embed-check`)을 돌려 easy 와 hard 가 어떻게 맞바뀌는지 표로 볼 수 있음.
#
# ## 2026-08-25 결정: 이 도구는 남기되 쓰지 않음
#
# 배율을 관측 결과에 맞춰 고르는 것은 **사후 조정**이라, 개발용에서 좋아 보이는 배율이
# 시험용에서도 맞을 보장이 없음. 고칠 자리는 배율이 아니라 학습 자료였음 - easy 질문을
# 같은 방법으로 만들어 넣고 다시 학습하는 쪽을 골랐음(ISSUE #51).
#
# 남겨 두는 이유: easy 를 넣어 다시 학습해도 잊어버림이 남으면 그때 마지막 수단이 됨.


def cmd_blend(args) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"원래 모델: {args.base}")
    base = SentenceTransformer(args.base)
    print(f"학습한 모델: {args.tuned}")
    tuned = SentenceTransformer(args.tuned)

    sb = base[0].auto_model.state_dict()
    st = tuned[0].auto_model.state_dict()
    only_base, only_tuned = set(sb) - set(st), set(st) - set(sb)
    if only_base or only_tuned:
        raise RuntimeError(f"두 모델의 가중치 이름이 다르다 "
                           f"(원래에만 {len(only_base)}개, 학습한 쪽에만 {len(only_tuned)}개)")

    a = float(args.scale)
    n_changed, max_delta = 0, 0.0
    blended = {}
    for k in sb:
        vb, vt = sb[k], st[k]
        if vb.dtype.is_floating_point and vb.shape == vt.shape:
            d = (vt.float() - vb.float())
            if d.abs().max().item() > 0:
                n_changed += 1
                max_delta = max(max_delta, d.abs().max().item())
            blended[k] = (vb.float() + a * d).to(vb.dtype)
        else:
            blended[k] = vt
    print(f"가중치 {len(sb):,}개 중 달라진 것 {n_changed:,}개 · 최대 변화량 {max_delta:.5f}")
    if not n_changed:
        raise RuntimeError("두 모델의 가중치가 완전히 같다 - 학습이 안 반영된 모델이다")

    base[0].auto_model.load_state_dict(blended)
    base.max_seq_length = tuned.max_seq_length
    base.save(args.out)
    print(f"배율 {a} 로 섞어 저장: {args.out}")


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="학습 (변환기: sft -> dpo · 검색 모델: embed)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sft", help="1단계: 지도 미세조정")
    s.add_argument("--data", default="data/training/train_query_translator_sft.jsonl")
    s.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    s.add_argument("--output-dir", default="models/query-translator-sft")
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
                   help="4비트 양자화 켜기. 기본은 끔 - 16GB VRAM 에서 bf16(약 9.7GB)이 "
                        "충분히 들어가고, 양자화는 성능을 깎기 때문. 메모리가 부족할 때만")
    s.set_defaults(func=cmd_sft)

    d = sub.add_parser("dpo", help="2단계: 선호 학습 (SFT 어댑터 위에)")
    d.add_argument("--data", default="data/training/train_query_translator_dpo.jsonl")
    d.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    d.add_argument("--sft-adapter", default="models/query-translator-sft/checkpoint-54",
                   help="SFT로 학습한 LoRA 어댑터. 그 위에 이어서 DPO 학습한다")
    d.add_argument("--output-dir", default="models/query-translator-dpo")
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

    e = sub.add_parser("embed", help="검색 모델(임베딩) 미세조정 - 1차 검색을 고치는 것")
    e.add_argument("--data", default="data/training/train_retriever.jsonl")
    e.add_argument("--base-model", default="BAAI/bge-m3",
                   help="지금 색인을 만든 모델과 같아야 한다")
    e.add_argument("--output-dir", default="models/retriever-ft",
                   help="기본값은 지금 있는 모델을 덮어씀. 옛것을 남기려면 다른 이름을 줄 것")
    e.add_argument("--epochs", type=float, default=1.0,
                   help="대조 학습은 문항이 2만 개대면 1회로도 충분한 것이 보통")
    e.add_argument("--batch-size", type=int, default=8,
                   help="문항 하나가 글 8개(질문+정답+오답 6)를 통과하므로 크게 잡으면 넘침")
    e.add_argument("--grad-accum", type=int, default=1)
    e.add_argument("--lr", type=float, default=1e-4, help="보조 행렬 학습의 통상값")
    e.add_argument("--lora-r", type=int, default=32)
    e.add_argument("--negatives", type=int, default=6, help="문항당 쓸 오답 편수")
    e.add_argument("--max-len", type=int, default=512,
                   help="색인을 만든 값과 반드시 같아야 한다")
    e.add_argument("--grad-checkpoint", action="store_true",
                   help="메모리를 아끼는 대신 느려짐. 묶음 크기를 못 키울 때만")
    e.add_argument("--logging-steps", type=int, default=50)
    e.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (속도 재기용)")
    e.add_argument("--seed", type=int, default=42)
    e.set_defaults(func=cmd_embed)

    r = sub.add_parser("rerank", help="재정렬기 미세조정 - 2단계(줄 세우기)를 고치는 것")
    r.add_argument("--data", default="data/training/train_reranker.jsonl")
    r.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3",
                   help="서비스가 쓰는 재정렬기. 모델 교체는 ISSUE #41 에서 반증됐으므로 바꾸지 말 것")
    r.add_argument("--output-dir", default="models/reranker-ft",
                   help="기본값은 지금 있는 모델을 덮어씀. ISSUE #55 의 근거 파일이므로 "
                        "손실을 바꿔 다시 학습할 때는 다른 이름을 줄 것")
    r.add_argument("--loss", choices=["mnrl", "ranknet"], default="mnrl",
                   help="mnrl 은 지금까지 쓰던 CachedMultipleNegativesRankingLoss "
                        "(묶음 안 다른 질문의 글도 오답으로 씀). ranknet 은 RankNetLoss "
                        "(그 질문에 딸린 글끼리만 견줌). 두 손실 모두 점수의 절대값은 "
                        "붙잡지 않음 - 자세한 것은 ISSUE #55 참고")
    r.add_argument("--negatives", type=int, default=4,
                   help="문항당 쓸 어려운 오답(정답을 뺀 검색 상위) 편수")
    r.add_argument("--in-batch-negatives", type=int, default=4,
                   help="묶음 안에서 뽑아 쓸 오답 수 (손실이 알아서 뽑음)")
    r.add_argument("--epochs", type=float, default=1.0)
    r.add_argument("--batch-size", type=int, default=32,
                   help="이 손실은 묶음이 클수록 오답이 많아져 학습이 세짐")
    r.add_argument("--mini-batch-size", type=int, default=16,
                   help="한 번에 모델을 통과시킬 쌍의 수. 그래픽카드 메모리를 정하는 값")
    r.add_argument("--lr", type=float, default=1e-4, help="보조 행렬 학습의 통상값")
    r.add_argument("--lora-r", type=int, default=32)
    r.add_argument("--max-len", type=int, default=512,
                   help="서비스의 CrossEncoderReranker 와 같은 값이어야 함")
    r.add_argument("--grad-checkpoint", action="store_true",
                   help="메모리가 모자랄 때. 학습 내용은 그대로이고 느려짐 (ISSUE #52)")
    r.add_argument("--logging-steps", type=int, default=50)
    r.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (속도 재기용)")
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(func=cmd_rerank)

    c = sub.add_parser("embed-check", help="관문 1: 미세조정이 정답 등수를 올렸는지 값싸게 확인")
    c.add_argument("--model", nargs="+", default=["models/retriever-ft"],
                   help="여러 개를 주면 같은 부분집합에서 나란히 견줌 (배율 고를 때 씀)")
    c.add_argument("--pairs", default="data/training/val_retriever.jsonl",
                   help="학습에서 뺀 검증용 문항")
    c.add_argument("--queries", default="data/eval/dev.jsonl")
    c.add_argument("--corpus", default="data/corpus/corpus-cs2021.jsonl")
    c.add_argument("--index", default="data/embeddings/cs2021")
    c.add_argument("--depth", type=int, default=100,
                   help="문항마다 후보를 몇 편까지 부분집합에 넣을지")
    c.add_argument("--val-sample", type=int, default=500,
                   help="검증용 문항을 몇 개만 쓸지 (전부 쓰면 다시 임베딩할 논문이 너무 많음)")
    c.add_argument("--max-len", type=int, default=512)
    c.add_argument("--batch-size", type=int, default=250)
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_embed_check)

    b = sub.add_parser("blend", help="원래 모델과 학습한 모델을 배율로 섞음 (다시 학습 안 함)")
    b.add_argument("--base", default="BAAI/bge-m3")
    b.add_argument("--tuned", default="models/retriever-ft")
    b.add_argument("--scale", type=float, required=True,
                   help="0 이면 원래 모델, 1 이면 학습한 모델. 사이 값이 중간 지점")
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_blend)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
