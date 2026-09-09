"""쿼리 변환기 학습 - 지도 파인튜닝과 선호 학습, 그리고 검색 모델, 재정렬기 파인튜닝.

    # 1단계: "이런 검색어를 만들어라" 를 흉내 내게 함
    python -m training.train sft --data data/training/train_query_translator_sft.jsonl \\
        --output-dir models/query-translator-sft --epochs 8

    # 2단계: 1단계 어댑터 위에 "좋은 것과 나쁜 것의 차이" 를 얹음
    python -m training.train dpo --data data/training/train_query_translator_dpo.jsonl \\
        --sft-adapter models/query-translator-sft/checkpoint-54 \\
        --output-dir models/query-translator-dpo

    # 검색 모델 파인튜닝과 그 결과 확인
    python -m training.train embed --data data/training/train_retriever.jsonl
    python -m training.train embed-check --model models/retriever-ft

    # 재정렬기 파인튜닝
    python -m training.train rerank --data data/training/train_reranker.jsonl

순서가 중요함. 지도 파인튜닝 다음에 선호 학습이고, 선호 학습은 앞 단계의 어댑터 위에
이어서 함. 서비스가 쓰는 것은 2단계까지 끝낸 `models/query-translator-dpo` 임.

## 무엇을 학습하나 (쿼리 변환기)

    입력  사용자의 일상어 질문
    출력  그 질문의 정답 논문을 arXiv 에서 실제로 찾아낸 검색어

라벨은 사람이 고른 것이 아니라 `training/build_translator_pairs.py` 가 검색 성공 여부로
뽑아 놓은 것임. "학술적으로 그럴싸한 말" 이 아니라 "실제로 통하는 말" 을 배움.

## 왜 LoRA 인가

Qwen3-4B 는 값이 40억 개라 전부 학습시키려면 메모리가 매우 많이 필요함. LoRA 는 원래
모델을 고정하고 작은 LoRA 층(전체의 1% 미만)만 새로 학습해 끼우는 방식이라 16GB
그래픽카드로 돌아가고, 결과물도 수십 메가바이트로 작음.

전제: pip install transformers peft trl bitsandbytes accelerate datasets
"""

from __future__ import annotations

import argparse
import json
import random
import time

# 학습에 쓰는 지시문. 두 단계가 반드시 같아야 하고 서비스가 쓰는 것과도 같아야 함.
# 형식이 다르면 앞서 배운 것이 흐트러짐
INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)


def read_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ==========================================================================
# 1단계. 지도 파인튜닝
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

    # LoRA: 주의(attention)와 피드포워드 층에만 작은 LoRA 층을 붙여 학습함
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
    print(f"\n지도 파인튜닝 완료. 어댑터 저장: {args.output_dir}")
    print(f"다음 단계: 검증 손실이 가장 낮은 체크포인트를 골라 선호 학습으로 넘긴다.\n"
          f"  $PY -m training.train dpo --sft-adapter {args.output_dir}/checkpoint-<번호>")


# ==========================================================================
# 2단계. 선호 학습
# ==========================================================================

def load_preference_dataset(path: str):
    """선호 학습 형식으로 불러온다: prompt / chosen / rejected."""
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
    print(f"앞 단계 어댑터: {args.sft_adapter}")
    print("양자화 없음 - bf16 전체 정밀도")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto"
    )
    # 앞 단계 어댑터를 얹고, 그 위에서 이어서 학습할 수 있도록 열어 둠
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
    print(f"\n선호 학습 완료. 어댑터 저장: {args.output_dir}")
    print("서비스와 평가에서 쓰려면 변환기 이름을 'dpo' 로 부르면 된다.")



# ==========================================================================
# 검색 모델(임베딩) 파인튜닝
# ==========================================================================
#
# 무엇을 학습하나:
#
#     질문        "조건이 많은 문제를 아주 적은 메모리로 대충 잘 푸는 방법이 있나"
#     정답 논문    그 질문을 만들어 낸 논문의 제목 + 초록      <- 가깝게
#     오답 논문 6편  재정렬기가 무관하다고 한 논문             <- 멀게
#
# 학습 쌍은 `training/build_retrieval_pairs.py` 가 만듦. 오답 고르는 규칙이 학습의 성패를
# 가르므로 그 파일 설명글을 반드시 읽을 것.
#
# 반드시 지킬 것 네 가지.
# 1. 바탕 모델은 지금 색인을 만든 것과 같아야 함(`BAAI/bge-m3`). 다르면 학습한 것과
#    색인이 어긋남
# 2. 최대 길이 512. `local_index.build_embeddings` 가 512 로 색인을 만들었음
# 3. 저장은 어댑터를 합쳐서 함. `LocalDenseRetriever` 는 `SentenceTransformer(경로)` 로
#    올리므로 LoRA 만 저장하면 못 읽음
# 4. 학습이 끝나면 색인을 다른 이름으로 새로 만들 것. 같은 이름을 주면 기존 임베딩
#    2.93GB 를 덮어씀

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

    # 큰 모델은 고정하고 작은 LoRA 층만 학습함. 주의 층과 완전연결 층에 붙임.
    #
    # `SentenceTransformer.add_adapter` 가 아니라 `get_peft_model` 로 감싸는 이유:
    # 앞의 것은 transformers 의 자체 연동을 써서 본체가 `XLMRobertaModel` 그대로 남는데,
    # 그러면 학습이 끝난 뒤 LoRA 를 기본 모델에 합칠 방법(`merge_and_unload`)이 없음.
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

    # LoRA 를 기본 모델에 합쳐서 저장함. 합치지 않으면 LocalDenseRetriever 가 못 읽음.
    n_merged = merge_lora_into(model)
    if not n_merged:
        raise RuntimeError("합칠 LoRA 층을 하나도 못 찾았다 - 학습이 안 붙은 것이다")
    print(f"LoRA {n_merged}개 층을 본체에 합침")
    model.save(args.output_dir)
    print(f"모델 저장: {args.output_dir}")
    print("다음 단계 - 색인을 **다른 이름으로** 새로 만들 것:")
    print(f"  $PY -m src.retrieval.local_index --model {args.output_dir} "
          f"--out data/embeddings/cs2021-ft")



# ==========================================================================
# 재정렬기 파인튜닝 - 2단계(줄 세우기)를 고치는 것
# ==========================================================================
#
# 무엇을 고치려는 것인가: 일상어 질문에서 정답도 10등도 전부 무관 구간에 있음(정답 점수
# 중앙 0.0035, 10등 0.0152). 정답을 오답보다 낮게 매긴 것이 아니라 쓸 만한 점수 범위
# 자체를 못 만듦. 고칠 것은 순서가 아니라 점수 범위임.
#
# 이분 라벨(관련 1 / 무관 0)을 안 쓰는 이유: 정답을 밀어낸 논문의 77.1% 가 등급 2 이상,
# 즉 실제로 쓸모 있는 논문임. "무관" 라벨을 붙이면 좋은 논문을 내리라고 가르치게 됨.
# 그래서 순서 손실을 씀 - "정답이 이 논문들보다 위" 까지만 가르침.
#
# 오답 편수가 문항마다 다른 것: 정답이 이미 1등이면 밀어낸 논문이 없어 오답이 0편임.
# `CrossEncoderTrainer` 는 `DatasetDict` 를 받으므로 오답 편수로 갈라 담음.
#
#     pairs           (질문, 정답)                     오답이 모자란 문항
#     hard_negatives  (질문, 정답, 오답1, ..., 오답N)    오답이 N편 이상인 문항
#
# 양쪽 다 같은 손실을 씀. `pairs` 쪽도 묶음 안 다른 질문의 논문이 자동으로 오답이 됨.
#
# 껍데기를 씌우지 않고 층만 끼워 넣는 이유: `get_peft_model` 로 껍데기를 씌우면 본체의
# `forward` 인자 이름이 `(*args, **kwargs)` 로 바뀜. `CrossEncoder` 는 인자 이름을 보고
# 부를 방식을 정하므로, 이름이 사라지면 토큰 묶음을 통째로 첫 자리에 넣어 학습이
# 시작되자마자 멈춤. `inject_adapter_in_model` 로 층만 제자리에 끼우면 본체가
# `XLMRobertaForSequenceClassification` 그대로 남음. 대신 원래 값은 직접 고정해야 함.
#
# 학습이 끝나면 `app.py` 의 `MIN_RERANK_SCORE` 를 반드시 다시 재야 함. 그 값은 재정렬
# 모델의 점수 범위에 딸린 것이고, 이번 학습의 목적이 바로 그 점수 범위를 바꾸는 것임.

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
    # 나빠짐 - 검색 모델에서 easy 를 빼고 겪었음
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
    # inject_adapter_in_model 은 원래 값을 고정해 주지 않음. 직접 고정함.
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
        raise RuntimeError("합칠 LoRA 층을 하나도 못 찾았다 - 학습이 안 붙은 것이다")
    print(f"LoRA {n_merged}개 층을 본체에 합침")
    model.save_pretrained(args.output_dir)
    print(f"모델 저장: {args.output_dir}")
    print("다음 단계 - 개발용에서 견줄 것 (색인은 다시 안 만들어도 됨):")
    print(f"  $PY -m evaluation.pipeline_eval --queries data/eval/dev.jsonl \\")
    print(f"      --channels local_dense local_hyde \\")
    print(f"      --reuse-queries runs/dev_service_repro.jsonl \\")
    print(f"      --index data/embeddings/cs2021-ft \\")
    print(f"      --embed-model models/retriever-ft \\")
    print(f"      --k 100 --rerank cross --rerank-depth 100 --out runs/dev_rr_ft.jsonl")
    print("그리고 app.py 의 MIN_RERANK_SCORE 를 다시 잴 것 (점수 범위가 바뀌었음)")


# ==========================================================================
# 점검 1 - 파인튜닝이 정답 등수를 끌어올렸는가 (부분집합에서 빠르게 확인)
# ==========================================================================
#
# 71만 편을 다시 임베딩하는 데 3시간이 걸림. 그 전에 "오르긴 하는가" 를 빠르게 걸러냄.
# 부분집합은 정답 논문과 지금 색인이 데려온 상위 후보로 만듦.
#
# 이 점검은 그만둘지 판정할 때만 씀. 부분집합에는 파인튜닝한 모델이 새로 끌어올릴 엉뚱한 논문이
# 없어서 여기 값은 실제보다 좋게 나옴. 한 방향으로만 믿을 수 있음.
#
#     여기서 안 오름  ->  71만 편에서도 안 오름. 재색인하지 말고 접을 것
#     여기서 오름     ->  아직 모름. 재색인해서 점검 2 에서 판정할 것
#
# 두 무리를 나눠 보는 이유: 학습 자료는 코퍼스 비율대로 뽑아 cs 계열이 45.9% 인데, 개발용
# 평가셋은 분야를 고르게 뽑아 그 넷이 2.9% 뿐임. 검증용에서는 오르는데 개발용에서만 안
# 오르면 방법이 안 되는 것이 아니라 분야가 안 맞는 것임.
#
#     검증용   학습에서 뺀 논문 500편의 문항. 학습 자료와 같은 분야 분포
#     개발용   data/eval/dev.jsonl. 평가셋 분포

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

    # -- 파인튜닝 전 등수: 이미 있는 임베딩을 그대로 쓰므로 정확하고 공짜 ------
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
        print(f"\n{'=' * 74}\n파인튜닝 모델 불러오는 중: {model_path}", flush=True)
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

    _row("(파인튜닝 전)", ranks_before)
    for name, ranks in summary:
        _row(name, ranks)

    print("\n[점검 1 판정] 부분집합 안에서 잰 값이라 그만둘지 판정할 때만 씀")
    print("  기준: 검증용 등수 중앙값이 내려가면 재색인으로 넘어가고, "
          "안 내려가면 여기서 접음")



# ==========================================================================
# 두 모델 섞기 - 원래 모델과 학습한 모델의 중간 지점 만들기
# ==========================================================================
#
# 파인튜닝이 일상어 층은 크게 올렸는데 정확한 학술어 층을 떨어뜨렸음(easy 0.836 -> 0.638,
# hard 0.250 -> 0.776). 원인은 학습 자료에 easy 를 안 넣은 것임.
#
# LoRA 학습은 원래 가중치에 변화량을 더하는 방식이라, 그 변화량에 배율을 곱하면
# 중간 지점이 나옴. 다시 학습하지 않고 배율마다 점검 1 을 돌려 볼 수 있음.
#
#     섞은 모델 = 원래 모델 + 배율 x (학습한 모델 - 원래 모델)
#
# 이 도구는 남기되 쓰지 않기로 했음. 배율을 관측 결과에 맞춰 고르는 것은 사후 조정이라
# 개발용에서 좋아 보이는 배율이 시험용에서도 맞을 보장이 없음. 고칠 자리는 배율이 아니라
# 학습 자료였음.

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

    s = sub.add_parser("sft", help="1단계: 지도 파인튜닝")
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
                   help="LoRA 크기. 메모리 여유가 있으니 32로 표현력 확보")
    s.add_argument("--max-len", type=int, default=512,
                   help="실제 데이터가 최대 424글자라 512로 충분(길면 메모리만 낭비)")
    s.add_argument("--val-ratio", type=float, default=0.15, help="검증용으로 뗄 비율")
    s.add_argument("--use-4bit", action="store_true",
                   help="4비트 양자화 켜기. 기본은 끔 - 그래픽 메모리 16GB 에서 약 9.7GB 로 "
                        "충분히 들어가고, 양자화는 성능을 깎기 때문. 메모리가 부족할 때만")
    s.set_defaults(func=cmd_sft)

    d = sub.add_parser("dpo", help="2단계: 선호 학습 (앞 단계 어댑터 위에)")
    d.add_argument("--data", default="data/training/train_query_translator_dpo.jsonl")
    d.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    d.add_argument("--sft-adapter", default="models/query-translator-sft/checkpoint-54",
                   help="앞 단계에서 학습한 어댑터. 그 위에 이어서 선호 학습을 한다")
    d.add_argument("--output-dir", default="models/query-translator-dpo")
    d.add_argument("--epochs", type=int, default=2,
                   help="선호 학습은 앞 단계보다 적은 에폭으로 충분. 과하면 성능이 무너짐")
    d.add_argument("--batch-size", type=int, default=2)
    d.add_argument("--lr", type=float, default=5e-6,
                   help="선호 학습은 훨씬 낮은 학습률을 쓴다(1e-6~1e-5). 크면 모델이 무너짐")
    d.add_argument("--beta", type=float, default=0.1,
                   help="원본 모델에서 얼마나 벗어날지 조절. 작을수록 자유롭게 변함")
    d.add_argument("--max-len", type=int, default=512)
    d.add_argument("--val-ratio", type=float, default=0.15)
    d.set_defaults(func=cmd_dpo)

    e = sub.add_parser("embed", help="검색 모델(임베딩) 파인튜닝 - 1차 검색을 고치는 것")
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
    e.add_argument("--lr", type=float, default=1e-4, help="LoRA 학습률 통상값")
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

    r = sub.add_parser("rerank", help="재정렬기 파인튜닝 - 2단계(줄 세우기)를 고치는 것")
    r.add_argument("--data", default="data/training/train_reranker.jsonl")
    r.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3",
                   help="서비스가 쓰는 재정렬기. 모델을 바꿔도 나아지지 않는 것을 확인했음")
    r.add_argument("--output-dir", default="models/reranker-ft",
                   help="기본값은 지금 있는 모델을 덮어씀. 손실을 바꿔 다시 학습할 때는 "
                        "옛 결과를 지우지 않도록 다른 이름을 줄 것")
    r.add_argument("--loss", choices=["mnrl", "ranknet"], default="mnrl",
                   help="mnrl 은 지금까지 쓰던 CachedMultipleNegativesRankingLoss "
                        "(묶음 안 다른 질문의 글도 오답으로 씀). ranknet 은 RankNetLoss "
                        "(그 질문에 딸린 글끼리만 견줌). 두 손실 모두 점수의 절대값은 "
                        "붙잡지 않음. 순서는 가르쳐도 '무관한 것은 0 에 가깝게' 는 못 가르침")
    r.add_argument("--negatives", type=int, default=4,
                   help="문항당 쓸 어려운 오답(정답을 뺀 검색 상위) 편수")
    r.add_argument("--in-batch-negatives", type=int, default=4,
                   help="묶음 안에서 뽑아 쓸 오답 수 (손실이 알아서 뽑음)")
    r.add_argument("--epochs", type=float, default=1.0)
    r.add_argument("--batch-size", type=int, default=32,
                   help="이 손실은 묶음이 클수록 오답이 많아져 학습이 세짐")
    r.add_argument("--mini-batch-size", type=int, default=16,
                   help="한 번에 모델을 통과시킬 쌍의 수. 그래픽카드 메모리를 정하는 값")
    r.add_argument("--lr", type=float, default=1e-4, help="LoRA 학습률 통상값")
    r.add_argument("--lora-r", type=int, default=32)
    r.add_argument("--max-len", type=int, default=512,
                   help="서비스의 CrossEncoderReranker 와 같은 값이어야 함")
    r.add_argument("--grad-checkpoint", action="store_true",
                   help="메모리가 모자랄 때. 학습 내용은 그대로이고 느려짐")
    r.add_argument("--logging-steps", type=int, default=50)
    r.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (속도 재기용)")
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(func=cmd_rerank)

    c = sub.add_parser("embed-check", help="점검 1: 파인튜닝이 정답 등수를 올렸는지 빠르게 확인")
    c.add_argument("--model", nargs="+", default=["models/retriever-ft"],
                   help="여러 개를 주면 같은 부분집합에서 나란히 견줌 (배율 고를 때 씀)")
    c.add_argument("--pairs", default="data/training/val_retriever.jsonl",
                   help="학습에서 뺀 검증용 문항")
    c.add_argument("--queries", default="data/eval/dev.jsonl")
    c.add_argument("--corpus", default="data/corpus/corpus-cs2021.jsonl")
    c.add_argument("--index", default="data/embeddings/cs2021-ft")
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
