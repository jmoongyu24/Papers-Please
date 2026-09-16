"""
쿼리 변환기 학습 - 지도 파인튜닝, 선호 학습
검색 모델, 재정렬기 파인튜닝

    # 1단계: "이런 검색어를 만들어라"를 흉내 내게 함
    python -m training.train sft --data data/training/train_query_translator_sft.jsonl \\
        --output-dir models/query-translator-sft --epochs 8

    # 2단계: 1단계 어댑터 위에 "좋은 것과 나쁜 것의 차이"를 얹음
    python -m training.train dpo --data data/training/train_query_translator_dpo.jsonl \\
        --sft-adapter models/query-translator-sft/checkpoint-54 \\
        --output-dir models/query-translator-dpo

    # 검색 모델 파인튜닝과 그 결과 확인
    python -m training.train embed --data data/training/train_retriever.jsonl
    python -m training.train embed-check --model models/retriever-ft

    # 재정렬기 파인튜닝
    python -m training.train rerank --data data/training/train_reranker.jsonl

지도 파인튜닝 다음에 선호 학습이고, 선호 학습은 앞 단계의 어댑터 위에 이어서 함
서비스가 쓰는 것은 2단계까지 끝낸 'models/query-translator-dpo'
"""

from __future__ import annotations

import argparse
import json
import random

INSTRUCTION = (
    "사용자의 검색어를 arXiv에서 관련 논문을 잘 찾아내는 검색 쿼리로 변환하라. "
    "결과 쿼리만 출력한다."
)


def read_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def format_example(row: dict) -> dict:
    """학습 예시 한 건을 대화 형식으로 만듦"""
    return {
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def split_by_paper(path: str, val_ratio: float, seed: int = 42):
    """논문 단위로 학습/검증을 나눔

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
    return (Dataset.from_list([format_example(r) for r in train_rows]),
            Dataset.from_list([format_example(r) for r in val_rows]))


def cmd_sft(args) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer


    # 베이스 모델은 양자화 없이 bf16으로 학습
    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    # LoRA: attention과 피드포워드 층에만 작은 LoRA 층을 붙여 학습함
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
    )

    train_ds, eval_ds = split_by_paper(args.data, args.val_ratio)

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
            eval_strategy="epoch",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            bf16=True,
            report_to="none"
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model(args.output_dir)

# DPO
def load_preference_dataset(path: str):
    """선호 학습 형식 불러옴"""
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


    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto"
    )

    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)

    dataset = load_preference_dataset(args.data)
    split = dataset.train_test_split(test_size=args.val_ratio, seed=42)

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
            report_to="none"
        ),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model(args.output_dir)


"""
검색 모델(임베딩) 파인튜닝

학습하는 것:
    질문        "조건이 많은 문제를 아주 적은 메모리로 대충 잘 푸는 방법이 있나"
    정답 논문    그 질문을 만들어 낸 논문의 제목 + 초록      <- 가깝게
    오답 논문 6편  재정렬기가 무관하다고 한 논문             <- 멀게
"""
def merge_lora_into(model) -> int:
    """LoRA로 학습한 값을 원래 가중치에 더해 한 덩어리로 만들고, LoRA 층을 떼어냄

    학습 중에는 원래 층 옆에 LoRA 층이 따로 붙어 있음. 이대로 저장하면 어댑터가 있어야만
    쓸 수 있으므로, 값을 원래 가중치에 더하고 LoRA 층을 치워 단독 모델로 만듦. 합친 층 수를 리턴
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
    n_neg = min(min(len(r["negatives"]) for r in rows), args.negatives)

    cols = {"anchor": [r["query"] for r in rows],
            "positive": [r["positive"] for r in rows]}
    for j in range(n_neg):
        cols[f"negative_{j + 1}"] = [r["negatives"][j] for r in rows]
    train_ds = Dataset.from_dict(cols)

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_len

    from peft import get_peft_model
    model[0].auto_model = get_peft_model(model[0].auto_model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"], bias="none"))

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
        seed=args.seed
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs,
                                         train_dataset=train_ds, loss=loss)

    trainer.train()

    n_merged = merge_lora_into(model)
    if not n_merged:
        raise RuntimeError("합칠 LoRA 층을 하나도 못 찾았다 - 학습이 안 붙은 것이다")
    model.save(args.output_dir)




# 재정렬기 파인튜닝
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

    for r in rows:
        if "docs" not in r:
            raise SystemExit(
                f"{args.data}에 `docs` 열이 없다. 2026-08-28 이전 형식으로 보인다.\n"
                f"  build_retrieval_pairs.py --for-rerank로 다시 만들 것.")
    n_neg = min(min(len(r["labels"]) - 1 for r in rows), args.negatives)
    if n_neg == 0:
        raise SystemExit("오답이 0편인 문항이 있다. 이 자료로는 순서를 가르칠 수 없다.")

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

    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_len)

    inject_adapter_in_model(LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"], bias="none"), model.model)
    for name, prm in model.model.named_parameters():
        prm.requires_grad = "lora_" in name

    if args.loss == "ranknet":
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
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=args.seed
    )
    trainer = CrossEncoderTrainer(
        model=model, args=targs,
        train_dataset=DatasetDict(subsets) if len(subsets) > 1 else next(iter(subsets.values())),
        loss=loss)

    trainer.train()

    n_merged = merge_lora_into(model.model)
    if not n_merged:
        raise RuntimeError("합칠 LoRA 층을 하나도 못 찾았다 - 학습이 안 붙은 것이다")
    model.save_pretrained(args.output_dir)


def _rank_of(sub_emb, q_vec, gold_row: int) -> int:
    """부분집합 안에서 정답이 몇 등인지 (1등은 1)"""
    scores = sub_emb @ q_vec
    return int((scores > scores[gold_row]).sum()) + 1


def _report_ranks(name: str, ranks_before: list, ranks_after: list, groups: list) -> None:
    import numpy as np

    print(f"\n[{name}]")
    print(f"  {'구분':<12}{'문항':>7}{'등수 중앙값':>22}{'10등 안':>18}{'100등 안':>18}")
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

    from src.retrieval.corpus import normalize_paper_id
    from src.retrieval.local_index import LocalDenseRetriever
    from src.utils import read_jsonl
    from training.build_retrieval_pairs import doc_text, score_batch

    val = [r for r in read_rows(args.pairs)]
    if args.val_sample and args.val_sample < len(val):
        val = random.Random(args.seed).sample(val, args.val_sample)
    dev = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    items = ([{"text": r["query"], "gold_id": r["gold_id"],
               "group": "검증 " + r["difficulty"], "set": "val"} for r in val]
             + [{"text": r["text"], "gold_id": r["gold_id"],
                 "group": "개발 " + r["difficulty"], "set": "dev"} for r in dev])

    ret = LocalDenseRetriever(args.corpus, args.index)
    items = [it for it in items if normalize_paper_id(it["gold_id"]) in ret._pos]
    gold_pos = [ret._pos[normalize_paper_id(it["gold_id"])] for it in items]

    top_idx, _, _ = score_batch(ret, [it["text"] for it in items], args.depth, gold_pos,
                                batch_size=args.batch_size)

    subset = sorted(set(gold_pos) | {int(x) for row in top_idx for x in row})
    row_of = {p: i for i, p in enumerate(subset)}

    sub_before = np.asarray(ret.emb[subset], dtype=np.float32)
    qb = ret.embedder.encode([it["text"] for it in items], normalize_embeddings=True,
                             convert_to_numpy=True, batch_size=64).astype(np.float32)
    ranks_before = [_rank_of(sub_before, qb[i], row_of[gold_pos[i]])
                    for i in range(len(items))]
    del sub_before, qb

    texts = [doc_text(r) for r in ret.read_rows(subset)]
    del ret.emb                                   # 2.93GB를 비움

    import torch

    summary = []
    for model_path in args.model:
        kw = {"model_kwargs": {"dtype": torch.float16}} if torch.cuda.is_available() else {}
        ft = SentenceTransformer(model_path, **kw)
        ft.max_seq_length = args.max_len

        sub_after = ft.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                              batch_size=64, show_progress_bar=False).astype(np.float32)
        qa = ft.encode([it["text"] for it in items], normalize_embeddings=True,
                       convert_to_numpy=True, batch_size=64).astype(np.float32)
        ranks_after = [_rank_of(sub_after, qa[i], row_of[gold_pos[i]])
                       for i in range(len(items))]
        del sub_after, qa, ft
        torch.cuda.empty_cache()

        print(f"\n## {model_path}")
        for tag, which in (("검증용 - 학습 자료와 같은 분야 분포", "val"),
                           ("개발용 - 평가셋 분포", "dev")):
            sel = [i for i in range(len(items)) if items[i]["set"] == which]
            if sel:
                _report_ranks(tag, [ranks_before[i] for i in sel],
                              [ranks_after[i] for i in sel],
                              [items[i]["group"] for i in sel])
        summary.append((model_path, ranks_after))

    print(f"\n{'=' * 74}\n## 모델 견주기 - 개발용 {sum(1 for it in items if it['set'] == 'dev'):,}문항, 100등 안에 정답이 든 비율")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sft")
    s.add_argument("--data", default="data/training/train_query_translator_sft.jsonl")
    s.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    s.add_argument("--output-dir", default="models/query-translator-sft")
    s.add_argument("--epochs", type=int, default=8)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--grad-accum", type=int, default=1)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--lora-r", type=int, default=32)
    s.add_argument("--max-len", type=int, default=512)
    s.add_argument("--val-ratio", type=float, default=0.15)
    s.add_argument("--use-4bit", action="store_true")
    s.set_defaults(func=cmd_sft)

    d = sub.add_parser("dpo")
    d.add_argument("--data", default="data/training/train_query_translator_dpo.jsonl")
    d.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    d.add_argument("--sft-adapter", default="models/query-translator-sft/checkpoint-54")
    d.add_argument("--output-dir", default="models/query-translator-dpo")
    d.add_argument("--epochs", type=int, default=2)
    d.add_argument("--batch-size", type=int, default=2)
    d.add_argument("--lr", type=float, default=5e-6)
    d.add_argument("--beta", type=float, default=0.1)
    d.add_argument("--max-len", type=int, default=512)
    d.add_argument("--val-ratio", type=float, default=0.15)
    d.set_defaults(func=cmd_dpo)

    e = sub.add_parser("embed")
    e.add_argument("--data", default="data/training/train_retriever.jsonl")
    e.add_argument("--base-model", default="BAAI/bge-m3")
    e.add_argument("--output-dir", default="models/retriever-ft")
    e.add_argument("--epochs", type=float, default=1.0)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--grad-accum", type=int, default=1)
    e.add_argument("--lr", type=float, default=1e-4)
    e.add_argument("--lora-r", type=int, default=32)
    e.add_argument("--negatives", type=int, default=6)
    e.add_argument("--max-len", type=int, default=512)
    e.add_argument("--grad-checkpoint", action="store_true")
    e.add_argument("--logging-steps", type=int, default=50)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--seed", type=int, default=42)
    e.set_defaults(func=cmd_embed)

    r = sub.add_parser("rerank")
    r.add_argument("--data", default="data/training/train_reranker.jsonl")
    r.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    r.add_argument("--output-dir", default="models/reranker-ft")
    r.add_argument("--loss", choices=["mnrl", "ranknet"], default="mnrl")
    r.add_argument("--negatives", type=int, default=4)
    r.add_argument("--in-batch-negatives", type=int, default=4)
    r.add_argument("--epochs", type=float, default=1.0)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--mini-batch-size", type=int, default=16)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--lora-r", type=int, default=32)
    r.add_argument("--max-len", type=int, default=512)
    r.add_argument("--grad-checkpoint", action="store_true")
    r.add_argument("--logging-steps", type=int, default=50)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(func=cmd_rerank)

    c = sub.add_parser("embed-check")
    c.add_argument("--model", nargs="+", default=["models/retriever-ft"])
    c.add_argument("--pairs", default="data/training/val_retriever.jsonl")
    c.add_argument("--queries", default="data/eval/dev.jsonl")
    c.add_argument("--corpus", default="data/corpus/corpus-cs2021.jsonl")
    c.add_argument("--index", default="data/embeddings/cs2021-ft")
    c.add_argument("--depth", type=int, default=100)
    c.add_argument("--val-sample", type=int, default=500)
    c.add_argument("--max-len", type=int, default=512)
    c.add_argument("--batch-size", type=int, default=250)
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_embed_check)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
