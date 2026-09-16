"""
검색 모델과 재정렬기 파인튜닝용 학습 데이터쌍 만드는 코드
    질문        train_queries.jsonl의 한 문항
    정답 논문    그 문항을 만들어 낸 논문의 제목 + 초록
    오답 논문    "가깝긴 한데 정답은 아닌" 논문 여러 편
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src import config
from src.retrieval.corpus import normalize_paper_id
from src.utils import read_jsonl


def doc_text(row: dict) -> str:
    """제목과 초록을 임베딩할 때와 같은 방식으로 이어 붙임. 학습 자료와 색인의 글이 같아야 함"""
    return f"{row.get('title', '').strip()}\n{row.get('abstract', '').strip()}"


def score_batch(retriever, queries: list[str], top_k: int,
                gold_positions: list[int | None], batch_size: int = 256):
    """질문 여러 개를 한 번에 채점함

    Returns:
        top_idx    (질문 수, top_k)  상위 후보의 배열 위치
        top_score  (질문 수, top_k)  그 유사도
        gold_score (질문 수)        정답 논문의 유사도 (색인에 없으면 nan)
    """
    emb = retriever.emb
    n_q = len(queries)
    top_idx = np.zeros((n_q, top_k), dtype=np.int64)
    top_score = np.zeros((n_q, top_k), dtype=np.float32)
    gold_score = np.full(n_q, np.nan, dtype=np.float32)

    for s in range(0, n_q, batch_size):
        chunk = queries[s:s + batch_size]
        q = retriever.embedder.encode(chunk, normalize_embeddings=True,
                                      convert_to_numpy=True, batch_size=64)
        q = np.asarray(q, dtype=np.float32)
        scores = emb @ q.T                       # (논문 수, 이번 배치 질문 수)
        for j in range(len(chunk)):
            col = scores[:, j]
            part = np.argpartition(-col, top_k - 1)[:top_k]
            part = part[np.argsort(-col[part])]
            top_idx[s + j] = part
            top_score[s + j] = col[part]
            pos = gold_positions[s + j]
            if pos is not None:
                gold_score[s + j] = col[pos]
    return top_idx, top_score, gold_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/training/train_queries.jsonl")
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--index", default=str(config.DATA_DIR / "embeddings" / "cs2021-ft"))
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--out-dir", default="data/training")
    ap.add_argument("--negatives", type=int, default=6)
    ap.add_argument("--pool-from-en", action="store_true")
    ap.add_argument("--for-rerank", action="store_true")
    ap.add_argument("--neg-source", choices=["rerank", "retrieval"], default="rerank")
    ap.add_argument("--rerank-pool", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--val-papers", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    if args.limit:
        rows = rows[: args.limit]
    if args.sample and args.sample < len(rows):
        rows = random.Random(args.seed).sample(rows, args.sample)

    papers = sorted({normalize_paper_id(r["gold_id"]) for r in rows})
    rng = random.Random(args.seed)
    val_papers = set(rng.sample(papers, min(args.val_papers, len(papers))))

    from src.retrieval.local_index import LocalDenseRetriever, read_meta
    meta_model = read_meta(args.index).get("model")
    embed_model = args.embed_model or meta_model or config.EMBED_MODEL
    if meta_model and embed_model != meta_model:
        raise SystemExit(
            f"색인을 만든 모델과 질문을 찾을 모델이 다르다.\n"
            f"  색인({args.index})을 만든 모델: {meta_model}\n"
            f"  질문을 임베딩할 모델        : {embed_model}\n"
            f"이대로 쓰면 오류 없이 순위만 무너진다. --embed-model을 맞추거나 빼고 돌릴 것.")
    ret = LocalDenseRetriever(args.corpus, args.index, model_name=embed_model)

    rows = [r for r in rows if normalize_paper_id(r["gold_id"]) in ret._pos]

    gold_pos = [ret._pos[normalize_paper_id(r["gold_id"])] for r in rows]

    search_text = [r["text"] for r in rows]
    if args.pool_from_en:
        en_of = {r["pair_id"]: r["text"] for r in rows if r.get("lang") == "en"}
        for i, r in enumerate(rows):
            if r.get("lang") != "en" and r.get("pair_id") in en_of:
                search_text[i] = en_of[r["pair_id"]]

    top_idx, _, gold_score = score_batch(
        ret, search_text, args.top_k, gold_pos)

    rank_before = []
    for i in range(len(rows)):
        w = np.where(top_idx[i] == gold_pos[i])[0]
        rank_before.append(int(w[0]) + 1 if len(w) else None)

    if args.diagnose:
        return

    need = set(gold_pos)
    cheap = args.for_rerank and args.neg_source == "retrieval"
    n_pool = (args.negatives + 1) if cheap else (args.rerank_pool + 1)
    for i in range(len(rows)):
        need.update(int(x) for x in top_idx[i][: n_pool])
    need = sorted(need)
    body = {pos: doc_text(row) for pos, row in zip(need, ret.read_rows(need))}

    neg_pos: list[list[int]] = []
    neg_score: list[list[float]] = []
    gold_rr: list[float | None] = []
    gold_rr_rank: list[int | None] = []

    if cheap:
        for i in range(len(rows)):
            gp = gold_pos[i]
            neg_pos.append([int(x) for x in top_idx[i] if int(x) != gp][: args.negatives])
            neg_score.append([])
            gold_rr.append(None)
            gold_rr_rank.append(None)
        return_early = True
    else:
        return_early = False

    from src.retrieval.ranking import CrossEncoderReranker
    from src.schemas import ScoredPaper
    reranker = None if return_early else CrossEncoderReranker(batch_size=64)

    step = 200
    for s0 in ([] if return_early else range(0, len(rows), step)):
        chunk = list(range(s0, min(s0 + step, len(rows))))
        queries, cand_lists = [], []
        for i in chunk:
            gp = gold_pos[i]
            pos = [int(x) for x in top_idx[i] if int(x) != gp][: args.rerank_pool]
            if args.for_rerank:
                pos = pos + [gp]
            queries.append(rows[i]["text"])
            cand_lists.append([ScoredPaper(paper_id=str(p), score=0.0, rank=j + 1,
                                           title="", abstract=body[p])
                               for j, p in enumerate(pos)])
        ranked = reranker.rerank_batch(queries, cand_lists,
                                       top_k=args.rerank_pool + (1 if args.for_rerank else 0))
        for k, row_out in enumerate(ranked):
            scored = [(int(c.paper_id), float(c.score)) for c in row_out]
            if not args.for_rerank:
                pairs = sorted(scored, key=lambda t: t[1])
                neg_pos.append([p for p, _ in pairs[: args.negatives]])
                neg_score.append([round(sc, 6) for _, sc in pairs[: args.negatives]])
                continue

            gp = gold_pos[chunk[k]]
            gs = next((sc for pid, sc in scored if pid == gp), None)
            rank = next((j + 1 for j, (pid, _) in enumerate(scored) if pid == gp), None)
            others = [(pid, sc) for pid, sc in scored if pid != gp]
            neg_pos.append([pid for pid, _ in others[: args.negatives]])
            neg_score.append([round(sc, 6) for _, sc in others[: args.negatives]])
            gold_rr.append(None if gs is None else round(gs, 6))
            gold_rr_rank.append(rank)

    out_train, out_val = [], []
    for i, r in enumerate(rows):
        item = {
            "query_id": r["query_id"],
            "query": r["text"],
            "gold_id": normalize_paper_id(r["gold_id"]),
            "lang": r["lang"],
            "difficulty": r["difficulty"],
            "gold_score": round(float(gold_score[i]), 4),
            "rank_before": rank_before[i],
            "neg_rerank_scores": neg_score[i],
        }
        if args.for_rerank:
            item["docs"] = [body[gold_pos[i]]] + [body[p] for p in neg_pos[i]]
            item["labels"] = [1] + [0] * len(neg_pos[i])
            item["gold_rerank_score"] = gold_rr[i]
            item["gold_rerank_rank"] = gold_rr_rank[i]
        else:
            item["positive"] = body[gold_pos[i]]
            item["negatives"] = [body[p] for p in neg_pos[i]]
        (out_val if normalize_paper_id(r["gold_id"]) in val_papers
         else out_train).append(item)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    module = "reranker" if args.for_rerank else "retriever"
    for name, data in ((f"train_{module}.jsonl", out_train),
                       (f"val_{module}.jsonl", out_val)):
        fp = out_dir / name
        with open(fp, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
