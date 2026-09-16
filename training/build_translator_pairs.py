"""
쿼리 변환기 학습 데이터셋 만들기 - 실제로 논문을 찾아내는 검색어를 정답 라벨로

변환기의 문제는 지식 부족이 아니라 어떤 표현이 arXiv에서 실제로 통하는지 모른다는 것
그래서 사람이 좋아 보인다고 고른 라벨을 쓰지 않고 arXiv 검색 결과를 정답 신호로 씀

    질문 Q (정답 논문 P가 정해져 있음)
      -> 변환기가 후보 검색어를 N개 생성 (온도를 높여 서로 다르게)
      -> 각 후보로 실제 arXiv 검색
      -> 정답 논문 P를 가장 높은 순위로 찾아낸 후보 = 그 질문의 정답 라벨
      -> 하나도 못 찾으면 그 질문은 학습에서 제외

실패한 후보도 버리지 않고 선호 학습용 쌍으로 함께 저장함.

출력 (data/training/):
  train_query_translator_sft.jsonl   {"input": 질문, "output": 가장 잘 찾은 검색어}
  train_query_translator_dpo.jsonl   {"input": 질문, "chosen": ..., "rejected": ...}
  candidates.jsonl                   모든 후보와 점수
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.retrieval.arxiv_live import ArxivLiveRetriever
from src.retrieval.corpus import normalize_paper_id as normalize_id
from src.rewriter.base import OllamaClient
from src.rewriter.baselines import (
    OUTPUT_SCHEMA, SYSTEM, HierarchicalRewriter, build_arxiv_query, build_messages
)
from src.utils import read_jsonl, write_jsonl

OUT_DIR = config.DATA_DIR / "training"
CACHE_PATH = config.DATA_DIR / "cache" / "arxiv_search_cache.jsonl"

def score_query(query: str, gold_id: str, retriever, k: int = 30) -> tuple[float, int | None]:
    """후보 쿼리로 검색해 '정답 논문이 몇 등인가'로 점수를 측정함"""
    try:
        results = retriever.search(query, k=k)
    except Exception:
        return -1.0, None
    ids = [normalize_id(r.paper_id) for r in results]
    gold = normalize_id(gold_id)
    if gold in ids:
        rank = ids.index(gold) + 1
        return 1.0 / rank, rank
    return 0.0, None


def generate_candidates(rewriter: HierarchicalRewriter, question: str,
                        n: int, temperature: float) -> list[str]:
    """같은 질문에 대해 서로 다른 변환 후보를 n개 만듦"""
    seen, candidates = set(), []
    for i in range(n):
        temp = 0.0 if i == 0 else temperature
        try:
            data = rewriter.client.generate_json(
                build_messages(question), OUTPUT_SCHEMA, system=SYSTEM, temperature=temp
            )
            terms = list(data.get("academic_terms", []))
            q = build_arxiv_query(question, terms)
        except Exception:
            continue
        if q and q not in seen:
            seen.add(q)
            candidates.append(q)
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/eval/dev.jsonl")
    ap.add_argument("--n-candidates", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    rows = list(read_jsonl(args.queries))
    if args.limit:
        rows = rows[: args.limit]

    rewriter = HierarchicalRewriter(OllamaClient())
    retriever = ArxivLiveRetriever(cache_path=CACHE_PATH)

    sft, dpo, cand_log = [], [], []
    n_used = n_skipped = 0

    for i, row in enumerate(rows, 1):
        question, gold = row["text"], row["gold_id"]
        candidates = generate_candidates(rewriter, question, args.n_candidates,
                                         args.temperature)
        scored = []
        for q in candidates:
            s, rank = score_query(q, gold, retriever, args.k)
            scored.append({"query": q, "score": s, "gold_rank": rank})

        cand_log.append({"query_id": row["query_id"], "question": question,
                         "gold_id": gold, "candidates": scored})

        valid = [c for c in scored if c["score"] >= 0]
        best = max(valid, key=lambda c: c["score"], default=None)

        if best and best["score"] > 0:
            sft.append({"input": question, "output": best["query"],
                        "gold_rank": best["gold_rank"]})
            for c in valid:
                if c["score"] == 0:
                    dpo.append({"input": question, "chosen": best["query"],
                                "rejected": c["query"]})
            n_used += 1
        else:
            n_skipped += 1
    out = Path(args.out_dir)
    write_jsonl(out / "train_query_translator_sft.jsonl", sft)
    write_jsonl(out / "train_query_translator_dpo.jsonl", dpo)
    write_jsonl(out / "candidates.jsonl", cand_log)

if __name__ == "__main__":
    main()
