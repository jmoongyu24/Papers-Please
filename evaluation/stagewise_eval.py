"""계단식(stage-wise) 검색 평가 — 계층 변환의 각 단계로 검색해 성능이 오르는지 본다.

사용자 구상: 계층 변환은 쿼리에 대한 CoT다. 의도 → 개념 → 전문 용어로 단계를 밟을수록
검색이 잘 되어야 한다. 이 스크립트는 그 '계단'이 실재하는지 직접 측정한다.

단계별 검색어(계층 변환 결과에서 구성):
  S0 raw      : 사용자 원본 질문 (변환 전, 바닥)
  S1 intent   : 의도를 다시 쓴 문장
  S2 concepts : 핵심 개념 나열
  S3 terms    : 전문 학술 용어 나열 (= 최종, 정점)

각 단계 검색어를 BM25(단어 일치)·dense(의미) 양쪽으로 검색해 Recall@10과 MRR@10을 잰다.
백엔드를 고정하고 검색어(단계)만 바꾸므로, "쿼리 정제가 검색을 개선하는가"가 순수하게 드러난다.

실행:
  python -m evaluation.stagewise_eval \
      --rewrites data/eval/_rw_hierarchical_train.jsonl \
      --queries data/eval/train.jsonl --corpus data/corpus/corpus-v1.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src import config
from src.retrieval.bm25_simple import BM25Retriever
from src.retrieval.corpus import load_corpus
from src.retrieval.dense import DenseRetriever
from evaluation import metrics


def build_stage_queries(row: dict) -> dict[str, str]:
    """한 질문의 계층 변환 결과에서 단계별 검색어 문자열을 만든다."""
    concepts = row.get("concepts") or []
    terms = row.get("academic_terms") or []
    return {
        "S0_raw": row["raw"],
        "S1_intent": (row.get("intent") or row["raw"]).strip(),
        "S2_concepts": " ".join(concepts) if concepts else row["raw"],
        "S3_terms": " ".join(terms) if terms else row.get("sparse", row["raw"]),
    }


STAGES = ["S0_raw", "S1_intent", "S2_concepts", "S3_terms"]


def run_stage(retriever, rewrites: list[dict], stage: str, k: int) -> dict[str, list[str]]:
    """한 단계 검색어로 모든 질문을 검색해 질문별 논문 순위를 만든다."""
    run = {}
    for row in rewrites:
        q = build_stage_queries(row)[stage]
        run[row["query_id"]] = [x.paper_id for x in retriever.search(q, k=k)]
    return run


def subset_recall(qrels, run, ids, k=10) -> float:
    sub = {qid: qrels[qid] for qid in ids}
    return float(np.mean(list(metrics.per_query_recall(sub, run, k).values())))


def main() -> None:
    ap = argparse.ArgumentParser(description="계단식 검색 평가")
    ap.add_argument("--rewrites", default="data/eval/_rw_hierarchical_train.jsonl")
    ap.add_argument("--queries", default="data/eval/train.jsonl")
    ap.add_argument("--corpus", default="data/corpus/corpus-v1.jsonl")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    rewrites = [json.loads(l) for l in open(args.rewrites)]
    queries = [json.loads(l) for l in open(args.queries)]
    qrels = {q["query_id"]: {q["gold_id"]: 1} for q in queries}
    lang_of = {q["query_id"]: q["lang"] for q in queries}
    ko = [qid for qid, lg in lang_of.items() if lg == "ko"]
    en = [qid for qid, lg in lang_of.items() if lg == "en"]

    papers = load_corpus(args.corpus)
    retrievers = {
        "bm25": BM25Retriever(papers),
        "dense": DenseRetriever.build(args.corpus),
    }

    for be_name, retriever in retrievers.items():
        print(f"\n═══════ 백엔드: {be_name} ═══════")
        print(f"{'단계':>12} | {'Recall@10':>9} {'MRR@10':>7} | {'한국어R@10':>10} {'영어R@10':>9}")
        print("-" * 60)
        for stage in STAGES:
            run = run_stage(retriever, rewrites, stage, args.k)
            s = metrics.evaluate(qrels, run, k_values=(10,), mrr_k=10)
            print(f"{stage:>12} | {s['Recall@10']:>9.3f} {s['MRR@10']:>7.3f} | "
                  f"{subset_recall(qrels, run, ko):>10.3f} {subset_recall(qrels, run, en):>9.3f}")


if __name__ == "__main__":
    main()
