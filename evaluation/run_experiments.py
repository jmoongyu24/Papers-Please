"""여러 (변환 방식 × 검색 방식) 조합을 돌려 검색 결과를 저장하는 실험 러너.

한 조합(condition)이란 예를 들면 "변환 안 함 + BM25" 또는 "mock 변환 + BM25" 이다.
각 조합에 대해 평가셋의 모든 질문을 검색해, 질문마다 나온 논문 순위를 파일로 저장한다.

왜 저장(캐싱)하나: 검색을 다시 돌리지 않고도 언제든 점수를 다시 계산할 수 있고,
같은 결과가 재현되며, (진짜 언어 모델을 쓸 때) 느린 변환을 반복하지 않아도 된다.

저장 형식(runs/<이름>.json):
  {"condition": {...}, "run": {"q00001": ["논문id 1등", "2등", ...], ...}}

실행 예:
  # 데모: 샘플 데이터로 전체 파이프라인을 한 번에 돌리고 비교표까지 출력
  python -m evaluation.run_experiments --demo
  # 실제: 특정 코퍼스/평가셋으로 지정한 조합 실행
  python -m evaluation.run_experiments --corpus data/corpus/corpus-v1.jsonl \
      --queries data/eval/dev.jsonl --rewriters passthrough mock_expander --backends bm25
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.retrieval.bm25_simple import BM25Retriever
from src.retrieval.corpus import load_corpus
from src.rewriter.base import build_rewriter
from src.schemas import EvalQuery, Paper
from src.utils import read_jsonl, write_json

# 검색 방식 이름 -> (검색기 만드는 법, 변환 결과에서 꺼내 쓸 검색어 종류)
# 지금은 BM25(단어 일치)만. dense/hybrid 는 모듈 ②가 준비되면 여기에 추가한다.
BACKEND_QUERY_FAMILY = {"bm25": "sparse"}


def build_retrievers(papers: list[Paper], backends: list[str]) -> dict:
    retrievers = {}
    for name in backends:
        if name == "bm25":
            retrievers[name] = BM25Retriever(papers)
        else:
            raise ValueError(
                f"아직 준비 안 된 검색 방식: {name} (모듈 ②에서 구현 예정)"
            )
    return retrievers


def run_condition(
    queries: list[EvalQuery],
    rewriter_name: str,
    backend_name: str,
    retriever,
    k: int = config.TOP_K,
) -> dict[str, list[str]]:
    """한 조합(변환 × 검색)으로 모든 질문을 검색해, 질문별 논문 순위를 만든다."""
    rewriter = build_rewriter(rewriter_name)
    family = BACKEND_QUERY_FAMILY[backend_name]
    run: dict[str, list[str]] = {}
    for q in queries:
        rewritten = rewriter.rewrite(q.text)
        search_query = rewritten.query_for(family)
        results = retriever.search(search_query, k=k)
        run[q.query_id] = [r.paper_id for r in results]
    return run


def qrels_from_queries(queries: list[EvalQuery]) -> dict[str, dict[str, int]]:
    """평가셋에서 정답지(qrels)를 만든다. 질문마다 정답 논문 하나에 관련도 1."""
    return {q.query_id: {q.gold_id: 1} for q in queries}


def run_all(
    corpus_path: str,
    queries_path: str,
    rewriters: list[str],
    backends: list[str],
    out_dir: str,
    k: int = config.TOP_K,
) -> dict[str, dict[str, list[str]]]:
    """지정한 모든 조합을 실행하고 결과를 저장한다. {조합이름: run} 을 돌려준다."""
    papers = load_corpus(corpus_path)
    queries = [EvalQuery.from_dict(r) for r in read_jsonl(queries_path)]
    retrievers = build_retrievers(papers, backends)
    queryset = Path(queries_path).stem

    all_runs: dict[str, dict[str, list[str]]] = {}
    for backend in backends:
        for rw in rewriters:
            cond = f"{queryset}__{rw}__{backend}"
            run = run_condition(queries, rw, backend, retrievers[backend], k=k)
            all_runs[cond] = run
            write_json(
                Path(out_dir) / f"{cond}.json",
                {"condition": {"queryset": queryset, "rewriter": rw,
                               "backend": backend, "k": k},
                 "run": run},
            )
    return all_runs


def main() -> None:
    ap = argparse.ArgumentParser(description="변환×검색 조합 실험 실행")
    ap.add_argument("--corpus", default=str(config.SAMPLE_DIR / "corpus.sample.jsonl"))
    ap.add_argument("--queries", default=str(config.SAMPLE_DIR / "queries.sample.jsonl"))
    ap.add_argument("--rewriters", nargs="+", default=["passthrough", "mock_expander"])
    ap.add_argument("--backends", nargs="+", default=["bm25"])
    ap.add_argument("--out-dir", default=str(config.RUNS_DIR))
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--demo", action="store_true",
                    help="샘플 데이터로 실행하고 비교표까지 바로 출력")
    args = ap.parse_args()

    all_runs = run_all(
        args.corpus, args.queries, args.rewriters, args.backends, args.out_dir, args.k
    )
    print(f"조합 {len(all_runs)}개 실행 완료 → {args.out_dir}")

    if args.demo:
        # 데모에서는 곧바로 비교표를 출력해 결과를 눈으로 확인한다.
        from evaluation.report import report_from_runs
        queries = [EvalQuery.from_dict(r) for r in read_jsonl(args.queries)]
        print()
        report_from_runs(all_runs, queries, backends=args.backends,
                         rewriters=args.rewriters)


if __name__ == "__main__":
    main()
