"""실험 결과(run)들을 모아 점수를 내고, '변환 전 vs 변환 후' 비교표를 만든다.

이 프로젝트에서 가장 보고 싶은 것: **변환 전(passthrough)에 비해 변환 후가 얼마나 더
좋은 검색 결과를 주는가.** 그래서 이 리포트는 두 가지를 보여준다.
1) 조합별 지표 표 (각 변환 방식 × 검색 방식의 Recall@K, MRR).
2) 같은 검색 방식 안에서 '변환 전 → 변환 후'의 변화량과, 그 차이가 우연이 아닌지
   통계로 확인한 결과(p-value, 신뢰구간).

실행 예:
  # 저장된 run 파일들로 리포트
  python -m evaluation.report --runs-dir runs --queries data/sample/queries.sample.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.schemas import EvalQuery
from src.utils import read_json, read_jsonl
from evaluation import metrics

# '변환 전' 기준으로 삼을 변환기 이름
BEFORE_REWRITER = "passthrough"


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _print_metric_table(rows: list[dict], k_values, mrr_k: int) -> None:
    """조합별 지표 표를 보기 좋게 출력한다."""
    headers = ["변환 방식", "검색 방식"] + [f"R@{k}" for k in k_values] + [f"MRR@{mrr_k}"]
    print("── 조합별 지표 " + "─" * 40)
    print("  ".join(f"{h:>12}" for h in headers))
    for r in rows:
        cells = [r["rewriter"], r["backend"]]
        cells += [_fmt(r["scores"][f"Recall@{k}"]) for k in k_values]
        cells += [_fmt(r["scores"][f"MRR@{mrr_k}"])]
        print("  ".join(f"{c:>12}" for c in cells))


def _print_before_after(
    qrels, runs_by_cond: dict, backends, rewriters, mrr_k: int
) -> None:
    """검색 방식별로 '변환 전 → 변환 후' 변화와 통계 유의성을 출력한다."""
    print("\n── 변환 전 vs 변환 후 (Recall@10 기준) " + "─" * 20)
    print("  변환 전 기준:", BEFORE_REWRITER)
    for backend in backends:
        before_cond = _find_cond(runs_by_cond, BEFORE_REWRITER, backend)
        if before_cond is None:
            continue
        before_run = runs_by_cond[before_cond]
        before_pq = metrics.per_query_recall(qrels, before_run, 10)
        for rw in rewriters:
            if rw == BEFORE_REWRITER:
                continue
            after_cond = _find_cond(runs_by_cond, rw, backend)
            if after_cond is None:
                continue
            after_pq = metrics.per_query_recall(qrels, runs_by_cond[after_cond], 10)
            stat = metrics.paired_bootstrap(before_pq, after_pq)
            sig = "유의미함" if stat["p_value"] < 0.05 else "불확실(표본 작음일 수 있음)"
            print(
                f"  [{backend}] {rw}: "
                f"{_fmt(stat['mean_before'])} → {_fmt(stat['mean_after'])} "
                f"(변화 {stat['delta']:+.3f}, "
                f"95% 구간 [{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}], "
                f"p={stat['p_value']:.3f}, {sig}, 질문 {stat['n']}개)"
            )


def _find_cond(runs_by_cond: dict, rewriter: str, backend: str):
    """조합 이름에서 (변환기, 검색기)가 일치하는 것을 찾는다."""
    for cond in runs_by_cond:
        parts = cond.split("__")
        if len(parts) >= 3 and parts[-2] == rewriter and parts[-1] == backend:
            return cond
    return None


def report_from_runs(
    runs_by_cond: dict[str, dict],
    queries: list[EvalQuery],
    backends: list[str],
    rewriters: list[str],
    k_values=config.K_VALUES,
    mrr_k: int = 10,
) -> list[dict]:
    """메모리에 있는 run들로 리포트를 출력한다 (데모/테스트에서 바로 사용)."""
    from evaluation.run_experiments import qrels_from_queries
    qrels = qrels_from_queries(queries)

    rows = []
    for backend in backends:
        for rw in rewriters:
            cond = _find_cond(runs_by_cond, rw, backend)
            if cond is None:
                continue
            scores = metrics.evaluate(qrels, runs_by_cond[cond], k_values, mrr_k)
            rows.append({"rewriter": rw, "backend": backend, "scores": scores})

    _print_metric_table(rows, k_values, mrr_k)
    _print_before_after(qrels, runs_by_cond, backends, rewriters, mrr_k)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="저장된 run들로 성능 리포트 생성")
    ap.add_argument("--runs-dir", default=str(config.RUNS_DIR))
    ap.add_argument("--queries", required=True)
    args = ap.parse_args()

    queries = [EvalQuery.from_dict(r) for r in read_jsonl(args.queries)]
    queryset = Path(args.queries).stem

    # 이 평가셋에 해당하는 run 파일만 불러온다.
    runs_by_cond: dict[str, dict] = {}
    backends, rewriters = [], []
    for path in sorted(Path(args.runs_dir).glob(f"{queryset}__*.json")):
        obj = read_json(path)
        cond = path.stem
        runs_by_cond[cond] = obj["run"]
        c = obj.get("condition", {})
        if c.get("backend") and c["backend"] not in backends:
            backends.append(c["backend"])
        if c.get("rewriter") and c["rewriter"] not in rewriters:
            rewriters.append(c["rewriter"])

    if not runs_by_cond:
        print(f"'{queryset}' 에 해당하는 run 파일이 {args.runs_dir}에 없습니다. "
              f"먼저 run_experiments를 실행하세요.")
        return

    # 변환 전(passthrough)을 맨 앞으로 정렬해 보기 좋게.
    rewriters = sorted(rewriters, key=lambda r: (r != BEFORE_REWRITER, r))
    report_from_runs(runs_by_cond, queries, backends, rewriters)


if __name__ == "__main__":
    main()
