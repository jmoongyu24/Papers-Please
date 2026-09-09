"""검색 결과에 점수를 매기는 지표.

정답지가 "질문 하나에 정답 논문 하나" 구조라 그에 맞는 지표를 씀.

- Recall@K   정답 논문이 상위 K편 안에 들어왔으면 1점
- MRR        정답이 몇 등인지까지 반영. 1등 1점, 2등 1/2점
- nDCG@K     등급 정답지가 있을 때, 상위 K편이 얼마나 쓸모 있는가

질문별 점수를 남기는 이유는 두 방식을 같은 질문끼리 짝지어 비교하고 그 차이가 우연인지
통계로 확인하기 위함임 (`paired_bootstrap`).

    qrels  정답지.     {질문id: {정답논문id: 등급}}
    run    검색 결과.  {질문id: [1등 논문id, 2등 논문id, ...]}
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

Qrels = Dict[str, Dict[str, int]]
Run = Dict[str, List[str]]


# -- 질문 하나에 대한 점수 --------------------------------------------------

def _recall_at_k_single(ranked: List[str], gold: set[str], k: int) -> float:
    """상위 k편 안에서 찾은 정답 수 / 전체 정답 수."""
    if not gold:
        return 0.0
    topk = ranked[:k]
    found = sum(1 for g in gold if g in topk)
    return found / len(gold)


def reciprocal_rank_at_k_single(ranked: List[str], gold: set[str], k: int) -> float:
    """정답이 처음 나온 등수의 역수. 상위 k 안에 없으면 0."""
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


# -- 질문별 점수 (통계 검정에 쓰려고 개별 점수를 남김) ---------------------

def per_query_recall(qrels: Qrels, run: Run, k: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for qid, gold_map in qrels.items():
        gold = {d for d, rel in gold_map.items() if rel > 0}
        ranked = run.get(qid, [])
        out[qid] = _recall_at_k_single(ranked, gold, k)
    return out


def per_query_rr(qrels: Qrels, run: Run, k: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for qid, gold_map in qrels.items():
        gold = {d for d, rel in gold_map.items() if rel > 0}
        ranked = run.get(qid, [])
        out[qid] = reciprocal_rank_at_k_single(ranked, gold, k)
    return out


# -- 전체 평균 지표 --------------------------------------------------------

def evaluate(qrels: Qrels, run: Run, k_values=(1, 5, 10, 20), mrr_k: int = 10) -> Dict[str, float]:
    """한 검색 결과에 대해 여러 지표를 한 번에 계산함.

    Returns: {"Recall@1": ..., "Recall@10": ..., "MRR@10": ...}
    """
    scores: Dict[str, float] = {}
    for k in k_values:
        pq = per_query_recall(qrels, run, k)
        scores[f"Recall@{k}"] = float(np.mean(list(pq.values()))) if pq else 0.0
    rr = per_query_rr(qrels, run, mrr_k)
    scores[f"MRR@{mrr_k}"] = float(np.mean(list(rr.values()))) if rr else 0.0
    return scores


# -- 두 방식을 짝지어 비교하고 통계로 확인 ---------------------------------

def bootstrap_ci(scores: List[float], n_boot: int = 10000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """점수 평균과 그 95% 신뢰구간을 구함.

    질문 목록에서 중복을 허용해 같은 개수만큼 다시 뽑기를 n_boot 번 반복하고, 그때마다의
    평균이 어느 범위에 흩어지는지 봄. 분포를 가정하지 않아 비율 지표에도 안전함.
    표본이 작으면 구간이 넓게 나오므로, 같은 0.35 라도 무게가 다름을 볼 수 있음.

    Returns: (평균, 신뢰구간 하한, 신뢰구간 상한)
    """
    if not scores:
        return 0.0, 0.0, 0.0
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(arr.mean()),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def paired_bootstrap(
    before: Dict[str, float],
    after: Dict[str, float],
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """같은 질문들에 대한 두 방식의 점수 차이가 우연인지 통계로 확인함.

    질문 목록에서 중복을 허용해 다시 뽑기를 n_boot 번 반복해 '후 - 전' 평균 차이의
    분포를 봄. 그 분포가 0을 거의 넘지 않으면 차이가 우연이 아니라고 봄. 질문마다
    난이도가 제각각이므로 같은 질문끼리 짝지어 비교해야 공정함.

    Returns:
        mean_before, mean_after, delta(후-전), p_value, ci_low, ci_high.
    """
    qids = [q for q in before.keys() if q in after]
    b = np.array([before[q] for q in qids], dtype=np.float64)
    a = np.array([after[q] for q in qids], dtype=np.float64)
    diff = a - b
    n = len(diff)
    if n == 0:
        return {"mean_before": 0.0, "mean_after": 0.0, "delta": 0.0,
                "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))       # 반복마다 질문을 다시 뽑음
    boot_deltas = diff[idx].mean(axis=1)             # 각 반복의 평균 차이

    observed = float(diff.mean())
    # 양측 p 값 근사: 다시 뽑은 차이가 0의 반대편으로 얼마나 자주 가는지
    if observed >= 0:
        p = float(2 * np.mean(boot_deltas <= 0))
    else:
        p = float(2 * np.mean(boot_deltas >= 0))
    p = min(1.0, p)

    return {
        "mean_before": float(b.mean()),
        "mean_after": float(a.mean()),
        "delta": observed,
        "p_value": p,
        "ci_low": float(np.percentile(boot_deltas, 2.5)),
        "ci_high": float(np.percentile(boot_deltas, 97.5)),
        "n": n,
    }


# ==========================================================================
# 등급 정답지가 있을 때 쓰는 지표
#   qrels_graded : {질문id: {논문id: 등급}}  등급 0(무관)~3(정답급)
#   run          : {질문id: [1등 논문id, 2등, ...]}
# 정답이 1편뿐이면 Recall 로 붕괴하므로, 정답지를 넓힌 뒤에 씀.
# ==========================================================================

def dcg(gains: List[float]) -> float:
    """할인 누적 이득. 뒤로 갈수록 이득을 로그로 깎음."""
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k_single(ranked: List[str], gold_map: Dict[str, int], k: int) -> float:
    """nDCG@k - 관련도 등급과 순위를 함께 반영함.

    등급을 2^등급-1 로 바꿔 이득으로 쓰고(높은 등급을 더 크게 보상), 이상적 정렬 대비
    비율로 정규화하므로 0~1 사이임.
    """
    gains = [(2 ** gold_map.get(d, 0) - 1) for d in ranked[:k]]
    ideal = sorted((2 ** g - 1 for g in gold_map.values()), reverse=True)[:k]
    idcg = dcg([float(x) for x in ideal])
    return dcg([float(x) for x in gains]) / idcg if idcg > 0 else 0.0
