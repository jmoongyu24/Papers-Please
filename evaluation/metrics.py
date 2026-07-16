"""검색 결과에 점수를 매기는 지표 계산 모듈.

핵심 질문은 하나다: **정답 논문이 검색 결과에서 얼마나 위쪽에 나오는가?**
이 프로젝트의 정답지는 "질문 하나에 정답 논문이 딱 하나"인 구조라, 그 구조에
가장 잘 맞는 두 지표를 쓴다.

- 재현율 (Recall@K): 정답 논문이 상위 K개 안에 들어왔으면 1점, 아니면 0점.
- 평균 역순위 (MRR):  정답이 몇 등인지까지 반영. 1등=1점, 2등=1/2점, 3등=1/3점 …

두 지표 모두 "질문 하나당 점수"를 먼저 구하고, 그것들을 평균낸다.
질문별 점수를 남겨 두는 이유는, '변환 전 vs 변환 후'를 **같은 질문끼리 짝지어**
비교하고 그 차이가 우연이 아닌지 통계로 확인하기 위해서다 (paired_bootstrap 참고).

용어 정리:
- qrels : 정답지. {질문id: {정답논문id: 1}}  (relevance judgements)
- run   : 검색 결과. {질문id: [1등 논문id, 2등 논문id, ...]}  (순위대로 나열)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

Qrels = Dict[str, Dict[str, int]]
Run = Dict[str, List[str]]


# ── 질문 하나에 대한 점수 ──────────────────────────────────────────────────

def _recall_at_k_single(ranked: List[str], gold: set[str], k: int) -> float:
    """상위 k개 안에 정답이 하나라도 있으면, (찾은 정답 수 / 전체 정답 수)."""
    if not gold:
        return 0.0
    topk = ranked[:k]
    found = sum(1 for g in gold if g in topk)
    return found / len(gold)


def _reciprocal_rank_single(ranked: List[str], gold: set[str], k: int) -> float:
    """정답이 처음 나온 등수의 역수. 상위 k 안에 없으면 0."""
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


# ── 질문별 점수 배열 (통계 검정에 쓰려고 개별 점수를 남긴다) ────────────────

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
        out[qid] = _reciprocal_rank_single(ranked, gold, k)
    return out


# ── 전체 평균 지표 ────────────────────────────────────────────────────────

def evaluate(qrels: Qrels, run: Run, k_values=(1, 5, 10, 20), mrr_k: int = 10) -> Dict[str, float]:
    """한 검색 결과(run)에 대해 여러 지표를 한 번에 계산한다.

    Returns: {"Recall@1": ..., "Recall@10": ..., "MRR@10": ...}
    """
    scores: Dict[str, float] = {}
    for k in k_values:
        pq = per_query_recall(qrels, run, k)
        scores[f"Recall@{k}"] = float(np.mean(list(pq.values()))) if pq else 0.0
    rr = per_query_rr(qrels, run, mrr_k)
    scores[f"MRR@{mrr_k}"] = float(np.mean(list(rr.values()))) if rr else 0.0
    return scores


# ── '변환 전 vs 변환 후' 짝지어 비교 + 통계 검정 ──────────────────────────

def paired_bootstrap(
    before: Dict[str, float],
    after: Dict[str, float],
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """같은 질문들에 대한 두 방식의 점수 차이가 우연인지 통계로 확인한다.

    방법(부트스트랩): 질문 목록에서 무작위로(중복 허용) 다시 뽑기를 n_boot번 반복해,
    '변환 후 - 변환 전' 평균 차이가 매번 어떻게 나오는지 분포를 본다. 그 분포가 0을
    거의 넘지 않으면(=거의 항상 변환 후가 낫다면) 차이가 우연이 아니라고 본다.

    왜 이 방법인가: 질문마다 난이도가 제각각이라, 두 방식을 '같은 질문끼리' 짝지어
    비교해야 공정하다. 부트스트랩은 데이터 분포를 가정하지 않아 안전하다.

    Returns:
        mean_before, mean_after, delta(후-전), p_value(차이가 없을 확률의 추정),
        ci_low, ci_high (delta의 95% 신뢰구간).
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
    idx = rng.integers(0, n, size=(n_boot, n))       # 매 반복마다 질문을 다시 뽑음
    boot_deltas = diff[idx].mean(axis=1)             # 각 반복의 평균 차이

    observed = float(diff.mean())
    # 양측 p-value 근사: 부트스트랩 차이가 0의 반대편으로 얼마나 자주 가는지.
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
