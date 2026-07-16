"""지표 계산이 손으로 계산한 값과 맞는지 확인하는 테스트.

지표는 평가 전체의 기준이므로, 알려진 입력에 대해 정확한 값이 나오는지 못박아 둔다.
실행: /home/jmoongyu/venvs/paper_py310/bin/python -m pytest tests/test_metrics.py -q
"""

from evaluation import metrics


def test_recall_at_k_hit_and_miss():
    qrels = {"q1": {"pA": 1}, "q2": {"pB": 1}}
    # q1: 정답 pA가 2등 -> Recall@1=0, Recall@5=1
    # q2: 정답 pB가 결과에 없음 -> 0
    run = {"q1": ["pX", "pA", "pY"], "q2": ["pZ", "pW"]}

    assert metrics.evaluate(qrels, run, k_values=(1,), mrr_k=10)["Recall@1"] == 0.5 * 0  # q1 miss@1, q2 miss -> 0
    scores = metrics.evaluate(qrels, run, k_values=(1, 5), mrr_k=10)
    assert scores["Recall@1"] == 0.0     # 둘 다 1등 아님
    assert scores["Recall@5"] == 0.5     # q1만 상위5 안에 정답


def test_mrr_uses_rank():
    qrels = {"q1": {"pA": 1}, "q2": {"pB": 1}}
    run = {"q1": ["pA"], "q2": ["pX", "pB"]}  # q1 정답 1등(1.0), q2 정답 2등(0.5)
    scores = metrics.evaluate(qrels, run, k_values=(10,), mrr_k=10)
    assert abs(scores["MRR@10"] - (1.0 + 0.5) / 2) < 1e-9


def test_paired_bootstrap_detects_improvement():
    # 모든 질문에서 after가 before보다 확실히 좋으면 유의미(p 작음)해야 한다.
    before = {f"q{i}": 0.0 for i in range(30)}
    after = {f"q{i}": 1.0 for i in range(30)}
    stat = metrics.paired_bootstrap(before, after, n_boot=2000, seed=0)
    assert stat["delta"] == 1.0
    assert stat["p_value"] < 0.05
    assert stat["n"] == 30


def test_paired_bootstrap_no_difference():
    before = {f"q{i}": (i % 2) * 1.0 for i in range(20)}
    after = dict(before)  # 완전히 동일 -> 차이 0
    stat = metrics.paired_bootstrap(before, after, n_boot=1000, seed=0)
    assert stat["delta"] == 0.0
    assert stat["p_value"] >= 0.05
