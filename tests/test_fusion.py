"""채널 합치기(RRF)가 의도대로 동작하는지 못박아 두는 테스트.

특히 **논문 번호 표기 통일**을 집중해서 검사한다. arXiv 실시간 결과(`2103.00020v2`)와
로컬 색인 결과(`2103.00020`)가 같은 논문을 다르게 표기하면, 두 채널이 합의한 논문일수록
점수가 반으로 쪼개진다. 오류를 내지 않고 성능만 조용히 깎는 종류의 버그라 테스트로 막는다.

실행: /home/jmoongyu/venvs/paper_py310/bin/python -m pytest tests/test_fusion.py -q
      (프로젝트 가상환경에 pytest 가 없으면 `/home/jmoongyu/anaconda3/bin/python -m pytest`.
       이 테스트는 numpy·torch 없이 표준 라이브러리만으로 돌아간다.)
"""

from src.retrieval.fusion import FusedPaper, normalize_paper_id, rrf_fuse, rrf_fuse_ids
from src.schemas import ScoredPaper


def sp(pid: str, rank: int = 1, title: str = "", abstract: str = "") -> ScoredPaper:
    return ScoredPaper(paper_id=pid, score=0.0, rank=rank, title=title, abstract=abstract)


def ids_of(fused: list[FusedPaper]) -> list[str]:
    return [p.paper_id for p in fused]


# ── 번호 표기 통일 ────────────────────────────────────────────────────────
def test_normalize_strips_only_trailing_version():
    assert normalize_paper_id("2103.00020v2") == "2103.00020"
    assert normalize_paper_id("2103.00020") == "2103.00020"
    assert normalize_paper_id("1706.03762v12") == "1706.03762"
    assert normalize_paper_id(" 2103.00020v1 ") == "2103.00020"


def test_normalize_does_not_break_old_style_ids_containing_v():
    # `split("v")[0]` 로 구현하면 'sol' 이 되어 버리는 옛 형식 번호들
    assert normalize_paper_id("solv-int/9611001v1") == "solv-int/9611001"
    assert normalize_paper_id("solv-int/9611001") == "solv-int/9611001"
    assert normalize_paper_id("cs/0501001v3") == "cs/0501001"


def test_same_paper_from_two_channels_merges_despite_version_tag():
    """버전 표기만 다른 같은 논문은 하나로 합쳐지고, 점수도 두 채널 몫이 더해져야 한다."""
    fused = rrf_fuse({
        "arxiv": [sp("2103.00020v2", 1), sp("1111.1111v1", 2)],
        "local_dense": [sp("2222.2222", 1), sp("2103.00020", 2)],
    }, k=60, top_n=10)

    assert ids_of(fused)[0] == "2103.00020"        # 두 채널이 합의했으므로 1등
    assert len(fused) == 3                          # 4줄이 들어왔지만 고유 논문은 3편
    assert fused[0].channels == {"arxiv": 1, "local_dense": 2}
    # 1/(60+1) + 1/(60+2) 가 다른 어떤 논문의 단일 채널 점수보다 크다
    assert abs(fused[0].score - (1 / 61 + 1 / 62)) < 1e-12


def test_version_mismatch_would_split_score_without_normalization():
    """정규화가 빠지면 어떤 일이 벌어지는지 반대 방향으로 확인한다.

    같은 논문이 두 채널에서 2등·2등으로 나오고, 다른 논문 하나가 한 채널에서 1등으로
    나온 상황. 번호를 통일하면 합의된 논문이 이기고, 통일하지 않으면 진다.
    """
    fused = rrf_fuse({
        "arxiv": [sp("9999.9999", 1), sp("2103.00020v2", 2)],
        "local_dense": [sp("8888.8888", 1), sp("2103.00020", 2)],
    }, k=60, top_n=10)
    assert ids_of(fused)[0] == "2103.00020"
    assert fused[0].score > fused[1].score


# ── 합치기 규칙 ───────────────────────────────────────────────────────────
def test_rank_comes_from_list_order_not_stale_rank_field():
    """목록을 잘라 쓰면 rank 필드가 옛 값으로 남는다 — 순서를 믿어야 한다."""
    # rank 필드에는 엉뚱한 값(50, 99)이 들어 있지만 목록 순서는 A, B 다.
    fused = rrf_fuse({"c": [sp("A", 50), sp("B", 99)]}, k=60, top_n=10)
    assert ids_of(fused) == ["A", "B"]
    assert abs(fused[0].score - 1 / 61) < 1e-12
    assert abs(fused[1].score - 1 / 62) < 1e-12


def test_duplicate_inside_one_channel_counts_once():
    """한 채널이 같은 논문에 두 번 점수를 주면 그 채널의 영향력이 부풀려진다."""
    fused = rrf_fuse({"c": [sp("A", 1), sp("Av1", 2), sp("B", 3)]}, k=60, top_n=10)
    assert ids_of(fused) == ["A", "B"]
    assert abs(fused[0].score - 1 / 61) < 1e-12          # 1/61 만, 1/62 는 더하지 않는다
    assert fused[0].channels == {"c": 1}                  # 가장 앞선 등수만 남는다


def test_weights_shift_influence_between_channels():
    """가중치는 점수 눈금을 건드리지 않고 채널의 영향력만 바꾼다."""
    channels = {"arxiv": [sp("A", 1), sp("B", 2)],
                "local_dense": [sp("B", 1), sp("A", 2)]}
    even = rrf_fuse(channels, k=60, top_n=10)
    assert ids_of(even) == ["A", "B"]                     # 완전 대칭이면 먼저 등장한 쪽

    tilted = rrf_fuse(channels, k=60, top_n=10, weights={"local_dense": 3.0})
    assert ids_of(tilted) == ["B", "A"]                   # 의미 검색 쪽 1등이 올라온다


def test_missing_weight_defaults_to_one():
    channels = {"a": [sp("X", 1)], "b": [sp("Y", 1)]}
    fused = rrf_fuse(channels, k=60, top_n=10, weights={"a": 1.0})   # b 는 안 적음
    assert abs(fused[0].score - fused[1].score) < 1e-12


def test_metadata_prefers_a_channel_that_actually_has_text():
    """재정렬기는 제목·초록으로 판단한다. 빈 채로 넘어가면 조용히 품질이 떨어진다."""
    fused = rrf_fuse({
        "no_text": [sp("A", 1)],                                   # 제목·초록 없음
        "with_text": [sp("Av1", 1, title="제목", abstract="초록")],
    }, k=60, top_n=10)
    assert fused[0].title == "제목" and fused[0].abstract == "초록"


def test_top_n_limits_output_but_not_scoring():
    channels = {"c": [sp(f"P{i}", i) for i in range(1, 11)]}
    fused = rrf_fuse(channels, k=60, top_n=3)
    assert ids_of(fused) == ["P1", "P2", "P3"]
    assert [p.rank for p in fused] == [1, 2, 3]           # 등수는 1부터 다시 매긴다


def test_empty_and_missing_channels_are_safe():
    assert rrf_fuse({}) == []
    assert rrf_fuse({"a": [], "b": []}) == []
    fused = rrf_fuse({"a": [], "b": [sp("A", 1)]})
    assert ids_of(fused) == ["A"]


def test_fuse_ids_matches_full_fusion():
    """저장된 논문 번호만으로 합쳐도 결과 순서가 같아야 한다(재검색 없는 재계산의 근거)."""
    channels = {"arxiv": ["2103.00020v2", "1111.1111"],
                "local_dense": ["2222.2222", "2103.00020"]}
    assert rrf_fuse_ids(channels, k=60, top_n=10) == ["2103.00020", "2222.2222", "1111.1111"]


# ── 기존 하이브리드 검색기와의 호환 ────────────────────────────────────────
def test_hybrid_wrapper_still_works():
    from src.retrieval.hybrid import reciprocal_rank_fusion

    out = reciprocal_rank_fusion([[sp("A", 1), sp("B", 2)], [sp("B", 1), sp("C", 2)]],
                                 k=60, top_k=2)
    assert [p.paper_id for p in out] == ["B", "A"]
    assert out[0].rank == 1
