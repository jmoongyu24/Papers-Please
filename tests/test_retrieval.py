"""검색 쪽이 조용히 틀리는 자리 두 곳을 못박아 두는 테스트 (모델 없이 돌아감).

1부. 채널 합치기(RRF) 와 논문 번호 표기 통일
arXiv 실시간 결과(`2103.00020v2`)와 로컬 색인 결과(`2103.00020`)가 같은 논문을 다르게
표기하면, 두 채널이 합의한 논문일수록 점수가 반으로 쪼개짐. 오류를 내지 않고 성능만
조용히 깎는 종류의 버그라 테스트로 막음.

2부. 색인과 코퍼스의 짝 맞추기
줄 위치표는 그 코퍼스 파일 전용임. 다른 파일에 갖다 쓰면 `seek` 이 엉뚱한 줄에
떨어지는데 JSON 파싱은 그대로 성공함. 오류도 안 나고 결과도 그럴듯해 보이는 채로
다른 논문의 제목과 초록이 나옴. 서비스에서 나면 사용자에게 존재하지 않는 조합의
논문 정보를 보여주게 되고, 아무도 알아채지 못함. 그래서 "안 걸리는 경우"가 아니라
"반드시 걸려야 하는 경우" 를 검사함.

실행: $PY -m pytest tests/test_retrieval.py -q
"""

import json

import numpy as np
import pytest

from src.retrieval import local_index as li
from src.retrieval.corpus import normalize_paper_id
from src.retrieval.ranking import FusedPaper, rrf_fuse, rrf_fuse_ids
from src.schemas import ScoredPaper


def sp(pid: str, rank: int = 1, title: str = "", abstract: str = "") -> ScoredPaper:
    return ScoredPaper(paper_id=pid, score=0.0, rank=rank, title=title, abstract=abstract)


def ids_of(fused: list[FusedPaper]) -> list[str]:
    return [p.paper_id for p in fused]


# -- 번호 표기 통일 --------------------------------------------------------
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
    """버전 표기만 다른 같은 논문은 하나로 합쳐지고, 점수도 두 채널 몫이 더해져야 함."""
    fused = rrf_fuse({
        "arxiv": [sp("2103.00020v2", 1), sp("1111.1111v1", 2)],
        "local_dense": [sp("2222.2222", 1), sp("2103.00020", 2)],
    }, k=60, top_n=10)

    assert ids_of(fused)[0] == "2103.00020"        # 두 채널이 합의했으므로 1등
    assert len(fused) == 3                          # 4줄이 들어왔지만 고유 논문은 3편
    assert fused[0].channels == {"arxiv": 1, "local_dense": 2}
    # 1/(60+1) + 1/(60+2) 가 다른 어떤 논문의 단일 채널 점수보다 큼
    assert abs(fused[0].score - (1 / 61 + 1 / 62)) < 1e-12


def test_version_mismatch_would_split_score_without_normalization():
    """정규화가 빠지면 어떤 일이 벌어지는지 반대 방향으로 확인함.

    같은 논문이 두 채널에서 2등, 2등으로 나오고, 다른 논문 하나가 한 채널에서 1등으로
    나온 상황. 번호를 통일하면 합의된 논문이 이기고, 통일하지 않으면 짐.
    """
    fused = rrf_fuse({
        "arxiv": [sp("9999.9999", 1), sp("2103.00020v2", 2)],
        "local_dense": [sp("8888.8888", 1), sp("2103.00020", 2)],
    }, k=60, top_n=10)
    assert ids_of(fused)[0] == "2103.00020"
    assert fused[0].score > fused[1].score


# -- 합치기 규칙 -----------------------------------------------------------
def test_rank_comes_from_list_order_not_stale_rank_field():
    """목록을 잘라 쓰면 rank 필드가 옛 값으로 남음 - 순서를 믿어야 함."""
    # rank 필드에는 엉뚱한 값(50, 99)이 들어 있지만 목록 순서는 A, B 다.
    fused = rrf_fuse({"c": [sp("A", 50), sp("B", 99)]}, k=60, top_n=10)
    assert ids_of(fused) == ["A", "B"]
    assert abs(fused[0].score - 1 / 61) < 1e-12
    assert abs(fused[1].score - 1 / 62) < 1e-12


def test_duplicate_inside_one_channel_counts_once():
    """한 채널이 같은 논문에 두 번 점수를 주면 그 채널의 영향력이 부풀려짐."""
    fused = rrf_fuse({"c": [sp("A", 1), sp("Av1", 2), sp("B", 3)]}, k=60, top_n=10)
    assert ids_of(fused) == ["A", "B"]
    assert abs(fused[0].score - 1 / 61) < 1e-12          # 1/61 만, 1/62 는 더하지 않음
    assert fused[0].channels == {"c": 1}                  # 가장 앞선 등수만 남음


def test_weights_shift_influence_between_channels():
    """가중치는 점수 눈금을 건드리지 않고 채널의 영향력만 바꿈."""
    channels = {"arxiv": [sp("A", 1), sp("B", 2)],
                "local_dense": [sp("B", 1), sp("A", 2)]}
    even = rrf_fuse(channels, k=60, top_n=10)
    assert ids_of(even) == ["A", "B"]                     # 완전 대칭이면 먼저 등장한 쪽

    tilted = rrf_fuse(channels, k=60, top_n=10, weights={"local_dense": 3.0})
    assert ids_of(tilted) == ["B", "A"]                   # 의미 검색 쪽 1등이 올라옴


def test_missing_weight_defaults_to_one():
    channels = {"a": [sp("X", 1)], "b": [sp("Y", 1)]}
    fused = rrf_fuse(channels, k=60, top_n=10, weights={"a": 1.0})   # b 는 안 적음
    assert abs(fused[0].score - fused[1].score) < 1e-12


def test_metadata_prefers_a_channel_that_actually_has_text():
    """재정렬기는 제목, 초록으로 판단함. 빈 채로 넘어가면 조용히 품질이 떨어짐."""
    fused = rrf_fuse({
        "no_text": [sp("A", 1)],                                   # 제목, 초록 없음
        "with_text": [sp("Av1", 1, title="제목", abstract="초록")],
    }, k=60, top_n=10)
    assert fused[0].title == "제목" and fused[0].abstract == "초록"


def test_top_n_limits_output_but_not_scoring():
    channels = {"c": [sp(f"P{i}", i) for i in range(1, 11)]}
    fused = rrf_fuse(channels, k=60, top_n=3)
    assert ids_of(fused) == ["P1", "P2", "P3"]
    assert [p.rank for p in fused] == [1, 2, 3]           # 등수는 1부터 다시 매김


def test_empty_and_missing_channels_are_safe():
    assert rrf_fuse({}) == []
    assert rrf_fuse({"a": [], "b": []}) == []
    fused = rrf_fuse({"a": [], "b": [sp("A", 1)]})
    assert ids_of(fused) == ["A"]


def test_fuse_ids_matches_full_fusion():
    """저장된 논문 번호만으로 합쳐도 결과 순서가 같아야 함(재검색 없는 재계산의 근거)."""
    channels = {"arxiv": ["2103.00020v2", "1111.1111"],
                "local_dense": ["2222.2222", "2103.00020"]}
    assert rrf_fuse_ids(channels, k=60, top_n=10) == ["2103.00020", "2222.2222", "1111.1111"]


# -- 서비스와 평가가 같은 순위를 내는가 (ISSUE #10 · #13 · #39 가 난 자리) --------
#
# `app.py` 는 검색어 2개의 결과를 `fuse_local()` 로 합치고, 평가 하네스는 저장된
# 채널별 논문 번호를 `fused_ids_of()` 로 합침. 두 경로가 다른 순위를 내면 평가로 잰
# 값이 서비스의 값이 아니게 됨. 같은 종류의 어긋남을 세 번 겪었으므로 못박아 둠.
#
# 이 시험은 모델도 색인도 쓰지 않음 - 합치는 규칙만 봄.

def test_서비스와_평가의_후보_합치기가_같은_순위를_낸다():
    import app
    from evaluation.pipeline_eval import fused_ids_of
    from src import config

    literal = ["2103.00020", "1111.1111", "2222.2222v1", "3333.3333"]
    hyde = ["2222.2222", "2103.00020v2", "4444.4444", "1111.1111"]

    from_service = [p.paper_id for p in app.fuse_local(
        [sp(pid, i) for i, pid in enumerate(literal, 1)],
        [sp(pid, i) for i, pid in enumerate(hyde, 1)],
    )]
    from_eval = fused_ids_of(
        {"channels": {"local_dense": literal, "local_hyde": hyde}},
        rrf_k=config.RRF_K, top_n=app.RERANK_DEPTH, weights={},
    )

    assert from_service == from_eval, (
        f"서비스와 평가의 순위가 다름\n  서비스 {from_service}\n  평가   {from_eval}")


def test_화면에_넣는_제목은_한_줄로_펴진다():
    """arXiv 제목의 줄바꿈이 마크다운 머리글을 깨뜨리는 것을 막음.

    코퍼스 표본 20,000편 중 814편(4.1%)의 제목에 줄바꿈이 들어 있음. 그대로
    `#### 3. [{제목}]({주소})  `{판정}`` 에 끼우면 머리글이 첫 줄에서 끝나고, 나머지
    제목과 링크 문법과 판정 표시가 본문 글씨로 떨어짐 - 논문마다 글씨 크기가 달라지고
    링크가 안 걸림. 2026-08-28 에 사용자가 화면에서 발견했음.
    """
    import app

    제목 = "ReinDiffuse: Crafting Physically Plausible Motions with Reinforced\n  Diffusion Model"
    편 = app.one_line(제목)
    assert "\n" not in 편, f"줄바꿈이 남아 있음: {편!r}"
    assert 편 == ("ReinDiffuse: Crafting Physically Plausible Motions with Reinforced "
                 "Diffusion Model"), 편
    # 실제로 만드는 마크다운 한 줄이 정말 한 줄인지 확인함
    head = f"#### 3. [{편}](https://arxiv.org/abs/2410.07296)  `관련 있음`"
    assert head.count("\n") == 0
    assert head.startswith("#### 3. [") and head.endswith("`관련 있음`")
    # 탭과 여러 칸 띄어쓰기도 한 칸으로 모음
    assert app.one_line("A\t\tB   C") == "A B C"
    assert app.one_line("") == "" and app.one_line(None) == ""


def test_서비스와_평가의_순위_합치기가_같은_순서를_낸다():
    """재정렬 순위와 검색 순위를 합치는 계산이 두 경로에서 같아야 함.

    같은 종류의 어긋남을 이미 네 번 겪었음 (#10, #13, #39, #40). 서비스가 한 가중치로
    합치고 평가가 다른 가중치로 합치면 오류 없이 값만 달라지고, 그 값으로 채택을 정하게 됨.
    """
    import app
    from src import config
    from src.retrieval.ranking import rrf_fuse_ids

    # 재정렬이 매긴 순서와 검색이 매긴 순서가 다른 상황
    rerank_order = ["aaa", "bbb", "ccc", "ddd", "eee"]
    search_order = ["eee", "ddd", "aaa", "ccc", "bbb"]

    ranked = [sp(pid, i) for i, pid in enumerate(rerank_order, 1)]
    candidates = [sp(pid, i) for i, pid in enumerate(search_order, 1)]

    from_service = [p.paper_id for p in app.fuse_with_search(ranked, candidates)]
    from_eval = rrf_fuse_ids(
        {"rerank": rerank_order, "search": search_order},
        k=config.RRF_K, top_n=len(rerank_order),
        weights={"rerank": app.FUSE_RERANK_WEIGHT, "search": 1.0})

    assert from_service == from_eval, (
        f"서비스와 평가의 순위 합치기가 다름\n  서비스 {from_service}\n  평가   {from_eval}")
    # 합치기를 껐을 때는 재정렬 순서 그대로여야 함
    old = app.FUSE_RERANK_WEIGHT
    try:
        app.FUSE_RERANK_WEIGHT = 0.0
        assert [p.paper_id for p in app.fuse_with_search(ranked, candidates)] == rerank_order
    finally:
        app.FUSE_RERANK_WEIGHT = old


def test_순위_합치기가_재정렬_점수를_잃지_않는다():
    """합친 뒤에도 각 논문의 재정렬 점수가 따라가야 함.

    `MIN_RERANK_SCORE` 로 '못 찾았다' 를 판정하는 자리가 그 점수를 씀. 순서만 바꾸고
    점수를 잃으면 그 판정이 조용히 망가짐.
    """
    import app

    ranked = [sp("aaa", 1), sp("bbb", 2), sp("ccc", 3)]
    for p, v in zip(ranked, (0.9, 0.5, 0.001)):
        p.score = v
    candidates = [sp("ccc", 1), sp("bbb", 2), sp("aaa", 3)]

    out = app.fuse_with_search(ranked, candidates)
    assert {p.paper_id for p in out} == {"aaa", "bbb", "ccc"}
    assert {p.paper_id: p.score for p in out} == {"aaa": 0.9, "bbb": 0.5, "ccc": 0.001}


def test_가상_초록이_비면_그_채널은_검색하지_않는다():
    """생성이 실패했을 때 원본 질문으로 대신 찾으면 로컬 채널이 표를 두 번 던짐."""
    from evaluation.pipeline_eval import LocalHydeChannel

    class 부르면안됨:
        def search(self, query, k=100):
            raise AssertionError("빈 검색어로 색인을 찾으면 안 됨")

    ch = LocalHydeChannel(부르면안됨())
    assert ch.search("", k=10) == []
    assert ch.search("   ", k=10) == []


def test_저장된_검색어를_그대로_다시_쓴다(tmp_path):
    """색인만 바꿔 견줄 때, 가상 초록을 새로 만들면 생성 흔들림이 섞여 판정이 흐려짐."""
    import json
    from evaluation.pipeline_eval import ReplayRewriter

    run = tmp_path / "run.jsonl"
    run.write_text(json.dumps({
        "query_id": "q1", "text": "사진 보고 글로 설명해주는 AI",
        "search_queries": {"local_dense": "image captioning models",
                           "local_hyde": "We propose a vision-language model ..."},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    rw = ReplayRewriter(run)
    got = rw.rewrite("사진 보고 글로 설명해주는 AI")

    # 채널이 자기 이름으로 찾을 때 저장된 검색어가 나와야 함.
    assert got.query_for("local_dense") == "image captioning models"
    assert got.query_for("local_hyde").startswith("We propose")
    assert rw.n_missing == 0

    # 저장에 없는 질문은 세어 두어야 함 (다른 평가셋 파일을 준 경우를 잡기 위함)
    missed = rw.rewrite("저장에 없는 질문")
    assert missed.parse_ok is False and rw.n_missing == 1


def test_색인을_만든_모델로_질문을_임베딩한다():
    """미세조정한 색인을 옛 모델로 찾으면 오류 없이 검색 결과만 무너짐."""
    from evaluation import pipeline_eval as pe

    받은인자 = {}

    class 가짜색인:
        def __init__(self, *a, **kw):
            받은인자.update(kw)

    import src.retrieval.local_index as li_mod
    진짜 = li_mod.LocalDenseRetriever
    li_mod.LocalDenseRetriever = 가짜색인
    try:
        class Args:
            corpus, index, mmap, no_cache = "c", "i", False, True
            embed_model = "models/retriever-ft"
        pe.build_channels(["local_dense"], Args())
    finally:
        li_mod.LocalDenseRetriever = 진짜

    assert 받은인자.get("model_name") == "models/retriever-ft", (
        f"색인을 만든 모델이 안 넘어갔음: {받은인자}")


def test_평가_채널을_만들_때_로컬_색인은_한_벌만_올린다():
    """임베딩이 2.93GB 라 채널마다 새로 올리면 시스템 메모리 15GB 에서 터짐."""
    from evaluation import pipeline_eval as pe

    만든횟수 = []

    class 가짜색인:
        def __init__(self, *a, **kw):
            만든횟수.append(1)

    import src.retrieval.local_index as li_mod
    진짜 = li_mod.LocalDenseRetriever
    li_mod.LocalDenseRetriever = 가짜색인
    try:
        class Args:
            corpus, index, mmap, no_cache = "c", "i", False, True
            embed_model = None
        chans = pe.build_channels(["local_dense", "local_hyde"], Args())
    finally:
        li_mod.LocalDenseRetriever = 진짜

    assert len(만든횟수) == 1, f"색인을 {len(만든횟수)}번 올렸음"
    assert chans["local_hyde"].retriever is chans["local_dense"]



# ==========================================================================
# 색인과 코퍼스의 짝 맞추기
# ==========================================================================

def write_corpus(path, rows):
    """논문 목록을 jsonl 로 쓰고, 줄 위치와 번호 목록을 함께 돌려줌."""
    ids, offsets, pos = [], [], 0
    with open(path, "wb") as f:
        for r in rows:
            raw = (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")
            offsets.append(pos)
            pos += len(raw)
            ids.append(str(r["id"]))
            f.write(raw)
    return ids, np.asarray(offsets, dtype=np.int64)


def make_rows(n, tag=""):
    return [{"id": f"2101.{i:05d}", "title": f"제목{i}{tag}", "abstract": f"초록{i}{tag}"}
            for i in range(n)]


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    ids, offsets = write_corpus(path, make_rows(20))
    return path, ids, offsets, tmp_path / "idx"


def test_짝이_맞으면_통과하고_지문을_기록한다(corpus):
    path, ids, offsets, prefix = corpus
    li.write_meta(prefix, {"count": len(ids)})          # 지문 없는 옛 색인 흉내

    ok, why = li.check_pairing(path, prefix, ids, offsets)

    assert ok, why
    meta = li.read_meta(prefix)
    # 옛 색인은 표본을 읽어 확인하고 통과하면 지문을 채워 둠.
    # 71만 편을 다시 훑는 데 세 시간이 걸리므로, 되는 색인을 버리게 만들면 안 됨.
    assert meta["corpus_size"] == path.stat().st_size
    assert meta["corpus_name"] == "corpus.jsonl"


def test_코퍼스를_다시_만들면_이름이_같아도_걸린다(corpus):
    """가장 현실적인 사고 시나리오. 파일 이름 비교만으로는 절대 못 잡음."""
    path, ids, offsets, prefix = corpus
    li.check_pairing(path, prefix, ids, offsets)        # 지문을 기록해 둠

    write_corpus(path, make_rows(20, tag="-다시만듦") + make_rows(5, tag="-추가"))

    ok, why = li.check_pairing(path, prefix, ids, offsets)
    assert not ok
    assert "크기" in why


def test_번호와_위치_개수가_다르면_걸린다(corpus):
    path, ids, offsets, prefix = corpus
    ok, why = li.check_pairing(path, prefix, ids[:-1], offsets)
    assert not ok
    assert "수가 다르다" in why


def test_위치표가_어긋나면_표본_확인에서_걸린다(tmp_path):
    """지문이 우연히 같아도(크기 동일) 내용이 밀리면 잡아야 함."""
    path = tmp_path / "corpus.jsonl"
    rows = make_rows(20)
    ids, offsets = write_corpus(path, rows)

    # 줄 길이가 모두 같으므로, 한 칸 민 위치표는 크기 검사를 통과함
    shifted = np.roll(offsets, 1)
    assert li.sample_matches(path, ids, offsets, n=10)
    assert not li.sample_matches(path, ids, shifted, n=10)


def test_짝이_안_맞는_색인은_다시_훑는다(tmp_path, capsys):
    """scan_corpus 가 낡은 위치표를 조용히 재사용하면 안 됨."""
    path = tmp_path / "corpus.jsonl"
    prefix = tmp_path / "idx"
    write_corpus(path, make_rows(10))
    ids1, off1 = li.scan_corpus(path, prefix)
    assert len(ids1) == 10

    write_corpus(path, make_rows(30))                   # 같은 이름으로 코퍼스를 키웠음
    ids2, off2 = li.scan_corpus(path, prefix)

    assert len(ids2) == 30, "낡은 위치표를 그대로 돌려주면 안 된다"
    assert "짝이 맞지 않는다" in capsys.readouterr().out
