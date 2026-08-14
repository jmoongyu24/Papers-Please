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
