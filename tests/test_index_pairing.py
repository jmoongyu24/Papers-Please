"""색인과 코퍼스가 짝이 맞는지 확인하는 규칙을 못박는 테스트 (모델 없이 돈다).

이 검사가 없으면 무슨 일이 나는가:
줄 위치표는 그 코퍼스 파일 전용이다. 다른 파일에 갖다 쓰면 `seek` 이 엉뚱한 줄에
떨어지는데 **JSON 파싱은 그대로 성공한다.** 오류도 안 나고 결과도 그럴듯해 보이는 채로
다른 논문의 제목과 초록이 나온다. 서비스에서 나면 사용자에게 존재하지 않는 조합의
논문 정보를 보여주게 되고, 아무도 알아채지 못한다.

그래서 이 테스트는 "안 걸리는 경우"가 아니라 **"반드시 걸려야 하는 경우"** 를 검사한다.

실행: python -m pytest tests/test_index_pairing.py -q
"""

import json

import numpy as np
import pytest

from src.retrieval import large_index as li


def write_corpus(path, rows):
    """논문 목록을 jsonl 로 쓰고, 줄 위치와 번호 목록을 함께 돌려준다."""
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
    # 옛 색인은 표본을 읽어 확인하고 통과하면 지문을 채워 둔다.
    # 71만 편을 다시 훑는 데 세 시간이 걸리므로, 되는 색인을 버리게 만들면 안 된다.
    assert meta["corpus_size"] == path.stat().st_size
    assert meta["corpus_name"] == "corpus.jsonl"


def test_코퍼스를_다시_만들면_이름이_같아도_걸린다(corpus):
    """가장 현실적인 사고 시나리오. 파일 이름 비교만으로는 절대 못 잡는다."""
    path, ids, offsets, prefix = corpus
    li.check_pairing(path, prefix, ids, offsets)        # 지문을 기록해 둔다

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
    """지문이 우연히 같아도(크기 동일) 내용이 밀리면 잡아야 한다."""
    path = tmp_path / "corpus.jsonl"
    rows = make_rows(20)
    ids, offsets = write_corpus(path, rows)

    # 줄 길이가 모두 같으므로, 한 칸 민 위치표는 크기 검사를 통과한다
    shifted = np.roll(offsets, 1)
    assert li.sample_matches(path, ids, offsets, n=10)
    assert not li.sample_matches(path, ids, shifted, n=10)


def test_짝이_안_맞는_색인은_다시_훑는다(tmp_path, capsys):
    """scan_corpus 가 낡은 위치표를 조용히 재사용하면 안 된다."""
    path = tmp_path / "corpus.jsonl"
    prefix = tmp_path / "idx"
    write_corpus(path, make_rows(10))
    ids1, off1 = li.scan_corpus(path, prefix)
    assert len(ids1) == 10

    write_corpus(path, make_rows(30))                   # 같은 이름으로 코퍼스를 키웠다
    ids2, off2 = li.scan_corpus(path, prefix)

    assert len(ids2) == 30, "낡은 위치표를 그대로 돌려주면 안 된다"
    assert "짝이 맞지 않는다" in capsys.readouterr().out
