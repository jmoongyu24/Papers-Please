"""다채널 평가 하네스의 집계 규칙을 못박아 두는 테스트 (검색·모델 없이 돈다).

가장 조용히 틀리는 자리 두 곳을 집중해서 검사한다.
1. **오류·결과 0건 문항을 실패로 세는가.** 제외해 버리면 기준선이 부풀려진다(ISSUE #23).
2. **저장된 채널 결과만으로 융합을 다시 계산할 수 있는가.** 이 하네스의 존재 이유다.

실행: /home/jmoongyu/anaconda3/bin/python -m pytest tests/test_pipeline_eval.py -q
"""

import json

from evaluation import pipeline_eval as pe


def row(qid, gold, channels, **extra):
    base = {"query_id": qid, "text": f"질문 {qid}", "gold_id": gold,
            "lang": "ko", "difficulty": "easy", "channels": channels}
    base.update(extra)
    return base


ROWS = [
    # 두 채널 모두 정답을 1등으로
    row("q1", "A", {"arxiv": ["A", "X"], "local_dense": ["A", "Y"]}),
    # arXiv 는 못 찾고 의미 검색만 3등으로 찾음 → 합집합에는 들어온다
    row("q2", "B", {"arxiv": ["X", "Y"], "local_dense": ["X", "Y", "B"]}),
    # 두 채널 다 못 찾음
    row("q3", "C", {"arxiv": ["X"], "local_dense": ["Y"]}),
    # 변환 실패 (오류 문항) — 검색을 아예 못 했다
    row("q4", "D", {"arxiv": [], "local_dense": []}, error="rewrite_failed: X"),
    # 한 채널만 오류, 다른 채널은 정답을 찾음
    row("q5", "E", {"arxiv": [], "local_dense": ["E"]},
        channel_errors={"arxiv": "HTTPError: 503"}),
]


# ── 실패를 실패로 세는가 ──────────────────────────────────────────────────
def test_missing_and_error_rows_count_as_failures():
    """분모는 항상 전체 문항 수여야 한다 (오류 문항을 빼면 안 된다)."""
    ranks = [pe.rank_in((r["channels"] or {}).get("arxiv") or [], r["gold_id"]) for r in ROWS]
    assert ranks == [1, None, None, None, None]
    hits = pe.hits_at(ranks, 10)
    assert hits == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert len(hits) == len(ROWS)            # 5문항 전부가 분모에 남아 있다
    assert sum(hits) / len(hits) == 0.2      # 제외했다면 1/2=0.5 로 부풀려졌을 값


def test_rank_respects_k():
    ranks = [pe.rank_in((r["channels"] or {}).get("local_dense") or [], r["gold_id"])
             for r in ROWS]
    assert ranks == [1, 3, None, None, 1]
    assert pe.hits_at(ranks, 1) == [1.0, 0.0, 0.0, 0.0, 1.0]      # 3등은 @1에 안 든다
    assert pe.hits_at(ranks, 3) == [1.0, 1.0, 0.0, 0.0, 1.0]


# ── 합집합 상한 ───────────────────────────────────────────────────────────
def test_union_is_the_ceiling_and_at_least_as_good_as_each_channel():
    union = pe.hits_at([pe.union_rank(r, 100) for r in ROWS], 1)
    assert union == [1.0, 1.0, 0.0, 0.0, 1.0]        # q2 는 의미 검색 덕분에 상한에 들어온다

    for name in ("arxiv", "local_dense"):
        ch = pe.hits_at([pe.rank_in((r["channels"] or {}).get(name) or [], r["gold_id"])
                         for r in ROWS], 100)
        assert all(u >= c for u, c in zip(union, ch)), f"{name} 이 합집합보다 좋을 수 없다"


def test_union_respects_depth():
    r = row("q", "B", {"arxiv": ["X", "Y", "B"], "local_dense": ["Z"]})
    assert pe.union_rank(r, 2) is None      # 상위 2편 안에는 없다
    assert pe.union_rank(r, 3) == 1


# ── 저장된 결과만으로 융합 재계산 ──────────────────────────────────────────
def test_fusion_recomputed_from_stored_ids():
    """검색을 다시 하지 않고 채널 결과만으로 융합 순위가 나와야 한다."""
    r = row("q", "B", {"arxiv": ["X", "B"], "local_dense": ["B", "Y"]})
    fused = pe.fused_ids_of(r, rrf_k=60, top_n=10, weights={})
    assert fused[0] == "B"                  # 두 채널이 합의한 논문이 1등

    # 가중치를 바꾸면 순서가 바뀐다 (검색은 여전히 0회)
    tilted = pe.fused_ids_of(row("q", "B", {"arxiv": ["X"], "local_dense": ["B"]}),
                             rrf_k=60, top_n=10, weights={"local_dense": 5.0})
    assert tilted[0] == "B"


def test_fusion_merges_version_tagged_ids_across_channels():
    """arXiv 는 버전 표기가 붙어 오고 로컬 색인은 붙지 않는다 — 합쳐져야 한다."""
    r = row("q", "2103.00020",
            {"arxiv": ["9999.9999", "2103.00020v2"], "local_dense": ["8888.8888", "2103.00020"]})
    fused = pe.fused_ids_of(r, rrf_k=60, top_n=10, weights={})
    assert fused[0] == "2103.00020"
    assert pe.rank_in(fused, r["gold_id"]) == 1


def test_gold_id_is_normalized_when_recording():
    class Rewriter:
        name = "t"

        def rewrite(self, raw):
            from src.schemas import RewriteResult
            return RewriteResult(raw_query=raw, queries={})

    class Channel:
        def search(self, query, k):
            from src.schemas import ScoredPaper
            return [ScoredPaper(paper_id="2103.00020v3", score=1.0, rank=1)]

    out = pe.evaluate_one({"query_id": "q", "text": "t", "gold_id": "2103.00020v1"},
                          Rewriter(), {"local_dense": Channel()}, k=10)
    assert out["gold_id"] == "2103.00020"
    assert out["channels"]["local_dense"] == ["2103.00020"]
    assert pe.rank_in(out["channels"]["local_dense"], out["gold_id"]) == 1


def test_channel_error_does_not_stop_other_channels():
    class Rewriter:
        name = "t"

        def rewrite(self, raw):
            from src.schemas import RewriteResult
            return RewriteResult(raw_query=raw, queries={})

    class Broken:
        def search(self, query, k):
            raise RuntimeError("503")

    class Fine:
        def search(self, query, k):
            from src.schemas import ScoredPaper
            return [ScoredPaper(paper_id="A", score=1.0, rank=1)]

    out = pe.evaluate_one({"query_id": "q", "text": "t", "gold_id": "A"},
                          Rewriter(), {"arxiv": Broken(), "local_dense": Fine()}, k=10)
    assert out["channels"]["arxiv"] == []
    assert out["channels"]["local_dense"] == ["A"]
    assert "arxiv" in out["channel_errors"]
    assert "error" not in out             # 문항 전체가 죽은 것은 아니다


# ── 이어하기 ──────────────────────────────────────────────────────────────
def test_resume_retries_rows_with_any_channel_error(tmp_path):
    path = tmp_path / "run.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in [row("ok", "A", {"arxiv": ["A"]}),
                  row("err", "B", {"arxiv": []}, channel_errors={"arxiv": "503"}),
                  row("dead", "C", {"arxiv": []}, error="rewrite_failed"),
                  {"_meta": True}]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    done = pe.load_done(path)
    assert set(done) == {"ok"}            # 오류 문항 둘은 다시 시도 대상


# ── 본문 조회 (재정렬 후보 복원) ───────────────────────────────────────────
def test_text_lookup_falls_back_to_scanning_corpus(tmp_path):
    corpus = tmp_path / "mini.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        for pid in ["1111.1111", "2222.2222"]:
            f.write(json.dumps({"id": pid, "title": f"제목 {pid}",
                                "abstract": f"초록 {pid}"}) + "\n")

    lookup = pe.TextLookup(index_prefix=None, corpus_path=corpus, arxiv_cache=None)
    got = lookup.fetch({"1111.1111", "3333.3333"})
    assert got["1111.1111"] == ("제목 1111.1111", "초록 1111.1111")
    assert "3333.3333" not in got          # 못 찾은 것은 조용히 빠진다 → 하네스가 경고를 찍는다


def test_text_lookup_reads_arxiv_cache(tmp_path):
    cache = tmp_path / "cache.jsonl"
    with open(cache, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query": "q", "k": 100, "results": [
            {"paper_id": "2103.00020v2", "score": 1.0, "rank": 1,
             "title": "T", "abstract": "A"}]}) + "\n")

    lookup = pe.TextLookup(index_prefix=None, corpus_path=None, arxiv_cache=cache)
    got = lookup.fetch({"2103.00020"})     # 버전 표기를 뗀 번호로도 찾아야 한다
    assert got["2103.00020"] == ("T", "A")


# ── 보고가 터지지 않는가 (오류 문항·빈 결과 포함) ──────────────────────────
def test_report_runs_on_rows_with_errors(capsys):
    pe.print_report(ROWS, "테스트", k_values=(1, 10), rrf_k=60, weights={})
    text = capsys.readouterr().out
    assert "합집합(상한)" in text
    assert "재정렬" in text


def test_report_runs_when_nothing_is_found(capsys):
    rows = [row("q1", "A", {"arxiv": [], "local_dense": []}, error="x")]
    pe.print_report(rows, "전부 실패", k_values=(10,), rrf_k=60, weights={})
    assert "상한이 0이라 계산 불가" in capsys.readouterr().out


# ── 상한을 어느 집합으로 재는가 (ISSUE #26 이 재발한 자리) ─────────────────
#
# 합집합과 '재정렬이 실제로 본 후보' 는 다른 집합이다. 융합이 후보를 줄이기 때문이다.
# 합집합을 상한이라 부르면 **융합이 흘린 몫까지 재정렬 탓으로 넘어가** 처방이 뒤바뀐다.
# 실측에서 시험용 300문항 기준 0.823 대 0.797 로 정답 8편이 그렇게 넘어가 있었다.

FUSION_DROPS_GOLD = row(
    "q9", "GOLD",
    # 정답은 의미 검색 3등이라 합집합@3 에는 들어온다.
    # 그러나 arXiv 가 올린 P·Q 가 RRF 점수에서 앞서, 융합 상위 3편에서는 밀려난다.
    {"local_dense": ["X", "Y", "GOLD"], "arxiv": ["P", "Q", "R"]},
)


def test_합집합에는_있지만_융합_상위에서는_밀려난다():
    assert pe.union_rank(FUSION_DROPS_GOLD, depth=3) == 1
    assert pe.rerank_pool_rank(FUSION_DROPS_GOLD, rrf_k=60, depth=3, weights={}) is None


def test_융합이_흘린_몫과_재정렬이_못_건진_몫을_따로_보고한다(capsys):
    rows = [dict(FUSION_DROPS_GOLD, reranked_ids=["X", "P", "Y"], rerank_depth=3)]
    pe.print_report(rows, "융합 손실", k_values=(10,), rrf_k=60, weights={}, pool_depth=3)
    text = capsys.readouterr().out

    assert "채널 합집합" in text
    assert "재정렬이 실제로 본 후보" in text
    assert "융합이 흘린 몫" in text
    # 합집합 1.000 → 융합 후 0.000 이므로 손실 전부가 융합 몫으로 잡혀야 한다
    assert "융합이 흘린 몫  (① → ②)         : -1.000" in text
    assert "재정렬이 못 건진 몫 (② → ③)      : +0.000" in text


def test_파일에_새겨진_재정렬_깊이를_명령줄보다_우선한다(capsys):
    """재집계할 때 기본값으로 상한을 재면 ISSUE #26 이 되살아난다."""
    rows = [dict(FUSION_DROPS_GOLD, reranked_ids=["GOLD"], rerank_depth=3)]
    pe.print_report(rows, "깊이 새김", k_values=(10,), rrf_k=60, weights={}, pool_depth=200)
    text = capsys.readouterr().out

    assert "깊이 3 로 재정렬돼 있다" in text
    assert "채널 합집합 @3" in text
