"""재정렬 효과 측정 — 이미 저장된 검색 결과를 다시 줄 세워 본다 (arXiv 호출 0회).

왜 arXiv 를 다시 안 불러도 되는가:
평가 하네스가 검색된 논문 번호를 순위대로 전부 저장하고(`retrieved_ids`), arXiv 검색
캐시(`data/cache/arxiv_search_cache.jsonl`)에 그 논문들의 제목·초록이 들어 있다. 둘을
합치면 후보 100편을 그대로 복원할 수 있다. 그래서 **재정렬 방식을 바꿔 가며 몇 번을
실험해도 arXiv 요청이 한 건도 발생하지 않는다.**

무엇을 재는가:
재정렬 전후의 Recall@1/5/10/30 을 같은 질문끼리 짝지어 비교한다. 재정렬은 후보 집합을
바꾸지 않으므로 **Recall@100(후보 확보율)은 변하지 않고 그 안의 순위만 바뀐다.**
따라서 재정렬의 이론적 상한은 그 실행의 Recall@100 이다.

실행 예:
  python -m evaluation.rerank_eval --run runs/test300_grounded_dpo.jsonl
  python -m evaluation.rerank_eval --run runs/test300_grounded_dpo.jsonl --method embedding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import bootstrap_ci, paired_bootstrap
from src import config
from src.retrieval.fusion import normalize_paper_id
from src.schemas import ScoredPaper
from src.utils import read_jsonl

CACHE_PATH = config.DATA_DIR / "cache" / "arxiv_search_cache.jsonl"


def load_cache(path: Path) -> dict[tuple[str, int], list[ScoredPaper]]:
    """arXiv 검색 캐시를 (검색어, k) -> 결과 목록 으로 읽는다."""
    cache: dict[tuple[str, int], list[ScoredPaper]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[(row["query"], row["k"])] = [
                ScoredPaper(paper_id=p["paper_id"], score=p["score"], rank=p["rank"],
                            title=p.get("title", ""), abstract=p.get("abstract", ""))
                for p in row["results"]
            ]
    return cache


def norm(pid: str) -> str:
    """버전 표기를 뗀다. `split("v")[0]` 은 옛 형식 번호를 잘랐다(ISSUE #27)."""
    return normalize_paper_id(pid)


def recall_at(ranks: dict[str, int | None], k: int) -> dict[str, float]:
    return {q: (1.0 if (r and r <= k) else 0.0) for q, r in ranks.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="재정렬 효과 측정 (arXiv 호출 없음)")
    ap.add_argument("--run", required=True, help="평가 결과 jsonl (retrieved_ids 포함)")
    ap.add_argument("--cache", default=str(CACHE_PATH))
    ap.add_argument("--method", default="cross", choices=["cross", "embedding"],
                    help="cross=교차 인코더(권장) · embedding=임베딩 유사도(비교군)")
    ap.add_argument("--model", default=None, help="재정렬 모델 이름 (기본은 방식별 기본값)")
    ap.add_argument("--depth", type=int, default=100, help="재정렬에 넣을 후보 수")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None, help="재정렬 결과 저장 경로 (선택)")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.run) if not r.get("_meta") and not r.get("error")]
    cache = load_cache(Path(args.cache))
    print(f"결과 {len(rows)}문항 · 캐시 {len(cache):,}건 로드")

    # 후보 복원 — 검색어로 캐시를 찾고, 없으면 그 문항은 건너뛴다
    queries, cand_lists, kept = [], [], []
    missing = 0
    for r in rows:
        sq = r.get("search_query", "")
        # 같은 검색어라도 실행할 때 쓴 깊이(k)가 다르면 캐시 키가 다르다.
        # 깊은 것부터 찾아 가장 많은 후보를 확보한다.
        cands = next((cache[(sq, kk)] for kk in (300, 200, 100, 30) if (sq, kk) in cache), None)
        if not cands:
            missing += 1
            continue
        queries.append(r["text"])            # ★ 원본 질문으로 재정렬한다
        cand_lists.append(cands[: args.depth])
        kept.append(r)
    print(f"후보 복원 {len(kept)}문항 (캐시에 없어 제외 {missing}문항)")
    if not kept:
        print("복원된 문항이 없다. --cache 경로와 검색 깊이를 확인할 것.")
        return

    # 재정렬 전 순위
    before = {r["query_id"]: r.get("gold_rank") for r in kept}

    # 재정렬
    print(f"재정렬 방식: {args.method}")
    if args.method == "cross":
        from src.retrieval.rerank import CrossEncoderReranker, DEFAULT_RERANKER
        rr = CrossEncoderReranker(args.model or DEFAULT_RERANKER,
                                  batch_size=args.batch_size)
        reranked = rr.rerank_batch(queries, cand_lists, top_k=args.depth)
    else:
        from src.retrieval.dense import Embedder
        from src.retrieval.rerank import rerank_by_similarity
        emb = Embedder()
        reranked = [rerank_by_similarity(q, c, emb, top_k=args.depth)
                    for q, c in zip(queries, cand_lists)]

    after: dict[str, int | None] = {}
    detail = []
    for r, ranked in zip(kept, reranked):
        ids = [norm(p.paper_id) for p in ranked]
        g = norm(r["gold_id"])
        after[r["query_id"]] = (ids.index(g) + 1) if g in ids else None
        detail.append({**{k: r[k] for k in ("query_id", "text", "gold_id", "lang",
                                            "difficulty")},
                       "rank_before": r.get("gold_rank"), "rank_after": after[r["query_id"]]})

    # ── 보고 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"■ 재정렬 전후 (같은 {len(kept)}문항, 부트스트랩 95% 신뢰구간)")
    print(f"\n   {'K':>5}{'재정렬 전':>22}{'재정렬 후':>22}{'변화':>9}{'p값':>8}")
    for k in (1, 5, 10, 30):
        b = recall_at(before, k)
        a = recall_at(after, k)
        mb, lb, hb = bootstrap_ci(list(b.values()))
        ma, la, ha = bootstrap_ci(list(a.values()))
        res = paired_bootstrap(b, a)
        mark = " *" if res["p_value"] < 0.05 else ""
        print(f"   {k:>5}{mb:>9.3f} [{lb:.3f},{hb:.3f}]{ma:>9.3f} [{la:.3f},{ha:.3f}]"
              f"{res['delta']:>+9.3f}{res['p_value']:>8.3f}{mark}")
    ceiling = sum(1 for v in before.values() if v) / len(before)
    print(f"\n   재정렬의 상한(=재정렬 전 Recall@{args.depth}): {ceiling:.3f}")
    print("   (재정렬은 후보를 바꾸지 않고 순서만 바꾸므로 이 값을 넘을 수 없다)")

    # 언어별
    print("\n■ 언어별 Recall@10")
    for lang in sorted({r.get("lang") for r in kept}):
        ids = [r["query_id"] for r in kept if r.get("lang") == lang]
        b = sum(recall_at(before, 10)[q] for q in ids) / len(ids)
        a = sum(recall_at(after, 10)[q] for q in ids) / len(ids)
        print(f"   {lang} (n={len(ids)}): {b:.3f} → {a:.3f}  ({a-b:+.3f})")

    if args.out:
        from src.utils import write_jsonl
        write_jsonl(args.out, detail)
        print(f"\n문항별 결과 저장: {args.out}")


if __name__ == "__main__":
    main()
