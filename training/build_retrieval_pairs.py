"""검색 모델과 재정렬기 파인튜닝용 학습 쌍 만들기.

    질문        train_queries.jsonl 의 한 문항
    정답 논문    그 문항을 만들어 낸 논문의 제목 + 초록
    오답 논문    "가깝긴 한데 정답은 아닌" 논문 여러 편

대조 학습은 정답을 가깝게 오답을 멀게 당기는 방식이라 오답을 어떻게 고르느냐가 성패를
가름. 오답이 너무 쉬우면 배울 것이 없고, 너무 어려우면 그 안에 진짜 좋은 논문이 섞여
"좋은 논문을 내려라" 를 가르치게 됨.

## 오답 고르는 규칙

교차 인코더에게 후보를 채점시켜 점수가 가장 낮은 것부터 오답으로 씀. 등급 정답지로
규칙별 오염도를 재서 정한 것임.

    오답 6편을 뽑는 규칙        전체 등급2+   일상어 등급2+   6편 못 채움
    상위 30등에서                 0.609        0.626        0.000
    재정렬 점수 낮은 것부터         0.160        0.172        0.000

상위 30등에서 그냥 집는 것보다 오염이 4분의 1로 줄고 오답이 항상 채워짐.

처음 계획한 "정답보다 유사도 0.05 낮은 것만" 은 쓸 수 없었음. 로컬 색인의 상위 목록은
점수가 좁은 구간에 몰려 있어(1등 0.642, 1000등 0.543, 정답 0.565) 규칙에 걸리는 오답이
거의 안 나오고, 문항의 80.0% 에서 오답이 한 편도 안 나왔음.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src import config
from src.retrieval.corpus import normalize_paper_id
from src.utils import read_jsonl


def doc_text(row: dict) -> str:
    """색인을 만들 때와 **글자 그대로 같은** 방식으로 논문 글을 만듦.

    `local_index.build_embeddings` 가 쓰는 것과 다르면, 학습할 때 본 글과 색인에 들어간
    글이 달라져서 학습 효과가 그대로 전달되지 않음. 오류는 안 나고 성능만 조용히 떨어짐.
    """
    return f"{row.get('title', '').strip()}\n{row.get('abstract', '').strip()}"


def score_batch(retriever, queries: list[str], top_k: int,
                gold_positions: list[int | None], batch_size: int = 256):
    """질문 여러 개를 한 번에 채점함.

    한 개씩 돌리면 71만 편과의 내적을 질문마다 따로 하게 되어 24,000문항에 시간이 많이 듦.
    질문을 묶어 한 번의 행렬 곱으로 처리하면 훨씬 빠름.

    Returns:
        top_idx    (질문 수, top_k)  상위 후보의 배열 위치
        top_score  (질문 수, top_k)  그 유사도
        gold_score (질문 수,)        정답 논문의 유사도 (색인에 없으면 nan)
    """
    emb = retriever.emb
    n_q = len(queries)
    top_idx = np.zeros((n_q, top_k), dtype=np.int64)
    top_score = np.zeros((n_q, top_k), dtype=np.float32)
    gold_score = np.full(n_q, np.nan, dtype=np.float32)

    for s in range(0, n_q, batch_size):
        chunk = queries[s:s + batch_size]
        q = retriever.embedder.encode(chunk, normalize_embeddings=True,
                                      convert_to_numpy=True, batch_size=64)
        q = np.asarray(q, dtype=np.float32)
        scores = emb @ q.T                       # (논문 수, 이번 묶음 질문 수)
        for j in range(len(chunk)):
            col = scores[:, j]
            part = np.argpartition(-col, top_k - 1)[:top_k]
            part = part[np.argsort(-col[part])]
            top_idx[s + j] = part
            top_score[s + j] = col[part]
            pos = gold_positions[s + j]
            if pos is not None:
                gold_score[s + j] = col[pos]
        print(f"  채점 {min(s + batch_size, n_q):,}/{n_q:,}", flush=True)
    return top_idx, top_score, gold_score


def main() -> None:
    ap = argparse.ArgumentParser(description="검색 모델 파인튜닝용 학습 쌍 만들기")
    ap.add_argument("--queries", default="data/training/train_queries.jsonl")
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--index", default=str(config.DATA_DIR / "embeddings" / "cs2021-ft"))
    ap.add_argument("--embed-model", default=None,
                    help="질문을 임베딩할 모델. 안 주면 색인을 만든 모델을 그대로 씀. "
                         "색인과 다른 모델을 주면 멈춤")
    ap.add_argument("--out-dir", default="data/training")
    ap.add_argument("--negatives", type=int, default=6, help="문항당 오답 편수")
    ap.add_argument("--pool-from-en", action="store_true",
                    help="후보를 같은 짝(pair_id)의 영어 질문으로 뽑는다. 서비스가 한국어를 "
                         "영어로 옮겨 검색하는 것과 조건을 맞춘다")
    ap.add_argument("--for-rerank", action="store_true",
                    help="재정렬기 파인튜닝용 학습 쌍을 만든다. 오답은 '정답을 뺀 상위 N편' 이고 "
                         "결과를 train/val_reranker.jsonl 에 query/docs/labels 로 저장한다")
    ap.add_argument("--neg-source", choices=["rerank", "retrieval"], default="rerank",
                    help="오답을 어느 순서에서 고를지. rerank 는 교차 인코더로 후보를 다시 "
                         "채점해 그 순서를 씀(느림, 33,000문항에 4시간 20분). retrieval 은 "
                         "색인 등수를 그대로 씀(빠름, 약 20~30분). --for-rerank 에서만 뜻이 있음")
    ap.add_argument("--rerank-pool", type=int, default=24,
                    help="재정렬기에게 채점시킬 상위 후보 수. 이 중 점수가 가장 낮은 "
                         "--negatives 편을 오답으로 씀")
    ap.add_argument("--top-k", type=int, default=200,
                    help="정답이 지금 몇 등인지 보려고 훑는 깊이 (오답은 --rerank-pool 에서 고름)")
    ap.add_argument("--batch-size", type=int, default=250, help="질문을 몇 개씩 묶어 채점할지")
    ap.add_argument("--val-papers", type=int, default=500,
                    help="학습에서 뺄 논문 수 (그 논문의 문항 전부가 검증용이 됨)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N문항만 (점검용)")
    ap.add_argument("--sample", type=int, default=None,
                    help="무작위 N문항만 (분포를 보려면 앞에서 자르지 말고 이쪽을 쓸 것)")
    ap.add_argument("--diagnose", action="store_true",
                    help="학습 쌍을 만들지 않고 오답이 몇 편이나 나오는지만 셈 (본문을 안 읽어 빠름)")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    if args.limit:
        rows = rows[: args.limit]
    if args.sample and args.sample < len(rows):
        rows = random.Random(args.seed).sample(rows, args.sample)
    print(f"문항 {len(rows):,}개")

    # -- 논문 단위로 학습/검증 가르기 ------------------------------------
    papers = sorted({normalize_paper_id(r["gold_id"]) for r in rows})
    rng = random.Random(args.seed)
    val_papers = set(rng.sample(papers, min(args.val_papers, len(papers))))
    print(f"논문 {len(papers):,}편 -> 학습 {len(papers) - len(val_papers):,}편 "
          f", 검증 {len(val_papers):,}편")

    # -- 색인 올리기 ------------------------------------------------------
    #
    # 색인을 만든 모델과 질문을 임베딩하는 모델이 다르면 오류 없이 순위만 무너짐.
    # 평가 코드에서 이미 한 번 겪은 자리임. 여기서도 막음
    from src.retrieval.local_index import LocalDenseRetriever, read_meta
    meta_model = read_meta(args.index).get("model")
    embed_model = args.embed_model or meta_model or config.EMBED_MODEL
    if meta_model and embed_model != meta_model:
        raise SystemExit(
            f"색인을 만든 모델과 질문을 찾을 모델이 다르다.\n"
            f"  색인({args.index})을 만든 모델: {meta_model}\n"
            f"  질문을 임베딩할 모델        : {embed_model}\n"
            f"이대로 쓰면 오류 없이 순위만 무너진다. --embed-model 을 맞추거나 빼고 돌릴 것.")
    print(f"색인 불러오는 중... (짝 확인 포함, 모델 {embed_model})", flush=True)
    t0 = time.time()
    ret = LocalDenseRetriever(args.corpus, args.index, model_name=embed_model)
    print(f"색인 준비 완료 ({time.time() - t0:.0f}초, 논문 {len(ret.ids):,}편)", flush=True)

    # 정답 논문이 색인에 없는 문항은 학습 쌍을 만들 수 없음
    missing = [r for r in rows if normalize_paper_id(r["gold_id"]) not in ret._pos]
    if missing:
        print(f"경고: 정답 논문이 색인에 없는 문항 {len(missing)}개를 버림")
    rows = [r for r in rows if normalize_paper_id(r["gold_id"]) in ret._pos]

    gold_pos = [ret._pos[normalize_paper_id(r["gold_id"])] for r in rows]

    # -- 채점 -------------------------------------------------------------
    #
    # 후보를 뽑는 데 쓰는 글과, 재정렬기에 넣는 글은 서로 다를 수 있음.
    # 서비스가 한국어를 영어로 옮겨 검색하기 때문임 (--pool-from-en 설명글 참고).
    search_text = [r["text"] for r in rows]
    if args.pool_from_en:
        en_of = {r["pair_id"]: r["text"] for r in rows if r.get("lang") == "en"}
        n_swap = 0
        for i, r in enumerate(rows):
            if r.get("lang") != "en" and r.get("pair_id") in en_of:
                search_text[i] = en_of[r["pair_id"]]
                n_swap += 1
        print(f"후보 뽑기용 글을 영어 짝으로 바꾼 문항 {n_swap:,}개 "
              f"(재정렬기에 넣는 질문은 원문 그대로)")

    print(f"질문 {len(rows):,}개를 색인 {len(ret.ids):,}편과 대조하는 중...", flush=True)
    t0 = time.time()
    top_idx, top_score, gold_score = score_batch(
        ret, search_text, args.top_k, gold_pos)
    print(f"채점 완료 ({time.time() - t0:.0f}초)", flush=True)

    # -- 진단: 정답이 지금 몇 등인가 (파인튜닝 전 기준값) --------------------
    rank_before = []
    for i in range(len(rows)):
        w = np.where(top_idx[i] == gold_pos[i])[0]
        rank_before.append(int(w[0]) + 1 if len(w) else None)

    print(f"\n[파인튜닝 전 정답 등수] 색인 {len(ret.ids):,}편 안에서 상위 {args.top_k}등까지 훑음")
    print(f"  {'난이도':<8}{'문항':>8}{'10등내':>9}{'100등내':>9}"
          f"{str(args.top_k)+'등내':>9}{'든 것의 중앙값':>14}")
    for d in ("easy", "medium", "hard", None):
        sel = [i for i, r in enumerate(rows) if d is None or r["difficulty"] == d]
        if not sel:
            continue
        rr = [rank_before[i] for i in sel]
        f = [x for x in rr if x is not None]
        print(f"  {(d or '전체'):<8}{len(rr):>8,}"
              f"{sum(1 for x in f if x <= 10) / len(rr):>9.3f}"
              f"{sum(1 for x in f if x <= 100) / len(rr):>9.3f}"
              f"{len(f) / len(rr):>9.3f}"
              f"{(int(np.median(f)) if f else 0):>14,}")

    if args.diagnose:
        return

    # -- 오답 후보의 본문을 꺼내 읽기 ---------------------------------------
    #
    # 재정렬기는 제목과 초록을 읽어야 채점할 수 있으므로, 채점할 후보의 본문이 먼저 필요함.
    # 정답 논문과 상위 --rerank-pool 편만 읽음 (200등까지 전부 읽으면 낭비임).
    need = set(gold_pos)
    # 색인 등수로 오답을 고를 때는 상위 (오답 수 + 1)편만 있으면 됨. 교차 인코더로 고를
    # 때는 채점할 후보 전체(--rerank-pool)의 본문이 필요함.
    cheap = args.for_rerank and args.neg_source == "retrieval"
    n_pool = (args.negatives + 1) if cheap else (args.rerank_pool + 1)
    for i in range(len(rows)):
        need.update(int(x) for x in top_idx[i][: n_pool])
    need = sorted(need)
    print(f"\n본문을 꺼낼 논문 {len(need):,}편", flush=True)
    t0 = time.time()
    body = {pos: doc_text(row) for pos, row in zip(need, ret.read_rows(need))}
    print(f"본문 읽기 완료 ({time.time() - t0:.0f}초)", flush=True)

    # -- 재정렬기로 오답 고르기 ---------------------------------------------
    #
    # 각 문항의 상위 후보를 교차 인코더에게 채점시켜, **점수가 가장 낮은** 편부터 오답으로 씀.
    # 왜 이 규칙인지는 이 파일 맨 위 설명글 참고 (상위 등수에서 그냥 집으면 뽑힌 오답의
    # 60.9% 가 실제로는 쓸모 있는 논문이었음).
    neg_pos: list[list[int]] = []
    neg_score: list[list[float]] = []
    gold_rr: list[float | None] = []             # --for-rerank 에서만 채움
    gold_rr_rank: list[int | None] = []

    if cheap:
        # 색인 등수 그대로 정답을 뺀 상위 N편을 오답으로 씀. 교차 인코더를 한 번도 안 부름.
        # 이 논문들은 서비스가 재정렬기에게 실제로 넘기는 후보와 같은 자리에서 나온 것임.
        print(f"오답을 색인 등수에서 고름 (정답을 뺀 상위 {args.negatives}편, 교차 인코더 안 부름)")
        for i in range(len(rows)):
            gp = gold_pos[i]
            neg_pos.append([int(x) for x in top_idx[i] if int(x) != gp][: args.negatives])
            neg_score.append([])
            gold_rr.append(None)
            gold_rr_rank.append(None)
        print(f"오답 고르기 완료 (0초)")
        return_early = True
    else:
        return_early = False

    if not return_early:
        print("재정렬기를 올리는 중...", flush=True)
    from src.retrieval.ranking import CrossEncoderReranker
    from src.schemas import ScoredPaper
    reranker = None if return_early else CrossEncoderReranker(batch_size=64)

    t0 = time.time()
    step = 200                                   # 문항 200개씩 묶어 채점
    for s0 in ([] if return_early else range(0, len(rows), step)):
        chunk = list(range(s0, min(s0 + step, len(rows))))
        queries, cand_lists, cand_pos = [], [], []
        for i in chunk:
            gp = gold_pos[i]
            pos = [int(x) for x in top_idx[i] if int(x) != gp][: args.rerank_pool]
            if args.for_rerank:
                # 정답도 함께 채점해야 '정답보다 위' 를 가릴 수 있음.
                pos = pos + [gp]
            cand_pos.append(pos)
            queries.append(rows[i]["text"])
            cand_lists.append([ScoredPaper(paper_id=str(p), score=0.0, rank=j + 1,
                                           title="", abstract=body[p])
                               for j, p in enumerate(pos)])
        # top_k 를 후보 수 전체로 주어 모든 후보의 점수를 받음.
        ranked = reranker.rerank_batch(queries, cand_lists,
                                       top_k=args.rerank_pool + (1 if args.for_rerank else 0))
        for k, row_out in enumerate(ranked):
            # rerank_batch 는 점수가 높은 순으로 돌려줌.
            scored = [(int(c.paper_id), float(c.score)) for c in row_out]
            if not args.for_rerank:
                pairs = sorted(scored, key=lambda t: t[1])
                neg_pos.append([p for p, _ in pairs[: args.negatives]])
                neg_score.append([round(sc, 6) for _, sc in pairs[: args.negatives]])
                continue

            gp = gold_pos[chunk[k]]
            gs = next((sc for pid, sc in scored if pid == gp), None)
            rank = next((j + 1 for j, (pid, _) in enumerate(scored) if pid == gp), None)
            # 정답을 뺀 상위 N편. 2026-08-28 이전에는 여기에 `sc > gs`(정답보다 점수가
            # 높은 것) 조건이 있었는데, 정답의 재정렬 등수 중앙값이 1등이라 문항의 64.5%가
            # 오답 0편이 됐음. 그 문항들은 학습 때 묶음 안 다른 질문의 논문(쉬운 오답)만
            # 상대하게 되어 "주제만 겹치면 높은 점수" 를 배웠음
            others = [(pid, sc) for pid, sc in scored if pid != gp]
            neg_pos.append([pid for pid, _ in others[: args.negatives]])
            neg_score.append([round(sc, 6) for _, sc in others[: args.negatives]])
            gold_rr.append(None if gs is None else round(gs, 6))
            gold_rr_rank.append(rank)
        done = min(s0 + step, len(rows))
        el = time.time() - t0
        print(f"  오답 고르기 {done:,}/{len(rows):,}  경과 {el/60:.1f}분 "
              f",  남은 예상 {(len(rows)-done)/max(done,1)*el/60:.1f}분", flush=True)

    # -- 학습 쌍 만들기 ------------------------------------------------------
    out_train, out_val = [], []
    n_short = 0
    for i, r in enumerate(rows):
        if len(neg_pos[i]) < args.negatives:
            n_short += 1
        item = {
            "query_id": r["query_id"],
            "query": r["text"],
            "gold_id": normalize_paper_id(r["gold_id"]),
            "lang": r["lang"],
            "difficulty": r["difficulty"],
            "gold_score": round(float(gold_score[i]), 4),
            "rank_before": rank_before[i],
            "neg_rerank_scores": neg_score[i],
        }
        if args.for_rerank:
            # `RankNetLoss` / `LambdaLoss` 가 받는 모양임: (질문, 글 목록) + 라벨 목록.
            # 정답을 맨 앞에 둠. 라벨은 순서만 가르치는 값이라 1 과 0 이면 충분함 -
            # "이 논문은 무관하다" 를 가르치는 것이 아니라 "정답이 이 논문들보다 위" 만 가르침.
            # `CachedMultipleNegativesRankingLoss` 로 학습할 때는 train.py 가 이 두 열에서
            # anchor / positive / negative_N 을 만들어 씀.
            item["docs"] = [body[gold_pos[i]]] + [body[p] for p in neg_pos[i]]
            item["labels"] = [1] + [0] * len(neg_pos[i])
            item["gold_rerank_score"] = gold_rr[i]
            item["gold_rerank_rank"] = gold_rr_rank[i]
        else:
            item["positive"] = body[gold_pos[i]]
            item["negatives"] = [body[p] for p in neg_pos[i]]
        (out_val if normalize_paper_id(r["gold_id"]) in val_papers
         else out_train).append(item)

    # -- 저장 ----------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 파일 이름은 어느 모듈을 가르치는 자료인지로 지음 (docs/ARTIFACTS.md 4절).
    module = "reranker" if args.for_rerank else "retriever"
    for name, data in ((f"train_{module}.jsonl", out_train),
                       (f"val_{module}.jsonl", out_val)):
        fp = out_dir / name
        with open(fp, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"-> {fp}  {len(data):,}줄")

    # -- 보고 ----------------------------------------------------------------
    flat = [sc for row_sc in neg_score for sc in row_sc]
    print("\n[학습 쌍 요약]")
    print(f"  학습 {len(out_train):,}문항 · 검증 {len(out_val):,}문항")
    print(f"  오답이 {args.negatives}편에 모자란 문항 {n_short:,}개")
    if args.for_rerank:
        if cheap:
            print(f"  색인 등수에서 정답을 뺀 상위 {args.negatives}편을 오답으로 씀 "
                  f"(교차 인코더 안 부름)")
        else:
            print(f"  상위 {args.rerank_pool}편과 정답을 함께 채점해 정답을 뺀 "
                  f"상위 {args.negatives}편을 씀")
        cnt = [len(x) for x in neg_pos]
        print(f"  오답 편수 분포: " + " · ".join(
            f"{k}편 {sum(1 for c in cnt if c == k):,}개" for k in range(args.negatives + 1)))
        zero = sum(1 for c in cnt if c == 0) / len(cnt)
        print(f"  오답 0편 비율 {zero:.3f}  <- 이 값이 크면 학습 신호가 통째로 빔")
        print(f"  저장 형식: query / docs(정답 맨 앞) / labels([1, 0, ...])")
        gr = [x for x in gold_rr if x is not None]
        if gr:
            a = np.asarray(gr)
            print(f"  정답의 재정렬 점수: 중앙값 {np.median(a):.4f} · "
                  f"0.002 미만 {np.mean(a < 0.002):.3f}  <- 낮을수록 재정렬기가 못 알아본 것")
        rk = [x for x in gold_rr_rank if x is not None]
        if rk:
            b = np.asarray(rk)
            print(f"  재정렬 뒤 정답 등수: 중앙값 {np.median(b):.0f} · "
                  f"10등 안 {np.mean(b <= 10):.3f}  <- 파인튜닝 전 기준값")
    else:
        print(f"  상위 {args.rerank_pool}편을 채점해 점수가 낮은 {args.negatives}편을 씀")
    if flat:
        a = np.asarray(flat)
        print(f"  뽑힌 오답의 재정렬 점수: 중앙값 {np.median(a):.4f} · "
              f"0.002 미만 {np.mean(a < 0.002):.3f} · 0.02 미만 {np.mean(a < 0.02):.3f}")


if __name__ == "__main__":
    main()
