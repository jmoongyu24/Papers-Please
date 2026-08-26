"""검색 모델(임베딩) 미세조정용 학습 쌍 만들기 - 오답을 유사도 간격으로 고름.

## 무엇을 만드는가

    질문        ft_queries.jsonl 의 한 문항
    정답 논문    그 문항을 만들어 낸 논문의 제목 + 초록
    오답 논문    "가깝긴 한데 정답은 아닌" 논문 여러 편

대조 학습은 정답을 가깝게, 오답을 멀게 당기는 방식이라 **오답을 어떻게 고르느냐가
학습의 성패를 가름.** 오답이 너무 쉬우면(무작위 논문) 배울 것이 없고, 너무 어려우면
(상위권에서 그냥 집으면) 그 안에 진짜 좋은 논문이 섞여 "좋은 논문을 내려라"를 가르침.

## 오답 고르는 규칙 (2026-08-25 실측으로 다시 정함)

**처음 계획한 "정답보다 유사도 0.05 낮은 것만" 은 이 자리에서 쓸 수 없음.** 그 규칙은
등급 정답지의 후보 묶음(짝당 71편)에서 재서 정한 것인데, 로컬 색인의 상위 목록은 점수가
훨씬 좁은 구간에 몰려 있어서 규칙에 걸리는 오답이 거의 안 나옴. 문항 4,000개로 확인함.

    등수별 유사도 중앙값   1등 0.642 · 10등 0.610 · 100등 0.579 · 1000등 0.543
    정답의 유사도 중앙값   0.565   (즉 정답이 대개 100~200등 언저리 점수임)

정답보다 0.05 낮으려면 0.515 아래여야 하는데 1000등이 0.543 임. 그래서:

    오답이 한 편도 안 나온 문항   전체 80.0% (medium 73.4% · hard 86.6%)
    정답이 상위 200 밖인 문항     100.0% 에서 오답 0편

**정작 고치려던 일상어 층에서 학습 신호가 통째로 비는 규칙임.** 그대로 쓰면 오답 자리를
무작위 논문으로 채우게 되어 학습이 안 됨.

### 그래서 재정렬기로 거름

교차 인코더(`bge-reranker-v2-m3`)에게 후보를 채점시켜 **점수가 가장 낮은 것부터** 오답으로
씀. 넓힌 등급 정답지로 규칙별 오염도를 다시 쟀음(`runs/dev_service_repro.jsonl` 의 저장된
재정렬 점수 + `grades_dev.jsonl`, 새 계산 0회).

    오답 6편을 뽑는 규칙          전체 등급2+   hard 등급2+   판정받은 비율   6편 못 채움
    상위 30등에서                  0.609       0.626        1.000       0.000
    재정렬 점수 0.002 미만           0.249       0.380        0.199       0.233
    재정렬 점수 낮은 것부터           0.160       0.172        0.090       0.000

재정렬 점수와 등급의 관계는 후보 100편 전부에서 확인했음(판정 12,910건, 빈틈 없음).

    재정렬 점수      등급0    등급1    등급2    등급3     등급2 이상
    0    ~0.002    0.278   0.497   0.199   0.025    0.224
    0.002~0.02     0.143   0.509   0.296   0.051    0.347
    0.02 ~0.2      0.070   0.477   0.343   0.111    0.454
    0.2  ~1.01     0.021   0.308   0.416   0.254    0.670

점수가 낮을수록 좋은 논문일 확률이 확실히 낮아짐. 상위 30등에서 그냥 집는 것(0.609)보다
오염이 4분의 1로 줄고, **오답이 항상 6편 채워짐.**

이 규칙이 이 시스템에서 정당한 이유가 하나 더 있음. 서비스는 재정렬 점수가 0.002 아래인
논문을 사용자에게 접어서 안 보여 줌(`app.py` 의 `MIN_RERANK_SCORE`). 즉 **재정렬기가
무관하다고 한 논문은 이 시스템이 이미 무관하다고 정의한 논문임.** 그것보다 정답을 위에
놓으라고 가르치는 것은 시스템 자신의 기준과 어긋나지 않음.

**한계도 적어 둠:** 0.160 이라는 값은 뽑힌 오답의 9.0% 만 등급을 받아서 잰 것이라 흔들림이
큼. 빈틈없이 잰 것은 위의 점수-등급 표(0.224)이고, 그쪽도 같은 방향임.

## 검증용을 떼어 두는 이유 (반드시 읽을 것)

학습 자료와 평가셋의 **분야 분포가 다름.** `evaluation/dataset.py` 의 `sample_papers`
설명글에 적혀 있듯이, 코퍼스의 45.9%인 cs.CV·cs.LG·cs.CL·cs.AI 가 개발용 평가셋에서는
2.9%(348문항 중 10개)뿐임. 학습 자료는 코퍼스 비율 그대로 뽑았으므로 그 분야가 절반 가까움.

그래서 평가셋에서 이득이 안 보였을 때 두 가지를 구분할 수 없음.

    ⓐ 방법 자체가 안 되는 것
    ⓑ 되긴 하는데 평가셋에 그 분야가 거의 없어서 안 보이는 것

논문 500편(문항 2,000개)을 학습에서 빼 두면 ⓐ 와 ⓑ 를 가를 수 있음. 검증용에서도
안 오르면 ⓐ 이므로 거기서 접음. 검증용에서는 오르는데 평가셋에서만 안 오르면 ⓑ 이고,
그때는 분야를 맞춘 자료를 더 만드는 것이 다음 수임.

**나누는 기준은 문항이 아니라 논문임.** 문항으로 나누면 같은 논문에서 나온 한국어판과
영어판이 학습과 검증에 갈라져 들어가 검증이 새 버림.

## 실행

    $PY -m training.build_embed_pairs --queries data/training/ft_queries.jsonl \
        --out-dir data/training --negatives 6 --rerank-pool 24

전제: 지금 색인(data/embeddings/cs2021)과 재정렬 모델이 있어야 함.
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
    ap = argparse.ArgumentParser(description="임베딩 미세조정용 학습 쌍 만들기")
    ap.add_argument("--queries", default="data/training/ft_queries.jsonl")
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--index", default=str(config.DATA_DIR / "embeddings" / "cs2021"))
    ap.add_argument("--out-dir", default="data/training")
    ap.add_argument("--negatives", type=int, default=6, help="문항당 오답 편수")
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
    from src.retrieval.local_index import LocalDenseRetriever
    print("색인 불러오는 중... (짝 확인 포함)", flush=True)
    t0 = time.time()
    ret = LocalDenseRetriever(args.corpus, args.index)
    print(f"색인 준비 완료 ({time.time() - t0:.0f}초, 논문 {len(ret.ids):,}편)", flush=True)

    # 정답 논문이 색인에 없는 문항은 학습 쌍을 만들 수 없음
    missing = [r for r in rows if normalize_paper_id(r["gold_id"]) not in ret._pos]
    if missing:
        print(f"경고: 정답 논문이 색인에 없는 문항 {len(missing)}개를 버림")
    rows = [r for r in rows if normalize_paper_id(r["gold_id"]) in ret._pos]

    gold_pos = [ret._pos[normalize_paper_id(r["gold_id"])] for r in rows]

    # -- 채점 -------------------------------------------------------------
    print(f"질문 {len(rows):,}개를 색인 {len(ret.ids):,}편과 대조하는 중...", flush=True)
    t0 = time.time()
    top_idx, top_score, gold_score = score_batch(
        ret, [r["text"] for r in rows], args.top_k, gold_pos)
    print(f"채점 완료 ({time.time() - t0:.0f}초)", flush=True)

    # -- 진단: 정답이 지금 몇 등인가 (미세조정 전 기준값) --------------------
    rank_before = []
    for i in range(len(rows)):
        w = np.where(top_idx[i] == gold_pos[i])[0]
        rank_before.append(int(w[0]) + 1 if len(w) else None)

    print(f"\n[미세조정 전 정답 등수] 색인 {len(ret.ids):,}편 안에서 상위 {args.top_k}등까지 훑음")
    print(f"  {'난이도':<8}{'문항':>8}{'10등내':>9}{'100등내':>9}"
          f"{str(args.top_k)+'등내':>9}{'든 것의 중앙값':>14}")
    for d in ("medium", "hard", None):
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
    for i in range(len(rows)):
        need.update(int(x) for x in top_idx[i][: args.rerank_pool + 1])
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
    print("재정렬기를 올리는 중...", flush=True)
    from src.retrieval.ranking import CrossEncoderReranker
    from src.schemas import ScoredPaper
    reranker = CrossEncoderReranker(batch_size=64)

    neg_pos: list[list[int]] = []
    neg_score: list[list[float]] = []
    t0 = time.time()
    step = 200                                   # 문항 200개씩 묶어 채점
    for s0 in range(0, len(rows), step):
        chunk = list(range(s0, min(s0 + step, len(rows))))
        queries, cand_lists, cand_pos = [], [], []
        for i in chunk:
            gp = gold_pos[i]
            pos = [int(x) for x in top_idx[i] if int(x) != gp][: args.rerank_pool]
            cand_pos.append(pos)
            queries.append(rows[i]["text"])
            cand_lists.append([ScoredPaper(paper_id=str(p), score=0.0, rank=j + 1,
                                           title="", abstract=body[p])
                               for j, p in enumerate(pos)])
        # top_k 를 후보 수 전체로 주어 모든 후보의 점수를 받음. 그중 낮은 쪽을 씀.
        ranked = reranker.rerank_batch(queries, cand_lists, top_k=args.rerank_pool)
        for row_out in ranked:
            pairs = sorted(((int(c.paper_id), c.score) for c in row_out), key=lambda t: t[1])
            neg_pos.append([p for p, _ in pairs[: args.negatives]])
            neg_score.append([round(sc, 6) for _, sc in pairs[: args.negatives]])
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
            "positive": body[gold_pos[i]],
            "negatives": [body[p] for p in neg_pos[i]],
            "gold_id": normalize_paper_id(r["gold_id"]),
            "lang": r["lang"],
            "difficulty": r["difficulty"],
            "gold_score": round(float(gold_score[i]), 4),
            "rank_before": rank_before[i],
            "neg_rerank_scores": neg_score[i],
        }
        (out_val if normalize_paper_id(r["gold_id"]) in val_papers
         else out_train).append(item)

    # -- 저장 ----------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("embed_pairs_train.jsonl", out_train),
                       ("embed_pairs_val.jsonl", out_val)):
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
    print(f"  상위 {args.rerank_pool}편을 채점해 점수가 낮은 {args.negatives}편을 씀")
    if flat:
        a = np.asarray(flat)
        print(f"  뽑힌 오답의 재정렬 점수: 중앙값 {np.median(a):.4f} · "
              f"0.002 미만 {np.mean(a < 0.002):.3f} · 0.02 미만 {np.mean(a < 0.02):.3f}")


if __name__ == "__main__":
    main()
