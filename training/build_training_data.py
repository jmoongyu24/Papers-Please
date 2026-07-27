"""쿼리 변환기 학습 데이터 생성 — '실제로 논문을 찾아내는 쿼리'를 정답 라벨로 만든다.

핵심 아이디어 (왜 이 방식인가):
지금 변환기의 문제는 지식 부족이 아니라, **"어떤 표현이 arXiv에서 실제로 통하는지" 모른다**는
것이다(ISSUE #7). 그래서 사람이 "좋아 보이는" 라벨을 쓰지 않고, **arXiv 검색 결과를 정답
신호로 삼는다**. 이를 best-of-N rejection sampling 이라고 한다.

    질문 Q(정답 논문 P가 정해져 있음)
      → 변환기가 후보 쿼리를 N개 생성 (온도를 높여 서로 다르게)
      → 각 후보로 실제 arXiv 검색
      → 정답 논문 P를 가장 높은 순위로 찾아낸 후보 = 그 질문의 정답 라벨 ★
      → 하나도 못 찾으면 그 질문은 학습에서 제외(잘못된 패턴을 배우지 않도록)

이렇게 만들면 라벨이 "학술적으로 그럴싸한 표현"이 아니라 **"진짜 찾아내는 표현"** 이 되어,
문제의 정곡을 찌른다. 실패한 후보도 버리지 않고 선호 학습(DPO)용 쌍으로 함께 저장한다.

출력 (data/training/):
  sft_pairs.jsonl   : {"input": 질문, "output": 가장 잘 찾은 쿼리}  ← 지도 미세조정(SFT)용
  dpo_pairs.jsonl   : {"input": 질문, "chosen": 잘 찾은 쿼리, "rejected": 못 찾은 쿼리} ← DPO용
  candidates.jsonl  : 모든 후보와 점수(분석·재사용용)

실행 예:
  python -m training.build_training_data --queries data/eval/train.jsonl \
      --n-candidates 5 --limit 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src import config
from src.retrieval.arxiv_live import ArxivLiveRetriever
from src.rewriter.hierarchical import HierarchicalRewriter, build_arxiv_query
from src.rewriter.llm_client import OllamaClient
from src.rewriter.prompts import OUTPUT_SCHEMA, SYSTEM, build_messages
from src.utils import read_jsonl, write_jsonl

OUT_DIR = config.DATA_DIR / "training"
CACHE_PATH = config.DATA_DIR / "cache" / "arxiv_search_cache.jsonl"


def normalize_id(paper_id: str) -> str:
    return paper_id.split("v")[0]


def score_query(query: str, gold_id: str, retriever, k: int = 30) -> tuple[float, int | None]:
    """후보 쿼리로 검색해 '정답 논문이 몇 등인가'로 점수를 매긴다.

    점수 = 1 / 순위 (1등이면 1.0, 2등이면 0.5 …). 못 찾으면 0.
    단순히 찾았나/못 찾았나보다 세밀해서 후보 간 우열을 잘 가른다.
    """
    try:
        results = retriever.search(query, k=k)
    except Exception:
        return -1.0, None          # 오류는 '점수 없음'(-1)으로 표시해 라벨 선정에서 제외
    ids = [normalize_id(r.paper_id) for r in results]
    gold = normalize_id(gold_id)
    if gold in ids:
        rank = ids.index(gold) + 1
        return 1.0 / rank, rank
    return 0.0, None


def generate_candidates(rewriter: HierarchicalRewriter, question: str,
                        n: int, temperature: float) -> list[str]:
    """같은 질문에 대해 서로 다른 변환 후보를 n개 만든다.

    온도를 높여 매번 다른 학술 용어가 나오게 한다. 어떤 표현이 arXiv에서 통할지 모르므로
    여러 개를 만들어 실제로 시험해 보기 위함이다.
    """
    seen, candidates = set(), []
    for i in range(n):
        # 첫 후보는 서비스와 동일하게 온도 0(결정적), 나머지는 다양성을 위해 온도를 올린다
        temp = 0.0 if i == 0 else temperature
        try:
            data = rewriter.client.generate_json(
                build_messages(question), OUTPUT_SCHEMA, system=SYSTEM, temperature=temp,
            )
            terms = list(data.get("academic_terms", []))
            q = build_arxiv_query(question, terms)
        except Exception:
            continue
        if q and q not in seen:
            seen.add(q)
            candidates.append(q)
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser(description="검색 성공을 정답 신호로 쓰는 학습 데이터 생성")
    ap.add_argument("--queries", default="data/eval/train.jsonl")
    ap.add_argument("--n-candidates", type=int, default=5, help="질문당 후보 쿼리 수")
    ap.add_argument("--temperature", type=float, default=0.9, help="후보 다양성 온도")
    ap.add_argument("--k", type=int, default=30, help="채점 시 볼 검색 결과 수")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    rows = list(read_jsonl(args.queries))
    if args.limit:
        rows = rows[: args.limit]

    rewriter = HierarchicalRewriter(OllamaClient())
    retriever = ArxivLiveRetriever(cache_path=CACHE_PATH)   # 디스크 캐시로 중단·재개 가능

    sft, dpo, cand_log = [], [], []
    n_used = n_skipped = 0
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        question, gold = row["text"], row["gold_id"]
        candidates = generate_candidates(rewriter, question, args.n_candidates,
                                         args.temperature)
        scored = []
        for q in candidates:
            s, rank = score_query(q, gold, retriever, args.k)
            scored.append({"query": q, "score": s, "gold_rank": rank})

        cand_log.append({"query_id": row["query_id"], "question": question,
                         "gold_id": gold, "candidates": scored})

        valid = [c for c in scored if c["score"] >= 0]          # 오류 후보 제외
        best = max(valid, key=lambda c: c["score"], default=None)

        if best and best["score"] > 0:
            # 정답 논문을 실제로 찾아낸 쿼리만 학습 라벨로 쓴다
            sft.append({"input": question, "output": best["query"],
                        "gold_rank": best["gold_rank"]})
            # 못 찾은 후보가 있으면 선호쌍(잘 찾음 > 못 찾음)으로 저장
            for c in valid:
                if c["score"] == 0:
                    dpo.append({"input": question, "chosen": best["query"],
                                "rejected": c["query"]})
            n_used += 1
        else:
            n_skipped += 1     # 어떤 후보도 못 찾음 → 잘못된 패턴 학습 방지를 위해 제외

        if i % 10 == 0:
            print(f"  {i}/{len(rows)} · 라벨 확보 {n_used} · 제외 {n_skipped} "
                  f"({time.time()-t0:.0f}초)", flush=True)

    out = Path(args.out_dir)
    write_jsonl(out / "sft_pairs.jsonl", sft)
    write_jsonl(out / "dpo_pairs.jsonl", dpo)
    write_jsonl(out / "candidates.jsonl", cand_log)

    print("\n" + "=" * 60)
    print(f"질문 {len(rows)}개 처리")
    print(f"  학습 라벨 확보: {n_used}개 ({n_used/len(rows):.1%})")
    print(f"  제외(정답 못 찾음): {n_skipped}개")
    print(f"  SFT 쌍: {len(sft)}개 → {out/'sft_pairs.jsonl'}")
    print(f"  DPO 선호쌍: {len(dpo)}개 → {out/'dpo_pairs.jsonl'}")


if __name__ == "__main__":
    main()
