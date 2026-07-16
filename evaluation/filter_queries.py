"""생성한 질문 중 품질이 나쁜 것을 걸러내는 스크립트.

이 단계가 평가 신뢰도를 좌우하는 가장 중요한 부분이다. 언어 모델이 질문을 만들 때
초록의 전문 용어를 그대로 베끼는 일이 잦은데, 그러면 '일상어 질문'이 아니라 사실상
정답을 미리 알려주는 질문이 되어 평가가 무의미해진다.

여기서 자동으로 거르는 것:
1) 전문 용어 베끼기(누수) — 질문이 정답 논문의 '드문 단어'를 그대로 담고 있으면 버림.
2) 너무 짧거나 긴 질문 — 검색어로 부적절.
3) 사실상 중복인 질문 — 서로 너무 비슷하면 하나만 남김.

'드문 단어'의 판단: 코퍼스 전체에서 몇 개 문서에 나오는지(문서빈도 df)를 센다.
아주 적은 문서에만 나오는 단어일수록 특정 논문을 콕 집어내는 전문 용어다. 이런 단어가
질문과 정답 논문에 함께 있으면 '베낀 것'으로 본다. (이 원리를 IDF라고 부른다.)

실행 예:
  python -m evaluation.filter_queries \
      --corpus data/sample/corpus.sample.jsonl \
      --queries data/eval/queries.raw.jsonl \
      --out data/eval/queries.filtered.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from src.retrieval.base import tokenize
from src.retrieval.corpus import load_corpus
from src.schemas import EvalQuery, Paper
from src.utils import read_jsonl, write_jsonl

# 아주 흔한 기능어는 '드문 단어' 판정에서 빼서 오탐을 줄인다.
_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "is",
    "are", "we", "this", "that", "using", "based", "via", "from", "by", "as",
    "논문", "연구", "관련", "대한", "찾아줘", "있어", "어떻게", "무엇", "방법",
}


def build_doc_freq(papers: list[Paper]) -> dict[str, int]:
    """단어마다 '몇 개 문서에 나오는지(df)'를 센다."""
    df: dict[str, int] = defaultdict(int)
    for p in papers:
        for term in set(tokenize(p.text)):
            df[term] += 1
    return df


def rare_terms_of(
    paper: Paper, df: dict[str, int], n_docs: int, max_df: int, max_df_ratio: float
) -> set[str]:
    """정답 논문 안의 '드문 전문 용어' 집합.

    '드문 단어'는 코퍼스에서 아주 적은 문서에만 나오는 단어다. 두 조건을 함께 만족해야 한다.
    - 절대 개수: max_df개 이하 문서에만 등장 (큰 코퍼스에서 '드묾'의 기준).
    - 상대 비율: 전체 문서의 max_df_ratio 이하에만 등장 (작은 코퍼스에서 과판정 방지).
    두 조건을 모두 쓰는 이유: 코퍼스가 커도 작아도 '진짜 드문 용어'만 잡기 위해서다.
    (예: 30000편이면 절대 기준 5가 주도, 12편이면 비율 기준이 주도해 유일 등장 단어만 잡음.)
    """
    ratio_cap = max(1, int(n_docs * max_df_ratio))
    out = set()
    for term in set(tokenize(paper.text)):
        if term in _STOPWORDS or len(term) < 3:
            continue
        d = df.get(term, 0)
        if d <= max_df and d <= ratio_cap:
            out.add(term)
    return out


def leakage_count(
    query: EvalQuery, paper: Paper, df: dict[str, int],
    n_docs: int, max_df: int, max_df_ratio: float,
) -> int:
    """질문이 정답 논문의 드문 전문 용어를 몇 개나 그대로 담고 있는지 센다."""
    q_terms = set(tokenize(query.text))
    return len(q_terms & rare_terms_of(paper, df, n_docs, max_df, max_df_ratio))


def jaccard(a: set[str], b: set[str]) -> float:
    """두 단어 집합이 얼마나 겹치는지 (0~1). 중복 질문 판단에 쓴다."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_queries(
    queries: list[EvalQuery],
    papers: list[Paper],
    max_df: int = 5,
    max_df_ratio: float = 0.1,
    max_leak: int = 1,
    min_tokens: int = 2,
    max_tokens: int = 40,
    dedup_threshold: float = 0.9,
) -> tuple[list[EvalQuery], dict[str, int]]:
    """질문 목록을 걸러 통과한 것만 돌려주고, 각 사유별 탈락 수도 함께 준다.

    Args:
        max_df: 이 수 이하 문서에만 나오는 단어를 '드문 전문 용어'로 본다(절대 기준).
        max_df_ratio: 전체 문서의 이 비율 이하에만 나오는 단어(상대 기준). 작은 코퍼스 보정용.
        max_leak: 정답 논문의 드문 용어를 이 개수를 '넘게' 담으면 누수로 보고 버린다.
        min_tokens/max_tokens: 질문 길이(단어 수) 허용 범위.
        dedup_threshold: 이미 통과한 질문과 이만큼 겹치면 중복으로 보고 버린다.
    """
    papers_by_id = {p.id: p for p in papers}
    df = build_doc_freq(papers)
    n_docs = len(papers)

    kept: list[EvalQuery] = []
    kept_token_sets: list[set[str]] = []
    reasons = defaultdict(int)

    for q in queries:
        q_tokens = tokenize(q.text)
        # 1) 길이 검사
        if not (min_tokens <= len(q_tokens) <= max_tokens):
            reasons["length"] += 1
            continue
        # 2) 정답 논문이 코퍼스에 있는지
        gold = papers_by_id.get(q.gold_id)
        if gold is None:
            reasons["gold_not_in_corpus"] += 1
            continue
        # 3) 전문 용어 베끼기(누수) 검사
        if leakage_count(q, gold, df, n_docs, max_df, max_df_ratio) > max_leak:
            reasons["term_leakage"] += 1
            continue
        # 4) 중복 검사
        q_set = set(q_tokens)
        if any(jaccard(q_set, s) >= dedup_threshold for s in kept_token_sets):
            reasons["duplicate"] += 1
            continue

        kept.append(q)
        kept_token_sets.append(q_set)

    reasons["kept"] = len(kept)
    reasons["input"] = len(queries)
    return kept, dict(reasons)


def main() -> None:
    ap = argparse.ArgumentParser(description="생성한 질문 품질 필터링")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-df", type=int, default=5)
    ap.add_argument("--max-df-ratio", type=float, default=0.1)
    ap.add_argument("--max-leak", type=int, default=1)
    args = ap.parse_args()

    papers = load_corpus(args.corpus)
    queries = [EvalQuery.from_dict(r) for r in read_jsonl(args.queries)]
    kept, reasons = filter_queries(
        queries, papers, max_df=args.max_df,
        max_df_ratio=args.max_df_ratio, max_leak=args.max_leak,
    )
    write_jsonl(args.out, (q.to_dict() for q in kept))
    print(f"입력 {reasons['input']}개 → 통과 {reasons['kept']}개  ({args.out})")
    print("탈락 사유:", {k: v for k, v in reasons.items()
                      if k not in ("kept", "input")})


if __name__ == "__main__":
    main()
