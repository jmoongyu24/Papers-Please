"""관련도 등급 판정 — 풀의 각 후보 논문을 원본 질문 기준 0~3 등급으로 매긴다 (LLM 저지).

만족도 평가용 '등급 정답지'를 만드는 유일한 유료 단계다. 설계 원칙:
- **원본(일상어) 질문 기준**으로 판정한다. 변환된 쿼리가 아니라 사용자의 원래 의도를
  만족시켰는지가 관심사이기 때문.
- 원 출처 논문은 자동 3등급(판정 생략, 비용 절약).
- **생성 모델과 다른 판정자**를 쓴다(자기 질문을 자기가 후하게 채점하는 편향 방지).
- 의미 검색에 쓴 bge-m3를 판정에 쓰지 않는다(텍스트 LLM 판정이라 자연히 회피).
- 이어하기(resume): 이미 판정한 (질문,논문)은 건너뛴다.

등급 기준(루브릭):
  3 정답급 — 질문 주제를 정면으로 다룸, 매우 만족
  2 관련   — 주제에 맞고 유용
  1 약간   — 주변적으로만 관련
  0 무관   — 주제 벗어남

실행(확정 후):
  python -m evaluation.judge_relevance --pool data/eval/_pool_train.jsonl \
      --corpus data/corpus/corpus-v1.jsonl --out data/eval/qrels_graded_train.jsonl \
      --model gpt-4o-mini --limit 90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config
from src.retrieval.corpus import load_corpus
from src.utils import read_jsonl

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "integer", "enum": [0, 1, 2, 3]},
        "reason": {"type": "string"},
    },
    "required": ["grade"],
}

SYSTEM = (
    "너는 학술 검색 결과의 관련도를 매기는 평가자다. 사용자의 '원래 검색 의도'와 후보 논문을 "
    "보고, 이 논문이 그 사용자를 얼마나 만족시킬지 0~3 등급으로 판정하라.\n"
    "3=주제를 정면으로 다룸(매우 만족), 2=주제에 맞고 유용, 1=주변적으로만 관련, 0=무관.\n"
    "논문 제목·초록의 실제 내용만 보고 판단하고, grade 정수만 정확히 낸다."
)


def build_prompt(query: str, title: str, abstract: str) -> str:
    return (
        f"[사용자 원래 검색어] {query}\n\n"
        f"[후보 논문 제목] {title}\n"
        f"[후보 논문 초록] {abstract[:1500]}\n\n"
        '이 논문의 관련도 등급(0~3)을 JSON으로 답하라. 형식: {"grade": 정수}'
    )


class RelevanceJudge:
    def __init__(self, model: str):
        from openai import OpenAI
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 없음 (data/API_KEY.env 확인)")
        self.model = model
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def judge(self, query: str, title: str, abstract: str) -> int:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": build_prompt(query, title, abstract)}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=120,
        )
        data = json.loads(resp.choices[0].message.content)
        g = int(data.get("grade", 0))
        return max(0, min(3, g))


def load_existing(out_path: str) -> dict[str, dict[str, int]]:
    """이어하기: 이미 저장된 등급 정답지를 불러온다."""
    p = Path(out_path)
    if not p.exists():
        return {}
    qrels = {}
    for row in read_jsonl(out_path):
        qrels[row["query_id"]] = {k: int(v) for k, v in row["grades"].items()}
    return qrels


def save(out_path: str, qrels: dict[str, dict[str, int]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, grades in qrels.items():
            f.write(json.dumps({"query_id": qid, "grades": grades},
                               ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="관련도 등급 LLM 판정 (유료)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="판정 모델. 생성 모델(gpt-4o-mini)과 다른/강한 모델 권장")
    ap.add_argument("--limit", type=int, default=None,
                    help="앞에서부터 이 개수의 질문만 판정(프로토타입용)")
    args = ap.parse_args()

    papers = {p.id: p for p in load_corpus(args.corpus)}
    pool = list(read_jsonl(args.pool))
    if args.limit:
        pool = pool[: args.limit]

    qrels = load_existing(args.out)
    judge = RelevanceJudge(args.model)

    n_judged = 0
    for i, row in enumerate(pool, 1):
        qid, query, gold = row["query_id"], row["raw"], row["gold_id"]
        grades = qrels.get(qid, {})
        grades[gold] = 3  # 원 출처 논문은 자동 정답급
        for doc_id in row["candidates"]:
            if doc_id == gold or doc_id in grades:
                continue
            p = papers.get(doc_id)
            if p is None:
                continue
            grades[doc_id] = judge.judge(query, p.title, p.abstract)
            n_judged += 1
        qrels[qid] = grades
        if i % 10 == 0:
            save(args.out, qrels)
            print(f"  질문 {i}/{len(pool)} 처리, 누적 판정 {n_judged}건", flush=True)

    save(args.out, qrels)
    print(f"완료: 질문 {len(qrels)}개, 새로 판정 {n_judged}건 → {args.out}")


if __name__ == "__main__":
    main()
