"""평가용 질문을 '거꾸로' 만들어내는 스크립트.

기본 아이디어: 정답 논문을 먼저 정하고, 그 논문(제목+초록)을 언어 모델에게 보여 주며
"이 분야를 잘 모르는 사람이 이 논문을 찾을 때 평소 말로 어떻게 검색할까?"를 물어
질문을 만들게 한다. 그러면 (질문, 정답 논문) 쌍이 자동으로 생긴다.

논문 한 편당 질문을 3개(easy/mid/hard) 만든다. 같은 정답 논문을 놓고 난이도만 다른
세 검색어로 검색했을 때 결과가 어떻게 달라지는지 비교하기 위해서다.

질문 생성기는 두 종류를 준비한다. 같은 인터페이스라 서로 바꿔 끼울 수 있다.
- OpenAIGenerator : 진짜 유료 언어 모델(OpenAI)로 생성. 품질이 좋다. (API 키 필요)
- MockGenerator   : 규칙 기반의 가짜 생성기. 인터넷/키 없이 파이프라인을 점검할 때 쓴다.

실행 예 (--n 은 '논문 수'. 질문은 그 3배로 생성된다):
  # 오프라인 점검 (키 불필요)
  python -m evaluation.generate_queries --corpus data/sample/corpus.sample.jsonl \
      --out data/eval/queries.raw.jsonl --n 20 --mock
  # 진짜 생성 (OPENAI_API_KEY 필요, data/API_KEY.env 로도 읽음) — 논문 100편 -> 질문 300개
  python -m evaluation.generate_queries --corpus data/corpus/corpus-v1.jsonl \
      --out data/eval/queries.raw.jsonl --n 100
"""

from __future__ import annotations

import argparse
import random
from typing import Optional, Protocol

from src import config
from src.retrieval.corpus import load_corpus, stratified_sample
from src.schemas import EvalQuery, Paper
from src.utils import read_jsonl, write_jsonl

# 만들 질문의 난이도·언어 조합
DIFFICULTIES = ("easy", "mid", "hard")   # 일상어 / 어설픈 전문어 / 정확한 전문어에 가까움
LANGS = ("ko", "en")

# 언어 모델에게 주는 안내문(프롬프트). 난이도·언어에 따라 지시를 바꾼다.
_DIFF_GUIDE = {
    "easy": (
        "도메인 지식이 없는 일반인이 일상어와 현상 기술만으로 표현한다.논문의 핵심 고유명사나 특정 기술명은 전면 배제하고 일반적인 표현으로 Paraphrasing한다."
    ),
    "mid": (
        "상위 범주의 범용 기술 용어(Generic term)는 사용하되, 본문의 핵심 타겟 기술명(Specific term)은 유의어나 유사 개념으로 변경하여 불완전한 문장으로 작성한다."
    ),
    "hard": (
        "해당 분야 연구자가 사용하는 정확한 학술적 키워드와 도메인 지식을 포함하되, 제목의 자구와 완전히 일치하지 않도록 단어 순서를 바꾸거나 동의어를 결합하여 전문적인 검색어로 구성한다."
    ),
}
_LANG_GUIDE = {"ko": "한국어로", "en": "영어로"}


def build_prompt(paper: Paper, difficulty: str, lang: str) -> str:
    """한 논문에 대해 질문을 만들라는 안내문을 생성한다."""
    return (
        "너는 학술 논문 검색 성능 평가용 데이터를 만드는 전문가다.\n"
        "아래 논문을 '찾고 싶어 하는 사용자'가 웹 검색창에 직접 입력할 법한 짧고 목적 지향적인 검색 쿼리(질문) 하나를 생성하라.\n"
        f"- 제한조건: {_DIFF_GUIDE[difficulty]} {_LANG_GUIDE[lang]} 작성한다.\n"
        "- 형태: 부연 설명이나 미사여구 없이, 한 문장(어절 기준 20단어 이내)으로만 출력하라.\n\n"
        f"[제목] {paper.title}\n[초록] {paper.abstract}\n\n"
        "질문:"
    )

class QueryGenerator(Protocol):
    def generate(self, paper: Paper, difficulty: str, lang: str) -> str:
        ...


class MockGenerator:
    """키 없이 도는 가짜 생성기. 제목 앞부분을 재료로 그럴싸한 질문 흉내만 낸다.

    실제 평가에는 쓰지 않고, 오직 파이프라인(생성→필터→분할→실험)이 끝까지
    도는지 점검하는 용도다.
    """

    model_name = "mock"

    def generate(self, paper: Paper, difficulty: str, lang: str) -> str:
        head = paper.title.split(":")[0]
        words = head.split()
        short = " ".join(words[:4])
        if lang == "ko":
            base = {"easy": f"{short} 관련해서 쉽게 설명한 연구 있어?",
                    "mid": f"{short} 기법을 다룬 논문 찾아줘",
                    "hard": f"{short} 관련 최신 방법론 논문"}[difficulty]
        else:
            base = {"easy": f"papers that explain {short} in simple terms",
                    "mid": f"research on {short} methods",
                    "hard": f"recent approaches to {short}"}[difficulty]
        return base


class OpenAIGenerator:
    """진짜 OpenAI 모델로 질문을 생성한다. openai 패키지와 API 키가 필요하다."""

    def __init__(self, model: str = config.QUERY_GEN_MODEL):
        self.model_name = model
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai 패키지가 필요합니다. `pip install openai` 후, "
                "환경변수 OPENAI_API_KEY를 설정하세요. "
                "(키 없이 점검하려면 --mock 옵션을 쓰세요.)"
            ) from e
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 비어 있습니다.")
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, paper: Paper, difficulty: str, lang: str) -> str:
        prompt = build_prompt(paper, difficulty, lang)
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip().strip('"')


def generate_dataset(
    papers: list[Paper],
    generator: QueryGenerator,
    n_papers: int,
    seed: int = 42,
    categories: Optional[list[str]] = None,
    prior_category_counts: Optional[dict[str, int]] = None,
) -> list[EvalQuery]:
    """논문들을 뽑아, **논문 한 편당 easy/mid/hard 질문을 3개씩** 만든다.

    왜 3개씩인가: 같은 정답 논문 하나를 놓고 "일상어로 검색했을 때 / 어설픈 전문 용어로
    검색했을 때 / 정확한 전문 용어로 검색했을 때" 검색 결과가 어떻게 달라지는지를 같은
    논문 기준으로 비교할 수 있다. 논문마다 난이도를 하나씩만 뽑으면 이 비교가 불가능하다.

    언어는 논문 단위로 하나를 배정한다(그 논문의 easy/mid/hard 3개가 같은 언어).
    질문 하나하나마다 언어를 바꾸면 조합이 3배(난이도3×언어2)로 늘어나 생성 비용이
    커지고, "논문당 3개"라는 요청과도 어긋나기 때문이다. 대신 논문마다 언어를 번갈아
    배정해 전체적으로는 한국어/영어가 고르게 섞이게 한다.

    categories를 주면 분야별로 고르게 논문을 뽑는다(`stratified_sample`). 그냥
    무작위로 뽑으면 코퍼스에 원래 많은 분야가 평가셋에도 그대로 많이 뽑히기 때문이다.
    """
    if categories:
        chosen = stratified_sample(
            papers, n_papers, categories, seed=seed, prior_counts=prior_category_counts
        )
    else:
        rng = random.Random(seed)
        chosen = rng.sample(papers, min(n_papers, len(papers)))

    rows: list[EvalQuery] = []
    qnum = 0
    for i, paper in enumerate(chosen):
        lang = LANGS[i % len(LANGS)]
        for difficulty in DIFFICULTIES:
            text = generator.generate(paper, difficulty, lang)
            qnum += 1
            rows.append(
                EvalQuery(
                    query_id=f"q{qnum:05d}",
                    gold_id=paper.id,
                    text=text,
                    difficulty=difficulty,
                    lang=lang,
                    source="synthetic",
                    gen_model=getattr(generator, "model_name", "unknown"),
                )
            )
    return rows


def load_used_gold_ids(paths: list[str]) -> set[str]:
    """이미 만들어 둔 질문 파일(들)에서 정답 논문 id 목록을 뽑는다.

    데이터셋을 나눠서(배치별로) 생성할 때, 이미 쓴 논문을 다시 뽑지 않으려고 쓴다.
    """
    used: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            used.add(str(row["gold_id"]))
    return used


def category_counts_of(paper_ids: set[str], corpus_papers: list[Paper],
                        categories: list[str]) -> dict[str, int]:
    """주어진 논문 id 집합이 분야별로 몇 편씩인지 센다 (균형 맞추기의 '현재 상태' 계산용)."""
    from collections import Counter
    from src.retrieval.corpus import primary_bucket
    by_id = {p.id: p for p in corpus_papers}
    counts = Counter()
    for pid in paper_ids:
        p = by_id.get(pid)
        if p:
            counts[primary_bucket(p, categories)] += 1
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description="논문에서 평가용 질문을 거꾸로 생성")
    ap.add_argument("--corpus", required=True, help="코퍼스 JSON Lines 경로")
    ap.add_argument("--out", required=True, help="생성 결과 저장 경로")
    ap.add_argument("--n", type=int, default=20,
                     help="질문을 만들 '논문' 수 (논문 1편당 easy/mid/hard 질문 3개가 나온다)")
    ap.add_argument("--mock", action="store_true", help="키 없이 가짜 생성기로 점검")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", nargs="*", default=[],
                     help="이 파일(들)에 이미 나온 정답 논문은 다시 뽑지 않음 "
                          "(배치를 나눠 이어서 생성할 때 사용)")
    ap.add_argument("--categories", nargs="*", default=list(config.CORPUS_CATEGORIES),
                     help="이 분야들에서 최대한 고르게 뽑는다 (물 채우기 방식). "
                          "빈 목록을 주면 순수 무작위 추출로 되돌아간다")
    args = ap.parse_args()

    all_papers = load_corpus(args.corpus)
    papers = all_papers
    prior_counts = None
    if args.exclude:
        used = load_used_gold_ids(args.exclude)
        before = len(papers)
        papers = [p for p in all_papers if p.id not in used]
        print(f"이미 사용된 논문 {len(used)}편 제외 ({before} → {len(papers)}편 중에서 추출)")
        if args.categories:
            prior_counts = category_counts_of(used, all_papers, args.categories)
            print(f"이전 배치 분야별 분포: {prior_counts} (이 분포를 참고해 부족한 분야를 우선 채움)")

    generator: QueryGenerator = MockGenerator() if args.mock else OpenAIGenerator()
    rows = generate_dataset(
        papers, generator, n_papers=args.n, seed=args.seed,
        categories=args.categories or None, prior_category_counts=prior_counts,
    )
    n = write_jsonl(args.out, (r.to_dict() for r in rows))
    print(f"논문 {args.n}편 × 3난이도 = 질문 {n}개 생성 → {args.out} "
          f"(생성기: {'mock' if args.mock else 'openai'})")


if __name__ == "__main__":
    main()
