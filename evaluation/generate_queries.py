"""평가용 질문을 '거꾸로' 만들어내는 스크립트.

기본 아이디어: 정답 논문을 먼저 정하고, 그 논문(제목+초록)을 언어 모델에게 보여 주며
"이 분야를 잘 모르는 사람이 이 논문을 찾을 때 평소 말로 어떻게 검색할까?"를 물어
일상어 질문을 만들게 한다. 그러면 (질문, 정답 논문) 쌍이 자동으로 생긴다.

질문 생성기는 두 종류를 준비한다. 같은 인터페이스라 서로 바꿔 끼울 수 있다.
- OpenAIGenerator : 진짜 유료 언어 모델(OpenAI)로 생성. 품질이 좋다. (API 키 필요)
- MockGenerator   : 규칙 기반의 가짜 생성기. 인터넷/키 없이 파이프라인을 점검할 때 쓴다.

실행 예:
  # 오프라인 점검 (키 불필요)
  python -m evaluation.generate_queries --corpus data/sample/corpus.sample.jsonl \
      --out data/eval/queries.raw.jsonl --n 20 --mock
  # 진짜 생성 (OPENAI_API_KEY 필요)
  python -m evaluation.generate_queries --corpus data/corpus/corpus-v1.jsonl \
      --out data/eval/queries.raw.jsonl --n 500
"""

from __future__ import annotations

import argparse
import random
from typing import Protocol

from src import config
from src.retrieval.corpus import load_corpus
from src.schemas import EvalQuery, Paper
from src.utils import write_jsonl

# 만들 질문의 난이도·언어 조합 (0절의 넓어진 사용자 정의를 반영)
DIFFICULTIES = ("easy", "mid", "hard")   # 일상어 / 어설픈 전문어 / 정확한 전문어에 가까움
LANGS = ("ko", "en")

# 언어 모델에게 주는 안내문(프롬프트). 난이도·언어에 따라 지시를 바꾼다.
_DIFF_GUIDE = {
    "easy": "그 분야를 전혀 모르는 학부생처럼, 전문 용어를 하나도 쓰지 말고 완전한 일상어로",
    "mid": "그 분야를 조금 아는 사람처럼, 전문 용어를 어설프게 쓰되 논문이 실제로 쓰는 "
           "정확한 용어와는 살짝 어긋나게",
    "hard": "그 분야를 어느 정도 아는 대학원생처럼, 비교적 정확한 전문 용어를 써서",
}
_LANG_GUIDE = {"ko": "한국어로", "en": "영어로"}


def build_prompt(paper: Paper, difficulty: str, lang: str) -> str:
    """한 논문에 대해 질문을 만들라는 안내문을 조립한다."""
    return (
        "너는 학술 검색 평가용 데이터를 만드는 조수다.\n"
        "아래 논문을 '찾고 싶어 하는 사용자'가 검색창에 칠 법한 짧은 질문 하나를 만들어라.\n"
        f"- {_DIFF_GUIDE[difficulty]} {_LANG_GUIDE[lang]} 작성한다.\n"
        "- 논문 제목이나 초록의 문장을 그대로 베끼지 마라. 특히 드문 전문 용어를 그대로 "
        "옮기면 안 된다 (그러면 평가가 무의미해진다).\n"
        "- 한 문장, 20단어 이내.\n\n"
        f"[제목] {paper.title}\n[초록] {paper.abstract[:1200]}\n\n"
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
) -> list[EvalQuery]:
    """논문들을 뽑아, 논문마다 (난이도·언어를 바꿔가며) 질문을 만든다."""
    rng = random.Random(seed)
    chosen = rng.sample(papers, min(n_papers, len(papers)))

    rows: list[EvalQuery] = []
    qnum = 0
    for paper in chosen:
        # 논문마다 난이도 하나, 언어 하나를 골라 질문 1개 생성 (조합을 고루 순환)
        difficulty = DIFFICULTIES[qnum % len(DIFFICULTIES)]
        lang = LANGS[qnum % len(LANGS)]
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


def main() -> None:
    ap = argparse.ArgumentParser(description="논문에서 평가용 질문을 거꾸로 생성")
    ap.add_argument("--corpus", required=True, help="코퍼스 JSON Lines 경로")
    ap.add_argument("--out", required=True, help="생성 결과 저장 경로")
    ap.add_argument("--n", type=int, default=20, help="질문을 만들 논문 수")
    ap.add_argument("--mock", action="store_true", help="키 없이 가짜 생성기로 점검")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    papers = load_corpus(args.corpus)
    generator: QueryGenerator = MockGenerator() if args.mock else OpenAIGenerator()
    rows = generate_dataset(papers, generator, n_papers=args.n, seed=args.seed)
    n = write_jsonl(args.out, (r.to_dict() for r in rows))
    print(f"질문 {n}개 생성 → {args.out} (생성기: {'mock' if args.mock else 'openai'})")


if __name__ == "__main__":
    main()
