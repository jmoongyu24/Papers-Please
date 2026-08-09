"""어휘 조회를 붙인 변환기 — 모델이 만든 용어 + 실제 논문에서 가져온 용어.

왜 '대체'가 아니라 '보태기'인가:
실측에서 둘은 서로 다른 강점을 보였다.
  - 모델이 만든 용어: 정답 논문에 걸리는 비율 **0.793** (넓은 용어 위주, 중앙값 3,191편)
  - 조회한 용어    : 걸리는 비율 0.350 이지만 **좁다** (목표 구간 20~1,000편)
즉 모델은 폭을, 조회는 정밀도를 담당한다. 조회 쪽 1차 검색이 주제를 못 맞히는 경우가
있으므로(문항의 65%) 모델 용어를 빼면 오히려 손해다.

절을 늘리는 것이 왜 해롭지 않은가 (예상과 달랐던 실측):
처음에는 "넓은 절이 정답을 파묻는다"고 보고 넓은 절을 쳐내려 했다. 그런데 test 300문항에서
검색어 전체의 합집합 크기와 성공률 사이에 단조 관계가 없었다(0~5천편 0.411 / 5천~2만 0.263 /
2만~5만 0.360 / 5만~15만 0.293 / 15만+ 0.397). arXiv 는 여러 절에 동시에 걸리는 논문을
위로 올리므로, 넓은 절은 아래쪽에 논문을 더할 뿐 정답을 밀어내지 않는다.
라벨 실험도 같은 방향이었다: 절이 4.0개일 때 Recall@10 0.933, 2.92개일 때 0.656,
2.0개일 때 0.467 로 **절이 많을수록 좋았다.**
→ 그래서 이 변환기는 절을 **줄이지 않고 합친다.** 다만 300자 제한이 있으므로 좁은 절부터 넣는다.

동작:
    질문 → (1) 기존 변환기가 학술 용어 생성
         → (2) 어휘 조회가 실제 논문에서 용어 수집
         → (3) 문서 빈도가 낮은(좁은) 것부터 300자까지 채워 arXiv 검색어 조립
"""

from __future__ import annotations

import re

from src.rewriter.base import BACKENDS, Rewriter, build_rewriter
from src.schemas import RewriteResult

ARXIV_MAX_QUERY_LEN = 300


def assemble_query(raw_query: str, terms_with_df: list[tuple[str, int]],
                   max_len: int = ARXIV_MAX_QUERY_LEN) -> str:
    """arXiv 검색어를 조립한다. 좁은 용어(문서 빈도가 낮은 것)를 앞에 둔다.

    앞에 두는 이유는 300자에서 잘릴 때 **살아남는 쪽이 좁은 절**이 되게 하기 위함이다.
    좁은 절이 정답을 가려내는 힘이 크다(성공 문항 중앙값 144편 vs 묻힘 6,166편).

    원본 질문 절은 사용자가 논문 제목을 그대로 친 경우를 잡기 위한 것이라, 라틴 문자가
    있을 때만 넣는다. 한국어만 있는 질문에 넣으면 arXiv 가 한글을 색인하지 않아 아무것도
    걸지 못하면서 글자 수만 먹는다.
    """
    parts: list[str] = []
    orig = (raw_query or "").replace('"', " ").strip()
    if orig and re.search(r"[A-Za-z]", orig):
        parts.append(f'all:"{orig}"')

    seen: set[str] = set()
    for term, _df in sorted(terms_with_df, key=lambda x: x[1]):
        t = str(term).replace('"', " ").strip()
        low = t.lower()
        if not t or low in seen:
            continue
        seen.add(low)
        parts.append(f'abs:"{t}"')

    query = ""
    for p in parts:
        candidate = p if not query else f"{query} OR {p}"
        if len(candidate) > max_len:
            break
        query = candidate
    return query


class GroundedRewriter:
    """기존 변환기 위에 어휘 조회를 얹는다."""

    name = "grounded"

    def __init__(self, base: str = "dpo", n_lookup: int = 6,
                 lookup=None, sim_threshold: float = 0.0, **lookup_kwargs):
        """
        Args:
            base: 바탕이 될 변환기 이름 (passthrough / hierarchical / dpo 등).
            n_lookup: 조회로 가져올 용어 수.
        """
        self.base_name = base
        self.base: Rewriter = build_rewriter(base)
        self.n_lookup = n_lookup
        if lookup is None:
            from src.rewriter.vocab_lookup import VocabularyLookup
            lookup = VocabularyLookup(sim_threshold=sim_threshold, **lookup_kwargs)
        self.lookup = lookup

    def rewrite(self, raw_query: str) -> RewriteResult:
        # (1) 기존 변환기의 용어 — 폭을 담당
        base_result = self.base.rewrite(raw_query)
        base_terms = _terms_of(base_result)
        base_scored = [(t, self.lookup.stats.df(t) or 10 ** 6) for t in base_terms]

        # (2) 실제 논문에서 가져온 용어 — 정밀도를 담당
        try:
            looked = self.lookup.lookup(raw_query, k=self.n_lookup)
        except Exception:
            looked = []          # 조회가 실패해도 검색은 계속되어야 한다

        merged = looked + base_scored
        query = assemble_query(raw_query, merged)
        if not query:
            query = raw_query

        return RewriteResult(
            raw_query=raw_query,
            queries={b: query for b in BACKENDS},
            intent=base_result.intent,
            concepts=base_result.concepts,
            academic_terms=[t for t, _ in merged],
            parse_ok=base_result.parse_ok,
        )


def _terms_of(result: RewriteResult) -> list[str]:
    """변환 결과에서 학술 용어를 꺼낸다.

    계층 변환기는 academic_terms 에 담아 주지만, 학습 모델은 완성된 검색어 문자열만
    내므로 거기서 abs:"..." 절을 다시 뽑아야 한다.
    """
    if result.academic_terms:
        return [str(t) for t in result.academic_terms]
    return re.findall(r'abs:"([^"]+)"', result.query_for("arxiv") or "")
