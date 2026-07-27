"""Papers-Please 웹 데모 (Streamlit) — 실시간 arXiv 검색 버전.

사용자가 일상어·한국어로 검색하면, Qwen3-4B가 정확한 영어 학술 용어로 변환하는 '과정'을
보여주고, 그 변환 검색어로 **arXiv에 실시간 질의**해 논문(제목·초록)을 가져온다. 로컬
데이터베이스가 필요 없다. 변환 전/후 결과를 나란히 비교해 변환의 효과를 눈으로 확인한다.

실행:
  /home/jmoongyu/venvs/paper_py310/bin/streamlit run app.py
(전제: Ollama 서버 + qwen3:4b 모델, 인터넷 연결)
"""

from __future__ import annotations

import streamlit as st

from src.recommend_agent.recommender import PaperRecommender
from src.retrieval.arxiv_live import ArxivLiveRetriever
from src.rewriter.hierarchical import HierarchicalRewriter
from src.rewriter.paper_resolver import PaperResolver, resolve_and_verify

EXAMPLES = [
    "사진 보고 글로 설명해주는 AI",
    "가짜 뉴스 걸러내는 방법",
    "AI가 사람처럼 대화하게 만들기",
    "Attention is all you need",
]

st.set_page_config(page_title="Papers-Please", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="arXiv 검색기 준비 중…")
def load_arxiv() -> ArxivLiveRetriever:
    return ArxivLiveRetriever()


@st.cache_resource(show_spinner="언어 모델(Qwen3-4B) 연결 중…")
def load_rewriter() -> HierarchicalRewriter:
    return HierarchicalRewriter()


@st.cache_resource
def load_resolver() -> PaperResolver:
    return PaperResolver()


@st.cache_resource
def load_recommender() -> PaperRecommender:
    return PaperRecommender()


def arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def safe_search(query: str, k: int):
    """arXiv 검색. (결과, 오류메시지) 튜플. 오류를 '결과 없음'과 구분한다."""
    try:
        return arxiv.search(query, k=k), None
    except Exception as e:
        return None, (f"arXiv 검색 중 오류가 발생했습니다 (일시적일 수 있어요, "
                      f"잠시 후 다시 시도해 주세요). [{type(e).__name__}]")


def render_results(results, error: str | None = None) -> None:
    if error:
        st.warning(error)
        return
    if not results:
        st.info("검색 결과가 없습니다. (변환된 검색어에 맞는 논문이 arXiv에 없을 수 있어요)")
        return
    for r in results:
        st.markdown(f"**{r.rank}. [{r.title}]({arxiv_url(r.paper_id)})**")
        with st.expander("초록 보기"):
            st.write(r.abstract)


# ── 화면 ──────────────────────────────────────────────────────────────────
st.title("📄 Papers, Please")
st.caption("일상어, 한국어로 검색해도, arXiv에서 논문을 찾아 드립니다.")

arxiv = load_arxiv()

TOP_K = 10   # 검색 결과 10개 고정 (에이전트가 이 10개를 분석)

# arXiv 호출을 아끼기 위해, 한 번의 검색에서 arXiv를 최대 1~2회만 부른다.
# (변환 전/후 비교 검색은 호출이 2배가 되므로 기본으로 끄고, 개발용 옵션으로만 남긴다)
with st.sidebar:
    st.header("설정")
    st.caption("검색 대상: arXiv 실시간 · 결과 10개 고정")
    st.caption("arXiv 요청 제한을 지키기 위해 한 검색당 호출을 최소화합니다.")
    compare = st.checkbox("변환 전/후 비교 (개발용, arXiv 호출 2배)", value=False)

if "query" not in st.session_state:
    st.session_state.query = ""

st.write("**예시로 시작하기:**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.query = ex

query = st.text_input("검색어", value=st.session_state.query,
                      placeholder="예: 사진 보고 글로 설명해주는 AI")

if st.button("🔍 검색", type="primary") and query.strip():
    # 방안 A: 특정 유명 논문을 '설명'으로 찾는 질문이면, 그 논문을 짚어서 먼저 보여준다
    # (LLM 지식으로 제목 추정 → arXiv에서 검증된 경우에만)
    with st.spinner("특정 논문을 가리키는 질문인지 확인 중…"):
        try:
            resolved_paper, _ = resolve_and_verify(query, load_resolver(), arxiv)
        except Exception:
            resolved_paper = None   # arXiv 오류 등은 배너 생략(치명적 아님)
    if resolved_paper:
        st.success("🎯 이 논문을 찾으시는 것 같습니다")
        st.markdown(f"### [{resolved_paper.title}]({arxiv_url(resolved_paper.paper_id)})")
        with st.expander("초록 보기"):
            st.write(resolved_paper.abstract)
        st.divider()

    rewriter = load_rewriter()
    with st.spinner("Qwen3-4B가 학술 용어로 변환 중…"):
        rw = rewriter.rewrite(query)

    # 1) 변환 과정 (핵심 UX)
    st.subheader("🧭 쿼리 변환 과정")
    if not rw.parse_ok:
        st.warning("변환에 실패해 원본 검색어로 검색합니다.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1) 의도 파악**")
    c1.info(rw.intent or "—")
    c2.markdown("**2) 핵심 개념**")
    c2.info("　·　".join(rw.concepts) if rw.concepts else "—")
    c3.markdown("**3) 학술 용어**")
    c3.success("　·　".join(rw.academic_terms) if rw.academic_terms else "—")

    # 2) arXiv 실시간 검색 결과 (10개 고정)
    st.divider()
    st.subheader("🔍 검색된 논문 (arXiv 실시간, 10개)")
    transformed_query = rw.query_for("arxiv")   # arXiv 전용 쿼리(코드 구성: 원본 phrase OR 학술용어)
    with st.spinner("arXiv 검색 중…"):
        after_results, after_err = safe_search(transformed_query, k=TOP_K)
        before_results, before_err = safe_search(query, k=TOP_K) if compare else (None, None)
    if compare:
        before_col, after_col = st.columns(2)
        with before_col:
            st.markdown("**변환 전** (원본 그대로)")
            st.caption(f"검색어: `{query}`")
            render_results(before_results, before_err)
        with after_col:
            st.markdown("**변환 후** (학술 용어)")
            st.caption(f"검색어: `{transformed_query}`")
            render_results(after_results, after_err)
    else:
        st.caption(f"검색어: `{transformed_query}`")
        render_results(after_results, after_err)

    # 3) 에이전트: 검색된 10개를 사용자 의도와 대조해 추천 + 이유
    st.divider()
    st.subheader("🤖 에이전트 추천 (의도에 맞는 논문 선별)")
    if after_err:
        st.warning("검색 오류로 추천을 생성할 수 없습니다.")
    elif after_results:
        with st.spinner("에이전트(Qwen3-4B)가 의도에 맞는 논문을 고르는 중…"):
            rec = load_recommender().recommend(query, after_results)
        if rec["summary"]:
            st.info(f"**종합:** {rec['summary']}")
        badge = {"high": "🟢 강력 추천", "medium": "🟡 관련", "low": "⚪ 거의 무관"}
        for r in rec["recommendations"]:
            p = after_results[r["index"] - 1]
            st.markdown(f"{badge.get(r['relevance'], '')}　**[{p.title}]({arxiv_url(p.paper_id)})**")
            st.caption(f"↳ 추천 이유: {r['reason']}")
    else:
        st.info("검색 결과가 없어 추천할 논문이 없습니다.")
