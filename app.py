"""Papers-Please 웹 데모 (Streamlit).

비전문가/한국어 검색어를 Qwen3-4B가 정확한 영어 학술 용어로 바꾸는 '변환 과정'을 보여주고,
그 변환으로 arXiv 논문(제목·초록)을 찾아 준다. 변환 전/후 결과를 나란히 비교해, 변환이
검색을 어떻게 바꾸는지 눈으로 확인할 수 있다.

실행:
  /home/jmoongyu/venvs/paper_py310/bin/streamlit run app.py
(전제: Ollama 서버 실행 + qwen3:4b 모델, data/corpus/corpus-v1.jsonl + 임베딩 캐시)
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.retrieval.bm25_simple import BM25Retriever
from src.retrieval.corpus import load_corpus
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.rewriter.base import PassthroughRewriter
from src.rewriter.hierarchical import HierarchicalRewriter

CORPUS_PATH = str(config.CORPUS_DIR / "corpus-v1.jsonl")
EXAMPLES = [
    "사진 보고 글로 설명해주는 AI",
    "가짜 뉴스 걸러내는 방법",
    "AI가 사람처럼 대화하게 만들기",
    "graph data neural network for labeling",
]

st.set_page_config(page_title="Papers-Please", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="검색 시스템 로딩 중… (논문 3만 편·임베딩, 최초 1회 수십 초)")
def load_search_system(corpus_path: str) -> dict:
    """코퍼스·세 검색기를 한 번만 로드해 재사용한다."""
    papers = load_corpus(corpus_path)
    bm25 = BM25Retriever(papers)
    dense = DenseRetriever.build(corpus_path)
    hybrid = HybridRetriever(bm25, dense, rrf_k=config.RRF_K)
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid, "n": len(papers)}


@st.cache_resource(show_spinner="언어 모델(Qwen3-4B) 연결 중…")
def load_rewriter() -> HierarchicalRewriter:
    return HierarchicalRewriter()


def arxiv_url(paper_id: str) -> str:
    # 코퍼스 id는 "2101.00001" 형태. arXiv 페이지 주소로 변환.
    return f"https://arxiv.org/abs/{paper_id}"


def search_with(system: dict, backend: str, rw, k: int):
    """변환 결과(rw)로 선택한 검색 방식에 맞춰 검색한다."""
    if backend == "hybrid":
        return system["hybrid"].search(rw.query_for("sparse"), rw.query_for("dense"), k=k)
    family = "sparse" if backend == "bm25" else "dense"
    return system[backend].search(rw.query_for(family), k=k)


def render_results(results) -> None:
    if not results:
        st.info("검색 결과가 없습니다. (단어가 전혀 안 겹치면 BM25는 결과가 없을 수 있어요)")
        return
    for r in results:
        st.markdown(f"**{r.rank}. [{r.title}]({arxiv_url(r.paper_id)})**　·　유사도 {r.score:.3f}")
        with st.expander("초록 보기"):
            st.write(r.abstract)


# ── 화면 ──────────────────────────────────────────────────────────────────
st.title("📄 Papers-Please")
st.caption("일상어·한국어로 검색해도, 학술 용어로 바꿔서 arXiv 논문을 찾아 드립니다.")

system = load_search_system(CORPUS_PATH)

with st.sidebar:
    st.header("설정")
    backend = st.radio("검색 방식", ["hybrid", "dense", "bm25"],
                       format_func={"hybrid": "혼합(추천)", "dense": "의미 기반",
                                    "bm25": "단어 일치"}.get)
    topk = st.slider("결과 개수", 5, 20, 10)
    compare = st.checkbox("변환 전/후 비교", value=True)
    st.caption(f"코퍼스: 논문 {system['n']:,}편")

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

    # 2) 검색 결과
    st.divider()
    if compare:
        before_col, after_col = st.columns(2)
        with before_col:
            st.subheader("변환 전 (원본 그대로)")
            st.caption(f"검색어: `{query}`")
            render_results(search_with(system, backend, PassthroughRewriter().rewrite(query), topk))
        with after_col:
            st.subheader("변환 후 (학술 용어)")
            fam = "sparse" if backend == "bm25" else "dense"
            st.caption(f"검색어: `{rw.query_for(fam)}`")
            render_results(search_with(system, backend, rw, topk))
    else:
        st.subheader("검색 결과")
        render_results(search_with(system, backend, rw, topk))
