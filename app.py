"""Papers-Please 웹 데모 (Streamlit) - 두 채널 검색 + 교차 인코더 재정렬.

사용자가 일상어나 한국어로 검색하면, Qwen3-4B 가 정확한 영어 학술 용어로 바꾸는 과정을
보여주고, 두 갈래로 논문을 찾아 사용자 의도에 맞게 다시 줄 세워 보여준다.

## 왜 검색 통로가 둘인가

    질문
     ├─ 채널 A  쿼리 변환기 -> arXiv 실시간 검색   (글자가 정확히 맞아야 걸림)
     └─ 채널 B  로컬 의미 검색 71만 편              (뜻으로 찾음)
            |
       합치고 중복 제거
            |
       교차 인코더 재정렬 (원본 질문과 대조)
            |
       사용자에게 보여줄 10편

arXiv 키워드 검색은 정확한 문자열 일치를 요구해서, 실패한 검색어의 63% 가 거의 맞음으로
빗나갔다 (ISSUE 21). 뜻으로 찾는 두 번째 통로를 나란히 두어 그 벽을 넘는다.

시험용 300문항 실측으로 로컬 의미 검색 채널이 성능의 거의 전부를 만든다는 것을 확인했다
(로컬 단독 0.677 대 두 채널 0.680, p=0.889). 그런데도 arXiv 채널을 남기는 이유는,
평가셋의 정답 논문이 전부 로컬 색인 안에 있어서 그 평가가 색인 밖 논문을 못 찾는 약점을
구조적으로 잴 수 없기 때문이다. 실제 사용자는 어제 올라온 논문도 묻는다.
arXiv 채널의 역할은 정확도가 아니라 최신성과 범위다 (ISSUE 29).

실행:
  streamlit run app.py
(전제: Ollama 서버 + qwen3:4b, 로컬 색인, 인터넷 연결)
"""

from __future__ import annotations

import time

import streamlit as st

from src import config
from src.recommend_agent.recommender import PaperRecommender
from src.retrieval.arxiv_live import ArxivLiveRetriever
from src.retrieval.corpus import normalize_paper_id
from src.rewriter.base import build_rewriter
from src.rewriter.paper_resolver import PaperResolver, resolve_and_verify

EXAMPLES = [
    "사진 보고 글로 설명해주는 AI",
    "가짜 뉴스 걸러내는 방법",
    "AI가 사람처럼 대화하게 만들기",
    "Attention is all you need",
]

# ── 검색 예산 ──────────────────────────────────────────────────────────────
TOP_K = 10              # 사용자에게 보여줄 최대 논문 수
DEPTH_LOCAL = 300       # 로컬 의미 검색에서 받아올 후보 수
DEPTH_ARXIV = 100       # arXiv 에서 받아올 후보 수 (page_size 가 100 이라 호출 1회로 끝남)
RERANK_DEPTH = 300      # 재정렬에 넣을 최대 후보 수

# 이 깊이는 평가에서 Recall@10 = 0.680 을 낸 설정과 같다. 서비스 속도를 위해 줄이고 싶은
# 유혹이 있지만, 줄이기 전에 반드시 질문 하나 기준 응답 시간을 실측한다.
#
# 줄일 근거는 이미 있다. 정답의 1차 등수별 재정렬 회수율이 1~10등 0.956, 11~30등 0.938
# 인데 101~200등 0.350, 201~300등 0.125 로 절벽처럼 떨어진다. 즉 재정렬은 이미 위에 있는
# 정답만 지키고 깊은 곳에서는 끌어올리지 못하므로, 깊이를 줄여도 잃는 것이 적다.
# 다만 그것은 추정이고, 얼마나 느린지는 재봐야 안다. 재기 전에 줄이면 얻는 것도 잃는
# 것도 모르는 채로 바꾸는 것이다. 사이드바의 '단계별 소요 시간 보기'로 실측한다.

st.set_page_config(page_title="Papers-Please", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="arXiv 검색기 준비 중…")
def load_arxiv() -> ArxivLiveRetriever:
    return ArxivLiveRetriever()


@st.cache_resource(show_spinner="쿼리 변환기(Qwen3-4B) 연결 중…")
def load_rewriter():
    """검색 성공을 보상으로 학습한 변환기를 쓴다.

    계층 변환기(hierarchical)가 아니라 dpo 를 쓰는 이유는 실측 차이가 크기 때문이다.
    시험용 300문항 Recall@10 기준 변환 안 함 0.167, 계층 0.220, 학습 후 0.343 이다.
    """
    return build_rewriter("dpo")


@st.cache_resource(show_spinner="논문 71만 편 색인 불러오는 중… (첫 실행은 1분 정도 걸립니다)")
def load_local_index():
    """로컬 의미 검색기. 반드시 cache_resource 로 감싼다.

    Streamlit 은 사용자가 무언가 누를 때마다 스크립트를 처음부터 다시 실행한다.
    캐시하지 않으면 검색할 때마다 임베딩 2.93GB 를 새로 올려 메모리가 바로 터진다.
    """
    from src.retrieval.local_index import LocalDenseRetriever
    return LocalDenseRetriever(
        corpus_path=config.CORPUS_DIR / "corpus-cs2021.jsonl",
        out_prefix=config.DATA_DIR / "embeddings" / "cs2021",
    )


@st.cache_resource
def load_resolver() -> PaperResolver:
    return PaperResolver()


@st.cache_resource
def load_recommender() -> PaperRecommender:
    """추천 에이전트. **쿼리 변환기가 이미 올려 둔 Qwen3-4B 를 그대로 빌려 쓴다.**

    따로 올리면 같은 모델이 두 벌이 되어 8.64GB + 3.54GB 를 쓴다. GPU 가 16GB 라
    임베더와 재정렬까지 더하면 자리가 모자라고, 그러면 추천 쪽 모델이 오류 없이
    조용히 CPU 로 밀려난다. 그 상태로 재보니 추천 한 번에 229.6초가 걸렸다.
    GPU 에 올라가면 같은 일이 10.5초다.
    """
    rewriter = load_rewriter()
    if hasattr(rewriter, "generate_json"):
        return PaperRecommender(client=rewriter)
    return PaperRecommender()          # 변환기가 Ollama 계열이면 예전처럼 따로 부른다


@st.cache_resource(show_spinner="재정렬 모델 준비 중…")
def load_reranker():
    """교차 인코더 재정렬기 (질문과 논문을 함께 읽고 관련도를 매긴다)."""
    from src.retrieval.ranking import CrossEncoderReranker
    return CrossEncoderReranker()


def arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def search_arxiv(query: str, k: int):
    """arXiv 검색. (결과, 오류메시지) 튜플. 오류를 결과 없음과 구분한다."""
    try:
        return load_arxiv().search(query, k=k), None
    except Exception as e:
        return None, (f"arXiv 검색 중 오류가 발생했습니다 (일시적일 수 있어요, "
                      f"잠시 후 다시 시도해 주세요). [{type(e).__name__}]")


def merge_channels(*channels) -> list:
    """채널별 결과를 합치고 같은 논문을 하나로 만든다.

    논문 번호 표기를 반드시 통일해야 한다. 통일하지 않으면 2103.00020 과 2103.00020v2 가
    서로 다른 논문으로 취급되어, 두 채널이 함께 찾아낸 논문일수록 중복으로 남는다.
    가장 확실한 정답이 가장 손해를 보는, 알아채기 어려운 고장이다 (ISSUE 27).
    """
    seen: dict[str, object] = {}
    for ch in channels:
        for p in ch or []:
            pid = normalize_paper_id(p.paper_id)
            if pid not in seen:
                seen[pid] = p
    return list(seen.values())


def render_results(results, error: str | None = None) -> None:
    if error:
        st.warning(error)
        return
    if not results:
        st.info("검색 결과가 없습니다.")
        return
    for i, r in enumerate(results, 1):
        st.markdown(f"**{i}. [{r.title}]({arxiv_url(r.paper_id)})**")
        with st.expander("초록 보기"):
            st.write(r.abstract)


# ── 화면 ──────────────────────────────────────────────────────────────────
st.title("📄 Papers, Please")
st.caption("일상어, 한국어로 검색해도, arXiv에서 논문을 찾아 드립니다.")

with st.sidebar:
    st.header("설정")
    use_local = st.checkbox("로컬 의미 검색 사용 (권장)", value=True,
                            help="논문 71만 편을 뜻으로 검색합니다. 첫 실행에 1분 정도 걸립니다.")
    use_arxiv = st.checkbox("arXiv 실시간 검색 사용", value=True,
                            help="최신 논문과 색인 밖 분야를 담당합니다.")
    show_timing = st.checkbox("단계별 소요 시간 보기", value=False)
    st.divider()
    st.caption("arXiv 요청 제한을 지키기 위해 한 검색당 호출을 최소화합니다.")

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
    if not (use_local or use_arxiv):
        st.error("검색 통로를 하나 이상 켜 주세요.")
        st.stop()

    timing: dict[str, float] = {}

    # 특정 유명 논문을 설명으로 찾는 질문이면, 그 논문을 짚어서 먼저 보여준다
    # (LLM 지식으로 제목 추정 -> arXiv 에서 실제 존재를 검증한 경우에만)
    if use_arxiv:
        t0 = time.time()
        with st.spinner("어떤 논문을 찾는 질문인지 확인 중…"):
            try:
                resolved_paper, _ = resolve_and_verify(query, load_resolver(), load_arxiv())
            except Exception:
                resolved_paper = None       # arXiv 오류 등은 배너 생략(치명적 아님)
        timing["논문 지목 확인"] = time.time() - t0
        if resolved_paper:
            st.success("🎯 이 논문을 찾으시는 것 같습니다")
            st.markdown(f"### [{resolved_paper.title}]({arxiv_url(resolved_paper.paper_id)})")
            with st.expander("초록 보기"):
                st.write(resolved_paper.abstract)
            st.divider()

    # 1) 변환 과정 (핵심 사용자 경험)
    t0 = time.time()
    with st.spinner("Qwen3-4B가 학술 용어로 변환 중…"):
        rw = load_rewriter().rewrite(query)
    timing["쿼리 변환"] = time.time() - t0

    st.subheader("🧭 쿼리 변환 과정")
    if not rw.parse_ok:
        st.warning("변환에 실패해 원본 검색어로 검색합니다.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1) 의도 파악**")
    c1.info(rw.intent or "—")
    c2.markdown("**2) 핵심 개념**")
    c2.info("　".join(rw.concepts) if rw.concepts else "—")
    c3.markdown("**3) 학술 용어**")
    c3.success("　".join(rw.academic_terms) if rw.academic_terms else "—")

    # 2) 두 채널로 후보 모으기
    st.divider()
    st.subheader("🔍 검색된 논문")

    local_hits, arxiv_hits, arxiv_err = None, None, None

    if use_local:
        t0 = time.time()
        with st.spinner(f"논문 71만 편에서 뜻으로 찾는 중… (후보 {DEPTH_LOCAL}편)"):
            # 로컬 의미 검색에는 변환된 검색어가 아니라 원본 질문을 넣는다.
            # 이 채널은 글자가 아니라 뜻으로 찾으므로 변환이 필요 없고, 실측에서도
            # 원본 질문 그대로가 가장 잘 나왔다.
            try:
                local_hits = load_local_index().search(query, k=DEPTH_LOCAL)
            except Exception as e:
                st.warning(f"로컬 색인을 쓸 수 없습니다: {e}")
        timing["로컬 의미 검색"] = time.time() - t0

    transformed_query = rw.query_for("arxiv")
    if use_arxiv:
        t0 = time.time()
        with st.spinner(f"arXiv 검색 중… (후보 {DEPTH_ARXIV}편)"):
            arxiv_hits, arxiv_err = search_arxiv(transformed_query, k=DEPTH_ARXIV)
        timing["arXiv 검색"] = time.time() - t0

    candidates = merge_channels(local_hits, arxiv_hits)[:RERANK_DEPTH]

    # 3) 재정렬 - 원본 질문과 대조해 다시 줄 세운다
    results = None
    if candidates:
        t0 = time.time()
        with st.spinner(f"사용자 의도와 대조해 순위 다시 매기는 중… (후보 {len(candidates)}편)"):
            # 변환된 검색어가 아니라 원본 질문으로 재정렬한다. 재정렬의 목적이
            # 사용자 의도와 맞는가를 보는 것이기 때문이다.
            results = load_reranker().rerank(query, candidates, top_k=TOP_K)
        timing["재정렬"] = time.time() - t0

    src = []
    if use_local:
        src.append(f"로컬 의미 검색 {len(local_hits or [])}편")
    if use_arxiv:
        src.append(f"arXiv {len(arxiv_hits or [])}편")
    st.caption(f"후보: {' + '.join(src)} → 중복 제거 {len(candidates)}편 → 상위 {TOP_K}편")
    if use_arxiv:
        st.caption(f"arXiv 검색어: `{transformed_query}`")
    render_results(results, arxiv_err if not candidates else None)

    # 4) 추천 - 후보를 사용자 의도와 대조해 고르고 이유를 붙인다
    #
    # 강력 추천과 관련만 보여준다. 10편을 억지로 채우면 무관한 논문까지 추천처럼 보여
    # 신뢰를 깎는다. 로컬 의미 검색은 어떤 질문에도 항상 10편을 채워 돌려주므로,
    # 이 걸러내기가 없으면 무관한 논문을 추천으로 포장하는 서비스가 된다.
    st.divider()
    st.subheader("🤖 최종 추천 논문")
    if results:
        t0 = time.time()
        with st.spinner("에이전트가 의도에 맞는 논문을 고르는 중…"):
            rec = load_recommender().recommend(query, results)
        timing["추천"] = time.time() - t0

        if rec["summary"]:
            st.info(f"**종합:** {rec['summary']}")
        badge = {"high": "🟢 강력 추천", "medium": "🟡 관련"}
        shown = [r for r in rec["recommendations"] if r["relevance"] in ("high", "medium")]
        if not shown:
            st.warning("의도에 맞는 논문을 찾지 못했습니다. 검색어를 바꿔 다시 시도해 보세요. "
                       "(위 '검색된 논문'에서 직접 확인하실 수 있습니다)")
        n_hidden = len(rec["recommendations"]) - len(shown)
        if n_hidden:
            st.caption(f"관련성이 낮은 {n_hidden}편은 숨겼습니다.")
        for r in shown:
            p = results[r["index"] - 1]
            st.markdown(f"{badge.get(r['relevance'], '')}　**[{p.title}]({arxiv_url(p.paper_id)})**")
            st.caption(f"↳ 추천 이유: {r['reason']}")
    else:
        st.info("검색 결과가 없어 추천할 논문이 없습니다.")

    if show_timing and timing:
        st.divider()
        st.subheader("⏱ 단계별 소요 시간")
        total = sum(timing.values())
        for k, v in timing.items():
            st.caption(f"{k}: {v:.1f}초")
        st.caption(f"**합계: {total:.1f}초**")

# ── arXiv 이용 약관에 따른 표기 (공개 전 필수) ─────────────────────────────
# arXiv 이용 약관은 arXiv 가 지원하거나 보증하는 것처럼 표현하는 것을 명시적으로 금지한다.
# 메타데이터(제목, 초록, 논문 번호)는 CC0 로 배포되어 저장과 재사용이 허용되지만,
# 논문 원문(PDF, 소스)은 우리 서버에서 제공하지 않고 arXiv 초록 페이지로 보낸다.
st.divider()
st.caption(
    "논문 메타데이터(제목, 초록, 논문 번호)는 arXiv.org 에서 가져왔습니다. "
    "arXiv 메타데이터는 CC0 1.0 으로 배포됩니다. "
    "논문 원문은 arXiv 에서 직접 확인해 주세요."
)
st.caption(
    "이 서비스는 arXiv 와 무관한 개인 프로젝트이며, **arXiv 의 후원이나 보증을 받지 "
    "않았습니다.** Thank you to arXiv for use of its open access interoperability."
)
