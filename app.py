"""
웹 화면 (Streamlit)
검색어 두 개로 로컬 색인에서 찾고, 재정렬해 10편을 보여 줌

    질문
     +- 검색어 1  원본 (한국어면 영어로 옮김)
     +- 검색어 2  질문에 답할 법한 가상 초록
            |
       각각 로컬 의미 검색 71만 편 -> 순위 합치기
            |
       교차 인코더 재정렬 (원본 질문과 대조)
            |
       재정렬 순위와 검색 순위를 다시 합침
            |
       추천 에이전트가 관련도와 이유를 붙임 -> 10편 추천

arXiv 실시간 검색은 이 목록에 섞지 않고 '최신 논문' 칸으로 따로 보여 줌.

실행:
  streamlit run app.py
  (Ollama 서버 + qwen3:4b, 로컬 색인, 인터넷 연결 확인 필요)
"""

from __future__ import annotations

import re
import time

import streamlit as st

from src import config
from src.gpu_pool import GpuPool
from src.recommend_agent.recommender import PaperRecommender
from src.retrieval.arxiv_live import ArxivLiveRetriever
from src.rewriter.base import build_rewriter
from src.rewriter.paper_resolver import PaperResolver, resolve_and_verify
from src.schemas import RewriteResult

TOP_K = 10              # 사용자에게 보여줄 최대 논문 수
DEPTH_LOCAL = 100       # 검색어마다 로컬 의미 검색에서 받아올 후보 수
DEPTH_ARXIV = 100       # arXiv에서 받아올 후보 수
RERANK_DEPTH = 100      # 재정렬에 넣을 최대 후보 수

# 재정렬 순위를 검색 순위와 한 번 더 합칠 때의 재정렬 쪽 가중치 (검색 쪽은 1.0)
# 0이면 안 합치고 재정렬 순위만 씀.
FUSE_RERANK_WEIGHT: float = 3.0

# 쓸 색인
# `cs2021-ft` bge-m3 를 파인튜닝한 모델로 만든 색인
LOCAL_INDEX = "cs2021-ft"

# "관련 논문을 못 찾았다"고 말할 기준 점수
MIN_RERANK_SCORE: float | None = 0.002

st.set_page_config(page_title="Papers, Please", layout="wide")

@st.cache_resource
def pool() -> GpuPool:
    """이 안에 든 모델만 올렸다 내림"""
    return GpuPool()


class _Qwen3Bundle:
    """qwen3:4b를 쓰는 모듈 네 가지를 하나로 묶음

    번역, 가상 초록, 논문 지목, 추천 이유가 전부 같은 모델을 부름. `OllamaClient` 하나를
    나눠 쓰면 pool이 내릴 때 그 하나만 내리면 되고, 같은 모델이 여러 벌 올라갈 일도 없음.
    """

    def __init__(self):
        from src.rewriter.base import OllamaClient
        from src.rewriter.baselines import HydeRewriter, TranslateRewriter

        self.client = OllamaClient()
        self.translator = TranslateRewriter(client=self.client)
        self.hyde = HydeRewriter(client=self.client)
        self.resolver = PaperResolver(client=self.client)
        self.recommender = PaperRecommender(client=self.client)

    def unload(self) -> None:
        """`GpuPool.release`가 부름. Ollama는 이 프로세스 밖이라 따로 내려야 함"""
        self.client.unload()


def _qwen3_bundle() -> _Qwen3Bundle:
    return _Qwen3Bundle()


@st.cache_resource(show_spinner="arXiv 검색기 준비 중...")
def load_arxiv() -> ArxivLiveRetriever:
    return ArxivLiveRetriever()


def load_rewriter():
    """쿼리 변환기. arXiv용 검색어를 만듦"""
    return pool().get("ollama:rewriter", lambda: build_rewriter("dpo"))


@st.cache_resource(show_spinner="논문 71만 편 색인 불러오는 중... (첫 실행은 1분 정도 걸립니다)")
def load_local_index():
    """로컬 의미 검색기"""
    from src.retrieval.local_index import LocalDenseRetriever, read_meta

    prefix = config.DATA_DIR / "embeddings" / LOCAL_INDEX
    model = read_meta(prefix).get("model") or config.EMBED_MODEL
    return LocalDenseRetriever(
        corpus_path=config.CORPUS_DIR / "corpus-cs2021.jsonl",
        out_prefix=prefix,
        model_name=model,
    )


def load_resolver() -> PaperResolver:
    return pool().get("ollama:qwen3", _qwen3_bundle).resolver


def load_recommender() -> PaperRecommender:
    """추천 이유 생성 모델. qwen3:4b를 씀."""
    return pool().get("ollama:qwen3", _qwen3_bundle).recommender


def load_reranker():
    """교차 인코더 재정렬 모델. 질문과 논문을 함께 읽고 관련도 순위를 매김"""
    def make():
        from src.retrieval.ranking import CrossEncoderReranker
        return CrossEncoderReranker()
    return pool().get("reranker", make)


def load_translator():
    """한국어 질문을 영어로 옮기는 번역 모델. qwen3:4b를 씀"""
    return pool().get("ollama:qwen3", _qwen3_bundle).translator


def load_hyde():
    """
    질문에 답할 법한 가상의 영어 초록을 만드는 변환기. qwen3:4b를 씀
    일상어 질문은 번역만으로는 안 됨. 번역은 일상어를 일상어인 채로 옮기기 때문임.
    """
    return pool().get("ollama:qwen3", _qwen3_bundle).hyde


def one_line(text: str) -> str:
    """
    제목을 한 줄로 펼침
    """
    return " ".join((text or "").split())


def arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


_ABS_TERM = re.compile(r'abs:"([^"]+)"')


def academic_terms_of(rw, raw_query: str) -> list[str]:
    """
    변환기가 만든 영어 학술 용어를 추출함
    학습한 변환기는 arXiv 문법 문자열 하나만 내놓아 'academic_terms' 필드가 빔.
    용어는 문자열 안에 `abs:"..."`에 들어 있으므로 거기서 추출함
    """
    if rw.academic_terms:
        return list(rw.academic_terms)
    found = _ABS_TERM.findall(rw.query_for("arxiv") or "")
    return [t for t in dict.fromkeys(found) if t.strip() and t.strip() != raw_query.strip()]


def fuse_local(literal_hits, hyde_hits) -> list:
    """두 검색어로 얻은 로컬 결과를 순위로 합침"""
    from src.retrieval.ranking import rrf_fuse

    channels = {"literal": literal_hits or []}
    if hyde_hits:
        channels["hyde"] = hyde_hits
    return rrf_fuse(channels, k=config.RRF_K, top_n=RERANK_DEPTH)


def fuse_with_search(ranked: list, candidates: list) -> list:
    """재정렬 순위와 검색 순위를 한 번 더 합침"""
    if FUSE_RERANK_WEIGHT <= 0 or not ranked:
        return ranked
    from src.retrieval.ranking import rrf_fuse_ids

    order = rrf_fuse_ids(
        {"rerank": [p.paper_id for p in ranked],
         "search": [c.paper_id for c in candidates]},
        k=config.RRF_K, top_n=len(ranked),
        weights={"rerank": FUSE_RERANK_WEIGHT, "search": 1.0})
    by_id = {p.paper_id: p for p in ranked}
    return [by_id[pid] for pid in order if pid in by_id]


def run_search(query: str, use_local: bool, use_arxiv: bool, status) -> dict:
    """검색 한 번을 끝까지 수행하고 결과를 모아 돌려줌"""
    out: dict = {"timing": {}, "arxiv_error": None, "resolved": None,
                 "local_hits": None, "hyde_hits": None, "arxiv_hits": None,
                 "results": None, "recommendation": None}

    if use_arxiv:
        status.write("어떤 논문을 찾는 질문인지 확인하는 중...")
        t0 = time.time()
        try:
            out["resolved"], _ = resolve_and_verify(query, load_resolver(), load_arxiv())
        except Exception:
            out["resolved"] = None
        out["timing"]["논문 지목 확인"] = time.time() - t0

    from src.rewriter.baselines import TranslateRewriter
    out["search_text"] = query
    if use_local and TranslateRewriter.has_hangul(query):
        status.write("한국어 질문을 영어로 번역하는 중...")
        t0 = time.time()
        tr = load_translator().rewrite(query)
        if tr.parse_ok:
            out["search_text"] = tr.query_for("dense")
        else:
            out["translate_error"] = tr.intent
        out["timing"]["한국어를 영어로"] = time.time() - t0

    out["hyde_text"] = None
    if use_local:
        status.write("질문에 맞는 논문이 어떻게 쓰여 있을지 떠올리는 중...")
        t0 = time.time()
        hy = load_hyde().rewrite(query)
        if hy.parse_ok:
            out["hyde_text"] = hy.query_for("dense")
        else:
            out["hyde_error"] = hy.intent
        out["timing"]["두 번째 검색어 만들기"] = time.time() - t0

    if use_arxiv:
        status.write("검색어를 학술 용어로 바꾸는 중...")
        t0 = time.time()
        pool().release("ollama:qwen3")
        out["rewrite"] = load_rewriter().rewrite(query)
        out["timing"]["쿼리 변환"] = time.time() - t0
    else:
        out["rewrite"] = RewriteResult(
            raw_query=query,
            queries={"dense": out["search_text"], "arxiv": query},
            intent="", parse_ok=True)

    pool().release("ollama:qwen3", "ollama:rewriter")

    if use_local:
        status.write("코퍼스에서 찾는 중...")
        t0 = time.time()
        try:
            index = load_local_index()
            out["local_hits"] = index.search(out["search_text"], k=DEPTH_LOCAL)
            if out["hyde_text"]:
                out["hyde_hits"] = index.search(out["hyde_text"], k=DEPTH_LOCAL)
        except Exception as e:
            out["local_error"] = str(e)
        out["timing"]["로컬 의미 검색"] = time.time() - t0

    if use_arxiv:
        status.write("arXiv에서 논문을 찾는 중...")
        t0 = time.time()
        try:
            out["arxiv_hits"] = load_arxiv().search(
                out["rewrite"].query_for("arxiv"), k=DEPTH_ARXIV)
        except Exception as e:
            out["arxiv_error"] = (f"arXiv 검색 중 오류가 발생했습니다. 잠시 후 "
                                  f"다시 시도해 주세요. [{type(e).__name__}]")
        out["timing"]["arXiv 검색"] = time.time() - t0

    candidates = fuse_local(out["local_hits"], out["hyde_hits"])
    out["n_candidates"] = len(candidates)

    if candidates:
        status.write("질문 의도를 분석해 관련 논문 순위를 다시 매기는 중...")
        t0 = time.time()
        ranked = load_reranker().rerank(query, candidates, top_k=len(candidates))
        out["results"] = fuse_with_search(ranked, candidates)[:TOP_K]
        out["timing"]["재정렬"] = time.time() - t0

    if out["results"]:
        status.write("각 논문을 왜 추천하는지 정리하는 중...")
        t0 = time.time()
        pool().release("reranker")
        try:
            out["recommendation"] = load_recommender().recommend(query, out["results"])
        except Exception as e:
            out["recommend_error"] = str(e)
        out["timing"]["추천 이유 생성"] = time.time() - t0

    return out


def confident_results(results, min_score: float | None):
    numbered = list(enumerate(results or [], start=1))
    if min_score is None:
        return numbered, []
    keep = [(i, p) for i, p in numbered if p.score >= min_score]
    drop = [(i, p) for i, p in numbered if p.score < min_score]
    return keep, drop


HYDE_PREVIEW_LEN = 100

def render_understanding(query: str, state: dict) -> None:
    rw = state["rewrite"]
    translated = state.get("search_text")
    has_translation = bool(translated) and translated.strip() != query.strip()

    intent = (rw.intent or "").strip()
    has_intent = bool(intent) and intent != query.strip()
    terms = academic_terms_of(rw, query)
    hyde = (state.get("hyde_text") or "").strip()

    if not (has_translation or hyde or has_intent or terms):
        st.caption("입력하신 검색어를 그대로 뜻으로 검색했습니다.")
        return

    st.markdown("#### 이렇게 검색했습니다")
    n = 1 + int(has_translation) + int(bool(hyde)) + int(has_intent) + int(bool(terms))
    cols = st.columns(n)
    i = 0
    with cols[i]:
        st.caption("원본 질문")
        st.info(query)
    if has_translation:
        i += 1
        with cols[i]:
            st.caption("영어로 번역한 쿼리")
            st.success(translated)
    if hyde:
        i += 1
        with cols[i]:
            st.caption("쿼리를 바탕으로 작성한 가상 초록")
            if len(hyde) > HYDE_PREVIEW_LEN:
                st.success(hyde[:HYDE_PREVIEW_LEN] + "...")
                with st.expander("가상 초록 전체 보기"):
                    st.write(hyde)
            else:
                st.success(hyde)
    if has_intent:
        i += 1
        with cols[i]:
            st.caption("찾으시는 것")
            st.info(intent)
            if rw.concepts:
                st.caption("핵심 개념: " + ", ".join(rw.concepts))
    if terms:
        i += 1
        with cols[i]:
            st.caption("arXiv 검색에 사용한 키워드")
            st.success(", ".join(terms))
    if state.get("translate_error"):
        st.caption(f"번역에 실패해 원본 질문으로 검색했습니다. {state['translate_error']}")
    if not rw.parse_ok:
        st.caption("변환에 실패해 원본 질문으로 검색했습니다.")


_SUMMARY_INDEX = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\s*,\s*\d{1,2})+)\s*번(?!째)"
    r"|(?<![\d.])(\d{1,2})\s*번(?!째)"
    r"|index\s*(\d{1,2})"
)


def retitle_summary(summary: str, results: list) -> str:
    if not summary:
        return summary

    def title_of(num: str) -> str | None:
        i = int(num)
        return one_line(results[i - 1].title) if 1 <= i <= len(results) else None

    def swap(m: re.Match) -> str:
        listed, one, indexed = m.groups()
        if listed:
            titles = [title_of(n.strip()) for n in listed.split(",")]
            if any(t is None for t in titles):
                return m.group(0)      # 목록 밖 번호가 섞이면 손대지 않음
            return ", ".join(f"'{t}'" for t in titles)
        title = title_of(one or indexed)
        return f"'{title}'" if title else m.group(0)

    return _SUMMARY_INDEX.sub(swap, summary)


RELEVANCE = {
    "high":   ("관련성 높음", st.success),
    "medium": ("관련성 있음", st.warning),
    "low":    ("관련성 낮음", st.error),
}
RELEVANCE_ORDER = ("high", "medium", "low")


def render_paper(rank: int, paper, judgement: dict | None) -> None:
    """결과 한 편. 추천 에이전트의 판단을 보여 줌"""
    grade = (judgement or {}).get("relevance", "")
    head = f"#### {rank}. [{one_line(paper.title)}]({arxiv_url(paper.paper_id)})"
    if grade in RELEVANCE:
        label, block = RELEVANCE[grade]
        st.markdown(f"{head}  `{label}`")
        block(judgement.get("reason") or label)
    else:
        st.markdown(head)
    with st.expander("초록 보기"):
        st.write(paper.abstract or "(초록 없음)")


def render_results(state: dict) -> None:
    """결과를 목록 하나로 보여줌"""
    results = state.get("results")
    if not results:
        if state.get("arxiv_error"):
            st.warning(state["arxiv_error"])
        else:
            st.info("관련 논문을 찾지 못했습니다. 질문을 바꿔 다시 시도해 보세요.")
        return

    rec = state.get("recommendation") or {}
    by_index = {r["index"]: r for r in rec.get("recommendations", [])}
    keep, dropped = confident_results(results, MIN_RERANK_SCORE)
    judged = [j for j in (by_index.get(i, {}).get("relevance")
                          for i in range(1, len(results) + 1)) if j]
    nothing_good = bool(judged) and all(j == "low" for j in judged)

    if not keep or nothing_good:
        st.warning("관련 논문을 찾지 못했습니다. 질문을 바꿔 다시 시도해 보세요.")
        return

    if rec.get("summary"):
        st.info(retitle_summary(rec["summary"], results))

    def relevance_rank(item) -> int:
        grade = by_index.get(item[0], {}).get("relevance", "")
        return RELEVANCE_ORDER.index(grade) if grade in RELEVANCE_ORDER else len(RELEVANCE_ORDER)

    keep = sorted(keep, key=relevance_rank)

    for shown, (orig, p) in enumerate(keep, 1):
        render_paper(shown, p, by_index.get(orig))

def render_recent(state: dict, use_arxiv: bool) -> None:
    """
    arXiv 실시간 검색 결과를 '최신 논문'으로 따로 보여 줌
    본 목록에 이미 있는 논문은 빼서 같은 논문이 두 번 나오지 않게 함.
    """
    hits = state.get("arxiv_hits") or []
    if not (use_arxiv and hits):
        return
    from src.retrieval.corpus import normalize_paper_id

    shown = {normalize_paper_id(p.paper_id) for p in (state.get("results") or [])}
    fresh = [p for p in hits if normalize_paper_id(p.paper_id) not in shown][:5]
    if not fresh:
        return

    st.divider()
    st.markdown(f"#### arXiv에서 직접 찾은 논문 {len(fresh)}편")
    st.caption("코퍼스에 없는 논문입니다. 관련 정도가 아닌 arXiv가 전달해준 순서이고, 추천 논문과 달리 "
               "관련도를 매기지 않았습니다.")
    for i, p in enumerate(fresh, 1):
        render_paper(i, p, None)


st.title("Papers, Please")
st.caption("사용자의 질문에 맞춰 arXiv에서 논문을 찾아 드립니다.")

with st.sidebar:
    st.header("설정")
    st.caption("하나 이상의 검색 옵션을 선택해주세요.")
    use_local = st.checkbox("코퍼스에서 의미 검색 (권장)", value=True,
                            help="코퍼스에서 의미 기반으로 검색합니다.")
    use_arxiv = st.checkbox("코퍼스에 없는 arXiv 논문도 검색", value=True,
                            help="코퍼스에 없는 논문을 찾아 아래에 따로 보여줍니다. "
                                 "추천 목록 순위에는 영향을 주지 않습니다.")

    st.divider()
    st.caption("arXiv 호출 제한이 있으므로, 짧은 시간 내 과도한 검색 시 속도 저하가 발생할 수 있습니다")

if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input(
    "무엇을 찾으시나요?", value=st.session_state.query,
    placeholder="예: 사진을 보고 글로 설명해주는 AI 관련 논문을 찾아줘.",
    label_visibility="collapsed")

go = st.button("검색", type="primary", use_container_width=True)

if go and query.strip():
    if not (use_local or use_arxiv):
        st.error("검색 옵션을 하나 이상 선택해 주세요.")
        st.stop()

    with st.status("논문을 찾는 중입니다...", expanded=True) as status:
        with pool().session():
            state = run_search(query, use_local, use_arxiv, status)
        status.update(label="검색을 마쳤습니다.", state="complete", expanded=False)

    st.divider()

    if state.get("resolved"):
        p = state["resolved"]
        st.success("이 논문을 찾으시는 것 같습니다")
        st.markdown(f"### [{one_line(p.title)}]({arxiv_url(p.paper_id)})")
        with st.expander("초록 보기"):
            st.write(p.abstract)
        st.divider()

    render_understanding(query, state)
    st.divider()

    st.markdown("#### 찾은 논문")
    render_results(state)
    render_recent(state, use_arxiv)

    timing = state.get("timing") or {}
    if timing:
        st.caption("총 소요 시간 ({:.1f}초)".format(sum(timing.values())))

st.divider()
st.caption(
    "본 서비스는 논문 검색 시 참고용으로만 사용해주시길 바랍니다."
)
st.caption(
    "arXiv 메타데이터는 arXiv에서 가져왔습니다. "
    "arXiv 메타데이터는 CC0 1.0으로 배포됩니다. "
    "논문 원문은 arXiv에서 직접 확인해 주세요."
)
st.caption(
    "이 서비스는 arXiv와 무관한 프로젝트이며, arXiv의 후원이나 보증을 받지 "
    "않았습니다."
)
st.caption(
    "Thank you to arXiv for use of its open access interoperability."
    "This service was not reviewed or approved by, nor does it necessarily express or reflect the policies or opinions of, arXiv."
)

