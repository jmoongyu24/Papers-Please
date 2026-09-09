"""웹 화면 (Streamlit). 검색어 두 개로 로컬 색인을 찾고, 재정렬해 10편을 보여 줌.

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
       추천 에이전트가 관련도와 이유를 붙임 -> 10편

arXiv 실시간 검색은 이 목록에 섞지 않고 '최신 논문' 칸으로 따로 보여 줌. 섞으면 비율을
어떻게 잡아도 만족도가 떨어졌고(전부 p<0.001), arXiv 채널의 가치는 정확도가 아니라
색인에 없는 최신 논문이기 때문임.

검색어 두 개를 쓰는 이유: 번역만 쓰면 일상어가 일상어인 채로 넘어가 분야 용어에 닿지
못하고, 가상 초록만 쓰면 4B 모델이 모르는 분야에서 엉뚱한 초록을 지어냄. 둘을 합치면
양쪽을 다 얻음 (개발용 348문항 Recall@10 0.566 · 0.555 -> 0.586).

한국어 질문을 먼저 영어로 옮기는 이유: arXiv 논문은 제목도 초록도 영어임. 번역해서
넣으니 시험용 342문항에서 한국어 Recall@10 이 0.456 에서 0.567 로 올랐고(p=0.001),
손대지 않은 영어는 그대로였음.

재정렬에는 번역문이 아니라 원본 질문을 씀. 번역문으로도 재봤는데 이득이 없었고,
재정렬의 목적이 사용자 의도와 맞는지 보는 것이기 때문임.

실행:
  streamlit run app.py
  (전제: Ollama 서버 + qwen3:4b, 로컬 색인, 인터넷 연결)
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

# -- 검색 예산 --------------------------------------------------------------
#
# 후보 깊이 100 은 실측으로 정한 값임. 300 으로 늘리면 후보 안에 정답이 있는 비율은
# 오르지만(0.635 -> 0.721) 최종 성능은 안 따라오고(p=0.214) 만족도는 확실히
# 떨어짐(-0.036, p<0.001). 후보가 늘면 재정렬기가 엉뚱한 논문을 올리는 일도 함께 늚.
TOP_K = 10              # 사용자에게 보여줄 최대 논문 수
DEPTH_LOCAL = 100       # 검색어마다 로컬 의미 검색에서 받아올 후보 수
DEPTH_ARXIV = 100       # arXiv 에서 받아올 후보 수 (한 번 호출로 끝남)
RERANK_DEPTH = 100      # 재정렬에 넣을 최대 후보 수

# -- 재정렬 순위를 검색 순위와 한 번 더 합칠 때의 재정렬 쪽 가중치 (검색 쪽은 1.0) ----
#
# 0 이면 안 합치고 재정렬 순위만 씀.
#
# 왜 합치는가: 일상어 질문에서 재정렬기가 후보 100편 전부에 '관련 없음' 에 해당하는 값을
# 줌. 그 구간 안의 순서에는 근거가 없는데, 검색 순위는 같은 문항에서 다른 논문을 맞히고
# 있음. 개발용 348문항에서 가중치 3 이 전체와 일상어 층 모두 가장 좋았음(0.618 -> 0.635).
#
# 파인튜닝 색인(`cs2021-ft`)과 한 묶음임. 원본 색인에서는 효과가 없음(p=0.583).
# 다시 정하려면 감이 아니라 이 값을 바꿔 가며 재계산할 것:
#   python -m evaluation.pipeline_eval --report-only <실행결과> --rerank cross \
#       --rerank-depth 100 --fuse-rerank N
FUSE_RERANK_WEIGHT: float = 3.0

# -- 쓸 색인 ----------------------------------------------------------------
#
# `cs2021`    원본 `BAAI/bge-m3` 로 만든 색인
# `cs2021-ft` 파인튜닝한 `models/retriever-ft` 로 만든 색인   <- 지금 쓰는 것
#
# 질문을 임베딩할 모델은 `load_local_index` 가 색인 파일에 적힌 것을 읽어 씀.
# 손으로 맞추지 말 것 - 어긋나면 오류 없이 순위만 무너짐.
#
# 시험용 342문항 Recall@10 (순위 합치기와 한 묶음): 0.617 -> 0.658. 주 지표는 p=0.085 로
# 유의성을 확보하지 못했고, 채택 근거는 ⓐ 유의미하게 나빠진 무리가 하나도 없고
# ⓑ 한국어가 유의미하게 좋아졌다는 것임(+0.070, p=0.045).
#
# 원본 bge-m3 로 만든 색인(`cs2021`)을 쓰려면 이 값을 "cs2021" 로 바꾸고
# FUSE_RERANK_WEIGHT 를 0 으로 둘 것. 둘은 한 묶음임. 그 색인은 저장소에 없으므로
# README 의 색인 만들기를 먼저 할 것.
LOCAL_INDEX = "cs2021-ft"

# -- "못 찾았다"고 말할 기준선 ----------------------------------------------
#
# 로컬 의미 검색은 어떤 질문에도 후보를 채워서 돌려줌. 그대로 뿌리면 무관한 논문을
# 추천으로 포장하게 됨. 이 값 아래의 논문은 지우지 않고 '관련성이 낮아 접어 둔' 자리로
# 내려보내기만 함. None 이면 걸러내지 않음.
#
# 감으로 정하면 안 되고 등급 정답지로 실측해서 정함:
#   python -m evaluation.pipeline_eval --calibrate-threshold \
#       --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl
#
# 0.002 에서 무관한 논문의 44.4% 를 걸러내고 만족스러운 논문의 11.0% 를 잘못 버림.
# 일부러 느슨하게 잡았음 - 잘못 낸 "못 찾았습니다" 는 무관한 논문 한 편을 보여주는 것보다
# 훨씬 나쁜 경험이고, 같은 만족도라도 한국어 질문의 점수가 영어보다 낮게 나오기 때문임.
#
# 재정렬 모델을 바꾸면 점수 범위가 달라지므로 반드시 다시 재야 함.
MIN_RERANK_SCORE: float | None = 0.002

st.set_page_config(page_title="Papers, Please", layout="wide")


# ==========================================================================
# 무거운 부품
# ==========================================================================
#
# 그래픽 메모리를 쓰는 것은 `GpuPool` 에 맡겨 검색이 도는 동안에만 올림. 언어 모델
# 하나가 3.25GB 라 재정렬 모델(1.24GB)과 같은 시각에 올라가면 4.76GB 가 됨. 검색 한
# 번을 세 구간으로 나누고 구간이 바뀔 때 반대편을 내려 최대 3.52GB 로 맞춤.
#
# 그래픽 메모리를 안 쓰는 것(arXiv 검색기, 색인 본체)은 `st.cache_resource` 로 그대로
# 둠. 색인은 임베딩 2.8GB 를 시스템 메모리에 들고 있고 다시 올리는 데 1분 걸림.


@st.cache_resource
def pool() -> GpuPool:
    """앱당 하나. 이 안에 든 모델만 올렸다 내림."""
    return GpuPool()


class _Qwen3Bundle:
    """qwen3:4b 를 쓰는 네 가지를 한 덩어리로 묶음.

    번역, 가상 초록, 논문 지목, 추천 이유가 전부 같은 모델을 부름. `OllamaClient` 하나를
    나눠 쓰면 pool 이 내릴 때 그 하나만 내리면 되고, 같은 모델이 여러 벌 올라갈 일도 없음.
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
        """`GpuPool.release` 가 부름. Ollama 는 이 프로세스 밖이라 따로 내려야 함."""
        self.client.unload()


def _qwen3_bundle() -> _Qwen3Bundle:
    return _Qwen3Bundle()


@st.cache_resource(show_spinner="arXiv 검색기 준비 중...")
def load_arxiv() -> ArxivLiveRetriever:
    return ArxivLiveRetriever()


def load_rewriter():
    """검색 성공을 보상으로 학습한 변환기. arXiv 채널 검색어를 만듦.

    Ollama 4비트로 부름. transformers 로 올리면 8.27GB 인데 이쪽은 3.25GB 이고, 검색이
    끝나면 내려감. 이 변환기가 만든 문자열은 arXiv API 에만 들어감 - 로컬 의미 검색은
    번역기와 가상 초록 생성기가 만든 검색어를 씀.
    """
    return pool().get("ollama:rewriter", lambda: build_rewriter("dpo"))


@st.cache_resource(show_spinner="논문 71만 편 색인 불러오는 중... (첫 실행은 1분 정도 걸립니다)")
def load_local_index():
    """로컬 의미 검색기. 반드시 cache_resource 로 감쌈.

    Streamlit 은 사용자가 무언가 누를 때마다 스크립트를 처음부터 다시 실행함.
    캐시하지 않으면 검색할 때마다 임베딩 2.93GB 를 새로 올려 메모리가 모자람.

    이 안에서 그래픽 메모리를 쓰던 것은 질문 임베딩 모델뿐인데 CPU 로 옮겼음
    (`config.EMBED_DEVICE`). 나머지는 시스템 메모리라 계속 들고 있어도 됨.
    """
    from src.retrieval.local_index import LocalDenseRetriever, read_meta

    # 색인을 만든 모델과 질문을 임베딩하는 모델은 반드시 같아야 함. 어긋나면 오류 없이
    # 순위만 무너지므로, 색인 파일에 적힌 모델을 그대로 읽어 씀
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
    """추천 이유 생성기. 번역기, 가상 초록 생성기와 같은 qwen3:4b 를 씀."""
    return pool().get("ollama:qwen3", _qwen3_bundle).recommender


def load_reranker():
    """교차 인코더 재정렬기. 질문과 논문을 함께 읽고 관련도를 매김.

    미리 저장해 둔 float16 사본이 있으면 적재가 5.4~5.8초에서 1.8초로 줄어듦
    (`config.RERANKER_FP16_DIR`, `training/export.py fp16` 으로 만듦). 검색마다 올렸다
    내리므로 이 차이가 그대로 응답 시간이 됨.
    """
    def make():
        from src.retrieval.ranking import CrossEncoderReranker
        return CrossEncoderReranker()
    return pool().get("reranker", make)


def load_translator():
    """한국어 질문을 영어로 옮기는 변환기.

    arXiv 논문은 제목도 초록도 영어라 한국어 질문이 크게 뒤졌음. 번역해서 넣으니 시험용
    342문항에서 한국어 Recall@10 이 0.456 에서 0.567 로 올랐고(p=0.001), 손대지 않은
    영어는 그대로였음. 언어 격차가 58% 줄었음.

    Ollama 를 쓰는 이유는 앱이 이미 qwen3:4b 를 올려 두어 메모리가 더 안 들고, 평가도
    같은 경로로 쟀기 때문임.
    """
    return pool().get("ollama:qwen3", _qwen3_bundle).translator


def load_hyde():
    """질문에 답할 법한 가상의 영어 초록을 지어내는 변환기. 두 번째 검색어를 만듦.

    일상어 질문은 번역만으로는 안 됨. 번역은 일상어를 일상어인 채로 옮기기 때문임.
    "조건이 많은 문제를 아주 적은 메모리로 대충 잘 푸는 방법" 을 그대로 옮기면 정답
    논문이 457,938등이었음. 필요한 것은 번역이 아니라 "제약 충족 문제", "스케치 근사"
    같은 분야 용어이고, 가상 초록이 그것을 끌어냄.

    반드시 바꿔치기가 아니라 함께 쓸 것. 가상 초록으로 원본 검색어를 대체하면 일상어
    층만 좋아지고 나머지가 다 나빠짐(전체 0.583 -> 0.555). 두 검색어로 각각 찾아 순위를
    합치면 양쪽을 다 얻음(0.586).

    모델은 앱이 이미 올려 둔 qwen3:4b 라 메모리가 더 안 듦.
    """
    return pool().get("ollama:qwen3", _qwen3_bundle).hyde


# ==========================================================================
# 검색 파이프라인 (화면 그리기와 분리해 둠)
# ==========================================================================

def one_line(text: str) -> str:
    """제목을 한 줄로 폄.

    arXiv 제목의 4% 쯤은 줄이 접혀 있음. 그대로 마크다운 제목에 넣으면 줄바꿈 자리에서
    표시가 끊겨 뒷줄이 본문 크기로 나오고 링크 표기가 글자 그대로 드러남.
    """
    return " ".join((text or "").split())


def arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


_ABS_TERM = re.compile(r'abs:"([^"]+)"')


def academic_terms_of(rw, raw_query: str) -> list[str]:
    """변환기가 만든 영어 학술 용어를 뽑음.

    학습한 변환기는 arXiv 문법 문자열 하나만 내놓아 `academic_terms` 필드가 빔. 용어는
    그 문자열 안에 `abs:"..."` 조각으로 들어 있으므로 거기서 뽑아냄. `all:"..."` 조각은
    원본 질문이라 제외함.
    """
    if rw.academic_terms:
        return list(rw.academic_terms)
    found = _ABS_TERM.findall(rw.query_for("arxiv") or "")
    return [t for t in dict.fromkeys(found) if t.strip() and t.strip() != raw_query.strip()]


def fuse_local(literal_hits, hyde_hits) -> list:
    """두 검색어로 얻은 로컬 결과를 순위로 합침.

    점수가 아니라 등수를 더함. 가상 초록은 문장이 길어 유사도 값이 전반적으로 다르게
    나오므로, 점수를 더하면 옳은 쪽이 아니라 점수 범위가 큰 쪽이 이김.
    """
    from src.retrieval.ranking import rrf_fuse

    channels = {"literal": literal_hits or []}
    if hyde_hits:
        channels["hyde"] = hyde_hits
    return rrf_fuse(channels, k=config.RRF_K, top_n=RERANK_DEPTH)


def fuse_with_search(ranked: list, candidates: list) -> list:
    """재정렬 순위와 검색 순위를 한 번 더 합침. `FUSE_RERANK_WEIGHT` 설명 참고.

    재정렬 점수는 그대로 들고 감. `MIN_RERANK_SCORE` 판정이 그 값을 쓰기 때문임.
    """
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
    """검색 한 번을 끝까지 수행하고 결과를 모아 돌려줌.

    화면을 그리지 않고 자료만 만듦. 진행 상황은 `status.write` 로만 알림.
    """
    out: dict = {"timing": {}, "arxiv_error": None, "resolved": None,
                 "local_hits": None, "hyde_hits": None, "arxiv_hits": None,
                 "results": None, "recommendation": None}

    # 특정 논문을 설명으로 찾는 질문이면 그 논문을 짚어서 먼저 보여 줌.
    # 언어 모델로 제목을 추정하고 arXiv 에서 실제로 있는지 확인한 경우에만
    if use_arxiv:
        status.write("어떤 논문을 찾는 질문인지 확인하는 중...")
        t0 = time.time()
        try:
            out["resolved"], _ = resolve_and_verify(query, load_resolver(), load_arxiv())
        except Exception:
            out["resolved"] = None      # arXiv 오류는 치명적이지 않으므로 넘어감
        out["timing"]["논문 지목 확인"] = time.time() - t0

    # 한국어면 영어로 옮겨 로컬 의미 검색에 넣음. arXiv 논문이 영어라서임.
    # 영어 질문은 건드리지 않음. 번역기를 통과시키면 뜻이 미묘하게 바뀜
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

    # 두 번째 검색어: 질문에 답할 법한 가상의 영어 초록. 원본을 대체하지 않고 함께 씀
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

    # 학습한 변환기는 arXiv 문법 문자열만 만들므로 arXiv 채널을 쓸 때만 부름.
    # 로컬 검색에는 쓸 곳이 없음
    if use_arxiv:
        status.write("검색어를 학술 용어로 바꾸는 중...")
        t0 = time.time()
        # 같은 크기의 언어 모델 두 개가 함께 올라가면 6.5GB 가 됨. 앞 단계에서 쓴
        # qwen3:4b 는 여기서 할 일이 끝났으므로 자리를 넘김
        pool().release("ollama:qwen3")
        out["rewrite"] = load_rewriter().rewrite(query)
        out["timing"]["쿼리 변환"] = time.time() - t0
    else:
        out["rewrite"] = RewriteResult(
            raw_query=query,
            queries={"dense": out["search_text"], "arxiv": query},
            intent="", parse_ok=True)

    # 여기부터 검색 모델 차례임. 언어 모델은 할 일이 끝났으므로 전부 내려 재정렬 모델
    # (1.24GB)이 쓸 자리를 비움. 다시 올리는 데 2.4초 걸리지만, 함께 두면 4.76GB 가 됨
    pool().release("ollama:qwen3", "ollama:rewriter")

    if use_local:
        status.write(f"코퍼스에서 찾는 중...")
        t0 = time.time()
        try:
            index = load_local_index()
            out["local_hits"] = index.search(out["search_text"], k=DEPTH_LOCAL)
            if out["hyde_text"]:
                out["hyde_hits"] = index.search(out["hyde_text"], k=DEPTH_LOCAL)
        except Exception as e:
            out["local_error"] = str(e)
        out["timing"]["로컬 의미 검색"] = time.time() - t0

    # arXiv 결과는 재정렬 목록에 섞지 않고 따로 보여 줌. 섞으면 비율을 어떻게 잡아도
    # 만족도가 떨어짐(전부 p<0.001). 그래도 호출은 남김 - arXiv 채널의 가치는 정확도가
    # 아니라 색인에 없는 최신 논문이고, 그 가치는 평가셋으로 잴 수 없기 때문임
    if use_arxiv:
        status.write(f"arXiv에서 논문을 찾는 중...")
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
        status.write(f"질문 의도를 분석해 관련 논문 순위를 다시 매기는 중...")
        t0 = time.time()
        # 번역문이 아니라 원본 질문으로 재정렬함. 재정렬의 목적이 사용자 의도와 맞는지
        # 보는 것이기 때문임. 뒤에서 검색 순위와 합치려면 후보 전체를 채점받아야 함
        ranked = load_reranker().rerank(query, candidates, top_k=len(candidates))
        out["results"] = fuse_with_search(ranked, candidates)[:TOP_K]
        out["timing"]["재정렬"] = time.time() - t0

    if out["results"]:
        status.write("각 논문을 왜 추천하는지 정리하는 중...")
        t0 = time.time()
        # 재정렬은 끝났음. 자리를 비워야 추천 이유 생성 모델(3.25GB)이 들어감
        pool().release("reranker")
        try:
            out["recommendation"] = load_recommender().recommend(query, out["results"])
        except Exception as e:
            out["recommend_error"] = str(e)
        out["timing"]["추천 이유 생성"] = time.time() - t0

    return out


def confident_results(results, min_score: float | None):
    """관련도 기준선으로 결과를 둘로 가름. 기준선이 None 이면 전부 남김.

    원래 순번을 함께 돌려줌. 추천 에이전트가 재정렬 결과의 번호로 판단을 붙이는데,
    걸러낸 뒤 다시 번호를 매기면 그 판단이 엉뚱한 논문에 붙기 때문임.

    Returns: (남길 [(원래순번, 논문)], 접어 둘 [(원래순번, 논문)])
    """
    numbered = list(enumerate(results or [], start=1))
    if min_score is None:
        return numbered, []
    keep = [(i, p) for i, p in numbered if p.score >= min_score]
    drop = [(i, p) for i, p in numbered if p.score < min_score]
    return keep, drop


# ==========================================================================
# 화면 조각
# ==========================================================================

# 가상 초록은 초록 한 편 길이라 그대로 넣으면 줄이 세로로 길어짐. 앞부분만 보여 줌
HYDE_PREVIEW_LEN = 100


def render_understanding(query: str, state: dict) -> None:
    """무엇을 어떻게 알아들었는지 보여 줌. 검색 성능이 아니라 설명을 위한 화면임.

    실제로 일어난 것만 보여 줌. 칸을 고정으로 그리면 학습 변환기가 의도와 용어를 따로
    만들지 않는 경우에 "내가 쓴 말 / 내가 쓴 말" 처럼 빈 칸만 늘어놓게 됨.
    """
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


# 요약문에 남은 논문 번호를 잡아내는 자리.
#
# 추천 에이전트에게 제목을 쓰라고 지시하지만 4B 모델이 늘 지키지는 않음. 그리고 번호를
# 적는 꼴이 매번 다름 - "index 3", "3번 논문", "1, 3, 5번 논문", "1번, 4번입니다".
#
# 번호를 지우지 않고 **제목으로 바꿈.** 못 잡은 꼴이 남아도 번호만 보일 뿐 틀린 제목을
# 붙이지 않음. 나열 꼴을 맨 앞에 둬야 앞 번호까지 함께 잡음.
_SUMMARY_INDEX = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\s*,\s*\d{1,2})+)\s*번(?!째)"   # 1, 3, 5번
    r"|(?<![\d.])(\d{1,2})\s*번(?!째)"                       # 3번
    r"|index\s*(\d{1,2})"                                    # index 3
)


def retitle_summary(summary: str, results: list) -> str:
    """요약문의 논문 번호를 제목으로 바꿈.

    추천 에이전트가 붙이는 번호는 재정렬 직후 순서인데, 화면은 관련도 순으로 다시
    정렬하고 점수가 낮은 것을 걸러낸 뒤 번호를 새로 매김. 그대로 두면 "3번 논문을
    추천합니다" 가 화면의 3번이 아닌 논문을 가리킴.

    화면 번호로 바꾸지 않고 제목으로 바꾸는 이유는, 걸러내서 화면에 없는 논문을 요약문이
    가리킬 수도 있기 때문임. 제목은 그런 경우에도 뜻이 통함.
    """
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


# 관련도 세 단계의 표시 방법. 색은 Streamlit 이 정해 둔 것을 씀 -
# success 는 초록, warning 은 노랑, error 는 주황임.
RELEVANCE = {
    "high":   ("관련성 높음", st.success),
    "medium": ("관련성 있음", st.warning),
    "low":    ("관련성 낮음", st.error),
}
RELEVANCE_ORDER = ("high", "medium", "low")


def render_paper(rank: int, paper, judgement: dict | None) -> None:
    """결과 한 편. 추천 에이전트의 판단을 이 칸 안에 함께 보여 줌.

    관련도를 글자로만 적으면 목록을 훑을 때 눈에 안 들어옴. 이유 문장을 관련도 색
    블록에 담아 세 단계가 한눈에 갈리게 함. 판단이 없으면 블록 없이 제목만 보여 줌.
    """
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
    """결과를 목록 하나로 보여줌."""
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

    # 추천 에이전트가 전부 '관련성 낮음' 으로 봤으면 목록을 내밀기 전에 그렇게 말함.
    # 판단이 하나도 없는 경우(호출 실패, 빈 목록)를 반드시 갈라내야 함. 그걸 '전부
    # 관련 없음' 으로 읽으면 멀쩡한 결과를 두고 "못 찾았습니다" 라고 말하게 됨
    judged = [j for j in (by_index.get(i, {}).get("relevance")
                          for i in range(1, len(results) + 1)) if j]
    nothing_good = bool(judged) and all(j == "low" for j in judged)

    if not keep or nothing_good:
        st.warning("관련 논문을 찾지 못했습니다. 질문을 바꿔 다시 시도해 보세요.")
        return

    if rec.get("summary"):
        st.info(retitle_summary(rec["summary"], results))

    # 관련도 높음 -> 있음 -> 낮음 순으로 보여 줌. 같은 등급 안에서는 재정렬 순위를
    # 그대로 둠. 판단이 없는 것은 맨 뒤로 보냄 - 추천 에이전트가 빠뜨린 것이라
    # 관련이 없다는 뜻은 아니므로 '낮음' 과 섞지 않음
    def relevance_rank(item) -> int:
        grade = by_index.get(item[0], {}).get("relevance", "")
        return RELEVANCE_ORDER.index(grade) if grade in RELEVANCE_ORDER else len(RELEVANCE_ORDER)

    keep = sorted(keep, key=relevance_rank)

    # 화면에 보이는 번호는 1부터 다시 매기되, 추천 판단은 원래 순번으로 찾음
    for shown, (orig, p) in enumerate(keep, 1):
        render_paper(shown, p, by_index.get(orig))

def render_recent(state: dict, use_arxiv: bool) -> None:
    """arXiv 실시간 검색 결과를 '최신 논문' 으로 따로 보여 줌.

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
    # 추천 목록과 같은 모양으로 보여 줌. 논문마다 초록을 펼칠 수 있어야 하는데
    # Streamlit 은 expander 안에 expander 를 못 넣으므로 바깥을 절 제목으로 둠
    st.divider()
    st.markdown(f"#### arXiv에서 직접 찾은 논문 {len(fresh)}편")
    st.caption("코퍼스에 없는 논문입니다. 관련 정도가 아닌 arXiv가 전달해준 순서이고, 추천 논문과 달리 "
               "관련도를 매기지 않았습니다.")
    for i, p in enumerate(fresh, 1):
        render_paper(i, p, None)



# ==========================================================================
# 화면
# ==========================================================================

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
        # session 을 빠져나올 때 올려 둔 모델을 전부 내림. 오류가 나도 반드시 내려서,
        # 실패한 검색이 그래픽 메모리를 붙잡은 채 남지 않게 함
        with pool().session():
            state = run_search(query, use_local, use_arxiv, status)
        status.update(label="검색을 마쳤습니다.", state="complete", expanded=False)

    st.divider()

    # 특정 논문을 지목하는 질문이었으면 그 논문을 맨 위에 짚어 줌
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

# -- arXiv 이용 약관에 따른 표기 (공개 전 필수) -----------------------------
# arXiv 이용 약관은 arXiv 가 지원하거나 보증하는 것처럼 표현하는 것을 금지함.
# 제목, 초록, 논문 번호는 CC0 로 배포되어 저장과 재사용이 되지만, 논문 원문은 우리가
# 제공하지 않고 arXiv 초록 페이지로 보냄.
st.divider()
st.caption(
    "본 서비스는 논문 검색 시 참고용으로만 사용해주시길 바랍니다."
)
st.caption(
    "arXiv 메타데이터는 arXiv에서 가져왔습니다. "
    "arXiv 메타데이터는 CC0 1.0으로 배포됩니다. "
    "논문 원문은 arXiv 에서 직접 확인해 주세요."
)
st.caption(
    "이 서비스는 arXiv와 무관한 프로젝트이며, arXiv의 후원이나 보증을 받지 "
    "않았습니다."
)
st.caption(
    "Thank you to arXiv for use of its open access interoperability."
    "This service was not reviewed or approved by, nor does it necessarily express or reflect the policies or opinions of, arXiv."
)

