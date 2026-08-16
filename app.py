"""Papers-Please 웹 데모 (Streamlit) - 두 채널 검색 + 교차 인코더 재정렬.

사용자가 일상어나 한국어로 검색하면, 무엇을 어떻게 알아들었는지 보여주고, 두 갈래로
논문을 찾아 사용자 의도에 맞게 다시 줄 세워 보여줌.

## 검색어를 두 개 만들어 한 색인을 두 번 찾음 (2026-08-16 구조 변경)

    질문
     +- 검색어 1  원본 (한국어면 영어로 옮김)      사용자가 실제로 쓴 말
     \- 검색어 2  질문에 답할 법한 가상 초록        분야 용어로 건너가는 다리
            |
       각각 로컬 의미 검색 71만 편 -> 순위 합치기(RRF)
            |
       교차 인코더 재정렬 (원본 질문과 대조)
            |
       추천 에이전트가 관련도와 이유를 붙임
            |
       사용자에게 보여줄 10편

     * arXiv 실시간 검색은 이 목록에 섞지 않고 '최신 논문' 칸으로 따로 보여줌

왜 이렇게 바꿨는가 (개발용 348문항 실측):

    번역만                  Recall@10 0.566   nDCG@10 0.514
    가상 초록으로 대체        Recall@10 0.555   (hard 층만 좋아지고 나머지가 나빠짐)
    둘을 합침                Recall@10 0.586   nDCG@10 0.523 (+0.009, p=0.040)

옛 구조는 arXiv 채널과 로컬 채널을 합치는 것이었는데, 합치면 비율을 어떻게 잡아도
만족도가 떨어졌음 (1대1 -0.041, 2대1 -0.040, 3대1 -0.034, 전부 p<0.001). 게다가 옛
코드는 로컬 후보가 300칸을 다 채워 arXiv 후보를 한 편도 넣지 못하고 있었음 (ISSUE 39).
arXiv 채널의 가치는 정확도가 아니라 색인에 없는 최신 논문이고, 그 가치는 평가셋으로
잴 수 없으므로 (정답 논문이 전부 색인 안에 있음) 목록을 갈라 두는 것이 정직함.

## 화면 구성 원칙 (ISSUE 11)

옛 화면은 변환 과정, 검색된 논문 10편, 최종 추천 논문을 따로따로 세 덩어리로 쏟아냈음.
같은 논문이 두 목록에 겹쳐 나와서 어느 것을 봐야 하는지 알 수 없었음. 지금은 이렇게 함.

    1. 검색 전에는 입력창 하나만 보임
    2. 검색하면 "무엇을 어떻게 알아들었는지" 를 가로 3단 카드로 먼저 보여줌
    3. 결과는 목록 하나임. 추천 에이전트의 판단을 그 목록 카드 안에 녹임
    4. 진행 중에는 지금 어느 단계인지 보여줌 (한 번 검색에 10초 넘게 걸리므로)

## 한국어 질문을 먼저 영어로 옮기는 이유 (2026-08-14 실측)

arXiv 논문은 제목도 초록도 영어임. 로컬 색인이 다국어 임베딩(bge-m3)을 쓰는데도 한국어
질문이 영어보다 크게 뒤졌음. 번역해서 넣으니 시험용 342문항에서 한국어 Recall@10 이
0.456 -> 0.567 (+0.111, p=0.001), nDCG 가 0.400 -> 0.455 (+0.055, p<0.001) 로 올랐음.
영어 질문은 손대지 않으므로 대조군으로 +0.000 이 나왔음. 언어 격차가 58% 줄었음.

학습한 변환기(dpo)가 내놓는 arXiv 문법 문자열(all:"..." OR abs:"...")은 여기 쓰지 않음.
문장을 통째로 임베딩하는 의미 검색에 검색 문법을 넣으면 불리하기 때문임. 그 문자열은
arXiv 채널 전용임.

재정렬에는 번역문이 아니라 원본 질문을 씀. 번역문으로도 재봤는데 이득이 없었고
(Recall +0.006 p=0.678, nDCG -0.007 p=0.226), 재정렬의 목적이 '사용자 의도와 맞는가' 를
보는 것이라 사용자가 실제로 쓴 말을 쓰는 것이 원칙에 맞음.

실행:
  streamlit run app.py
(전제: Ollama 서버 + qwen3:4b, 로컬 색인, 인터넷 연결)
"""

from __future__ import annotations

import re
import time

import streamlit as st

from src import config
from src.recommend_agent.recommender import PaperRecommender
from src.retrieval.arxiv_live import ArxivLiveRetriever
from src.rewriter.base import build_rewriter
from src.rewriter.paper_resolver import PaperResolver, resolve_and_verify
from src.schemas import RewriteResult

EXAMPLES = [
    "사진 보고 글로 설명해주는 AI",
    "가짜 뉴스 걸러내는 방법",
    "AI가 사람처럼 대화하게 만들기",
    "Attention is all you need",
]

# -- 검색 예산 --------------------------------------------------------------
TOP_K = 10              # 사용자에게 보여줄 최대 논문 수
DEPTH_LOCAL = 100       # 로컬 의미 검색 한 번에 받아올 후보 수 (검색어마다)
DEPTH_ARXIV = 100       # arXiv 에서 받아올 후보 수 (page_size 가 100 이라 호출 1회로 끝남)
RERANK_DEPTH = 100      # 재정렬에 넣을 최대 후보 수

# 깊이를 왜 300 에서 100 으로 줄였는가 (2026-08-16 실측, 개발용 348문항)
#
#   후보 깊이   Recall@10   nDCG@10   상위 10편에 쓸모있는 논문이 한 편 이상
#     100        0.566      0.514              93.4% (평균 3.14편)
#     150        0.566      0.496                -
#     200        0.569      0.487                -
#     300        0.583      0.478              89.9% (평균 2.75편)
#
# 후보를 깊게 가져오면 후보 상한은 오르지만(0.635 -> 0.721) 최종 성능은 따라오지 않고
# (+0.017, p=0.214 판정 불가) 만족도는 확실히 떨어짐(-0.036, p<0.001). 후보가 늘면
# 재정렬기가 엉뚱한 논문을 상위로 올리는 일도 함께 늘기 때문임.
# 사용자 입장에서 중요한 "쓸모있는 논문을 한 편이라도 보여줬는가" 도 100 쪽이 높음.
#
# 깊이를 다시 만지려면 감이 아니라 이 표를 다시 만들 것:
#   python -m evaluation.pipeline_eval --report-only <실행결과> --rerank cross --rerank-depth N

# -- "못 찾았다"고 말할 기준선 ----------------------------------------------
#
# 로컬 의미 검색은 어떤 질문에도 항상 후보를 채워서 돌려줌. 아무리 무관한 질문이어도
# 유사도가 가장 높은 논문들이 나옴. 그대로 뿌리면 무관한 논문을 추천으로 포장하는
# 서비스가 됨. 옛 arXiv 키워드 검색은 못 찾으면 0건을 돌려줘서 이 문제가 없었는데,
# 구조를 바꾸면서 새로 생긴 위험임.
#
# 이 값은 감으로 정하면 안 됨. 등급 정답지로 실측해서 정함:
#   python -m evaluation.pipeline_eval --calibrate-threshold \
#       --queries data/eval/dev.jsonl --grades data/eval/grades_dev.jsonl \
#       --thresholds 0.001 0.002 0.005 0.01 0.02 0.05 0.10
#
# 2026-08-14 실측 (개발용 348문항, 등급 판정 7,112쌍, bge-reranker-v2-m3):
#
#   기준선   무관 걸러냄   만족 잘못 버림   (영어)   (한국어)
#   0.001      31.7%          4.3%        3.1%     5.5%
#   0.002      43.8%          7.7%        5.9%     9.5%   <- 이것을 고름
#   0.005      61.9%         14.2%       11.4%    17.1%
#   0.010      73.3%         22.5%       18.8%    26.2%
#
# 왜 0.002 인가 (일부러 느슨하게 잡음):
#   이 기준선의 목적은 결과를 깎아내는 것이 아니라 "정말 아무것도 못 찾았을 때 그렇게
#   말할 수 있게" 하는 것임. 잘못 낸 "못 찾았습니다" 는 무관한 논문 한 편을 보여주는 것보다
#   훨씬 나쁜 경험임. 그래서 무관을 절반쯤만 걸러내고 좋은 논문은 거의 안 버리는 쪽으로 잡음.
#   기준선 아래 논문도 지우지 않고 '관련성이 낮아 접어 둔' 자리로 내려보내기만 함.
#
# 한국어가 더 손해를 본다는 점을 알고 쓸 것:
#   같은 만족도의 논문이라도 한국어 질문의 점수가 영어보다 낮게 나옴 (만족 등급 중앙값
#   영어 0.17 대 한국어 0.07). 질문과 초록의 언어가 달라서임. 그래서 어느 기준선을 잡아도
#   한국어 쪽이 1.5배쯤 더 걸림. 느슨하게 잡은 이유의 절반이 이것임.
#
# None 이면 걸러내지 않음. 재정렬 모델을 바꾸면 점수 눈금이 달라지므로 반드시 다시 재야 함.
MIN_RERANK_SCORE: float | None = 0.002

st.set_page_config(page_title="Papers, Please", layout="wide")


# ==========================================================================
# 무거운 부품 (한 번만 올림)
# ==========================================================================

@st.cache_resource(show_spinner="arXiv 검색기 준비 중...")
def load_arxiv() -> ArxivLiveRetriever:
    return ArxivLiveRetriever()


@st.cache_resource(show_spinner="쿼리 변환기(Qwen3-4B) 연결 중...")
def load_rewriter():
    """검색 성공을 보상으로 학습한 변환기를 씀.

    계층 변환기(hierarchical)가 아니라 dpo 를 쓰는 이유는 실측 차이가 크기 때문임.
    옛 시험용 300문항 Recall@10 기준 변환 안 함 0.167, 계층 0.220, 학습 후 0.343 임.
    """
    return build_rewriter("dpo")


@st.cache_resource(show_spinner="논문 71만 편 색인 불러오는 중... (첫 실행은 1분 정도 걸립니다)")
def load_local_index():
    """로컬 의미 검색기. 반드시 cache_resource 로 감쌈.

    Streamlit 은 사용자가 무언가 누를 때마다 스크립트를 처음부터 다시 실행함.
    캐시하지 않으면 검색할 때마다 임베딩 2.93GB 를 새로 올려 메모리가 바로 터짐.
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
def load_recommender(borrow_dpo: bool = False) -> PaperRecommender:
    """추천 이유 생성기. 어느 모델을 쓸지는 지금 dpo 변환기가 올라와 있는지에 달림.

    - arXiv 를 켜면 dpo 변환기(transformers, 8.64GB)가 이미 올라와 있으므로 그것을
      빌려 씀. 안 빌리면 같은 모델이 두 벌이 되어 GPU 16GB 를 넘기고, 그러면 추천
      모델이 오류 없이 조용히 CPU 로 밀려나 한 번에 229.6초가 걸림(실측).
    - arXiv 를 끄면 dpo 를 아예 올리지 않음. 이때는 Ollama 의 qwen3:4b 를 씀.
      번역기와 가상 초록 생성기가 이미 쓰고 있는 모델이라 VRAM 이 더 들지 않음.

    빌릴 때 `borrow_dpo=True` 를 인자로 받는 이유는, 인자가 다르면 캐시가 갈라져서
    두 경우가 서로 덮어쓰지 않게 하기 위함임.
    """
    if borrow_dpo:
        rewriter = load_rewriter()
        if hasattr(rewriter, "generate_json"):
            return PaperRecommender(client=rewriter)
    return PaperRecommender()


@st.cache_resource(show_spinner="재정렬 모델 준비 중...")
def load_reranker():
    """교차 인코더 재정렬기 (질문과 논문을 함께 읽고 관련도를 매김)."""
    from src.retrieval.ranking import CrossEncoderReranker
    return CrossEncoderReranker()


@st.cache_resource
def load_translator():
    """한국어 질문을 영어로 옮기는 변환기.

    ## 왜 번역하는가 (2026-08-14 실측)

    arXiv 논문은 제목도 초록도 영어임. 로컬 색인이 다국어 임베딩(bge-m3)을 쓰는데도
    한국어 질문이 영어보다 크게 뒤졌음. 번역해서 넣으니 시험용 342문항에서:

        한국어 Recall@10   0.456 -> 0.567  (+0.111, p=0.001)
        한국어 nDCG@10     0.400 -> 0.455  (+0.055, p<0.001)
        영어 (대조군)       0.649 -> 0.649  (+0.000, 손대지 않았으므로)

    언어 격차가 0.193 에서 0.082 로 58% 줄었음. 학습은 한 번도 하지 않았고 프롬프트 한 줄임.
    같은 날 잰 학습한 변환기(dpo)는 Recall +0.018(판정 불가)에 nDCG -0.016(나쁨)이었음.

    ## 왜 Ollama 인가

    앱이 이미 `PaperResolver` 때문에 Ollama 의 qwen3:4b 를 올려 두고 있어서 VRAM 이 더
    들지 않음. 그리고 평가도 같은 경로로 쟀으므로 평가와 서비스가 어긋나지 않음.
    """
    from src.rewriter.baselines import TranslateRewriter
    return TranslateRewriter()


@st.cache_resource
def load_hyde():
    """질문에 답할 법한 가상의 영어 초록을 지어내는 변환기 (두 번째 검색어를 만듦).

    ## 왜 필요한가 (2026-08-16 실측)

    일상어로 묻는 hard 층은 번역만으로는 안 됨. 번역은 일상어를 일상어인 채로 옮기기
    때문임. 실제 실패 사례:

        질문      조건이 많은 문제를 아주 적은 메모리로 대충 잘 푸는 방법이 있나
        번역      Is there a way to solve a problem with many conditions with very little memory?
        정답 논문  Approximability of all Boolean CSPs with linear sketches  -> 457,938등

    필요한 것은 번역이 아니라 "조건이 많은 문제 = 제약 충족 문제", "적은 메모리로 대충 =
    스케치 근사" 같은 분야 지식임. 가상 초록은 그 지식을 끌어내는 장치임.

    ## 반드시 바꿔치기가 아니라 함께 쓸 것 (실측)

    개발용 348문항에서 가상 초록으로 원본 검색어를 대체하면 hard 층만 좋아지고
    (0.181 -> 0.198) 나머지가 다 나빠짐(전체 0.583 -> 0.555). 4B 모델은 모르는 분야에서
    그럴듯하지만 엉뚱한 초록을 지어내기 때문임. 두 검색어로 각각 검색해 순위를 합치면
    양쪽을 다 얻음:

        번역만          Recall@10 0.566   nDCG 0.514
        가상 초록만      Recall@10 0.555   nDCG   -
        둘을 합침        Recall@10 0.586   nDCG 0.523 (+0.009, p=0.040)

    모델은 Ollama 의 qwen3:4b 로, 앱이 이미 올려 둔 것을 쓰므로 VRAM 이 더 들지 않음.
    """
    from src.rewriter.baselines import HydeRewriter
    return HydeRewriter()


# ==========================================================================
# 검색 파이프라인 (화면 그리기와 분리해 둠)
# ==========================================================================

def arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


_ABS_TERM = re.compile(r'abs:"([^"]+)"')


def academic_terms_of(rw, raw_query: str) -> list[str]:
    """변환기가 만든 영어 학술 용어를 뽑음.

    계층 변환기는 용어를 `academic_terms` 에 따로 담지만, 서비스가 쓰는 학습 모델(dpo)은
    arXiv 문법 문자열 하나만 내놓아서 그 필드가 빔. 그런데 용어 자체는 문자열 안에
    `abs:"..."` 조각으로 들어 있음:

        all:"사진 보고 글로 설명해주는 AI" OR abs:"image description" OR abs:"automatic captioning"

    이걸 뽑아내지 않으면 화면이 "변환한 것이 없다"고 잘못 말하게 됨. 실제로는 있는데
    구조화된 자리에 없을 뿐임.

    `all:"..."` 조각은 원본 질문이라 제외함(용어가 아님).
    """
    if rw.academic_terms:
        return list(rw.academic_terms)
    found = _ABS_TERM.findall(rw.query_for("arxiv") or "")
    return [t for t in dict.fromkeys(found) if t.strip() and t.strip() != raw_query.strip()]


def fuse_local(literal_hits, hyde_hits) -> list:
    """두 검색어로 얻은 로컬 결과를 순위 합치기(RRF)로 합침.

    점수를 그대로 더하지 않고 등수를 더함. 두 결과가 같은 임베딩 모델에서 나오긴 하지만,
    가상 초록은 문장이 길어 유사도 값이 전반적으로 다르게 나옴. 점수를 더하면 어느 쪽이
    옳은가가 아니라 점수 눈금이 큰 쪽이 이김 (ISSUE 29 자리).

    논문 번호 표기 통일은 `rrf_fuse` 안에서 함. 통일하지 않으면 2103.00020 과
    2103.00020v2 가 다른 논문이 되어, 두 검색어가 함께 찾아낸 논문일수록 손해를 봄
    (ISSUE 27).
    """
    from src.retrieval.ranking import rrf_fuse

    channels = {"literal": literal_hits or []}
    if hyde_hits:
        channels["hyde"] = hyde_hits
    return rrf_fuse(channels, k=config.RRF_K, top_n=RERANK_DEPTH)


def run_search(query: str, use_local: bool, use_arxiv: bool, status) -> dict:
    """검색 한 번을 끝까지 수행하고 결과를 모아 돌려줌.

    화면을 그리지 않고 자료만 만듦. 진행 상황은 `status.write` 로만 알림.
    이렇게 나눠 두면 화면 구성을 바꿀 때 파이프라인을 건드리지 않아도 됨.
    """
    out: dict = {"timing": {}, "arxiv_error": None, "resolved": None,
                 "local_hits": None, "hyde_hits": None, "arxiv_hits": None,
                 "results": None, "recommendation": None}

    # 특정 유명 논문을 설명으로 찾는 질문이면, 그 논문을 짚어서 먼저 보여줌
    # (언어 모델 지식으로 제목 추정 -> arXiv 에서 실제 존재를 검증한 경우에만)
    if use_arxiv:
        status.write("어떤 논문을 찾는 질문인지 확인하는 중...")
        t0 = time.time()
        try:
            out["resolved"], _ = resolve_and_verify(query, load_resolver(), load_arxiv())
        except Exception:
            out["resolved"] = None      # arXiv 오류 등은 배너 생략(치명적 아님)
        out["timing"]["논문 지목 확인"] = time.time() - t0

    # 한국어면 영어로 옮겨 로컬 의미 검색에 넣음. arXiv 논문이 영어라서임.
    # 영어 질문은 건드리지 않음 (번역기를 통과시키면 뜻이 미묘하게 바뀌어 손해만 봄).
    from src.rewriter.baselines import TranslateRewriter
    out["search_text"] = query
    if use_local and TranslateRewriter.has_hangul(query):
        status.write("한국어 질문을 영어로 옮기는 중...")
        t0 = time.time()
        tr = load_translator().rewrite(query)
        if tr.parse_ok:
            out["search_text"] = tr.query_for("dense")
        else:
            out["translate_error"] = tr.intent
        out["timing"]["한국어를 영어로"] = time.time() - t0

    # 두 번째 검색어: 질문에 답할 법한 가상의 영어 초록. 일상어 질문을 분야 용어로
    # 건너게 하는 장치임 (load_hyde 설명글 참고). 원본 검색어를 대체하지 않고 함께 씀.
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

    # 학습한 변환기(dpo)는 arXiv 문법 문자열만 만듦. 그래서 arXiv 채널을 쓸 때만 부름.
    # 켜 두면 매 검색마다 GPU 8.64GB 와 3.5초를 쓰는데, 로컬 검색에는 쓸 곳이 없음
    # (ISSUE 39). arXiv 를 끄면 그 비용이 통째로 사라짐.
    if use_arxiv:
        status.write("arXiv 검색어를 학술 용어로 바꾸는 중...")
        t0 = time.time()
        out["rewrite"] = load_rewriter().rewrite(query)
        out["timing"]["쿼리 변환"] = time.time() - t0
    else:
        out["rewrite"] = RewriteResult(
            raw_query=query,
            queries={"dense": out["search_text"], "arxiv": query},
            intent="", parse_ok=True)

    if use_local:
        status.write(f"논문 71만 편에서 뜻으로 찾는 중... (검색어 2개 x 후보 {DEPTH_LOCAL}편)")
        t0 = time.time()
        try:
            index = load_local_index()
            out["local_hits"] = index.search(out["search_text"], k=DEPTH_LOCAL)
            if out["hyde_text"]:
                out["hyde_hits"] = index.search(out["hyde_text"], k=DEPTH_LOCAL)
        except Exception as e:
            out["local_error"] = str(e)
        out["timing"]["로컬 의미 검색"] = time.time() - t0

    # arXiv 결과는 재정렬 목록에 섞지 않고 따로 보여줌.
    #
    # 섞어 보고 판단한 결과임 (2026-08-16, 개발용 348문항). 로컬과 arXiv 를 순위 합치기로
    # 섞으면 비율을 어떻게 잡아도 만족도가 떨어짐: 1대1 -0.041, 2대1 -0.040, 3대1 -0.034
    # (전부 p<0.001). Recall 이득은 전부 판정 불가였음. 로컬 비중을 올릴수록 손해가 주는
    # 방향이라, 그 추세의 종점이 '섞지 않음' 임.
    # 그래도 호출은 남김. arXiv 채널의 가치는 정확도가 아니라 색인에 없는 최신 논문이고,
    # 그 가치는 평가셋으로 잴 수 없기 때문임 (평가셋 정답 논문이 전부 색인 안에 있음).
    if use_arxiv:
        status.write(f"arXiv 에서 최신 논문을 찾는 중... (후보 {DEPTH_ARXIV}편)")
        t0 = time.time()
        try:
            out["arxiv_hits"] = load_arxiv().search(
                out["rewrite"].query_for("arxiv"), k=DEPTH_ARXIV)
        except Exception as e:
            out["arxiv_error"] = (f"arXiv 검색 중 오류가 났습니다. 일시적일 수 있으니 잠시 후 "
                                  f"다시 시도해 주세요. [{type(e).__name__}]")
        out["timing"]["arXiv 검색"] = time.time() - t0

    candidates = fuse_local(out["local_hits"], out["hyde_hits"])
    out["n_candidates"] = len(candidates)

    if candidates:
        status.write(f"질문 의도와 대조해 순위를 다시 매기는 중... (후보 {len(candidates)}편)")
        t0 = time.time()
        # 번역문이 아니라 원본 질문으로 재정렬함.
        # 재정렬의 목적이 '사용자 의도와 맞는가' 를 보는 것이라 사용자가 실제로 쓴 말을 씀.
        # 번역문으로도 재봤는데 이득이 없었음 - 시험용 342문항에서 Recall +0.006(p=0.678),
        # nDCG -0.007(p=0.226) 으로 둘 다 잡음과 구분되지 않았음 (2026-08-14).
        out["results"] = load_reranker().rerank(query, candidates, top_k=TOP_K)
        out["timing"]["재정렬"] = time.time() - t0

    if out["results"]:
        status.write("각 논문이 왜 맞는지 정리하는 중...")
        t0 = time.time()
        try:
            out["recommendation"] = load_recommender(use_arxiv).recommend(query, out["results"])
        except Exception as e:
            out["recommend_error"] = str(e)
        out["timing"]["추천 이유 생성"] = time.time() - t0

    return out


def confident_results(results, min_score: float | None):
    """관련도 기준선으로 결과를 둘로 가름. 기준선이 없으면(측정 전) 전부 남김.

    원래 순번을 함께 돌려줌. 추천 에이전트는 재정렬 결과의 1-based 번호로 판단을
    붙이는데, 걸러낸 뒤 다시 번호를 매기면 그 판단이 엉뚱한 논문에 붙음. 오류가 나지
    않고 이유만 뒤바뀌는 종류의 고장이라 눈으로 알아채기 어려움.

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

def render_understanding(query: str, state: dict) -> None:
    """무엇을 어떻게 알아들었는지 보여줌 (ISSUE 11).

    이 화면이 검색 성능을 올리지는 않음. 사용자가 "내 말이 이렇게 바뀌어 검색됐구나" 를
    이해하게 하는 설명 기능이고, 그 점을 감추지 않음 (PLAN 0절).

    실제로 일어난 것만 보여줄 것 (실측으로 확인한 함정):
    학습 모델(dpo)은 arXiv 문법 문자열 하나만 내놓음. 의도, 개념, 학술 용어를 따로 만들지
    않아서 intent 자리에 원본 질문이 그대로 들어 있음. 그런데도 칸을 고정으로 그리면
    "내가 쓴 말 / 내가 쓴 말 / (원본을 그대로 썼습니다)" 가 되어 빈 칸만 늘어놓게 됨.
    """
    rw = state["rewrite"]
    translated = state.get("search_text")
    has_translation = bool(translated) and translated.strip() != query.strip()

    intent = (rw.intent or "").strip()
    has_intent = bool(intent) and intent != query.strip()
    terms = academic_terms_of(rw, query)

    if not (has_translation or has_intent or terms):
        st.caption("입력하신 말을 그대로 뜻으로 검색했습니다.")
        return

    st.markdown("#### 이렇게 알아들었습니다")
    n = 1 + int(has_translation) + int(has_intent) + int(bool(terms))
    cols = st.columns(n)
    i = 0
    with cols[i]:
        st.caption("내가 쓴 말")
        st.info(query)
    if has_translation:
        i += 1
        with cols[i]:
            st.caption("영어로 이렇게 옮겨 찾았습니다")
            st.success(translated)
            st.caption("arXiv 논문이 영어라, 한국어로 물으시면 먼저 영어로 옮깁니다.")
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
            st.caption("arXiv 에는 이 용어로 물었습니다")
            st.info(", ".join(terms))
    if state.get("translate_error"):
        st.caption(f"번역에 실패해 원본으로 검색했습니다. {state['translate_error']}")
    if not rw.parse_ok:
        st.caption("변환에 실패해 원본 검색어로 검색했습니다.")


def render_paper(rank: int, paper, judgement: dict | None) -> None:
    """결과 한 편. 추천 에이전트의 판단을 이 카드 안에 녹임.

    옛 화면은 '검색된 논문'과 '최종 추천 논문'을 따로 두어 같은 논문이 두 번 나왔음.
    사용자는 어느 목록을 봐야 하는지 알 수 없었음 (ISSUE 11).
    """
    label = {"high": "강력 추천", "medium": "관련 있음", "low": "관련성 낮음"}
    tag = label.get((judgement or {}).get("relevance", ""), "")
    head = f"#### {rank}. [{paper.title}]({arxiv_url(paper.paper_id)})"
    st.markdown(f"{head}  `{tag}`" if tag else head)
    if judgement and judgement.get("reason"):
        st.caption(judgement["reason"])
    with st.expander("초록 보기"):
        st.write(paper.abstract or "(초록 없음)")


def render_results(state: dict) -> None:
    """결과를 목록 하나로 보여줌."""
    results = state.get("results")
    if not results:
        if state.get("arxiv_error"):
            st.warning(state["arxiv_error"])
        else:
            st.info("검색 결과가 없습니다. 검색어를 바꿔 다시 시도해 보세요.")
        return

    rec = state.get("recommendation") or {}
    by_index = {r["index"]: r for r in rec.get("recommendations", [])}
    keep, dropped = confident_results(results, MIN_RERANK_SCORE)

    # 추천 에이전트가 전부 '관련성 낮음'으로 봤다면, 목록을 내밀기 전에 그렇게 말함.
    # 로컬 의미 검색은 어떤 질문에도 후보를 채워 돌려주므로 이 정직함이 없으면
    # 무관한 논문을 추천으로 포장하게 됨.
    #
    # 판단이 하나도 없을 때(에이전트 호출 실패, 빈 목록)를 반드시 갈라내야 함. 그걸
    # '전부 관련 없음' 으로 읽으면 멀쩡한 결과를 두고 "못 찾았습니다" 라고 말하게 됨.
    judged = [j for j in (by_index.get(i, {}).get("relevance")
                          for i in range(1, len(results) + 1)) if j]
    nothing_good = bool(judged) and all(j == "low" for j in judged)

    if not keep or nothing_good:
        st.warning("딱 맞는 논문을 찾지 못했습니다. 검색어를 조금 다르게 써 보시면 "
                   "결과가 달라질 수 있습니다.")
        with st.expander(f"그래도 가장 가까운 {len(results)}편 보기"):
            for i, p in enumerate(results, 1):
                render_paper(i, p, by_index.get(i))
        return

    if rec.get("summary"):
        st.info(rec["summary"])

    # 화면에 보이는 번호는 1부터 다시 매기되, 추천 판단은 원래 순번으로 찾음
    for shown, (orig, p) in enumerate(keep, 1):
        render_paper(shown, p, by_index.get(orig))

    if dropped:
        with st.expander(f"관련성이 낮아 접어 둔 {len(dropped)}편 보기"):
            for shown, (orig, p) in enumerate(dropped, len(keep) + 1):
                render_paper(shown, p, by_index.get(orig))


def render_recent(state: dict, use_arxiv: bool) -> None:
    """arXiv 실시간 검색 결과를 '최신 논문' 으로 따로 보여줌.

    본 목록에 섞지 않는 이유는 run_search 설명 참고 (섞으면 만족도가 떨어짐).
    본 목록에 이미 있는 논문은 빼서, 같은 논문이 두 번 나오지 않게 함 (ISSUE 11).
    """
    hits = state.get("arxiv_hits") or []
    if not (use_arxiv and hits):
        return
    from src.retrieval.corpus import normalize_paper_id

    shown = {normalize_paper_id(p.paper_id) for p in (state.get("results") or [])}
    fresh = [p for p in hits if normalize_paper_id(p.paper_id) not in shown][:5]
    if not fresh:
        return
    with st.expander(f"arXiv 에서 방금 찾은 논문 {len(fresh)}편 더 보기 (색인에 없는 최신 논문 포함)"):
        st.caption("이 목록은 관련도 순서가 아니라 arXiv 가 돌려준 순서입니다.")
        for p in fresh:
            st.markdown(f"- [{p.title}]({arxiv_url(p.paper_id)})")


def render_sources(state: dict, use_local: bool, use_arxiv: bool) -> None:
    """어디서 몇 편을 가져왔는지. 접어 두고, 궁금한 사람만 펼쳐 봄."""
    with st.expander("어떻게 찾았는지 보기"):
        src = []
        if use_local:
            src.append(f"뜻으로 찾기 {len(state.get('local_hits') or [])}편")
            if state.get("hyde_hits"):
                src.append(f"두 번째 검색어 {len(state['hyde_hits'])}편")
        st.caption(f"후보: {' + '.join(src)} -> 순위 합치기 {state.get('n_candidates', 0)}편 "
                   f"-> 재정렬해 상위 {TOP_K}편")
        if use_arxiv:
            st.caption(f"arXiv 는 최신 논문 칸으로 따로 {len(state.get('arxiv_hits') or [])}편")
        if use_local and state.get("search_text") != state["rewrite"].raw_query:
            st.caption(f"로컬 의미 검색어(영어로 옮김): `{state['search_text']}`")
        if state.get("hyde_text"):
            st.caption(f"두 번째 검색어(가상 초록): `{state['hyde_text'][:200]}`")
        if use_arxiv:
            st.caption(f"arXiv 검색어: `{state['rewrite'].query_for('arxiv')}`")
        if state.get("local_error"):
            st.caption(f"로컬 색인을 쓸 수 없었습니다: {state['local_error']}")
        if state.get("arxiv_error"):
            st.caption(state["arxiv_error"])
        if state.get("recommend_error"):
            st.caption(f"추천 이유를 만들지 못했습니다: {state['recommend_error']}")

        timing = state.get("timing") or {}
        if timing:
            st.caption("단계별 소요 시간 (합계 {:.1f}초)".format(sum(timing.values())))
            st.caption(" / ".join(f"{k} {v:.1f}초" for k, v in timing.items()))


# ==========================================================================
# 화면
# ==========================================================================

st.title("Papers, Please")
st.caption("한국어나 일상어로 물어보셔도 arXiv 에서 논문을 찾아 드립니다.")

with st.sidebar:
    st.header("설정")
    st.caption("기본값 그대로 두셔도 됩니다.")
    use_local = st.checkbox("로컬 의미 검색 사용 (권장)", value=True,
                            help="논문 71만 편을 뜻으로 검색합니다. 첫 실행에 1분 정도 걸립니다.")
    use_arxiv = st.checkbox("arXiv 최신 논문도 찾기", value=True,
                            help="색인에 없는 최신 논문을 따로 찾아 아래에 따로 보여줍니다. "
                                 "추천 목록의 순위에는 영향을 주지 않습니다. "
                                 "끄면 약 5초 빨라집니다.")
    st.divider()
    st.caption("arXiv 요청 제한을 지키기 위해 한 검색당 호출을 최소화합니다.")

if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input(
    "무엇을 찾으시나요?", value=st.session_state.query,
    placeholder="예: 사진 보고 글로 설명해주는 AI",
    label_visibility="collapsed")

st.caption("이런 것도 찾을 수 있습니다:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.query = ex
        st.rerun()

go = st.button("검색", type="primary", use_container_width=True)

if go and query.strip():
    if not (use_local or use_arxiv):
        st.error("검색 통로를 하나 이상 켜 주세요.")
        st.stop()

    with st.status("논문을 찾는 중입니다...", expanded=True) as status:
        state = run_search(query, use_local, use_arxiv, status)
        status.update(label="검색을 마쳤습니다.", state="complete", expanded=False)

    st.divider()

    # 특정 논문을 지목하는 질문이었으면 그 논문을 맨 위에 짚어 줌
    if state.get("resolved"):
        p = state["resolved"]
        st.success("이 논문을 찾으시는 것 같습니다")
        st.markdown(f"### [{p.title}]({arxiv_url(p.paper_id)})")
        with st.expander("초록 보기"):
            st.write(p.abstract)
        st.divider()

    render_understanding(query, state)
    st.divider()

    st.markdown("#### 찾은 논문")
    render_results(state)
    render_recent(state, use_arxiv)
    render_sources(state, use_local, use_arxiv)

# -- arXiv 이용 약관에 따른 표기 (공개 전 필수) -----------------------------
# arXiv 이용 약관은 arXiv 가 지원하거나 보증하는 것처럼 표현하는 것을 명시적으로 금지함.
# 메타데이터(제목, 초록, 논문 번호)는 CC0 로 배포되어 저장과 재사용이 허용되지만,
# 논문 원문(PDF, 소스)은 우리 서버에서 제공하지 않고 arXiv 초록 페이지로 보냄.
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
