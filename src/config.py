"""프로젝트 전역 설정값.

모델 이름, 파일 경로, 검색 개수 같은 상수를 여기서만 정의하고 다른 모듈은 가져다 씀.
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path


def _fix_libstdcxx() -> None:
    """torch 보다 먼저 새 C++ 표준 라이브러리를 올림.

    torch 가 먼저 잡는 시스템 libstdc++ 에는 CXXABI_1.3.15 가 없어서, 뒤이어 올라오는
    pyarrow 가 ImportError 로 깨짐. config 는 거의 모든 모듈이 맨 위에서 가져오므로
    여기서 미리 올려 둠. 라이브러리를 못 찾으면 조용히 넘어감.
    """
    if os.environ.get("PAPERS_SKIP_LIBSTDCXX_FIX"):
        return
    for pattern in ("/home/jmoongyu/anaconda3/lib/libstdc++.so.6*",
                    os.path.join(os.path.dirname(os.__file__), "..", "..", "libstdc++.so.6*")):
        found = sorted(glob.glob(pattern))
        if found:
            try:
                ctypes.CDLL(found[-1], mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            return


_fix_libstdcxx()

# -- 경로 ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"          # 논문 코퍼스 (대용량, git 제외)

# -- 검색 상수 --------------------------------------------------------------
TOP_K = 10                    # 검색 결과 기본 개수
RRF_K = 60                    # 순위 합치기 완충 상수

# 코퍼스로 쓰는 arXiv 분야
CORPUS_CATEGORIES = ("cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.AI", "stat.ML")

# -- 모델 이름 --------------------------------------------------------------
REWRITER_MODEL = "qwen3:4b"                       # Ollama 로 부르는 쿼리 변환 모델
# arXiv 검색어 변환기. LoRA 를 기본 모델에 합쳐 4비트로 만든 것.
# Ollama 라 검색이 도는 동안에만 올라감. `training/export.py ollama` 로 만듦.
#
# 이 변환기는 arXiv '최신 논문' 칸에만 쓰임. 추천 목록 10편은 번역기와 가상 초록
# 생성기(둘 다 qwen3:4b)가 만든 검색어로 찾으므로 이 변환기와 무관함.
#
# 정밀도를 8비트나 f16 으로 올리면 arXiv 채널이 조금 나아지지만(342문항 nDCG@10
# 0.385 -> 0.400 -> 0.405) 검색 최대치가 3.3GB 에서 5.0GB, 8.7GB 로 커짐. 바뀌는 것이
# '최신 논문' 칸뿐이라 4비트로 고정함. 등급별 실측은 results/README.md 참고.
ARXIV_REWRITER_MODEL = "papers-rewriter"

EMBED_MODEL = "BAAI/bge-m3"                        # 의미 검색용 임베딩 모델
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"        # 재정렬 교차 인코더

# -- 그래픽 메모리 --------------------------------------------------------
#
# 언어 모델 하나가 3.25GB 라 재정렬 모델(1.24GB)과 같은 시각에 올라가면 4.76GB 가 됨.
# 그래서 검색 한 번을 세 구간으로 나누고 구간이 바뀔 때 반대편을 내림. 최대 3.52GB.
#
# 질문 임베딩은 GPU 를 안 씀. 후보 상위 100편의 구성이 GPU float16 과 100% 같고
# (질문 벡터 차이 2.39e-04) 색인을 만든 정밀도가 float32 라 오히려 맞음. 대신 질문
# 2개 인코딩이 0.021초에서 0.208초로 늘어남.
EMBED_DEVICE = "cpu"

# 재정렬 모델을 float16 으로 미리 저장해 둔 폴더. 검색마다 올렸다 내리므로 적재 시간이
# 그대로 응답 시간이 됨. 원본 float32 파일 2.2GB 는 매번 5.4~5.8초가 걸리는데, 이 사본은
# 첫 검색 4.9초 뒤로는 1.8초임. 점수는 128쌍을 대조해 차이가 정확히 0 이고 순위도 같아서
# MIN_RERANK_SCORE 를 다시 잴 필요가 없음. `training/export.py fp16` 으로 만듦.
RERANKER_FP16_DIR = ROOT_DIR / "models" / "reranker-fp16"

# 검색이 끝나고 이 시간이 지나면 올려 둔 모델을 전부 내림. 0 이면 검색이 끝나는 즉시.
GPU_IDLE_SECONDS = 0.0

# -- OpenAI 키 --------------------------------------------------------------
# 환경변수 OPENAI_API_KEY 가 먼저이고, 없으면 data/API_KEY.env 를 읽음.
# 그 파일은 "OPENAI_API_KEY=..." 형식과 키 한 줄만 담긴 형식을 모두 받음.
_API_KEY_FILE = DATA_DIR / "API_KEY.env"


def _load_openai_key() -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    if _API_KEY_FILE.exists():
        raw = _API_KEY_FILE.read_text(encoding="utf-8").strip()
        if raw.startswith("OPENAI_API_KEY="):
            raw = raw.split("=", 1)[1].strip()
        return raw
    return ""


OPENAI_API_KEY = _load_openai_key()
