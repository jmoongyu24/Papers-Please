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
EMBED_MODEL = "BAAI/bge-m3"                        # 의미 검색용 임베딩 모델

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
