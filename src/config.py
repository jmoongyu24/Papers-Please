"""
프로젝트 전역 설정값

모델 이름, 파일 경로, 검색 개수 같은 상수를 여기서 정의
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path


def _fix_libstdcxx() -> None:
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

# 경로
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"

# RRF 상수
RRF_K = 60

# 코퍼스로 쓰는 arXiv 분야
CORPUS_CATEGORIES = ("cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.AI", "stat.ML")

# 모델 이름
REWRITER_MODEL = "qwen3:4b"                       # Ollama로 부르는 쿼리 변환 모델
ARXIV_REWRITER_MODEL = "papers-rewriter"

EMBED_MODEL = "BAAI/bge-m3"                       # 의미 검색용 임베딩 모델
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"        # 재정렬에 사용하는 크로스 인코더 모델

EMBED_DEVICE = "cpu"

# 재정렬 모델을 float16으로 미리 저장해 둔 폴더
RERANKER_FP16_DIR = ROOT_DIR / "models" / "reranker-fp16"

# OpenAI 키
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
