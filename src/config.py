"""프로젝트 전역 설정값을 한곳에 모아두는 파일.

모델 이름, 파일 경로, 검색 개수 같은 상수를 여기서만 정의하고
다른 모듈은 여기서 가져다 쓴다. 값을 바꿀 일이 생기면 이 파일만 고치면 된다.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────────────
# 이 파일 기준으로 저장소 최상위 폴더를 계산한다 (src/config.py -> 상위의 상위).
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"          # 캐글에서 받은 대용량 코퍼스 (gitignore)
EVAL_DIR = DATA_DIR / "eval"              # 평가셋 (git에 포함해 재현 가능)
SAMPLE_DIR = DATA_DIR / "sample"          # 데모·테스트용 작은 예시 데이터 (git 포함)
RUNS_DIR = ROOT_DIR / "runs"             # 실험 결과 캐시 (gitignore)

# ── 검색 관련 상수 ────────────────────────────────────────────────────────
TOP_K = 10                    # 검색 결과를 몇 개까지 가져올지 (기본값)
K_VALUES = (1, 5, 10, 20)     # 재현율을 어떤 K에서 잴지
RRF_K = 60                    # 혼합 검색(순위 합치기)에서 쓰는 상수

# 코퍼스로 쓸 arXiv 분야 (컴퓨터·인공지능 계열)
CORPUS_CATEGORIES = ("cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.AI", "stat.ML")

# ── 모델 이름 ────────────────────────────────────────────────────────────
REWRITER_MODEL = "qwen3:4b"                       # Ollama에서 쓸 쿼리 변환 모델
EMBED_MODEL = "BAAI/bge-m3"                        # 의미 기반 검색용 임베딩 모델
QUERY_GEN_MODEL = "gpt-4o-mini"                    # 평가셋 질문 생성용 (오프라인 단계 전용)

# ── 환경 변수 ────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def ensure_dirs() -> None:
    """필요한 폴더가 없으면 만든다."""
    for d in (DATA_DIR, CORPUS_DIR, EVAL_DIR, SAMPLE_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
