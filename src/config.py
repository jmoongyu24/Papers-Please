"""프로젝트 전역 설정값을 한곳에 모아두는 파일.

모델 이름, 파일 경로, 검색 개수 같은 상수를 여기서만 정의하고
다른 모듈은 여기서 가져다 쓴다. 값을 바꿀 일이 생기면 이 파일만 고치면 된다.
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path


def _fix_libstdcxx() -> None:
    """torch 보다 먼저 새 C++ 표준 라이브러리를 올려 둔다.

    무엇을 고치는가:
    torch 를 먼저 부르면 시스템의 `/lib/x86_64-linux-gnu/libstdc++.so.6` 가 먼저 잡힌다.
    그런데 그 파일에는 `CXXABI_1.3.15` 가 없어서, 뒤이어 pyarrow(sklearn 이 부른다)를
    올릴 때 아래 오류로 깨진다.

        ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6:
        version `CXXABI_1.3.15' not found (required by .../libarrow.so.2100)

    왜 이 방식인가:
    `LD_PRELOAD` 환경변수로도 되지만, 그건 실행할 때마다 사람이 붙여 줘야 한다. 잊으면
    어떤 경로에서는 되고 어떤 경로에서는 안 되는 상태가 된다. 여기서 미리 올려 두면
    config 를 가져오는 모든 실행 경로에 자동으로 적용된다.

    조건: **torch 가 올라오기 전에** 실행돼야 한다. config 는 거의 모든 모듈이 맨 위에서
    가져오므로 이 자리가 맞다. 라이브러리를 못 찾으면 조용히 넘어간다(원래 환경이 멀쩡한
    경우까지 깨뜨리지 않기 위함).
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

# ── 환경 변수 (OpenAI API 키) ─────────────────────────────────────────────
# 우선순위: 환경변수 OPENAI_API_KEY > data/API_KEY.env 파일.
# data/API_KEY.env 는 "KEY=VALUE" 형식이 아니라 키 값 한 줄만 담긴 파일도 허용한다.
# 이 파일은 .gitignore 에 등록되어 있어 절대 커밋되지 않는다.
_API_KEY_FILE = DATA_DIR / "API_KEY.env"


def _load_openai_key() -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    if _API_KEY_FILE.exists():
        raw = _API_KEY_FILE.read_text(encoding="utf-8").strip()
        # "OPENAI_API_KEY=sk-..." 형식이면 값만 뽑고, 아니면 파일 전체를 키로 본다.
        if raw.startswith("OPENAI_API_KEY="):
            raw = raw.split("=", 1)[1].strip()
        return raw
    return ""


OPENAI_API_KEY = _load_openai_key()


def ensure_dirs() -> None:
    """필요한 폴더가 없으면 만든다."""
    for d in (DATA_DIR, CORPUS_DIR, EVAL_DIR, SAMPLE_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
