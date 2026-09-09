"""검색이 도는 동안에만 그래픽 메모리를 쓰도록 모델을 올렸다 내리는 곳.

## 왜 필요한가

`app.py` 는 무거운 부품을 `st.cache_resource` 로 감쌌는데, 그것은 앱이 꺼질 때까지 놓지
않는다는 뜻임. 그래서 검색이 끝나도 13.83GB 를 계속 물고 있었음.

## 왜 구간을 나누는가

언어 모델 하나가 3.25GB 임. 4비트로 이미 줄인 값이고 추천 이유 프롬프트가 3,618토큰이라
context 4096 을 줄일 수도 없음. 여기에 재정렬 모델 1.24GB 와 CUDA 컨텍스트 0.27GB 를
더하면 4.76GB 가 됨. 4GB 안에 들어가려면 **언어 모델과 재정렬 모델이 같은 시각에 올라가
있으면 안 됨.**

검색 한 번은 서로 겹치지 않는 세 구간으로 나뉨:

    구간 1  논문 지목 · 번역 · 가상 초록 · arXiv 검색어    Ollama
    구간 2  로컬 의미 검색 · 재정렬                      재정렬 모델
    구간 3  추천 이유                                  Ollama

구간이 바뀔 때 반대편을 내림. 최대 3.52GB.

## 쓰는 법

    pool = GpuPool()
    with pool.session():                       # 검색 한 번을 통째로 감쌈
        rr = pool.get("reranker", CrossEncoderReranker)
        ...
        pool.release("reranker")               # 구간이 바뀌는 자리
    # session 을 빠져나오면 전부 내려감

## 등록하는 것은 반드시 `unload()` 를 가져야 함

참조를 끊는 것만으로는 자리가 안 돌아옴. 이유가 둘임.

- Ollama 모델은 이 프로세스 밖에 있음. 파이썬 객체를 없애도 서버는 모델을 들고 있음
- torch 모델은 부르는 쪽이 변수에 담아 두면 파이썬이 객체를 안 없앰. 실제로 겪었음 -
  `del` 만 했더니 재정렬 모델 1.24GB 가 안 돌아와 최대치가 3.51GB 가 아니라 4.56GB 였음

그래서 `unload()` 를 못 찾으면 조용히 넘어가지 않고 오류를 냄. 조용히 넘어가면 자리가
안 돌아온 것을 아무도 모른 채 최대치만 늘어남.
"""

from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from src import config


class GpuPool:
    """이름을 붙여 모델을 올려 두고, 다 쓴 시점에 내림.

    검색 한 번은 `session()` 안에서 돌아야 함. Streamlit 은 사용자가 무언가 누를 때마다
    스크립트를 다시 돌리므로, 앞 검색이 쓰는 모델을 뒤 검색이 내려 버릴 수 있음. 잠금으로
    한 번에 하나만 돌게 막음.
    """

    def __init__(self, idle_seconds: float | None = None):
        self._items: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._last_used = 0.0
        self.idle_seconds = (config.GPU_IDLE_SECONDS if idle_seconds is None
                             else idle_seconds)

    # -- 올리기 ------------------------------------------------------------

    def get(self, key: str, factory: Callable[[], Any]) -> Any:
        """이름으로 찾고, 없으면 `factory()` 로 만들어 둠."""
        with self._lock:
            if key not in self._items:
                self._items[key] = factory()
            self._last_used = time.time()
            return self._items[key]

    def loaded(self) -> list[str]:
        with self._lock:
            return sorted(self._items)

    # -- 내리기 ------------------------------------------------------------

    def release(self, *keys: str) -> None:
        """이름을 지정해 내림. 없는 이름은 조용히 넘어감."""
        with self._lock:
            dropped = False
            for key in keys:
                obj = self._items.pop(key, None)
                if obj is None:
                    continue
                dropped = True
                # 모든 등록 대상은 `unload()` 를 가져야 함. 참조를 끊는 것만으로는
                # 부족하기 때문임 - Ollama 모델은 이 프로세스 밖에 있고, torch 모델은
                # 부르는 쪽이 변수에 담아 두면 파이썬이 안 없앰
                unload = getattr(obj, "unload", None)
                if callable(unload):
                    unload()
                else:
                    raise TypeError(
                        f"'{key}' 에 unload() 가 없다. 없으면 자리가 안 돌아온다.")
                del obj
            if dropped:
                self._reclaim()

    def release_all(self) -> None:
        with self._lock:
            self.release(*list(self._items))

    def release_idle(self) -> bool:
        """마지막으로 쓴 지 `idle_seconds` 가 지났으면 전부 내림.

        `idle_seconds` 가 0 이면 이 함수는 아무 일도 안 함. 검색이 끝나는 즉시 내리는
        것은 `session()` 이 맡음.
        """
        with self._lock:
            if not self._items or self.idle_seconds <= 0:
                return False
            if time.time() - self._last_used < self.idle_seconds:
                return False
            self.release_all()
            return True

    @staticmethod
    def _reclaim() -> None:
        """torch 가 쥐고 있던 그래픽 메모리를 카드에 돌려줌.

        torch 를 아직 안 불러왔으면 GPU 를 쓴 적도 없으므로 넘어감. 여기서 import 하는
        것은 torch 를 안 쓰는 경로(평가 일부, 시험)에서 괜히 불러오지 않기 위함임.
        """
        gc.collect()
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- 검색 한 번 --------------------------------------------------------

    @contextmanager
    def session(self):
        """검색 한 번을 감쌈. 빠져나올 때 올려 둔 것을 전부 내림.

        오류가 나도 반드시 내림. 안 그러면 실패한 검색이 3.25GB 를 붙잡은 채 남음.
        """
        with self._lock:
            try:
                yield self
            finally:
                if self.idle_seconds <= 0:
                    self.release_all()
                else:
                    self._last_used = time.time()
