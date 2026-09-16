"""
검색이 진행되는 동안에만 그래픽 메모리를 쓰도록 관리하는 코드
"""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from typing import Any, Callable


class GpuPool:
    """이름을 붙여 모델을 올려 두고, 다 쓴 시점에 내림"""

    def __init__(self):
        self._items: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: str, factory: Callable[[], Any]) -> Any:
        """이름으로 찾고, 없으면 factory()로 만들어 둠."""
        with self._lock:
            if key not in self._items:
                self._items[key] = factory()
            return self._items[key]

    def release(self, *keys: str) -> None:
        """이름을 지정해 메모리에서 내림"""
        with self._lock:
            dropped = False
            for key in keys:
                obj = self._items.pop(key, None)
                if obj is None:
                    continue
                dropped = True
                unload = getattr(obj, "unload", None)
                if callable(unload):
                    unload()
                else:
                    raise TypeError(
                        f"'{key}'에 unload()가 없다. 없으면 그래픽 메모리가 안 비워진다.")
                del obj
            if dropped:
                self._reclaim()

    def release_all(self) -> None:
        with self._lock:
            self.release(*list(self._items))

    @staticmethod
    def _reclaim() -> None:
        """torch가 쥐고 있던 그래픽 메모리를 해제함"""
        gc.collect()
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def session(self):
        """검색 끝나면 전부 내림"""
        with self._lock:
            try:
                yield self
            finally:
                self.release_all()
