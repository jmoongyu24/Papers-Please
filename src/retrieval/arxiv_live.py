"""arXiv API 실시간 검색기.

다른 검색기와 같은 인터페이스를 따름: `search(query, k) -> list[ScoredPaper]`.

arXiv API 의 제약 세 가지를 코드가 처리함.
- 검색어는 300자에서 잘림
- 관련도 점수를 주지 않고 순위만 줌 -> 점수 자리에 1/순위 를 넣음
- 요청 간 3초 간격, 단일 연결. 같은 검색어는 캐싱해 재호출을 줄임
"""

from __future__ import annotations

import json
import random
import socket
import time
from pathlib import Path

from src.schemas import ScoredPaper

ARXIV_MAX_QUERY_LEN = 300

# 응답이 이 시간 안에 안 오면 끊고 다시 시도함 (초).
# `arxiv` 패키지는 안쪽 urllib 에 타임아웃을 걸지 않아서, 연결이 매달리면 영원히 기다림.
# 실제로 342문항 측정이 200번째에서 75분 동안 멈춘 적이 있음. 매달린 요청은 실패하지
# 않으므로 아래 재시도로는 못 푼다. 정상 응답이 1.5~7초라 90초로 잡음.
ARXIV_TIMEOUT_SEC = 90


class ArxivLiveRetriever:
    """arXiv API 로 실시간 검색."""

    name = "arxiv_live"

    def __init__(self, delay_seconds: float = 3.0, num_retries: int = 0,
                 max_attempts: int = 5, backoff_base: float = 15.0,
                 cache_path: str | Path | None = None):
        """
        Args:
            delay_seconds: 요청 사이 최소 간격. arXiv 권장값이 3초임.
            num_retries: arxiv 패키지 자체 재시도. 0으로 끔 - 간격을 늘리지 않고 같은
                속도로 계속 두드려서 차단을 오히려 길게 만듦. 대신 아래 백오프를 씀.
            max_attempts: 우리 쪽 최대 시도 횟수 (첫 시도 포함).
            backoff_base: 실패 시 대기 시간의 시작값(초). 실패할수록 2배씩 늘림.
            cache_path: 주면 검색 결과를 이 파일에 쌓아 프로그램을 껐다 켜도 재사용함.
                평가처럼 같은 검색을 반복하는 작업에서 씀. 웹 화면에서는 주지 않음.
        """
        import arxiv

        # arxiv 패키지 안쪽 urllib 에 타임아웃을 걸 방법이 없어 소켓 기본값으로 걸어 둠.
        current = socket.getdefaulttimeout()
        if current is None or current > ARXIV_TIMEOUT_SEC:
            socket.setdefaulttimeout(ARXIV_TIMEOUT_SEC)

        self._arxiv = arxiv
        self._client = arxiv.Client(
            page_size=100, delay_seconds=delay_seconds, num_retries=num_retries
        )
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._cache: dict[tuple[str, int], list[ScoredPaper]] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path and self._cache_path.exists():
            self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        """저장해 둔 검색 결과를 불러옴."""
        with open(self._cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["query"], row["k"])
                self._cache[key] = [
                    ScoredPaper(paper_id=p["paper_id"], score=p["score"], rank=p["rank"],
                                title=p["title"], abstract=p["abstract"])
                    for p in row["results"]
                ]

    def _append_disk_cache(self, query: str, k: int, results: list[ScoredPaper]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "query": query, "k": k,
                "results": [{"paper_id": r.paper_id, "score": r.score, "rank": r.rank,
                             "title": r.title, "abstract": r.abstract} for r in results],
            }, ensure_ascii=False) + "\n")

    def search(self, query: str, k: int) -> list[ScoredPaper]:
        """검색 결과를 돌려줌. 빈 목록은 '진짜로 결과 없음' 이고, 오류는 그대로 올림.

        오류를 빈 목록으로 삼키면 arXiv 의 일시적 오류와 진짜 '결과 없음' 을 구분할 수
        없어짐. 호출자가 오류를 처리함.
        """
        query = (query or "").strip()[:ARXIV_MAX_QUERY_LEN]
        if not query:
            return []
        key = (query, k)
        if key in self._cache:
            return self._cache[key]

        # 지수 백오프. 429(요청 과다)와 503(과부하)은 기다리면 대개 풀리므로 더 오래 쉼.
        # 짧은 간격으로 계속 두드리면 차단이 오히려 길어짐.
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                results = self._fetch(query, k)
                self._cache[key] = results      # 성공한 결과만 캐싱
                if self._cache_path is not None:
                    self._append_disk_cache(query, k, results)
                return results
            except Exception as e:
                last_error = e
                if attempt >= self._max_attempts - 1:
                    break
                wait = self._backoff_base * (2 ** attempt)
                msg = str(e)
                if "429" in msg or "503" in msg:
                    wait *= 2
                # 여러 요청이 같은 시점에 몰리지 않도록 무작위 지연을 섞음
                wait += random.uniform(0, self._backoff_base * 0.5)
                time.sleep(wait)
        raise last_error

    def _fetch(self, query: str, k: int) -> list[ScoredPaper]:
        """arXiv 에 한 번 요청해 결과를 우리 형식으로 바꿈."""
        search = self._arxiv.Search(
            query=query, max_results=k,
            sort_by=self._arxiv.SortCriterion.Relevance,
        )
        results: list[ScoredPaper] = []
        for rank, r in enumerate(self._client.results(search), start=1):
            results.append(ScoredPaper(
                paper_id=r.get_short_id(),          # 예: "1706.03762v7"
                score=1.0 / rank,                   # arXiv 은 점수를 안 주므로 순위 기반
                rank=rank,
                title=r.title.strip(),
                abstract=r.summary.strip(),
            ))
            if len(results) >= k:
                break
        return results
