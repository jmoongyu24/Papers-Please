"""arXiv API를 이용하는 검색기"""

from __future__ import annotations

import json
import random
import socket
import time
from pathlib import Path

from src.schemas import ScoredPaper

ARXIV_MAX_QUERY_LEN = 300

# 검색 응답이 이 안에 안 오면 끊고 다시 시도
ARXIV_TIMEOUT_SEC = 90


class ArxivLiveRetriever:
    """arXiv API로 검색"""

    name = "arxiv_live"

    def __init__(self, delay_seconds: float = 3.0, num_retries: int = 0,
                 max_attempts: int = 5, backoff_base: float = 15.0,
                 cache_path: str | Path | None = None):
        """
        Args:
            delay_seconds: 검색 요청 사이 대기 시간
            num_retries: arxiv 검색 재시도 횟수
            max_attempts: 최대 검색 시도 횟수
            backoff_base: 검색 실패 시 대기 시간의 시작값. 실패할수록 2배씩 늘림
            cache_path: 검색 결과를 이 파일 경로에 저장해두고, 프로그램을 껐다 켜도 재사용함. 평가에 사용함
        """
        import arxiv

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
        """저장해 둔 검색 결과를 불러옴"""
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
        """검색 결과를 리턴함. 빈 목록: '검색 결과 없음', 오류: 그대로 리턴"""
        query = (query or "").strip()[:ARXIV_MAX_QUERY_LEN]
        if not query:
            return []
        key = (query, k)
        if key in self._cache:
            return self._cache[key]

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                results = self._fetch(query, k)
                self._cache[key] = results      # 검색 성공한 결과만 캐싱
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

                wait += random.uniform(0, self._backoff_base * 0.5)
                time.sleep(wait)
        raise last_error

    def _fetch(self, query: str, k: int) -> list[ScoredPaper]:
        """arXiv로부터 받아온 검색 결과를 형식에 맞게 가공"""
        search = self._arxiv.Search(
            query=query, max_results=k,
            sort_by=self._arxiv.SortCriterion.Relevance,
        )
        results: list[ScoredPaper] = []
        for rank, r in enumerate(self._client.results(search), start=1):
            results.append(ScoredPaper(
                paper_id=r.get_short_id(),          # ex: "1706.03762v7"
                score=1.0 / rank,
                rank=rank,
                title=r.title.strip(),
                abstract=r.summary.strip(),
            ))
            if len(results) >= k:
                break
        return results
