"""실시간 arXiv 검색기 - 변환된 검색어로 arXiv API에 직접 질의해 논문을 가져옴.

로컬 코퍼스(평가 전용)와 달리, 이건 실제 서비스용임. 데이터베이스가 필요 없고
검색할 때마다 arXiv에서 최신 논문까지 가져옴. 다른 검색기와 같은 인터페이스
(`search(query, k) -> list[ScoredPaper]`)를 따름.

코드로 확인한 주의사항 반영(reference/langchain_community README):
- 결과 개수는 max_results 로 지정(우리가 k로).
- 검색어는 300자에서 잘림 -> 최종 검색어만 넘김.
- arXiv API는 관련도 점수를 주지 않고 '순위'만 줌 -> score는 1/순위로 채움.
- 요청 간 3초 간격 권장(arxiv 패키지 Client 기본값). 같은 검색어는 캐싱해 재호출을 줄임.
"""

from __future__ import annotations

import json
import random
import socket
import time
from pathlib import Path

from src.schemas import ScoredPaper

ARXIV_MAX_QUERY_LEN = 300

# 응답이 이 시간 안에 안 오면 끊고 재시도함 (초).
#
# 왜 필요한가 (2026-08-14 실측으로 확인한 사고):
# `arxiv` 패키지는 안쪽에서 `urllib` 로 요청하는데 타임아웃을 걸지 않음. 그래서 연결이
# 끊기지도 응답이 오지도 않는 상태가 되면 그대로 영원히 기다림. 실제로 시험용 342문항
# 측정이 200번째 문항에서 멈춰 75분 동안 한 걸음도 못 나갔음. 프로세스는 살아 있고
# CPU 도 쓰고 있어서 오류도 로그도 안 남고, 겉보기에는 그냥 느린 것과 구분이 안 됨.
#
# 백오프 재시도(15->30->60->120초)는 이 상황에 아무 도움이 안 됨. 재시도는 요청이
# '실패해야' 도는데, 매달린 요청은 실패하지 않기 때문임.
#
# 값의 근거: 정상 응답이 1.5~7초였음(같은 날 실측). 90초면 느린 응답도 넉넉히 받고,
# 매달린 연결은 확실히 끊음.
ARXIV_TIMEOUT_SEC = 90


class ArxivLiveRetriever:
    """arXiv API로 실시간 검색."""

    name = "arxiv_live"

    def __init__(self, delay_seconds: float = 3.0, num_retries: int = 0,
                 max_attempts: int = 5, backoff_base: float = 15.0,
                 cache_path: str | Path | None = None):
        """
        Args:
            delay_seconds: 요청 사이 최소 간격(arXiv 권장 3초). arxiv 패키지가 자동으로 지킴.
            num_retries: arxiv 패키지 자체 재시도. 0으로 끔 - 이 패키지는 간격을 늘리지 않고
                같은 속도로 계속 두드려서, 오히려 arXiv 차단을 길게 만듦. 대신 아래 백오프를 씀.
            max_attempts: 우리 쪽 최대 시도 횟수(첫 시도 포함). 429/503은 기다리면 대개 풀리므로
                넉넉히 5회로 둠.
            backoff_base: 실패 시 대기 시간의 시작값(초). 실패할수록 2배씩 늘리고(15->30->60->120),
                429, 503이면 다시 2배로 더 기다림. arXiv가 요청 과다로 막았을 때는 짧게 여러 번
                두드리는 것보다 충분히 쉬는 편이 훨씬 잘 풀림.
            cache_path: 검색 결과를 저장할 파일 경로. 주면 디스크 캐싱을 켬.
                평가, 학습 데이터 생성처럼 같은 검색을 반복하는 작업에서, 프로그램을 껐다 켜도
                이미 한 검색은 arXiv를 다시 부르지 않아 중단 지점부터 이어서 할 수 있음.
                웹 서비스에서는 주지 않아(None) 메모리 캐시만 씀.
        """
        import arxiv

        # arxiv 패키지가 안에서 쓰는 urllib 에 타임아웃을 걸 방법이 없어서 소켓 기본값으로 걸어 둠
        # (위 ARXIV_TIMEOUT_SEC 설명 참고). 이미 더 짧게 잡혀 있으면 건드리지 않음.
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

    # -- 디스크 캐시 --------------------------------------------------------
    def _load_disk_cache(self) -> None:
        """저장해 둔 검색 결과를 불러옴(있으면 arXiv를 다시 부르지 않음)."""
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
        """검색해서 결과를 돌려줌.

        성공 시: 결과 목록(비어 있으면 '진짜로 결과 없음'). 오류 시: 예외를 그대로 올림
        (예전엔 오류를 빈 목록으로 삼켜 '결과 없음'처럼 보였는데, arXiv 일시적 오류와 진짜
        '결과 없음'을 구분해야 하므로 더 이상 삼키지 않음. 호출자가 오류를 처리함).
        """
        query = (query or "").strip()[:ARXIV_MAX_QUERY_LEN]
        if not query:
            return []
        key = (query, k)
        if key in self._cache:
            return self._cache[key]

        # 지수 백오프로 재시도. 실패할수록 대기를 2배로 늘려(15->30->60->120초) arXiv에 회복할
        # 시간을 줌. 짧은 간격으로 계속 두드리면 차단이 오히려 길어짐.
        #
        # 실제로 겪은 오류 두 가지(둘 다 '요청이 너무 잦다'는 신호이며 쿼리 내용과 무관):
        #   - HTTP 429 Too Many Requests : 속도 제한에 걸림
        #   - HTTP 503 Service Unavailable : 서버가 일시적으로 처리 불가(과부하)
        # 이 둘은 기다리면 대개 풀리므로, 다른 오류보다 더 오래 기다림.
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
                    wait *= 2          # 속도 제한/과부하는 더 오래 쉼
                # 여러 요청이 동시에 같은 시점에 재시도하지 않도록 무작위 지연을 섞음
                wait += random.uniform(0, self._backoff_base * 0.5)
                time.sleep(wait)
        raise last_error       # 모두 실패하면 호출자에게 오류를 알림

    def _fetch(self, query: str, k: int) -> list[ScoredPaper]:
        """arXiv에 실제로 한 번 요청해 결과를 우리 형식으로 바꿈."""
        search = self._arxiv.Search(
            query=query, max_results=k,
            sort_by=self._arxiv.SortCriterion.Relevance,
        )
        results: list[ScoredPaper] = []
        for rank, r in enumerate(self._client.results(search), start=1):
            results.append(ScoredPaper(
                paper_id=r.get_short_id(),          # 예: "1706.03762v7"
                score=1.0 / rank,                   # arXiv은 점수 없음 -> 순위 기반
                rank=rank,
                title=r.title.strip(),
                abstract=r.summary.strip(),
            ))
            if len(results) >= k:
                break
        return results
