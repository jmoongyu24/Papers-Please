"""의미 기반 검색 (Dense retrieval): bge-m3 임베딩 + FAISS.

동작 원리:
1. 각 논문(제목+초록)을 bge-m3 모델로 '의미를 담은 숫자 목록(벡터)'으로 바꾼다.
   - bge-m3는 다국어 모델이라, 한국어 질문과 영어 초록을 같은 의미 공간에 놓는다.
     (그래서 단어가 하나도 안 겹치는 한국어 질문으로도 영어 논문을 찾을 수 있다.)
   - 벡터를 길이 1로 정규화하므로, 벡터 간 내적(inner product)이 곧 코사인 유사도가 된다.
2. 3만 편의 벡터를 FAISS 색인(IndexFlatIP = 전부 정확히 비교)에 넣어 둔다.
3. 검색 시: 질문도 같은 방식으로 벡터화 → 색인에서 유사도 높은 상위 k편을 찾는다.

임베딩 계산은 무거우므로(GPU로 몇 분) 한 번만 하고 디스크에 저장(캐싱)한다.
검색은 가벼우므로(질문당 수 밀리초) FAISS를 CPU로 돌린다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src import config
from src.retrieval.corpus import load_corpus
from src.schemas import Paper, ScoredPaper
from src.utils import read_json, write_json

EMBED_DIR = config.DATA_DIR / "embeddings"


class Embedder:
    """문장을 정규화된 벡터로 바꾸는 부분 (bge-m3 래핑)."""

    def __init__(self, model_name: str = config.EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        # device=None이면 GPU가 있으면 자동으로 GPU 사용
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        """문장 목록 → (문장 수, 차원) 크기의 float32 벡터 배열. 길이 1로 정규화됨."""
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,       # 코사인 유사도를 내적으로 계산하기 위함
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return emb.astype("float32")


def build_corpus_embeddings(
    papers: list[Paper], embedder: Embedder, cache_path: str | Path,
    batch_size: int = 64,
) -> np.ndarray:
    """코퍼스 전체를 벡터화한다. 이미 캐시가 있고 조건이 같으면 다시 계산하지 않는다."""
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("count") == len(papers) and meta.get("model") == embedder.model_name:
            return np.load(cache_path)

    texts = [p.text for p in papers]
    print(f"임베딩 계산 중: 논문 {len(texts)}편 × {embedder.model_name} ...")
    emb = embedder.encode(texts, batch_size=batch_size, show_progress=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    write_json(meta_path, {"count": len(papers), "model": embedder.model_name,
                           "dim": int(emb.shape[1])})
    print(f"저장: {cache_path} (shape={emb.shape})")
    return emb


class DenseRetriever:
    """코퍼스 위에서 의미 기반 검색을 한다."""

    def __init__(self, papers: list[Paper], embeddings: np.ndarray, embedder: Embedder):
        import faiss

        self.name = "dense"
        self.papers = papers
        self.embedder = embedder
        # IndexFlatIP: 모든 벡터와 내적(정규화했으므로 코사인 유사도)을 정확히 비교
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    @classmethod
    def build(cls, corpus_path: str | Path, cache_path: str | Path | None = None,
              model_name: str = config.EMBED_MODEL,
              embedder: Embedder | None = None) -> "DenseRetriever":
        """코퍼스 파일에서 검색기를 만든다 (임베딩은 캐시 사용/생성)."""
        papers = load_corpus(corpus_path)
        if cache_path is None:
            stem = Path(corpus_path).stem
            safe_model = model_name.replace("/", "_")
            cache_path = EMBED_DIR / f"{stem}.{safe_model}.npy"
        embedder = embedder or Embedder(model_name)
        emb = build_corpus_embeddings(papers, embedder, cache_path)
        return cls(papers, emb, embedder)

    def search(self, query: str, k: int) -> list[ScoredPaper]:
        q = self.embedder.encode([query])            # (1, dim), 정규화됨
        scores, idxs = self.index.search(q, k)       # 상위 k편의 유사도·위치
        results: list[ScoredPaper] = []
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
            if i < 0:
                continue
            p = self.papers[i]
            results.append(ScoredPaper(paper_id=p.id, score=float(s), rank=rank,
                                       title=p.title, abstract=p.abstract))
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description="코퍼스 임베딩 색인 빌드 (1회성)")
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-v1.jsonl"))
    ap.add_argument("--model", default=config.EMBED_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    papers = load_corpus(args.corpus)
    stem = Path(args.corpus).stem
    cache_path = EMBED_DIR / f"{stem}.{args.model.replace('/', '_')}.npy"
    embedder = Embedder(args.model)
    emb = build_corpus_embeddings(papers, embedder, cache_path, args.batch_size)
    print(f"완료: {len(papers)}편, 차원 {emb.shape[1]} → {cache_path}")


if __name__ == "__main__":
    main()
