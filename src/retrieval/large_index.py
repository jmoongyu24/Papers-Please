"""대규모 논문 색인 — arXiv 컴퓨터·인공지능 계열 71만 편의 의미 기반 검색.

arXiv 키워드 검색은 정확한 문자열 일치를 요구해 후보 확보율이 0.49에서 막혔다(ISSUE #21).
의미 검색은 그 요구가 없고 bge-m3 가 한국어를 번역 없이 처리하므로, 기존 경로를 대체하지
않고 두 번째 채널로 나란히 둔다.

대상은 "cs·stat.ML·eess 계열, 2021년 이후" 716,183편 전부다. 기존 평가용 3만 편이 바로 이
모집단에서 뽑은 표본이라, 모집단 전체를 색인해야 표본추출 왜곡 없이 실력을 잰다.

메모리가 15GB뿐이라 세 가지를 지킨다. 논문 본문은 줄 위치만 기록해 필요한 것만 꺼내 읽고,
FAISS 대신 numpy 내적을 써서 벡터 복사본을 만들지 않고, 임베딩은 조각으로 나눠 계산해
디스크에 바로 쓴다(중단 시 이어하기 가능).

색인 한 벌은 네 파일이다: `.ids.txt`(논문 번호) `.offsets.npy`(줄 위치)
`.emb.npy`(임베딩) `.meta.json`(진행 상황).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import config
from src.schemas import ScoredPaper


def index_paths(out_prefix: str | Path) -> dict[str, Path]:
    p = Path(out_prefix)
    return {
        "ids": p.with_suffix(".ids.txt"),
        "offsets": p.with_suffix(".offsets.npy"),
        "emb": p.with_suffix(".emb.npy"),
        "meta": p.with_suffix(".meta.json"),
    }


# ── 1단계: 논문 번호와 줄 위치 기록 ────────────────────────────────────────
def scan_corpus(corpus_path: str | Path, out_prefix: str | Path) -> tuple[list[str], np.ndarray]:
    """논문 번호와 각 줄의 시작 위치를 기록한다 (seek 한 번으로 특정 논문만 읽기 위함)."""
    paths = index_paths(out_prefix)
    if paths["ids"].exists() and paths["offsets"].exists():
        ids = paths["ids"].read_text(encoding="utf-8").splitlines()
        return ids, np.load(paths["offsets"])

    ids: list[str] = []
    offsets: list[int] = []
    pos = 0
    with open(corpus_path, "rb") as f:
        for raw in f:
            offsets.append(pos)
            pos += len(raw)
            row = json.loads(raw)
            ids.append(str(row["id"]))

    paths["ids"].parent.mkdir(parents=True, exist_ok=True)
    paths["ids"].write_text("\n".join(ids), encoding="utf-8")
    off = np.asarray(offsets, dtype=np.int64)
    np.save(paths["offsets"], off)
    print(f"논문 번호·줄 위치 기록 완료: {len(ids):,}편 → {paths['ids'].name}")
    return ids, off


# ── 2단계: 임베딩 계산 (조각으로 나눠, 이어하기 지원) ──────────────────────
def build_embeddings(corpus_path: str | Path, out_prefix: str | Path,
                     model_name: str = config.EMBED_MODEL,
                     batch_size: int = 64, chunk_size: int = 4096,
                     max_seq_length: int = 512) -> None:
    """코퍼스 전체를 임베딩해 디스크 배열에 써넣는다.

    max_seq_length: bge-m3 기본값 8192 는 초록(250~400 토큰)에 과하다. 512 로 제한하면
    품질 손실 없이 몇 배 빨라진다.
    """
    from sentence_transformers import SentenceTransformer

    paths = index_paths(out_prefix)
    ids, _ = scan_corpus(corpus_path, out_prefix)
    n = len(ids)

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    dim = model.get_sentence_embedding_dimension()
    print(f"모델 {model_name} · 차원 {dim} · 장치 {model.device} · 최대 길이 {max_seq_length}")
    if str(model.device) == "cpu":
        print("경고: GPU 를 못 잡았다. CPU 로 71만 편을 돌리면 며칠 걸린다. 중단할 것.")

    # 이어하기: 이미 몇 편까지 계산했는지 확인한다
    done = 0
    if paths["meta"].exists():
        meta = json.loads(paths["meta"].read_text())
        if meta.get("count") == n and meta.get("model") == model_name and paths["emb"].exists():
            done = int(meta.get("done", 0))
            print(f"이어하기: {done:,}편까지 이미 계산됨")

    mode = "r+" if paths["emb"].exists() and done > 0 else "w+"
    emb = np.lib.format.open_memmap(paths["emb"], mode=mode, dtype=np.float32, shape=(n, dim))

    t0 = time.time()
    with open(corpus_path, "r", encoding="utf-8") as f:
        # 이미 끝낸 부분은 읽고 버린다 (파일 순서와 배열 순서를 맞추기 위함)
        for _ in range(done):
            f.readline()

        i = done
        while i < n:
            texts = []
            while len(texts) < chunk_size and i + len(texts) < n:
                line = f.readline()
                if not line:
                    break
                row = json.loads(line)
                texts.append(f"{row.get('title', '').strip()}\n{row.get('abstract', '').strip()}")
            if not texts:
                break

            vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
            emb[i:i + len(texts)] = vecs.astype(np.float32)
            i += len(texts)

            emb.flush()
            paths["meta"].write_text(json.dumps(
                {"count": n, "dim": int(dim), "model": model_name, "done": i,
                 "corpus": str(corpus_path), "max_seq_length": max_seq_length},
                ensure_ascii=False), encoding="utf-8")

            elapsed = time.time() - t0
            speed = (i - done) / max(elapsed, 1e-6)
            eta = (n - i) / max(speed, 1e-6)
            print(f"  {i:,}/{n:,}편 · {speed:.0f}편/초 · 경과 {elapsed/60:.1f}분 "
                  f"· 남은 예상 {eta/60:.1f}분", flush=True)

    del emb
    print(f"임베딩 완료: {n:,}편 → {paths['emb']}")


# ── 검색기 ────────────────────────────────────────────────────────────────
class LargeDenseRetriever:
    """71만 편 위에서 의미 기반 검색. 인터페이스는 다른 검색기와 동일하다."""

    name = "local_dense"

    def __init__(self, corpus_path: str | Path, out_prefix: str | Path,
                 model_name: str = config.EMBED_MODEL, embedder=None,
                 mmap: bool = False, max_seq_length: int = 512):
        """mmap=True 면 임베딩을 디스크에 둔 채 읽는다(메모리 절약, 느림)."""
        paths = index_paths(out_prefix)
        self.corpus_path = Path(corpus_path)
        self.ids = paths["ids"].read_text(encoding="utf-8").splitlines()
        self.offsets = np.load(paths["offsets"])
        meta = json.loads(paths["meta"].read_text())
        if meta.get("done", 0) < meta["count"]:
            raise RuntimeError(
                f"임베딩이 아직 다 안 됐다: {meta['done']:,}/{meta['count']:,}편. "
                f"build_embeddings 를 마저 돌릴 것.")
        self.emb = np.load(paths["emb"], mmap_mode="r" if mmap else None)

        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(model_name)
            embedder.max_seq_length = max_seq_length
        self.embedder = embedder
        # 논문 번호 -> 배열 위치 (재정렬 후보의 본문을 꺼낼 때 쓴다)
        self._pos = {pid: i for i, pid in enumerate(self.ids)}

    def encode_query(self, query: str) -> np.ndarray:
        v = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)[0]

    def search(self, query: str, k: int = 100) -> list[ScoredPaper]:
        """질문과 뜻이 가까운 논문 상위 k편을 돌려준다."""
        q = self.encode_query(query)
        scores = self._scores(q)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]   # 상위 k개만 추린 뒤 그 안에서 정렬
        top = top[np.argsort(-scores[top])]
        return self._to_papers(top, scores)

    def _scores(self, q: np.ndarray) -> np.ndarray:
        """모든 논문과의 유사도(정규화돼 있으므로 내적 = 코사인 유사도)."""
        if isinstance(self.emb, np.memmap):
            out = np.empty(self.emb.shape[0], dtype=np.float32)
            step = 100_000
            for s in range(0, self.emb.shape[0], step):
                out[s:s + step] = np.asarray(self.emb[s:s + step]) @ q
            return out
        return self.emb @ q

    def _to_papers(self, idxs: np.ndarray, scores: np.ndarray) -> list[ScoredPaper]:
        rows = self.read_rows([int(i) for i in idxs])
        out = []
        for rank, (i, row) in enumerate(zip(idxs, rows), start=1):
            out.append(ScoredPaper(
                paper_id=self.ids[int(i)], score=float(scores[int(i)]), rank=rank,
                title=row.get("title", "").strip(), abstract=row.get("abstract", "").strip(),
            ))
        return out

    def read_rows(self, positions: list[int]) -> list[dict]:
        """배열 위치 목록에 해당하는 논문 본문을 파일에서 직접 꺼내 읽는다."""
        out = []
        with open(self.corpus_path, "rb") as f:
            for i in positions:
                f.seek(int(self.offsets[i]))
                out.append(json.loads(f.readline()))
        return out

    def get_by_ids(self, paper_ids: list[str]) -> dict[str, dict]:
        """논문 번호로 본문을 꺼낸다 (색인에 없는 번호는 빠진다)."""
        pos = [(pid, self._pos[pid]) for pid in paper_ids if pid in self._pos]
        rows = self.read_rows([p for _, p in pos])
        return {pid: row for (pid, _), row in zip(pos, rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="대규모 의미 검색 색인 구축 (1회성, 이어하기 지원)")
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--out", default=str(config.DATA_DIR / "embeddings" / "cs2021"))
    ap.add_argument("--model", default=config.EMBED_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--max-seq-length", type=int, default=512)
    args = ap.parse_args()

    build_embeddings(args.corpus, args.out, args.model, args.batch_size,
                     args.chunk_size, args.max_seq_length)


if __name__ == "__main__":
    main()
