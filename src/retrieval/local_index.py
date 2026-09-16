"""
로컬 의미 검색 색인 - 논문 71만 편을 임베딩한 후, 의미 기반으로 찾는 검색기
대상은 arXiv의 cs, stat.ML, eess 계열 2021년 이후 논문
"""

from __future__ import annotations

import argparse
import json
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


def corpus_fingerprint(corpus_path: str | Path) -> dict:
    """코퍼스 파일 식별자 (경로, 이름, 크기, 수정시각)"""
    p = Path(corpus_path)
    st = p.stat()
    return {"corpus": str(p), "corpus_name": p.name,
            "corpus_size": int(st.st_size), "corpus_mtime": int(st.st_mtime)}


def read_meta(out_prefix: str | Path) -> dict:
    paths = index_paths(out_prefix)
    return json.loads(paths["meta"].read_text()) if paths["meta"].exists() else {}


def write_meta(out_prefix: str | Path, meta: dict) -> None:
    paths = index_paths(out_prefix)
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def sample_matches(corpus_path: str | Path, ids: list[str], offsets: np.ndarray,
                   n: int = 5) -> bool:
    """n편을 꺼내 읽어 논문 번호가 맞는지 확인함. 색인과 코퍼스가 짝이 맞는지 확인할 때 씀"""
    if not len(ids):
        return False
    picks = np.linspace(0, len(ids) - 1, num=min(n, len(ids)), dtype=int)
    try:
        with open(corpus_path, "rb") as f:
            for i in picks:
                f.seek(int(offsets[i]))
                if str(json.loads(f.readline())["id"]) != ids[int(i)]:
                    return False
    except Exception:
        return False
    return True


def check_pairing(corpus_path: str | Path, out_prefix: str | Path,
                  ids: list[str], offsets: np.ndarray) -> tuple[bool, str]:
    """짝이 맞는지 확인하고 (맞음 여부, 읽을 이유)를 돌려줌"""
    fp = corpus_fingerprint(corpus_path)
    meta = read_meta(out_prefix)

    if len(ids) != len(offsets):
        return False, f"논문 번호 {len(ids):,}개와 offset {len(offsets):,}개의 수가 다르다"

    if "corpus_size" in meta:
        if meta["corpus_size"] != fp["corpus_size"]:
            return False, (f"코퍼스 크기가 다름: 색인을 만들 때 {meta['corpus_size']:,}바이트, "
                           f"현재 {fp['corpus_size']:,}바이트")
        if meta.get("corpus_name") not in (None, fp["corpus_name"]):
            return False, f"색인: '{meta['corpus_name']}' 용이지만 '{fp['corpus_name']}'를 받음"
        return True, "식별자 일치"

    if not sample_matches(corpus_path, ids, offsets):
        return False, "표본 확인 실패"
    write_meta(out_prefix, {**meta, **fp})
    return True, "표본 확인 성공"


def scan_corpus(corpus_path: str | Path, out_prefix: str | Path) -> tuple[list[str], np.ndarray]:
    """논문 번호와 각 줄이 시작하는 offset을 기록함. 특정 논문만 꺼내 읽을 때 씀"""
    paths = index_paths(out_prefix)
    if paths["ids"].exists() and paths["offsets"].exists():
        ids = paths["ids"].read_text(encoding="utf-8").splitlines()
        offsets = np.load(paths["offsets"])
        ok, _ = check_pairing(corpus_path, out_prefix, ids, offsets)
        if ok:
            return ids, offsets

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
    write_meta(out_prefix, {**read_meta(out_prefix), **corpus_fingerprint(corpus_path)})
    return ids, off


def build_embeddings(corpus_path: str | Path, out_prefix: str | Path,
                     model_name: str = config.EMBED_MODEL,
                     batch_size: int = 64, chunk_size: int = 4096,
                     max_seq_length: int = 512) -> None:
    """
    코퍼스 전체를 임베딩해 .emb.npy로 저장함
    max_seq_length  제목과 초록을 몇 토큰까지 읽을지. 넘는 부분은 버림
    """
    from sentence_transformers import SentenceTransformer

    paths = index_paths(out_prefix)
    ids, _ = scan_corpus(corpus_path, out_prefix)
    n = len(ids)

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    dim = model.get_sentence_embedding_dimension()

    emb = np.lib.format.open_memmap(paths["emb"], mode="w+", dtype=np.float32, shape=(n, dim))

    with open(corpus_path, "r", encoding="utf-8") as f:
        i = 0
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

    write_meta(out_prefix,
               {"count": n, "dim": int(dim), "model": model_name,
                "max_seq_length": max_seq_length, **corpus_fingerprint(corpus_path)})
    del emb


class LocalDenseRetriever:
    """임베딩한 논문으로부터 의미 검색을 수행하는 검색기"""

    name = "local_dense"

    def __init__(self, corpus_path: str | Path, out_prefix: str | Path,
                 model_name: str = config.EMBED_MODEL, embedder=None,
                 mmap: bool = False, max_seq_length: int = 512,
                 device: str | None = None):

        device = config.EMBED_DEVICE if device is None else device
        paths = index_paths(out_prefix)
        self.corpus_path = Path(corpus_path)
        self.ids = paths["ids"].read_text(encoding="utf-8").splitlines()
        self.offsets = np.load(paths["offsets"])

        ok, why = check_pairing(self.corpus_path, out_prefix, self.ids, self.offsets)
        if not ok:
            raise RuntimeError(
                    f"색인과 코퍼스 불일치: {why}\n"
                    f"  색인: {out_prefix}\n  코퍼스: {self.corpus_path}\n"
                )
        if not sample_matches(self.corpus_path, self.ids, self.offsets):
            raise RuntimeError(
                    "표시는 맞지만 표본 확인 실패. offset이 가리키는 논문이 번호 목록과 다름"
                )

        self.emb = np.load(paths["emb"], mmap_mode="r" if mmap else None)
        if len(self.emb) != len(self.ids):
            raise RuntimeError(
                f"임베딩 {len(self.emb):,}개와 논문 번호 {len(self.ids):,}개의 수가 다르다. "
                f"색인({out_prefix})을 다시 만들 것.")

        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(model_name, device=device)
            embedder.max_seq_length = max_seq_length
        self.embedder = embedder
        self._pos = {pid: i for i, pid in enumerate(self.ids)}

    def encode_query(self, query: str) -> np.ndarray:
        v = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)[0]

    def search(self, query: str, k: int = 100) -> list[ScoredPaper]:
        """질문과 의미가 유사한 논문 상위 k편을 돌려줌"""
        q = self.encode_query(query)
        scores = self._scores(q)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]   # 상위 k편만 추린 뒤 정렬
        top = top[np.argsort(-scores[top])]
        return self._to_papers(top, scores)

    def _scores(self, q: np.ndarray) -> np.ndarray:
        """모든 논문과 코사인 유사도 계산"""
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
                title=row.get("title", "").strip(), abstract=row.get("abstract", "").strip()
            ))
        return out

    def read_rows(self, positions: list[int]) -> list[dict]:
        """배열 위치에 해당하는 논문 본문을 파일에서 가져옴"""
        out = []
        with open(self.corpus_path, "rb") as f:
            for i in positions:
                f.seek(int(self.offsets[i]))
                out.append(json.loads(f.readline()))
        return out

    def get_by_ids(self, paper_ids: list[str]) -> dict[str, dict]:
        """논문 번호로 본문 가져옴"""
        pos = [(pid, self._pos[pid]) for pid in paper_ids if pid in self._pos]
        rows = self.read_rows([p for _, p in pos])
        return {pid: row for (pid, _), row in zip(pos, rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--out", default=str(config.DATA_DIR / "embeddings" / "cs2021-ft"))
    ap.add_argument("--model", default=config.EMBED_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--max-seq-length", type=int, default=512)
    args = ap.parse_args()

    build_embeddings(args.corpus, args.out, args.model, args.batch_size,
                     args.chunk_size, args.max_seq_length)


if __name__ == "__main__":
    main()
