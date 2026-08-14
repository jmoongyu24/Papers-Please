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


# ── 색인과 코퍼스가 짝이 맞는지 확인 ──────────────────────────────────────
#
# 줄 위치표는 **그 코퍼스 파일 전용**이다. 다른 파일(또는 같은 이름으로 다시 만든 파일)에
# 갖다 쓰면 `seek` 이 엉뚱한 줄에 떨어지는데, 그래도 JSON 파싱은 성공한다. 즉 **오류가 안 나고
# 결과도 그럴듯해 보이는 채로 다른 논문의 제목과 초록을 돌려준다.** 서비스에서 이게 나면
# 사용자에게 존재하지 않는 조합의 논문 정보를 보여주게 된다.
#
# 파일 이름 비교만으로는 부족하다. 실제 사고 시나리오가 "코퍼스를 같은 이름으로 다시 만드는
# 것"이기 때문이다. 그래서 크기·수정시각까지 지문으로 남기고, 마지막에 표본까지 확인한다.

def corpus_fingerprint(corpus_path: str | Path) -> dict:
    """코퍼스 파일의 지문. stat() 한 번이라 비용이 없다."""
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
    """위치표대로 몇 편 꺼내 읽어, 정말 그 논문이 나오는지 본다 (디스크 이동 n번)."""
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
    """짝이 맞는가. (맞음 여부, 사람이 읽을 이유) 를 돌려준다."""
    fp = corpus_fingerprint(corpus_path)
    meta = read_meta(out_prefix)

    if len(ids) != len(offsets):
        return False, f"논문 번호 {len(ids):,}개와 줄 위치 {len(offsets):,}개의 수가 다르다"

    if "corpus_size" in meta:                      # 지문이 있으면 그것으로 판정한다
        if meta["corpus_size"] != fp["corpus_size"]:
            return False, (f"코퍼스 크기가 다르다: 색인을 만들 때 {meta['corpus_size']:,}바이트, "
                           f"지금 {fp['corpus_size']:,}바이트")
        if meta.get("corpus_name") not in (None, fp["corpus_name"]):
            return False, f"색인은 '{meta['corpus_name']}' 용인데 '{fp['corpus_name']}' 를 받았다"
        return True, "지문 일치"

    # 지문이 없는 옛 색인. 다시 만들면 세 시간이 걸리므로, 표본을 읽어 확인하고 통과하면
    # 지문을 채워 넣는다(다음부터는 즉시 판정된다).
    if not sample_matches(corpus_path, ids, offsets):
        return False, "표본 확인 실패 — 위치표가 가리키는 논문이 번호 목록과 다르다"
    write_meta(out_prefix, {**meta, **fp})
    return True, "표본 확인 통과 (지문을 새로 기록했다)"


# ── 1단계: 논문 번호와 줄 위치 기록 ────────────────────────────────────────
def scan_corpus(corpus_path: str | Path, out_prefix: str | Path) -> tuple[list[str], np.ndarray]:
    """논문 번호와 각 줄의 시작 위치를 기록한다 (seek 한 번으로 특정 논문만 읽기 위함)."""
    paths = index_paths(out_prefix)
    if paths["ids"].exists() and paths["offsets"].exists():
        ids = paths["ids"].read_text(encoding="utf-8").splitlines()
        offsets = np.load(paths["offsets"])
        ok, why = check_pairing(corpus_path, out_prefix, ids, offsets)
        if ok:
            return ids, offsets
        # 저장해 둔 것이 이 코퍼스 것이 아니다. 그대로 쓰면 임베딩과 본문이 어긋난다.
        print(f"경고: 저장된 줄 위치 색인이 지금 코퍼스와 짝이 맞지 않는다 ({why}). 다시 훑는다.")

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
            # 지문을 함께 남긴다. 이게 없으면 나중에 이 색인이 어느 코퍼스 것인지 알 수 없다.
            write_meta(out_prefix,
                       {"count": n, "dim": int(dim), "model": model_name, "done": i,
                        "max_seq_length": max_seq_length, **corpus_fingerprint(corpus_path)})

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

        # 짝이 안 맞으면 **경고가 아니라 여기서 멈춘다.** 조용히 다른 논문의 제목과 초록을
        # 사용자에게 보여주는 것보다, 검색이 아예 안 뜨는 편이 낫다.
        ok, why = check_pairing(self.corpus_path, out_prefix, self.ids, self.offsets)
        if not ok:
            raise RuntimeError(
                f"색인과 코퍼스가 짝이 맞지 않는다: {why}\n"
                f"  색인: {out_prefix}\n  코퍼스: {self.corpus_path}\n"
                f"이대로 쓰면 오류 없이 엉뚱한 논문의 제목·초록을 돌려준다. "
                f"코퍼스를 원래 파일로 되돌리거나 색인을 다시 만들 것.")
        if not sample_matches(self.corpus_path, self.ids, self.offsets):
            raise RuntimeError(
                f"지문은 맞는데 표본 확인에 실패했다 — 위치표가 가리키는 논문이 번호 목록과 "
                f"다르다. 색인({out_prefix})을 다시 만들 것.")

        self.emb = np.load(paths["emb"], mmap_mode="r" if mmap else None)
        if len(self.emb) != len(self.ids):
            raise RuntimeError(
                f"임베딩 {len(self.emb):,}개와 논문 번호 {len(self.ids):,}개의 수가 다르다. "
                f"색인({out_prefix})을 다시 만들 것.")

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
