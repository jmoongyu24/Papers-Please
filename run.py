"""
설치 준비와 점검용 코드

    python run.py checklist     실행에 필요한 조건을 하나씩 확인해 보여줌
    python run.py init          빠진 것을 받아 채운 뒤 다시 점검함
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CORPUS = ROOT / "data" / "corpus" / "corpus-cs2021.jsonl"
INDEX_DIR = ROOT / "data" / "embeddings"
INDEX_PREFIX = INDEX_DIR / "cs2021-ft"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:4b"

HF_CORPUS = "GreenBed4725/arxiv-cs2021-corpus"
HF_INDEX = "GreenBed4725/arxiv-cs2021-embeddings-bge-m3"
HF_RETRIEVER = "GreenBed4725/bge-m3-arxiv-cs-retriever"

NEED_DISK_GB = 8

PACKAGES = ["streamlit", "arxiv", "numpy", "ollama",
            "sentence_transformers", "transformers", "torch"]


def _symbols() -> tuple[str, str, str]:
    """터미널이 못 그리는 글자면 대괄호 표시로 바꿈"""
    try:
        "✓✗–".encode(sys.stdout.encoding or "utf-8")
        return "✓", "✗", "–"
    except (UnicodeEncodeError, LookupError):
        return "[O]", "[X]", "[-]"


OK, NO, SKIP = _symbols()


def _width(text: str) -> int:
    """터미널에서 차지하는 칸 수. 한글과 한자는 두 칸을 씀"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, cols: int) -> str:
    return text + " " * max(0, cols - _width(text))


class Row:
    """점검 한 줄"""

    def __init__(self, name: str, state: str, detail: str = "", fix: str = ""):
        self.name = name
        self.state = state          # "ok" | "no" | "option"
        self.detail = detail
        self.fix = fix

    @property
    def mark(self) -> str:
        return {"ok": OK, "no": NO, "option": SKIP}[self.state]


# ── 점검 항목 ─────────────────────────────────────────────────────────────

def check_python() -> Row:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        return Row("파이썬 3.11 이상", "ok", got)
    return Row("파이썬 3.11 이상", "no", got, "파이썬 3.11 이상을 설치하십시오")


def check_packages() -> Row:
    import importlib.util
    missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    if not missing:
        return Row("파이썬 패키지", "ok", f"{len(PACKAGES)}개 확인")
    return Row("파이썬 패키지", "no", f"빠짐: {', '.join(missing)}",
               "pip install -r requirements.txt")


def check_disk() -> Row:
    free = shutil.disk_usage(ROOT).free / 1e9
    if free >= NEED_DISK_GB:
        return Row(f"디스크 여유 {NEED_DISK_GB}GB", "ok", f"{free:.1f}GB 남음")
    return Row(f"디스크 여유 {NEED_DISK_GB}GB", "no", f"{free:.1f}GB 남음",
               "공간을 확보한 뒤 다시 시도하십시오")


def ollama_tags() -> list[str] | None:
    """Ollama에 올라와 있는 모델 이름. 데몬이 없으면 None"""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return None


def check_ollama(tags: list[str] | None) -> Row:
    if tags is None:
        return Row("Ollama 데몬", "no", f"{OLLAMA_HOST} 응답 없음", "ollama serve")
    return Row("Ollama 데몬", "ok", f"모델 {len(tags)}개 등록됨")


def check_ollama_model(tags: list[str] | None, name: str) -> Row:
    label = f"Ollama {name}"
    if tags is None:
        return Row(label, "no", "데몬을 먼저 실행하십시오",
                   "ollama serve를 실행한 뒤 다시 확인하십시오")
    if any(t == name or t.startswith(name + ":") for t in tags):
        return Row(label, "ok", "등록됨")
    return Row(label, "no", "등록되지 않음", f"ollama pull {name}")


def check_corpus() -> Row:
    """있는지만 봅니다. 색인과 맞는지는 check_pairing_row가 봅니다"""
    if not CORPUS.exists():
        return Row("논문 코퍼스", "no", "없음", "python run.py init")
    return Row("논문 코퍼스", "ok", f"{CORPUS.stat().st_size / 1e9:.2f}GB")


def index_files() -> dict[str, Path]:
    p = INDEX_PREFIX
    return {"ids": p.with_suffix(".ids.txt"), "offsets": p.with_suffix(".offsets.npy"),
            "emb": p.with_suffix(".emb.npy"), "meta": p.with_suffix(".meta.json")}


def check_index() -> Row:
    files = index_files()
    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        return Row("의미 검색 색인", "no", f"빠진 파일: {', '.join(missing)}",
                   "python run.py init")
    try:
        meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Row("의미 검색 색인", "no", "meta.json을 읽지 못했습니다",
                   "python run.py init")
    gb = files["emb"].stat().st_size / 1e9
    return Row("의미 검색 색인", "ok", f"{meta.get('count', 0):,}편, {gb:.2f}GB")


def check_pairing_row() -> Row:
    files = index_files()
    if not (CORPUS.exists() and files["meta"].exists()):
        return Row("색인과 코퍼스의 짝", "no", "둘 다 있어야 확인합니다")
    try:
        meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Row("색인과 코퍼스의 짝", "no", "meta.json을 읽지 못했습니다")
    if meta.get("corpus_size") != CORPUS.stat().st_size:
        return Row("색인과 코퍼스의 짝", "no", "코퍼스 크기가 색인과 다릅니다",
                   "코퍼스와 색인을 같은 쌍으로 다시 받으십시오")
    if meta.get("corpus_name") not in (None, CORPUS.name):
        return Row("색인과 코퍼스의 짝", "no",
                   f"색인은 '{meta['corpus_name']}' 용입니다")
    return Row("색인과 코퍼스의 짝", "ok", "식별자 일치")


def check_embed_model() -> Row:
    files = index_files()
    if not files["meta"].exists():
        return Row("질문 임베딩 모델", "no", "색인을 먼저 준비하십시오")
    try:
        name = json.loads(files["meta"].read_text(encoding="utf-8")).get("model", "")
    except (OSError, ValueError):
        return Row("질문 임베딩 모델", "no", "meta.json을 읽지 못했습니다")
    if not name:
        return Row("질문 임베딩 모델", "no", "색인에 모델 이름이 없습니다")
    if Path(name).exists():
        return Row("질문 임베딩 모델", "ok", f"{name} (로컬)")
    from importlib.util import find_spec
    if find_spec("huggingface_hub") is None:
        return Row("질문 임베딩 모델", "option", f"{name} — 첫 실행 때 내려받습니다")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(name, local_files_only=True)
        return Row("질문 임베딩 모델", "ok", f"{name}")
    except Exception:
        return Row("질문 임베딩 모델", "option",
                   f"{name} — 첫 실행 때 약 1.2GB를 내려받습니다")


def collect() -> list[Row]:
    """실행에 반드시 필요한 것만 확인합니다"""
    tags = ollama_tags()
    return [
        check_python(),
        check_packages(),
        check_disk(),
        check_ollama(tags),
        check_ollama_model(tags, OLLAMA_MODEL),
        check_corpus(),
        check_index(),
        check_pairing_row(),
        check_embed_model(),
    ]


def cmd_checklist(_args) -> int:
    rows = collect()
    width = max(_width(r.name) for r in rows) + 2
    line = "  " + "-" * (width + 46)

    print()
    print("  실행 조건 확인")
    print(line)
    for r in rows:
        print(f"  {r.mark}  {_pad(r.name, width)}{r.detail}")
    print(line)

    blocked = [r for r in rows if r.state == "no"]
    if not blocked:
        print("\n  모두 준비되었습니다. streamlit run app.py로 실행하십시오.\n")
        return 0

    print(f"\n  준비되지 않은 항목 {len(blocked)}개\n")
    for r in blocked:
        print(f"  {NO} {r.name}")
        if r.fix:
            print(f"      {r.fix}")
    print()
    return 1


# ── init ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> bool:
    print(f"    $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def _download(repo: str, dest: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("    huggingface_hub가 없습니다. pip install huggingface_hub를 먼저 하십시오")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    print(f"    {repo} -> {dest}")
    snapshot_download(repo, repo_type="dataset", local_dir=str(dest))
    return True


def cmd_init(args) -> int:
    print("\n  준비를 시작합니다\n")

    for row in (check_python(), check_packages(), check_disk()):
        if row.state == "no":
            print(f"  {NO} {row.name}: {row.detail}")
            if row.fix:
                print(f"      {row.fix}")
            print("\n  위 조건을 갖춘 뒤 다시 실행하십시오.\n")
            return 1
        print(f"  {OK} {row.name}  {row.detail}")

    if not args.skip_ollama:
        print("\n  [1/3] Ollama 모델")
        if ollama_tags() is None:
            print(f"    {NO} 데몬이 응답하지 않습니다. 다른 터미널에서 ollama serve를 실행하십시오")
        else:
            tags = ollama_tags() or []
            if any(t.startswith(OLLAMA_MODEL) for t in tags):
                print(f"    {OK} {OLLAMA_MODEL} 이미 있습니다")
            elif not _run(["ollama", "pull", OLLAMA_MODEL]):
                print(f"    {NO} 다운로드에 실패했습니다")

    if not args.skip_download:
        print("\n  [2/3] 논문 코퍼스")
        if check_corpus().state == "ok":
            print(f"    {OK} 이미 있습니다")
        else:
            _download(HF_CORPUS, CORPUS.parent)

        print("\n  [3/3] 의미 검색 색인")
        if check_index().state == "ok" and check_pairing_row().state == "ok":
            print(f"    {OK} 이미 있습니다")
        else:
            _download(HF_INDEX, INDEX_DIR)

    print("\n  준비를 마쳤습니다. 다시 확인합니다.")
    return cmd_checklist(args)

def main() -> int:
    ap = argparse.ArgumentParser(description="Papers, Please 설치 준비 및 확인")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("checklist", help="실행 조건을 확인합니다")
    c.set_defaults(func=cmd_checklist)

    i = sub.add_parser("init", help="설치가 안 된 부분도 마저 설치합니다")
    i.add_argument("--skip-ollama", action="store_true")
    i.add_argument("--skip-download", action="store_true")
    i.set_defaults(func=cmd_init)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
