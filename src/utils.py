"""여러 모듈이 공통으로 쓰는 자잘한 도구 함수들.

지금은 주로 JSON Lines(줄마다 JSON 하나) 파일을 읽고 쓰는 함수를 담음.
평가셋, 코퍼스, 실험 결과를 전부 이 형식으로 저장하므로 한곳에 모아둠.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """JSON Lines 파일을 한 줄씩 읽어 딕셔너리로 돌려줌."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """딕셔너리들을 JSON Lines 파일로 저장함. 저장한 줄 수를 돌려줌."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
