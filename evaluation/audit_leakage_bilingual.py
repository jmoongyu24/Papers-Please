"""평가셋 누수 재감사 — 한국어 문항까지 처음으로 실제로 검사한다 (API 비용 0원).

## 왜 필요한가

기존 누수 검사 두 곳은 **한국어 문항에 대해 한 번도 작동한 적이 없다.**

    evaluation/audit_dataset.py:48   content_words()  = re.findall(r"[A-Za-z][A-Za-z0-9\\-]+", ...)
    evaluation/filter_queries.py:77  leakage_count()  = tokenize(query.text) ∩ 영어 전문 용어

둘 다 영어 낱말만 센다. 한국어 질문에서 뽑은 낱말 집합은 영어 논문 제목·용어 집합과
**항상 공집합**이므로, 겹침이 늘 0 으로 나온다. 즉 "한국어 문항은 깨끗하다"가 아니라
**"검사한 적이 없다"** 가 맞는 말이다.

실제로 이런 문항이 "겹침 0.024, 깨끗함"으로 통과해 있다.

    질문(한국어) "직접 음성 번역을 위한 순환 피드백을 활용한 자동 음성 인식 및 기계 번역 개선"
    정답 논문 제목  "Cascaded Models With Cyclic Feedback For Direct Speech Translation"

제목의 완전한 한국어 번역이다. 다국어 임베딩 모델(bge-m3)은 번역을 그대로 꿰뚫으므로,
이런 문항은 검색기에게 정답을 알려주고 푸는 것과 같다.

## 어떻게 푸는가

한국어 질문을 영어로 옮긴 뒤 영어끼리 비교한다. 번역은 **로컬 Qwen3-4B(Ollama)** 로 한다.

- 왜 로컬로 충분한가: 우리가 재는 것은 번역 품질이 아니라 **낱말이 겹치는지 여부**다.
  "순환 피드백" 이 "cyclic feedback" 으로만 나오면 목적을 다한다.
- 왜 이 모델을 써도 되는가: Qwen3-4B 는 쿼리 변환기로도 쓰지만, 여기서는 **검색 파이프라인
  바깥에서 감사만** 한다. 번역 결과가 검색에 들어가지 않으므로 자기참조 누수가 아니다.

## 무엇을 보고하는가

임계값 하나로 "누수/정상"을 가르지 않고 **겹침 구간별로 층화해서** 성능을 함께 보여준다.
임계값을 정하는 순간 그 값을 유리하게 고르고 싶은 유혹이 생기기 때문이다(사후 해석 방지).

실행:
  python -m evaluation.audit_leakage_bilingual --queries data/eval/test.jsonl \\
      --run runs/test300_two_channel.jsonl --out runs/leakage_audit_test300.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src import config
from src.retrieval.fusion import normalize_paper_id
from src.utils import read_jsonl, write_jsonl

from evaluation.audit_dataset import ACRONYM_RE, content_words

TRANSLATE_SYSTEM = (
    "You translate Korean academic search queries into English.\n"
    "Translate literally and completely. Keep every technical noun. Do not add, remove, "
    "or generalize any term. Do not explain. Output only the English sentence."
)

OLLAMA_HOST = "http://localhost:11434"


def ollama_json(model: str, system: str, user: str, timeout: int = 120) -> dict:
    """Ollama HTTP API 를 직접 부른다.

    `ollama` 파이썬 패키지를 안 쓰는 이유: 감사 도구는 의존성이 적을수록 좋다. 이 파일은
    "지금 이 평가셋을 믿어도 되는가"를 판정하는 도구라서, 나중에 다른 사람이 다른 환경에서
    그대로 돌려 볼 수 있어야 한다. 표준 라이브러리만 쓰면 그게 보장된다.
    """
    import urllib.request

    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "format": {"type": "object", "properties": {"english": {"type": "string"}},
                   "required": ["english"]},
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return json.loads(payload["message"]["content"])


def translate_batch(texts: list[str], model: str, cache_path: Path) -> dict[str, str]:
    """한국어 질문을 영어로 옮긴다. 이미 옮긴 것은 캐시에서 꺼내 쓴다.

    캐시를 두는 이유: 감사를 여러 번 돌리게 되는데, 번역은 매번 같은 결과여야
    비교가 성립한다. 그리고 300문항 번역에 몇 분이 걸린다.
    """
    cache: dict[str, str] = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            cache[row["ko"]] = row["en"]

    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if not todo:
        print(f"번역 캐시에서 전부 찾았다 ({len(cache)}건 보유)")
        return cache

    print(f"번역 시작: {len(todo)}건 (모델 {model}, 로컬 Ollama · 비용 0원)", flush=True)

    t0 = time.time()
    for i, text in enumerate(todo, 1):
        try:
            cache[text] = str(ollama_json(model, TRANSLATE_SYSTEM, text)
                              .get("english", "")).strip()
        except Exception as e:
            print(f"  번역 실패({i}): {e}")
            cache[text] = ""
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}초)", flush=True)

    write_jsonl(cache_path, [{"ko": k, "en": v} for k, v in cache.items()])
    print(f"번역 완료: {len(todo)}건, {time.time()-t0:.0f}초 → 캐시 {cache_path}")
    return cache


def title_overlap(query_en: str, title: str) -> float:
    """질문이 정답 논문 제목의 내용어를 얼마나 담고 있는가 (0~1).

    분모를 제목으로 잡는다. "제목의 몇 %가 질문에 이미 들어 있는가" 가 알고 싶은 것이므로.
    """
    t = content_words(title)
    if not t:
        return 0.0
    return len(content_words(query_en) & t) / len(t)


def method_name_hits(query_en: str, title: str, abstract: str) -> list[str]:
    """논문이 스스로 붙인 이름(대문자 섞인 토큰)이 질문에 들어왔는가.

    SequenceMatch, CODER 같은 것들이다. 이게 들어오면 사실상 정답을 적어 준 것이다.
    """
    names = {m.lower() for m in ACRONYM_RE.findall(f"{title} {abstract}") if len(m) > 3}
    q = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]+", query_en.lower()))
    return sorted(names & q)


BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]


def band_of(x: float) -> str:
    for lo, hi in BANDS:
        if lo <= x < hi:
            return f"{lo:.1f}~{hi if hi <= 1 else 1.0:.1f}"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description="평가셋 누수 재감사 (한국어 포함, API 비용 0원)")
    ap.add_argument("--queries", default="data/eval/test.jsonl")
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-v1.jsonl"))
    ap.add_argument("--run", default=None,
                    help="실행 결과(선택). 주면 겹침 구간별 Recall@10 을 함께 낸다")
    ap.add_argument("--model", default=config.REWRITER_MODEL)
    ap.add_argument("--cache", default="data/cache/ko_en_queries.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = list(read_jsonl(args.queries))
    gold_ids = {normalize_paper_id(r["gold_id"]) for r in rows}

    papers: dict[str, dict] = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            pid = normalize_paper_id(str(p["id"]))
            if pid in gold_ids:
                papers[pid] = p
    print(f"정답 논문 {len(papers)}/{len(gold_ids)}편을 코퍼스에서 찾았다")

    ko = [r["text"] for r in rows if r.get("lang") == "ko"]
    trans = translate_batch(ko, args.model, Path(args.cache)) if ko else {}

    # 실행 결과가 있으면 문항별 성공 여부를 붙인다
    hit: dict[str, float] = {}
    if args.run:
        for r in read_jsonl(args.run):
            if r.get("_meta"):
                continue
            ids = r.get("reranked_ids") or []
            hit[r["query_id"]] = 1.0 if normalize_paper_id(r["gold_id"]) in ids[:10] else 0.0

    out = []
    for r in rows:
        pid = normalize_paper_id(r["gold_id"])
        p = papers.get(pid)
        if not p:
            continue
        q_en = trans.get(r["text"], "") if r.get("lang") == "ko" else r["text"]
        ov = title_overlap(q_en, p["title"])
        out.append({
            "query_id": r["query_id"], "lang": r.get("lang"),
            "difficulty": r.get("difficulty"), "gold_id": pid,
            "text": r["text"], "text_en": q_en, "title": p["title"],
            "title_overlap": round(ov, 3), "band": band_of(ov),
            "method_names": method_name_hits(q_en, p["title"], p.get("abstract", "")),
            "hit10": hit.get(r["query_id"]),
        })

    report(out, bool(hit))
    if args.out:
        write_jsonl(args.out, out)
        print(f"\n문항별 결과 → {args.out}")


def report(out: list[dict], has_hits: bool) -> None:
    import numpy as np

    # 코드가 붙인 난이도 이름은 설계 의도와 뒤집혀 있다(ISSUE #32). 읽는 사람이
    # 헷갈리지 않도록 '누가 물었는가'를 함께 적는다.
    WHO = {"easy": "일상어·초심자", "mid": "어설픈 전문어·학부연구생",
           "hard": "정확한 전문어·대학원생"}

    print("\n" + "=" * 78)
    print("■ 정답 논문 제목과의 겹침 (한국어는 영어로 옮긴 뒤 비교)")
    print(f"\n{'난이도':<8}{'질문 유형':<26}{'언어':<6}{'n':>5}{'겹침 중앙값':>12}"
          f"{'≥0.4 비율':>11}{'방법명 포함':>11}" + (f"{'Recall@10':>11}" if has_hits else ""))
    for d in ("easy", "mid", "hard"):
        for lang in ("en", "ko"):
            g = [r for r in out if r["difficulty"] == d and r["lang"] == lang]
            if not g:
                continue
            ovs = [r["title_overlap"] for r in g]
            leak = sum(1 for r in g if r["title_overlap"] >= 0.4) / len(g)
            meth = sum(1 for r in g if r["method_names"]) / len(g)
            line = (f"{d:<8}{WHO[d]:<26}{lang:<6}{len(g):>5}{np.median(ovs):>12.3f}"
                    f"{leak:>10.1%}{meth:>11.1%}")
            if has_hits:
                hs = [r["hit10"] for r in g if r["hit10"] is not None]
                line += f"{np.mean(hs):>11.3f}" if hs else f"{'—':>11}"
            print(line)

    if not has_hits:
        return

    print("\n■ 겹침 구간별 Recall@10 — 누수가 성능을 얼마나 떠받치는가")
    print("   (임계값 하나로 자르지 않고 층으로 보여준다. 사후에 유리한 값을 고르지 못하게)")
    print(f"\n{'겹침 구간':<12}{'n':>5}{'Recall@10':>12}   문항 구성")
    for lo, hi in BANDS:
        b = band_of(lo)
        g = [r for r in out if r["band"] == b and r["hit10"] is not None]
        if not g:
            continue
        comp = {}
        for r in g:
            comp[r["difficulty"]] = comp.get(r["difficulty"], 0) + 1
        desc = " · ".join(f"{k} {v}" for k, v in sorted(comp.items()))
        print(f"{b:<12}{len(g):>5}{np.mean([r['hit10'] for r in g]):>12.3f}   {desc}")

    clean = [r for r in out if r["title_overlap"] < 0.4 and not r["method_names"]
             and r["hit10"] is not None]
    allr = [r for r in out if r["hit10"] is not None]
    print(f"\n■ 누수를 걷어낸 값 (겹침 <0.4 이고 방법명도 없는 문항만)")
    print(f"   전체        n={len(allr):<5} Recall@10 = {np.mean([r['hit10'] for r in allr]):.3f}")
    print(f"   누수 제외    n={len(clean):<5} Recall@10 = {np.mean([r['hit10'] for r in clean]):.3f}")
    print(f"   → 걸러진 문항 {len(allr) - len(clean)}건 ({1 - len(clean)/max(len(allr),1):.1%})")


if __name__ == "__main__":
    main()
