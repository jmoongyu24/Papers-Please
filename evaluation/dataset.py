"""
평가용 데이터셋 만들기 - 질문 생성, 등급 정답지 만들기 진행

    generate --> split --> pool --> grade --> (data/eval에 최종 4개 파일)

'data/eval/': 평가에 사용하는 데이터셋 저장
    - dev.jsonl, test.jsonl, grades_dev.jsonl, grades_test.jsonl

난이도 - 사용자가 특정 도메인에 대해 얼마나 아는가를 기준으로 함
    easy    대학원생      정확한 학술 용어를 씀
    medium  학부 연구생   범용 전문어와 일상어가 섞임
    hard    1~2학년      전문 용어를 못 쓰고 일상어로 두루뭉술하게 씀
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from src import config
from src.retrieval.corpus import normalize_paper_id
from src.utils import read_jsonl, write_jsonl


STOPWORDS = set("""
a an the of for and or in on at with to from using via is are be that this these those
what how can i my we you it its their there here does do done new novel study studies
method methods approach approaches based model models technique techniques
무엇 무엇인가 무엇인가요 어떻게 방법 방법은 위한 대한 있는 어떤 연구 논문 알고 싶어요
있을까요 있나요 궁금해요 그리고 또는 이런 그런 하는 되는
""".split())

ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")

_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
_LOWER_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")


def content_words(s: str) -> set[str]:
    """
    흔한 말과 2글자 이하는 제거
    """
    return {w for w in _ASCII_WORD.findall(s.lower())
            if w not in STOPWORDS and len(w) > 2}


def title_overlap(query_en: str, title: str) -> float:
    """
    제목의 키워드 중 얼마나 질문에 해당 키워드가 들어 있는지 계산함 (0~1)
    """
    t = content_words(title)
    return len(content_words(query_en) & t) / len(t) if t else 0.0


def method_name_hits(query_en: str, title: str, abstract: str) -> list[str]:
    """
    논문이 스스로 붙인 이름(대문자 섞인 토큰)이 질문에 있는지 확인함
    예를 들면 SequenceMatch, CODER, ACON 같은 것들. 이게 들어오면 사실상 정답을 적어 준 것
    """
    names = {m.lower() for m in ACRONYM_RE.findall(f"{title} {abstract}") if len(m) > 3}
    q = set(_ASCII_WORD.findall(query_en.lower()))
    return sorted(names & q)


def longest_copy(query_en: str, abstract: str) -> int:
    """초록에서 몇 단어까지 연속으로 일치하는지 계산"""
    def norm(s: str) -> list[str]:
        return _LOWER_WORD.findall(s.lower())

    a, q = norm(abstract), norm(query_en)
    aset = {" ".join(a[i:i + n]) for n in range(3, 9) for i in range(len(a) - n + 1)}
    best = 0
    for n in range(3, min(9, len(q) + 1)):
        for i in range(len(q) - n + 1):
            if " ".join(q[i:i + n]) in aset:
                best = max(best, n)
    return best


BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]


def band_of(x: float) -> str:
    for lo, hi in BANDS:
        if lo <= x < hi:
            return f"{lo:.1f}~{hi if hi <= 1 else 1.0:.1f}"
    return "?"


WHO = {"easy": "대학원생, 정확한 학술어", "medium": "학부연구생, 섞인 표현",
       "hard": "1~2학년, 일상어"}


PERSONAS = {
    "easy": (
        "이 분야 대학원생이다. 관련 논문을 여러 편 읽어 봤고, 그 분야에서 통용되는 정확한 학술 용어를 사용한다."
        "다만 이 논문은 아직 못 찾았다 이 논문이 존재하는지 모르고, 논문 저자들이 자기 방법에 붙인 이름은 모른다."
        "아는 것은 '이런 걸 다루는 연구가 있을 것 같다' 정도다. "
        "이 사람은 검색창에 8단어 이내로 짧게 친다. 자기에게 지금 필요한 논문을 한두 가지 학술 용어로 적는다."
    ),
    "medium": (
        "학부 연구생이다. 이 분야를 배우기 시작한 지 얼마 안 됐다. 널리 쓰이는 범용 전문 용어는 몇 개 알지만"
        "정확한 이름은 모르고, 일상어와 전문어가 섞인 다소 어설픈 문장을 쓴다. 이 논문의 존재는 모른다."
    ),
    "hard": (
        "학부 1~2학년이다. 이 분야를 거의 모른다. 전문 용어를 하나도 떠올리지 못하고, 겪고 있는 문제나 하고 싶은 일을"
        "일상어로 두루뭉술하게 표현한다. '이런 걸 어떻게 하는지 궁금하다' 정도의 말투다. 이 논문의 존재는 당연히 모른다."
    ),
}

GEN_SYSTEM = (
    "너는 학술 논문 검색 시스템의 평가 데이터를 만드는 전문가다. "
    "주어진 지시를 정확히 지키고, 요구된 JSON 형식으로만 답한다."
)

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "ko": {"type": "string"},
        "en": {"type": "string"},
        "knows": {"type": "array", "items": {"type": "string"}},
        "why_level": {"type": "string"},
    },
    "required": ["ko", "en", "knows", "why_level"],
    "additionalProperties": False,
}


def build_gen_prompt(abstract: str, difficulty: str) -> str:
    """초록만 주고 검색어를 만들게 함"""
    return f"""
    아래는 어떤 논문의 초록이다. 제목은 일부러 주지 않는다.

    [묻는 사람]
    {PERSONAS[difficulty]}

    이 사람이 검색창에 입력할 법한 검색어를 한국어와 영어로 하나씩 만들어라.

    [반드시 지킬 것]
    1. 초록의 표현을 5단어 이상 연속으로 그대로 옮기지 마라. 반드시 이 사람의 말로 바꿔 써라.
    2. 각각 한 문장, 20어절 이내로 쓴다.
    3. 한국어와 영어는 번역투가 아니라 같은 사람이 각 언어로 자연스럽게 물었을 때의 문장이어야 한다.
    뜻은 같아야 하지만 표현은 각 언어에서 자연스러운 쪽을 고른다.
    4. 위 [묻는 사람]의 전문성 수준을 정확히 지켜라. 이것이 이 데이터의 핵심이다. 대학원생이라면 정확한 학술 용어를, 1~2학년이라면 전문 용어를 하나도 쓰지 마라.
    5. 검색어는 논문 요약이 아니다. 이 사람은 초록을 본 적이 없다. 초록에 나온 요소를 여러 개 이어 붙이면 그것은 논문 요약이지 검색어가 아니다. 실제 사람은 그 요소들이 한 논문에 다 들어 있는지 모르는 상태에서 검색한다.
    -> 초록의 요소 중 두 가지 이하만 골라 쓴다. 나머지는 모르는 것으로 친다.

    [초록]
    {abstract}

    아래 JSON 형식으로만 답하라:
    {{"ko": "한국어 검색어", "en": "English query",
    "knows": ["이 사람이 안다고 가정한 것 1~3개"],
    "why_level": "왜 이 문장이 그 전문성 수준에 맞는지 한 문장"}}"""


def first_category(paper: dict) -> str:
    """
    논문의 첫 번째 분야. 분야별로 뽑을 때 사용
    """
    cats = paper.get("categories") or []
    if isinstance(cats, str):
        cats = cats.split()
    return cats[0] if cats else "?"


def sample_papers(corpus: str, n: int, exclude: set[str], seed: int,
                  min_abs_words: int = 80, sampling: str = "stratified") -> list[dict]:
    """
    코퍼스에서 논문을 뽑음
    sampling
      stratified   분야별로 고르게 뽑음
      proportional 코퍼스에 있는 분야 비율 그대로 뽑음
    """
    pool = []
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = normalize_paper_id(str(r["id"]))
            if pid in exclude:
                continue
            if len(r.get("abstract", "").split()) < min_abs_words:
                continue
            pool.append({"id": pid, "title": r["title"], "abstract": r["abstract"],
                         "categories": r.get("categories", "")})

    rng = random.Random(seed)
    if sampling == "proportional":
        out = rng.sample(pool, min(n, len(pool)))
    else:
        by_cat: dict[str, list[dict]] = {}
        for p in pool:
            by_cat.setdefault(first_category(p), []).append(p)
        out, cats = [], sorted(by_cat, key=lambda c: -len(by_cat[c]))
        i = 0
        while len(out) < n and cats:
            c = cats[i % len(cats)]
            if by_cat[c]:
                out.append(by_cat[c].pop(rng.randrange(len(by_cat[c]))))
            else:
                cats.remove(c)
                continue
            i += 1
        rng.shuffle(out)

    return out


def call_openai(client, model: str, prompt: str, effort: str = "") -> tuple[dict, dict]:
    """
    한 번 호출해 JSON과 토큰 사용량을 돌려줌.
    """
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": prompt}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "query", "strict": True, "schema": GEN_SCHEMA}}
    )
    if effort:
        kwargs["reasoning_effort"] = effort
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        if "unsupported" not in str(e).lower() and "unrecognized" not in str(e).lower():
            raise
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)

    u = resp.usage
    usage = {"in": u.prompt_tokens, "out": u.completion_tokens,
             "reasoning": getattr(getattr(u, "completion_tokens_details", None),
                                  "reasoning_tokens", 0) or 0}
    text = resp.choices[0].message.content or "{}"
    return json.loads(text), usage


def cmd_generate(args) -> None:
    papers = sample_papers(args.corpus, args.n_papers, set(), args.seed,
                           sampling=args.sampling)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    todo = [(p, d) for p in papers for d in args.difficulties]
    total_usage = {"in": 0, "out": 0, "reasoning": 0}

    for i, (p, d) in enumerate(todo, 1):
        try:
            got, usage = call_openai(client, args.model,
                                     build_gen_prompt(p["abstract"], d), args.effort)
        except Exception:
            continue
        for k in total_usage:
            total_usage[k] += usage[k]

        for lang, text in (("ko", got.get("ko", "")), ("en", got.get("en", ""))):
            rows.append({
                "query_id": f"v2-{p['id']}-{d}-{lang}",
                "text": text.strip(), "gold_id": p["id"],
                "lang": lang, "difficulty": d,
                "pair_id": f"v2-{p['id']}-{d}",
                "knows": got.get("knows", []), "why_level": got.get("why_level", ""),
                "gen_model": args.model,
                "title_overlap": round(title_overlap(got.get("en", ""), p["title"]), 3),
                "abstract_copy_words": longest_copy(got.get("en", ""), p["abstract"]),
                "_title": p["title"], "_categories": p["categories"],
            })

        if i % 10 == 0 or i == len(todo):
            write_jsonl(out_path, rows)

    meta = {"_meta": {"produced_by": "evaluation.dataset generate",
                      "model": args.model, "n_papers": len(papers),
                      "sampling": args.sampling, "effort": args.effort,
                      "difficulties": args.difficulties, "seed": args.seed,
                      "time": time.strftime("%Y-%m-%d %H:%M:%S"), "usage": total_usage}}
    write_jsonl(out_path, [meta] + rows)


MAX_ABSTRACT_COPY = 5
TEST_RATIO = 0.5
PER_LANG = 15


def drop_reasons(rows_of_paper: list[dict]) -> dict[str, str]:
    """논문별 탈락 사유"""
    bad: dict[str, str] = {}
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows_of_paper:
        by_pair[r["pair_id"]].append(r)

    for rs in by_pair.values():
        langs = {r["lang"] for r in rs}
        why = None
        if langs != {"ko", "en"}:
            why = "짝 없음(한 언어만 생성됨)"
        elif any(not r["text"].strip() for r in rs):
            why = "빈 검색어"
        elif any(r.get("abstract_copy_words", 0) >= MAX_ABSTRACT_COPY for r in rs):
            why = f"초록 {MAX_ABSTRACT_COPY}단어 이상 그대로 복사"
        if why:
            for r in rs:
                bad[r["query_id"]] = why
    return bad


def cmd_split(args) -> None:
    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_paper[r["gold_id"]].append(r)

    dropped: dict[str, str] = {}
    for rs in by_paper.values():
        dropped.update(drop_reasons(rs))
    kept = [r for r in rows if r["query_id"] not in dropped]

    by_paper = defaultdict(list)
    for r in kept:
        by_paper[r["gold_id"]].append(r)
    n_full = len({r["difficulty"] for r in kept}) * 2
    full = {p: rs for p, rs in by_paper.items() if len(rs) == n_full}

    papers = sorted(full)
    random.Random(args.seed).shuffle(papers)
    n_test = int(len(papers) * TEST_RATIO)
    test_papers, dev_papers = set(papers[:n_test]), set(papers[n_test:])
    assert not (dev_papers & test_papers), "논문이 양쪽에 들어갔다"

    def collect(ps: set[str]) -> list[dict]:
        return [r for p in sorted(ps) for r in sorted(full[p], key=lambda x: x["query_id"])]

    dev, test = collect(dev_papers), collect(test_papers)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "test.jsonl", test)


def pool_from_runs(run_paths: list[str], by_pair: dict[str, dict],
                   depth: int, stage: str = "rerank") -> dict[str, list[str]]:
    """
    실행 결과 파일들에서 각 짝의 상위 depth편을 추림
    """
    from evaluation.pipeline_eval import fused_ids_of
    qid_to_pair = {}
    for pair, langs in by_pair.items():
        for r in langs.values():
            qid_to_pair[r["query_id"]] = pair

    tally: dict[str, dict[str, list]] = defaultdict(dict)
    for fp in run_paths:
        n = 0
        for r in read_jsonl(fp):
            if r.get("_meta") or not r.get("query_id"):
                continue
            pair = qid_to_pair.get(r["query_id"])
            if pair is None:
                continue
            lists = []
            if stage in ("rerank", "both"):
                lists.append(r.get("reranked_ids") or [])
            if stage in ("fused", "both"):
                lists.append(fused_ids_of(r, rrf_k=60, top_n=depth, weights={}))
            for ids in lists:
                for rank, pid in enumerate(
                        [normalize_paper_id(x) for x in ids][:depth], 1):
                    v = tally[pair].setdefault(pid, [0, 0])
                    v[0] += 1
                    v[1] += rank
            n += 1

    pool: dict[str, list[str]] = {}
    for pair, papers in tally.items():
        pool[pair] = sorted(papers, key=lambda d: (-papers[d][0], papers[d][1]))
    return pool


def cmd_pool(args) -> None:
    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    by_pair: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_pair[r["pair_id"]][r["lang"]] = r

    if args.from_runs:
        from src.retrieval.local_index import LocalDenseRetriever
        picked = pool_from_runs(args.from_runs, by_pair, args.from_runs_depth,
                                args.from_runs_stage)
        ret = LocalDenseRetriever(args.corpus, args.index, embedder=False, mmap=True)

        out = []
        for pair, langs in sorted(by_pair.items()):
            gold = normalize_paper_id(next(iter(langs.values()))["gold_id"])
            ids = picked.get(pair, [])

            if args.depth:
                ids = ids[: args.depth]
            if gold not in ids:
                ids = ids + [gold]
            bodies = ret.get_by_ids(ids)
            cands = [{"paper_id": pid,
                      "title": (bodies.get(pid) or {}).get("title", ""),
                      "abstract": (bodies.get(pid) or {}).get("abstract", ""),
                      "found_by": ["run"]}
                     for pid in ids]
            out.append({
                "pair_id": pair, "gold_id": gold,
                "query_en": langs.get("en", {}).get("text", ""),
                "query_ko": langs.get("ko", {}).get("text", ""),
                "difficulty": next(iter(langs.values()))["difficulty"],
                "candidates": cands,
            })
        _report_pool(out, args.out)
        return

    from src.retrieval.local_index import LocalDenseRetriever
    ret = LocalDenseRetriever(args.corpus, args.index)

    for i, (pair, langs) in enumerate(sorted(by_pair.items()), 1):
        gold = normalize_paper_id(next(iter(langs.values()))["gold_id"])
        seen: dict[str, dict] = {}
        for lang, r in sorted(langs.items()):
            for p in ret.search(r["text"], k=PER_LANG):
                pid = normalize_paper_id(p.paper_id)
                if pid not in seen:
                    seen[pid] = {"paper_id": pid, "title": p.title,
                                 "abstract": p.abstract, "found_by": [lang]}
                elif lang not in seen[pid]["found_by"]:
                    seen[pid]["found_by"].append(lang)

        cands = list(seen.values())[: args.depth]
        if gold not in seen:
            cands.append({"paper_id": gold, "title": "", "abstract": "", "found_by": []})

        out.append({
            "pair_id": pair, "gold_id": gold,
            "query_en": langs.get("en", {}).get("text", ""),
            "query_ko": langs.get("ko", {}).get("text", ""),
            "difficulty": next(iter(langs.values()))["difficulty"],
            "candidates": cands,
        })

    _report_pool(out, args.out)


def _report_pool(out: list[dict], out_path: str) -> None:
    """후보 풀을 파일로 저장함"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, out)


PRICE = {"in": 0.40, "out": 1.60}

GRADE_SCHEMA = {
    "type": "object",
    "properties": {"grade": {"type": "integer", "enum": [0, 1, 2, 3]}},
    "required": ["grade"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "너는 학술 검색 결과의 관련도를 매기는 평가자다. 사용자의 원래 검색 의도와 후보 논문을 보고, "
    "이 논문이 그 사용자를 얼마나 만족시킬지 0~3 등급으로 판정하라.\n"
    "3 = 질문 주제를 정면으로 다룸, 사용자가 매우 만족\n"
    "2 = 주제에 맞고 쓸모 있음\n"
    "1 = 주변적으로만 관련\n"
    "0 = 무관\n"
    "논문 제목과 초록의 실제 내용만 보고 판단하라. 검색어에 단어가 겹치는지가 아니라, "
    "이 논문을 읽으면 사용자가 알고 싶던 것을 알 수 있는지를 기준으로 하라."
)


class Judge:
    """관련도 등급 판정기"""

    def __init__(self, model: str):
        from openai import OpenAI
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 없음 (data/API_KEY.env 확인)")
        self.model = model
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.usage = {"in": 0, "out": 0}

    def judge(self, query: str, title: str, abstract: str) -> int:
        prompt = (f"[사용자 검색어] {query}\n\n"
                  f"[후보 논문 제목] {title}\n"
                  f"[후보 논문 초록] {abstract[:1500]}\n\n"
                  '관련도 등급을 JSON으로 답하라. 형식: {"grade": 0~3 정수}')
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "grade", "strict": True,
                                             "schema": GRADE_SCHEMA}}
        )
        self.usage["in"] += resp.usage.prompt_tokens
        self.usage["out"] += resp.usage.completion_tokens
        g = int(json.loads(resp.choices[0].message.content).get("grade", 0))
        return max(0, min(3, g))

    def cost(self) -> float:
        return (self.usage["in"] * PRICE["in"] + self.usage["out"] * PRICE["out"]) / 1e6


def cmd_grade(args) -> None:
    pool = list(read_jsonl(args.pool))

    out_path = Path(args.out)
    qrels: dict[str, dict[str, int]] = {}
    judge = Judge(args.model)
    stopped = False

    def save() -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            for pid, g in qrels.items():
                f.write(json.dumps({"pair_id": pid, "grades": g}, ensure_ascii=False) + "\n")

    for i, row in enumerate(pool, 1):
        grades = qrels.setdefault(row["pair_id"], {})
        grades[row["gold_id"]] = 3
        query = row["query_en"] or row["query_ko"]
        for c in row["candidates"]:
            if c["paper_id"] == row["gold_id"] or c["paper_id"] in grades:
                continue
            if not c["title"]:
                continue
            try:
                grades[c["paper_id"]] = judge.judge(query, c["title"], c["abstract"])
            except Exception:
                continue
            if args.max_cost and judge.cost() >= args.max_cost:
                stopped = True
                break
        if stopped:
            break
        if i % 20 == 0:
            save()

    save()


TRANSLATE_SYSTEM = (
    "You translate Korean academic search queries into English.\n"
    "Translate literally and completely. Keep every technical noun. Do not add, remove, "
    "or generalize any term. Do not explain. Output only the English sentence."
)

OLLAMA_HOST = "http://localhost:11434"
TRANSLATE_CACHE = config.DATA_DIR / "cache" / "ko_en_queries.jsonl"


def ollama_json(model: str, system: str, user: str, timeout: int = 120) -> dict:
    """Ollama HTTP API 호출"""
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
    """한국어 질문을 영어로 옮김"""
    cache: dict[str, str] = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            cache[row["ko"]] = row["en"]

    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if not todo:
        return cache

    for text in todo:
        try:
            cache[text] = str(ollama_json(model, TRANSLATE_SYSTEM, text)
                              .get("english", "")).strip()
        except Exception:
            cache[text] = ""     # 번역 실패는 빈 문자열로 두고 넘어감

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache_path, [{"ko": k, "en": v} for k, v in cache.items()])
    return cache


def _gold_papers(rows: list[dict], corpus: str | None) -> dict[str, dict]:
    """
    정답 논문의 제목, 초록을 찾음
    """
    have = {normalize_paper_id(r["gold_id"]): {"title": r["_title"], "abstract": ""}
            for r in rows if r.get("_title")}
    want = {normalize_paper_id(r["gold_id"]) for r in rows} - set(have)
    if want and corpus and Path(corpus).exists():
        with open(corpus, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = normalize_paper_id(str(p["id"]))
                if pid in want:
                    have[pid] = {"title": p["title"], "abstract": p.get("abstract", "")}
    return have


def cmd_audit(args) -> None:
    import numpy as np

    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    papers = _gold_papers(rows, args.corpus)

    ko = [r["text"] for r in rows if r.get("lang") == "ko"]
    trans = translate_batch(ko, args.model, TRANSLATE_CACHE) if ko else {}

    hit: dict[str, float] = {}
    if args.run:
        for r in read_jsonl(args.run):
            if r.get("_meta"):
                continue
            ids = r.get("reranked_ids") or r.get("fused_ids") or []
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

    has_hits = bool(hit)
    print("\n" + "=" * 78)
    print("## 정답 논문 제목과의 겹침 (한국어는 영어로 옮긴 뒤 비교)")
    print(f"\n{'난이도':<12}{'질문 유형':<24}{'언어':<6}{'n':>5}{'겹침 중앙값':>12}"
          f"{'≥0.4 비율':>11}{'방법명 포함':>11}" + (f"{'Recall@10':>11}" if has_hits else ""))
    for d in sorted({r["difficulty"] for r in out}, key=lambda x: list(WHO).index(x)
                    if x in WHO else 99):
        for lang in ("en", "ko"):
            g = [r for r in out if r["difficulty"] == d and r["lang"] == lang]
            if not g:
                continue
            ovs = [r["title_overlap"] for r in g]
            leak = sum(1 for r in g if r["title_overlap"] >= 0.4) / len(g)
            meth = sum(1 for r in g if r["method_names"]) / len(g)
            line = (f"{d:<12}{WHO.get(d, '?'):<24}{lang:<6}{len(g):>5}"
                    f"{statistics.median(ovs):>12.3f}{leak:>10.1%}{meth:>11.1%}")
            if has_hits:
                hs = [r["hit10"] for r in g if r["hit10"] is not None]
                line += f"{np.mean(hs):>11.3f}" if hs else f"{'-':>11}"
            print(line)

    if has_hits:
        print("\n## 겹침 구간별 Recall@10")
        print(f"\n{'겹침 구간':<12}{'n':>5}{'Recall@10':>12} 문항 구성")
        for lo, _hi in BANDS:
            b = band_of(lo)
            g = [r for r in out if r["band"] == b and r["hit10"] is not None]
            if not g:
                continue
            comp = Counter(r["difficulty"] for r in g)
            desc = ", ".join(f"{k} {v}" for k, v in sorted(comp.items()))
            print(f"{b:<12}{len(g):>5}{np.mean([r['hit10'] for r in g]):>12.3f}   {desc}")

        clean = [r for r in out if r["title_overlap"] < 0.4 and not r["method_names"]
                 and r["hit10"] is not None]
        allr = [r for r in out if r["hit10"] is not None]
        print("\n## 누수를 걷어낸 값")
        print(f"   전체        n={len(allr):<5} Recall@10 = {np.mean([r['hit10'] for r in allr]):.3f}")
        print(f"   누수 제외    n={len(clean):<5} Recall@10 = {np.mean([r['hit10'] for r in clean]):.3f}")
        print(f"   걸러진 문항 {len(allr) - len(clean)}건 "
              f"({1 - len(clean)/max(len(allr),1):.1%})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out, out)

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    g.add_argument("--n-papers", type=int, default=200)
    g.add_argument("--model", default="gpt-5.4")
    g.add_argument("--difficulties", nargs="*", default=["easy", "medium", "hard"])
    g.add_argument("--sampling", choices=["stratified", "proportional"], default="stratified")
    g.add_argument("--effort", default="")
    g.add_argument("--seed", type=int, default=20260813)
    g.add_argument("--out", default="runs/queries_raw.jsonl")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("split")
    s.add_argument("--queries", default="runs/queries_raw.jsonl")
    s.add_argument("--out-dir", default="data/eval")
    s.add_argument("--seed", type=int, default=20260813)
    s.set_defaults(func=cmd_split)

    p = sub.add_parser("pool")
    p.add_argument("--queries", required=True)
    p.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    p.add_argument("--index", default=str(config.DATA_DIR / "embeddings" / "cs2021-ft"))
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--from-runs", nargs="*", default=[])
    p.add_argument("--from-runs-depth", type=int, default=10)
    p.add_argument("--from-runs-stage", choices=["rerank", "fused", "both"], default="rerank")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_pool)

    j = sub.add_parser("grade")
    j.add_argument("--pool", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--model", default="gpt-4.1-mini")
    j.add_argument("--max-cost", type=float, default=None)
    j.set_defaults(func=cmd_grade)

    a = sub.add_parser("audit")
    a.add_argument("--queries", default="data/eval/test.jsonl")
    a.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    a.add_argument("--run", default=None)
    a.add_argument("--model", default=config.REWRITER_MODEL)
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_audit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
