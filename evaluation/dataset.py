"""평가셋 만들기 - 질문 생성부터 등급 정답지까지, 그리고 만든 것을 검사하는 일까지.

    generate --> split --> pool --> grade --> (data/eval 에 최종 4개 파일)
       유료        무료      무료      유료
                                          audit  (언제든, 무료)

    python -m evaluation.dataset generate --n-papers 200 --out runs/queries_raw.jsonl
    python -m evaluation.dataset split    --queries runs/queries_raw.jsonl
    python -m evaluation.dataset pool     --queries data/eval/dev.jsonl --out runs/pool_dev.jsonl
    python -m evaluation.dataset grade    --pool runs/pool_dev.jsonl --out data/eval/grades_dev.jsonl
    python -m evaluation.dataset audit    --queries data/eval/test.jsonl --run runs/test_run.jsonl

중간 산출물은 `runs/` 에 둠. `data/eval/` 에는 실제로 평가에 쓰는 네 개만 남김 -
`dev.jsonl`, `test.jsonl`, `grades_dev.jsonl`, `grades_test.jsonl`.

## 누수를 막는 장치 세 가지

1. 제목을 모델에게 주지 않음. 초록만 주고 "이 논문을 아직 못 찾은 사람" 의 자리에서
   질문을 쓰게 함. 제목을 주고 어순만 바꾸게 했더니 대학원생 층 Recall@10 이 1.000
   이었음. 원인은 희귀 용어가 아니라 제목의 낱말 조합을 그대로 재현한 것이었음
2. 한 번의 호출로 한국어와 영어를 함께 만듦. 그러면 한국어 문항에도 영어판이 항상
   있어 모든 누수 검사가 영어 위에서 돌아감. 옛 검사기는 영어 낱말만 세어서 한국어
   문항은 검사받은 적이 없었음
3. 금지 어휘 목록을 두지 않음. 실제 사용자는 희귀한 논문도 찾기 때문임. 대신 문항마다
   제목 겹침과 희귀 용어 사용을 기록해 두고 성능을 그 값으로 나눠 보고함

## 난이도 축 - 표현 거리가 아니라 묻는 사람이 얼마나 아는가

    easy        대학원생        정확한 학술 용어를 씀
    medium      학부 연구생      범용 전문어와 일상어가 섞임
    hard        1~2학년         전문 용어를 못 쓰고 일상어로 두루뭉술하게 씀
    known_item  논문을 아는 사람  특정 논문을 지목해 찾음. 탐색형과 섞지 않고 따로 잼

"제목과 표현이 얼마나 다른가" 를 축으로 쓰면 의미 검색에서는 축이 뒤집힘. 전문 용어가
정확할수록 뜻으로 찾기가 쉽기 때문임. 지금 축은 검색 방식과 무관함.

## 언어를 짝짓는 이유

같은 논문, 같은 난이도를 두 언어로 만들어 다른 조건을 모두 고정함. `pair_id` 가 그 짝을
잇는 열쇠임. 옛 평가셋은 한국어 질문과 영어 질문이 서로 다른 논문에 대한 것이라
"한국어가 영어보다 낫다" 를 말할 수 없었음.
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


# ==========================================================================
# 공통 - 낱말 겹침을 재는 도구
# ==========================================================================

# 질문/제목 양쪽에서 빼고 볼 흔한 말들 (겹침 비율을 부풀리지 않게)
STOPWORDS = set("""
a an the of for and or in on at with to from using via is are be that this these those
what how can i my we you it its their there here does do done new novel study studies
method methods approach approaches based model models technique techniques
무엇 무엇인가 무엇인가요 어떻게 방법 방법은 위한 대한 있는 어떤 연구 논문 알고 싶어요
있을까요 있나요 궁금해요 그리고 또는 이런 그런 하는 되는
""".split())

# 대문자가 2개 이상 섞인 토큰 = 논문이 제안한 방법 이름인 경우가 많음 (SequenceMatch, CODER...)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")

_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
_LOWER_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")


def content_words(s: str) -> set[str]:
    """겹침을 잴 때 세는 낱말만 남김 (흔한 말과 2글자 이하는 뺌).

    영어 낱말만 셈. 한국어 질문에 그대로 쓰면 항상 공집합이 나와 "깨끗하다"고
    잘못 판정함. 한국어는 반드시 `audit` 이 영어로 옮긴 뒤에 넣을 것.
    """
    return {w for w in _ASCII_WORD.findall(s.lower())
            if w not in STOPWORDS and len(w) > 2}


def title_overlap(query_en: str, title: str) -> float:
    """제목 내용어 중 몇 %가 질문에 이미 들어 있는가 (0~1).

    분모를 제목으로 잡음. "제목의 몇 %가 질문에 이미 들어 있는가" 가 알고 싶은 것이므로.
    거르는 데 쓰지 않고 기록만 함 (위 설계 설명 참고).
    """
    t = content_words(title)
    return len(content_words(query_en) & t) / len(t) if t else 0.0


def method_name_hits(query_en: str, title: str, abstract: str) -> list[str]:
    """논문이 스스로 붙인 이름(대문자 섞인 토큰)이 질문에 들어왔는가.

    SequenceMatch, CODER 같은 것들임. 이게 들어오면 사실상 정답을 적어 준 것임.
    """
    names = {m.lower() for m in ACRONYM_RE.findall(f"{title} {abstract}") if len(m) > 3}
    q = set(_ASCII_WORD.findall(query_en.lower()))
    return sorted(names & q)


def longest_copy(query_en: str, abstract: str) -> int:
    """초록에서 몇 낱말까지 연속으로 그대로 옮겼는가."""
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


# 난이도 이름만 봐서는 누가 물었는지 알 수 없으므로 표에 함께 적음
WHO = {"easy": "대학원생, 정확한 학술어", "medium": "학부연구생, 섞인 표현",
       "hard": "1~2학년, 일상어", "known_item": "논문을 아는 사람"}


# ==========================================================================
# 1단계. generate - 논문에서 질문을 거꾸로 만듦 (유료, OpenAI)
# ==========================================================================

# 난이도 = 묻는 사람이 그 분야를 얼마나 아는가
PERSONAS = {
    "easy": (
        "이 분야 대학원생이다. 관련 논문을 여러 편 읽어 봤고, 그 분야에서 통용되는 정확한 "
        "학술 용어를 사용한다. 다만 **이 논문은 아직 못 찾았다** 이 논문이 존재하는지 "
        "모르고, 논문 저자들이 자기 방법에 붙인 이름은 모른다. 아는 것은 '이런 걸 다루는 연구가 "
        "있을 것 같다' 정도다. 이 사람은 **검색창에 8낱말 이내로 짧게 친다.** 자기에게 지금 필요한 "
        "논문을 **한두 가지** 학술 용어로 적는다."
    ),
    "medium": (
        "학부 연구생이다. 이 분야를 배우기 시작한 지 얼마 안 됐다. 널리 쓰이는 범용 전문 "
        "용어는 몇 개 알지만 정확한 이름은 모르고, 일상어와 전문어가 섞인 다소 어설픈 문장을 "
        "쓴다. 이 논문의 존재는 모른다."
    ),
    "hard": (
        "학부 1~2학년이다. 이 분야를 거의 모른다. 전문 용어를 하나도 떠올리지 못하고, "
        "**겪고 있는 현상이나 하고 싶은 일을 일상어로 두루뭉술하게** 표현한다. "
        "'이런 걸 어떻게 하는지 궁금하다' 정도의 말투다. 이 논문의 존재는 당연히 모른다."
    ),
    "known_item": (
        "이 논문을 어디선가 보거나 들어서 **알고 있는 사람**이다. 다시 찾으려고 검색한다. "
        "기억나는 특징(방법 이름, 다루는 문제, 결과 등)을 짚어 검색한다. "
        "이 층은 '주제로 논문을 발견하는 능력'이 아니라 '아는 논문을 다시 찾는 능력'을 재므로, "
        "다른 층과 점수를 섞지 않고 따로 본다."
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
    """초록만 주고 검색어를 만들게 함. 제목은 절대 주지 않음."""
    return f"""아래는 어떤 논문의 초록이다. 제목은 일부러 주지 않는다.

[묻는 사람]
{PERSONAS[difficulty]}

이 사람이 검색창에 입력할 법한 검색어를 한국어와 영어로 하나씩 만들어라.

[반드시 지킬 것]
1. 초록의 표현을 5낱말 이상 연속으로 그대로 옮기지 마라. 반드시 이 사람의 말로 바꿔 써라.
2. 각각 한 문장, 20어절 이내로 쓴다.
3. 한국어와 영어는 **번역투가 아니라** 같은 사람이 각 언어로 자연스럽게 물었을 때의 문장이어야
   한다. 뜻은 같아야 하지만 표현은 각 언어에서 자연스러운 쪽을 고른다.
4. 위 [묻는 사람]의 전문성 수준을 정확히 지켜라. 이것이 이 데이터의 핵심이다.
   대학원생이라면 정확한 학술 용어를, 1~2학년이라면 전문 용어를 하나도 쓰지 마라.
5. **검색어는 논문 요약이 아니다.** 이 사람은 초록을 본 적이 없다. 초록에 나온 요소를
   여러 개 이어 붙이면 그것은 논문 요약이지 검색어가 아니다. 실제 사람은 그 요소들이
   한 논문에 다 들어 있는지 모르는 상태에서 검색한다.
   → **초록의 요소 중 두 가지 이하만** 골라 쓴다. 나머지는 모르는 것으로 친다.

[초록]
{abstract}

아래 JSON 형식으로만 답하라:
{{"ko": "한국어 검색어", "en": "English query",
  "knows": ["이 사람이 안다고 가정한 것 1~3개"],
  "why_level": "왜 이 문장이 그 전문성 수준에 맞는지 한 문장"}}"""


def first_category(paper: dict) -> str:
    """논문의 첫 번째 분야. 분야별로 뽑을 때 층으로 씀.

    `categories` 는 리스트임(`['math.NA', 'cs.NA']`). 예전에는 `str(...).split()[0]` 로
    뽑았는데, 그러면 `"['math.NA',"` 와 `"['math.NA']"` 가 서로 다른 층이 됐음 - 분야가
    하나뿐인 논문과 여럿인 논문이 갈려서, 같은 분야가 두 층이 되고 두 배로 뽑혔음.
    코퍼스 71만 편에서 층이 156종이 아니라 197종으로 세어졌음.
    """
    cats = paper.get("categories") or []
    if isinstance(cats, str):
        cats = cats.split()
    return cats[0] if cats else "?"


def sample_papers(corpus: str, n: int, exclude: set[str], seed: int,
                  min_abs_words: int = 80, sampling: str = "stratified") -> list[dict]:
    """코퍼스에서 논문을 뽑음. 이미 평가에 쓰인 정답 논문은 제외함.

    제외하는 이유: 같은 논문에서 옛 질문과 새 질문이 나오면 둘이 쌍둥이가 되어, 옛
    평가셋으로 고른 설정이 새 평가셋에서도 유리해짐. 새로 만드는 뜻이 없어짐.

    초록이 너무 짧은 논문은 뺌 - 질문을 만들 재료가 부족해 층이 구분되지 않음.

    sampling
      stratified   분야마다 돌아가며 고르게 뽑음. 평가셋(v2)을 만든 방식임.
                   작은 분야도 큰 분야와 같은 수만큼 나옴.
      proportional 코퍼스에 있는 분야 비율 그대로 뽑음. 실제 검색 상황과 같은 분포임.
                   다만 평가셋은 stratified 로 만들어져 있어 분포가 서로 다름 -
                   코퍼스의 45.9%인 cs.CV·cs.LG·cs.CL·cs.AI 가 개발용 평가셋에서는
                   2.9%(348문항 중 10개)뿐임.
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
    print(f"후보 논문 {len(pool):,}편 (제외 {len(exclude)}편, 초록 {min_abs_words}낱말 미만 제외)")

    rng = random.Random(seed)
    if sampling == "proportional":
        # 코퍼스 비율 그대로. 그냥 무작위로 뽑으면 비율이 저절로 맞음.
        out = rng.sample(pool, min(n, len(pool)))
    else:
        # 분야가 한쪽으로 쏠리지 않게 첫 번째 분야 기준으로 층화 추출함
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

    got = Counter(first_category(p) for p in out)
    print(f"{sampling} 추출: {len(out)}편, 분야 {len(got)}종")
    print("   많은 분야: " + ", ".join(f"{k} {v}편({v/max(len(out),1):.1%})"
                                    for k, v in got.most_common(6)))
    return out


def call_openai(client, model: str, prompt: str, effort: str = "") -> tuple[dict, dict]:
    """한 번 호출해 JSON 과 토큰 사용량을 돌려줌.

    effort: gpt-5 계열의 추론 강도. 안 주면 medium 이 걸림. 실측(2026-08-18, gpt-5-mini,
    같은 프롬프트 20회)으로 minimal 은 추론 토큰 0, medium 은 1,382개였고 값이 6.9배
    차이 났음. 질문 품질은 눈으로 봐서 차이를 못 찾았음.
    """
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": prompt}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "query", "strict": True, "schema": GEN_SCHEMA}},
    )
    if effort:
        kwargs["reasoning_effort"] = effort
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # 추론 계열 모델은 일부 인자를 안 받음. 최소 인자로 한 번 더 시도함.
        if "unsupported" not in str(e).lower() and "unrecognized" not in str(e).lower():
            raise
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)

    u = resp.usage
    usage = {"in": u.prompt_tokens, "out": u.completion_tokens,
             # 추론 토큰은 눈에 안 보이지만 출력 요금으로 붙음. 예산이 여기서 무너짐.
             "reasoning": getattr(getattr(u, "completion_tokens_details", None),
                                  "reasoning_tokens", 0) or 0}
    text = resp.choices[0].message.content or "{}"
    return json.loads(text), usage


def cmd_generate(args) -> None:
    exclude = set()
    for p in args.exclude_from:
        if Path(p).exists():
            exclude |= {normalize_paper_id(r["gold_id"]) for r in read_jsonl(p)
                        if not r.get("_meta")}

    papers = sample_papers(args.corpus, args.n_papers, exclude, args.seed,
                           sampling=args.sampling)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if args.resume and out_path.exists():
        rows = [r for r in read_jsonl(out_path) if not r.get("_meta")]
        done = {(r["gold_id"], r["difficulty"]) for r in rows}
        print(f"이어하기: 이미 {len(done)}건 완료")

    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    todo = [(p, d) for p in papers for d in args.difficulties if (p["id"], d) not in done]
    total_usage = {"in": 0, "out": 0, "reasoning": 0}
    print(f"생성 시작: {len(todo)}회 호출, 모델 {args.model} "
          f"(한 번에 한국어, 영어 둘 다 생성 -> 문항 {len(todo)*2}개)\n", flush=True)

    t0 = time.time()
    for i, (p, d) in enumerate(todo, 1):
        try:
            got, usage = call_openai(client, args.model,
                                     build_gen_prompt(p["abstract"], d), args.effort)
        except Exception as e:
            print(f"  실패 {p['id']}/{d}: {e}")
            continue
        for k in total_usage:
            total_usage[k] += usage[k]

        for lang, text in (("ko", got.get("ko", "")), ("en", got.get("en", ""))):
            rows.append({
                "query_id": f"v2-{p['id']}-{d}-{lang}",
                "text": text.strip(), "gold_id": p["id"],
                "lang": lang, "difficulty": d,
                "pair_id": f"v2-{p['id']}-{d}",     # 같은 짝의 두 언어를 잇는 열쇠
                "knows": got.get("knows", []), "why_level": got.get("why_level", ""),
                "gen_model": args.model,
                # 거르지 않고 기록만 함. 나중에 원하는 기준으로 층화해 계산할 수 있게.
                "title_overlap": round(title_overlap(got.get("en", ""), p["title"]), 3),
                "abstract_copy_words": longest_copy(got.get("en", ""), p["abstract"]),
                "_title": p["title"], "_categories": p["categories"],
            })

        if i % 10 == 0 or i == len(todo):
            write_jsonl(out_path, rows)
            print(f"  {i}/{len(todo)}, {time.time()-t0:.0f}초, "
                  f"토큰 in {total_usage['in']:,} / out {total_usage['out']:,} "
                  f"(추론 {total_usage['reasoning']:,})", flush=True)

    meta = {"_meta": {"produced_by": "evaluation.dataset generate",
                      "model": args.model, "n_papers": len(papers),
                      "sampling": args.sampling, "effort": args.effort,
                      "difficulties": args.difficulties, "seed": args.seed,
                      "time": time.strftime("%Y-%m-%d %H:%M:%S"), "usage": total_usage}}
    write_jsonl(out_path, [meta] + rows)
    print(f"\n완료: 문항 {len(rows)}개 -> {out_path}")
    print(f"토큰: 입력 {total_usage['in']:,}, 출력 {total_usage['out']:,} "
          f"(그중 추론 {total_usage['reasoning']:,})")


# ==========================================================================
# 2단계. split - 개발용(dev)과 시험용(test)으로 나눔 (무료)
# ==========================================================================
#
# 질문이 아니라 정답 논문을 기준으로 나눔. 같은 논문에서 나온 질문이 양쪽에 흩어지면
# 개발하면서 시험 논문을 미리 보는 셈이 되어 성적이 부풀려짐. 논문 단위로 나누면
# 난이도와 언어는 저절로 균형이 맞음.
#
# 거름: 초록을 5낱말 이상 그대로 옮긴 문항, 한국어나 영어가 비었거나 짝이 깨진 문항.
# 거르지 않고 기록만 함: 제목 겹침. 대학원생이 정확한 용어로 물으면 겹치는 것이 정상이고,
# 여기서 임계값을 정하면 나중에 유리한 값을 고르고 싶은 유혹이 생김. 대신 분포를 남겨
# 두고 성능을 겹침 구간별로 나눠 보고함(`audit`).

MAX_ABSTRACT_COPY = 5      # 초록을 이만큼 연속으로 옮겼으면 사람이 쓴 검색어가 아님


def drop_reasons(rows_of_paper: list[dict]) -> dict[str, str]:
    """문항별 탈락 사유. 짝(pair)이 깨지면 그 짝 전체를 뺌."""
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
            why = f"초록 {MAX_ABSTRACT_COPY}낱말 이상 그대로 복사"
        if why:
            for r in rs:
                bad[r["query_id"]] = why
    return bad


def cmd_split(args) -> None:
    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_paper[r["gold_id"]].append(r)
    print(f"불러온 문항 {len(rows)}개, 정답 논문 {len(by_paper)}편")

    # -- 거르기 --
    dropped: dict[str, str] = {}
    for rs in by_paper.values():
        dropped.update(drop_reasons(rs))
    if dropped:
        print(f"\n## 탈락 {len(dropped)}문항")
        for why, n in Counter(dropped.values()).most_common():
            print(f"   {why}: {n}건")
    kept = [r for r in rows if r["query_id"] not in dropped]

    by_paper = defaultdict(list)
    for r in kept:
        by_paper[r["gold_id"]].append(r)
    # 난이도 × 언어 문항이 온전한 논문만 씀 - 층 균형이 저절로 맞음
    n_full = len({r["difficulty"] for r in kept}) * 2
    full = {p: rs for p, rs in by_paper.items() if len(rs) == n_full}
    partial = len(by_paper) - len(full)
    if partial:
        print(f"   문항이 {n_full}개가 안 되는 논문 {partial}편은 제외(층 균형 유지)")

    # -- 논문 단위로 나누기 --
    papers = sorted(full)
    random.Random(args.seed).shuffle(papers)
    n_test = int(len(papers) * args.test_ratio)
    test_papers, dev_papers = set(papers[:n_test]), set(papers[n_test:])
    assert not (dev_papers & test_papers), "논문이 양쪽에 들어갔다"

    def collect(ps: set[str]) -> list[dict]:
        return [r for p in sorted(ps) for r in sorted(full[p], key=lambda x: x["query_id"])]

    dev, test = collect(dev_papers), collect(test_papers)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "test.jsonl", test)

    # -- 보고 --
    for name, rs, ps in (("dev", dev, dev_papers), ("test", test, test_papers)):
        print(f"\n## {name}: 문항 {len(rs)}개, 논문 {len(ps)}편")
        print(f"   난이도 {dict(Counter(r['difficulty'] for r in rs))}")
        print(f"   언어   {dict(Counter(r['lang'] for r in rs))}")
        bands = Counter(band_of(r.get("title_overlap", 0.0)) for r in rs)
        print(f"   제목 겹침 분포 (거르지 않고 기록) {dict(sorted(bands.items()))}")

    print(f"\n-> {out_dir/'dev.jsonl'}, {out_dir/'test.jsonl'}")
    print("\n* 시험용(test)은 설정을 고르는 데 쓰지 않는다. 탐색은 개발용(dev)에서 하고,")
    print("   시험용으로는 확정 판정만 한다. 이 규칙을 어기면 시험용 평가셋이 소모된다.")


# ==========================================================================
# 3단계. pool - 등급을 매길 후보 풀을 만듦 (로컬 검색만, 비용 0원)
# ==========================================================================
#
# 단일 정답 지표는 "질문 하나에 정답 논문 딱 1편" 을 셈. 실제로는 한 질문에 맞는 논문이
# 여러 편이라 다른 좋은 논문을 1등에 올려도 0점을 받음. 그 한계를 풀려면 후보마다
# 관련도를 매긴 등급 정답지가 필요하고, 먼저 판정할 후보를 모아야 함.
#
# 판정은 한 짝(한국어, 영어)당 한 번만 함. 두 문항은 같은 정보 요구를 다른 언어로 쓴
# 것이므로 관련도의 답이 같음. 비용이 절반이 됨. 다만 후보 풀은 두 언어의 검색 결과를
# 합침 - 한쪽만 쓰면 그 언어에 유리한 풀이 되기 때문임.
#
# 원 출처 논문은 자동 3등급으로 두고 판정하지 않음(비용 절약). 정답 논문이 후보에 있는
# 것은 편향이 아님 - 실제 서비스도 그 논문을 결과로 내놓음. 금지되는 것은 정답 논문의
# 글을 읽고 검색어를 만드는 것이지, 검색 결과로 내놓는 것이 아님.

def pool_from_runs(run_paths: list[str], by_pair: dict[str, dict],
                   depth: int, stage: str = "rerank") -> dict[str, list[str]]:
    """실행 결과 파일들에서 각 짝의 상위 `depth` 편을 모음.

    ## 왜 필요한가

    `cmd_pool` 의 기본 방식은 후보를 "원본 질문으로 로컬 의미 검색한 상위 20편" 으로만
    만듦. 그러면 그 20편 밖으로 논문을 끌어올리는 구성일수록, 올린 논문이 실제로 좋아도
    등급이 없어 0점으로 세어져 손해를 봄(`metrics.ndcg_at_k_single` 은 등급 없는 논문을
    0 으로 셈). 그래서 만족도로 구성을 고를 수 없었음.

    비교할 구성들이 실제로 상위 10편에 올린 논문을 전부 판정 대상에 넣으면 그 손해가
    없어짐. 정보 검색에서 쓰는 표준 방식임.

    ## 재정렬 전후를 비교하려면 stage="both" 여야 함

    `reranked_ids` 만 모으면 '재정렬을 켠 구성' 의 상위 10편만 판정 대상이 됨. 그러면
    재정렬을 끈 구성이 올린 논문은 등급이 없어 0점이 되고, 재정렬이 실제보다 좋아 보임.
    stage="both" 는 순위 합치기 직후의 상위 N편도 함께 모아 그 비교를 가능하게 함.

    ## 새 구성을 비교할 때는 다시 돌려야 함

    여기 넣지 않은 구성이 올리는 논문은 여전히 등급이 없음. 파인튜닝한 검색기처럼 새
    구성을 비교하게 되면 그 구성의 상위 10편을 넣어 한 번 더 판정해야 함.
    """
    from evaluation.pipeline_eval import fused_ids_of
    qid_to_pair = {}
    for pair, langs in by_pair.items():
        for r in langs.values():
            qid_to_pair[r["query_id"]] = pair

    # 논문마다 (몇 개의 실행 결과가 상위에 올렸나, 등수 합) 을 셈. --depth 로 자를 때
    # 여러 구성이 함께 올린 논문부터 남기기 위함임 - 한 구성에만 나온 논문을 먼저 자르면
    # 그 구성만 등급을 못 받아 다시 손해를 봄(#42 가 생긴 것과 같은 자리).
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
        print(f"  {Path(fp).name:<28} 문항 {n}개 반영", flush=True)

    pool: dict[str, list[str]] = {}
    for pair, papers in tally.items():
        pool[pair] = sorted(papers, key=lambda d: (-papers[d][0], papers[d][1]))
    return pool


def cmd_pool(args) -> None:
    rows = [r for r in read_jsonl(args.queries) if not r.get("_meta")]
    by_pair: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_pair[r["pair_id"]][r["lang"]] = r
    print(f"문항 {len(rows)}개, 짝 {len(by_pair)}개")

    # 실행 결과에서 후보를 모으는 길. 검색을 새로 하지 않으므로 임베더가 필요 없음.
    if args.from_runs:
        from src.retrieval.local_index import LocalDenseRetriever
        print(f"실행 결과 {len(args.from_runs)}개에서 상위 {args.from_runs_depth}편씩 모음",
              flush=True)
        picked = pool_from_runs(args.from_runs, by_pair, args.from_runs_depth,
                                args.from_runs_stage)
        print("코퍼스 불러오는 중... (짝 확인 포함)", flush=True)
        t0 = time.time()
        # embedder 를 넘겨 주면 SentenceTransformer 를 안 올림. 본문만 꺼내면 되기 때문임.
        ret = LocalDenseRetriever(args.corpus, args.index, embedder=False, mmap=True)
        print(f"코퍼스 준비 완료 ({time.time()-t0:.0f}초)", flush=True)

        out = []
        for pair, langs in sorted(by_pair.items()):
            gold = normalize_paper_id(next(iter(langs.values()))["gold_id"])
            ids = picked.get(pair, [])
            # --depth 0 이면 자르지 않음. 여러 구성이 함께 올린 논문이 앞에 오도록
            # pool_from_runs 가 이미 정렬해 두었으므로 앞에서부터 자르면 됨.
            if args.depth:
                ids = ids[: args.depth]
            if gold not in ids:                    # 정답이 안 걸렸으면 반드시 넣음
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
    print("색인 불러오는 중... (짝 확인 포함)", flush=True)
    t0 = time.time()
    ret = LocalDenseRetriever(args.corpus, args.index)
    print(f"색인 준비 완료 ({time.time()-t0:.0f}초)", flush=True)

    out, t0 = [], time.time()
    for i, (pair, langs) in enumerate(sorted(by_pair.items()), 1):
        gold = normalize_paper_id(next(iter(langs.values()))["gold_id"])
        # 두 언어의 검색 결과를 합침 - 한쪽 언어에만 유리한 풀이 되지 않게
        seen: dict[str, dict] = {}
        for lang, r in sorted(langs.items()):
            for p in ret.search(r["text"], k=args.per_lang):
                pid = normalize_paper_id(p.paper_id)
                if pid not in seen:
                    seen[pid] = {"paper_id": pid, "title": p.title,
                                 "abstract": p.abstract, "found_by": [lang]}
                elif lang not in seen[pid]["found_by"]:
                    seen[pid]["found_by"].append(lang)

        cands = list(seen.values())[: args.depth]
        if gold not in seen:                       # 정답이 안 걸렸으면 반드시 넣음
            cands.append({"paper_id": gold, "title": "", "abstract": "", "found_by": []})

        out.append({
            "pair_id": pair, "gold_id": gold,
            "query_en": langs.get("en", {}).get("text", ""),
            "query_ko": langs.get("ko", {}).get("text", ""),
            "difficulty": next(iter(langs.values()))["difficulty"],
            "candidates": cands,
        })
        if i % 50 == 0:
            print(f"  {i}/{len(by_pair)} ({time.time()-t0:.0f}초)", flush=True)

    _report_pool(out, args.out)


def _report_pool(out: list[dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, out)
    n_cand = sum(len(r["candidates"]) for r in out)
    n_judge = sum(1 for r in out for c in r["candidates"] if c["paper_id"] != r["gold_id"])
    print(f"\n풀 {len(out)}짝, 후보 {n_cand:,}편")
    print(f"판정이 필요한 (질문,논문) 쌍: {n_judge:,}개 (정답 논문은 자동 3등급이라 제외)")
    print(f"예상 비용 (gpt-4.1-mini, 쌍당 약 $0.0002): 약 ${n_judge*0.0002:.2f}")
    print(f"   (이미 판정된 쌍은 grade 가 건너뛰므로 실제 청구액은 이보다 적음)")
    print(f"-> {out_path}")


# ==========================================================================
# 4단계. grade - 후보마다 관련도를 0~3 으로 매김 (유료, OpenAI)
# ==========================================================================
#
# ## 얻는 것 세 가지
#
#   1. nDCG - 지정 정답이 아니어도 좋은 논문을 위에 올리면 점수를 줌
#   2. 이 평가셋으로 도달할 수 있는 값을 새로 계산할 수 있음
#   3. 시스템이 못 찾은 것인지, 잘 찾았는데 지표가 못 알아본 것인지 구분됨
#
# ## 심판을 생성기와 다른 계열로 씀
#
# 질문을 만든 모델과 채점한 모델이 같으면, 자기가 만든 질문에 후하게 줘서 만족 논문 수가
# 부풀려짐. 지금은 생성기 gpt-5.4, 심판 gpt-4.1-mini 로 계열을 갈랐음. gpt-4.1-mini 는
# 비추론 모델이라 출력 토큰이 예측 가능함 - 추론 토큰이 붙으면 예산이 몇 배로 튐.

# gpt-4.1-mini 단가 (1M 토큰당 달러). 다른 모델을 쓰면 이 값을 바꿔야 비용 추정이 맞음.
PRICE = {"in": 0.40, "out": 1.60}

GRADE_SCHEMA = {
    "type": "object",
    "properties": {"grade": {"type": "integer", "enum": [0, 1, 2, 3]}},
    "required": ["grade"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "너는 학술 검색 결과의 관련도를 매기는 평가자다. 사용자의 원래 검색 의도와 후보 논문을 "
    "보고, 이 논문이 그 사용자를 얼마나 만족시킬지 0~3 등급으로 판정하라.\n"
    "3 = 질문 주제를 정면으로 다룸, 사용자가 매우 만족\n"
    "2 = 주제에 맞고 쓸모 있음\n"
    "1 = 주변적으로만 관련\n"
    "0 = 무관\n"
    "논문 제목과 초록의 실제 내용만 보고 판단하라. 검색어에 낱말이 겹치는지가 아니라, "
    "이 논문을 읽으면 사용자가 알고 싶던 것을 알 수 있는지를 기준으로 하라."
)


class Judge:
    """관련도 심판. 토큰 사용량과 누적 비용을 스스로 셈."""

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
                  '관련도 등급을 JSON 으로 답하라. 형식: {"grade": 0~3 정수}')
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "grade", "strict": True,
                                             "schema": GRADE_SCHEMA}},
        )
        self.usage["in"] += resp.usage.prompt_tokens
        self.usage["out"] += resp.usage.completion_tokens
        g = int(json.loads(resp.choices[0].message.content).get("grade", 0))
        return max(0, min(3, g))

    def cost(self) -> float:
        return (self.usage["in"] * PRICE["in"] + self.usage["out"] * PRICE["out"]) / 1e6


def cmd_grade(args) -> None:
    pool = list(read_jsonl(args.pool))
    if args.limit:
        pool = pool[: args.limit]

    # 이어하기
    out_path = Path(args.out)
    qrels: dict[str, dict[str, int]] = {}
    if out_path.exists():
        for r in read_jsonl(out_path):
            qrels[r["pair_id"]] = {k: int(v) for k, v in r["grades"].items()}
        print(f"이어하기: 이미 {len(qrels)}짝 판정됨")

    todo = sum(1 for r in pool for c in r["candidates"]
               if c["paper_id"] != r["gold_id"]
               and c["paper_id"] not in qrels.get(r["pair_id"], {}))
    est = todo * (440 * PRICE["in"] + 15 * PRICE["out"]) / 1e6
    print(f"판정할 (질문, 논문) 쌍 {todo:,}개, 모델 {args.model}, 예상 약 ${est:.2f}")
    if args.max_cost:
        print(f"안전장치: ${args.max_cost:.2f} 를 넘으면 중단한다")

    judge = Judge(args.model)
    n, t0, stopped = 0, time.time(), False

    def save() -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            for pid, g in qrels.items():
                f.write(json.dumps({"pair_id": pid, "grades": g}, ensure_ascii=False) + "\n")

    for i, row in enumerate(pool, 1):
        grades = qrels.setdefault(row["pair_id"], {})
        grades[row["gold_id"]] = 3                     # 출처 논문은 자동 정답급
        # 판정은 영어판으로만 함 (관련도는 언어가 아니라 뜻의 문제)
        query = row["query_en"] or row["query_ko"]
        for c in row["candidates"]:
            if c["paper_id"] == row["gold_id"] or c["paper_id"] in grades:
                continue
            if not c["title"]:
                continue
            try:
                grades[c["paper_id"]] = judge.judge(query, c["title"], c["abstract"])
                n += 1
            except Exception as e:
                print(f"  판정 실패 {row['pair_id']}/{c['paper_id']}: {e}")
            if args.max_cost and judge.cost() >= args.max_cost:
                stopped = True
                break
        if stopped:
            break
        if i % 20 == 0:
            save()
            print(f"  {i}/{len(pool)}짝, 판정 {n:,}건, {time.time()-t0:.0f}초 "
                  f",  누적 ${judge.cost():.3f}", flush=True)

    save()
    print(f"\n{'중단됨(비용 한도 도달)' if stopped else '완료'}: "
          f"짝 {len(qrels)}개, 새로 판정 {n:,}건")
    print(f"토큰 입력 {judge.usage['in']:,} / 출력 {judge.usage['out']:,} "
          f",  실제 비용 ${judge.cost():.3f}")
    print(f"-> {out_path}")


# ==========================================================================
# audit - 만든 평가셋에 정답이 새어 있는지 검사함 (로컬 번역, 비용 0원)
# ==========================================================================
#
# 한국어 질문을 영어로 옮긴 뒤 영어끼리 비교함. 번역은 로컬 Qwen3-4B 로 함. 재는 것이
# 번역 품질이 아니라 낱말이 겹치는지 여부라 로컬로 충분하고, 번역 결과가 검색에 들어가지
# 않으므로 자기참조 누수도 아님.
#
# ## 무엇을 보고하는가
#
# 임계값 하나로 "누수/정상"을 가르지 않고 겹침 구간별로 층화해서 성능을 함께 보여줌.
# 임계값을 정하는 순간 그 값을 유리하게 고르고 싶은 유혹이 생기기 때문임(사후 해석 방지).

TRANSLATE_SYSTEM = (
    "You translate Korean academic search queries into English.\n"
    "Translate literally and completely. Keep every technical noun. Do not add, remove, "
    "or generalize any term. Do not explain. Output only the English sentence."
)

OLLAMA_HOST = "http://localhost:11434"


def ollama_json(model: str, system: str, user: str, timeout: int = 120) -> dict:
    """Ollama HTTP API 를 직접 부름.

    `ollama` 파이썬 패키지를 안 쓰는 이유: 감사 도구는 의존성이 적을수록 좋음. 이 기능은
    "지금 이 평가셋을 믿어도 되는가"를 판정하는 도구라서, 나중에 다른 사람이 다른 환경에서
    그대로 돌려 볼 수 있어야 함. 표준 라이브러리만 쓰면 그게 보장됨.
    (실제로 의존성이 대거 깨졌을 때 이 도구만 살아 있었음.)
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
    """한국어 질문을 영어로 옮김. 이미 옮긴 것은 캐시에서 꺼내 씀.

    캐시를 두는 이유: 감사를 여러 번 돌리게 되는데, 번역은 매번 같은 결과여야
    비교가 성립함. 그리고 300문항 번역에 몇 분이 걸림.
    """
    cache: dict[str, str] = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            cache[row["ko"]] = row["en"]

    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if not todo:
        print(f"번역 캐시에서 전부 찾았다 ({len(cache)}건 보유)")
        return cache

    print(f"번역 시작: {len(todo)}건 (모델 {model}, 로컬 Ollama, 비용 0원)", flush=True)

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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache_path, [{"ko": k, "en": v} for k, v in cache.items()])
    print(f"번역 완료: {len(todo)}건, {time.time()-t0:.0f}초 -> 캐시 {cache_path}")
    return cache


def _gold_papers(rows: list[dict], corpus: str | None) -> dict[str, dict]:
    """정답 논문의 제목, 초록을 구함.

    평가셋 문항이 `_title` 을 갖고 있으면 그것을 씀(코퍼스 1GB 를 훑지 않아도 됨).
    없으면 코퍼스에서 찾음.
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
    print(f"정답 논문 {len(papers)}편의 제목을 확보했다")

    ko = [r["text"] for r in rows if r.get("lang") == "ko"]
    trans = translate_batch(ko, args.model, Path(args.cache)) if ko else {}

    # 실행 결과가 있으면 문항별 성공 여부를 붙임
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
        print("\n## 겹침 구간별 Recall@10 - 누수가 성능을 얼마나 떠받치는가")
        print("   (임계값 하나로 자르지 않고 층으로 보여준다. 사후에 유리한 값을 고르지 못하게)")
        print(f"\n{'겹침 구간':<12}{'n':>5}{'Recall@10':>12}   문항 구성")
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
        print("\n## 누수를 걷어낸 값 (겹침 <0.4 이고 방법명도 없는 문항만)")
        print(f"   전체        n={len(allr):<5} Recall@10 = {np.mean([r['hit10'] for r in allr]):.3f}")
        print(f"   누수 제외    n={len(clean):<5} Recall@10 = {np.mean([r['hit10'] for r in clean]):.3f}")
        print(f"   -> 걸러진 문항 {len(allr) - len(clean)}건 "
              f"({1 - len(clean)/max(len(allr),1):.1%})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out, out)
        print(f"\n문항별 결과 -> {args.out}")


# ==========================================================================
# 명령줄
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="평가셋 만들기와 검사 (generate -> split -> pool -> grade, 그리고 audit)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="논문에서 질문을 거꾸로 만든다 (유료)")
    g.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    g.add_argument("--n-papers", type=int, default=200)
    g.add_argument("--model", default="gpt-5.4")
    g.add_argument("--difficulties", nargs="*", default=["easy", "medium", "hard"])
    g.add_argument("--sampling", choices=["stratified", "proportional"], default="stratified",
                   help="stratified 는 분야마다 고르게(평가셋 v2 를 만든 방식), "
                        "proportional 은 코퍼스 분야 비율 그대로")
    g.add_argument("--effort", default="",
                   help="추론 모델의 추론 강도(minimal/low/medium/high). gpt-5 계열에서 "
                        "안 주면 medium 이 걸려 호출당 추론 토큰이 1,400개 붙음")
    g.add_argument("--exclude-from", nargs="*",
                   default=["data/eval/dev.jsonl", "data/eval/test.jsonl"],
                   help="이 평가셋들의 정답 논문은 뽑지 않는다")
    g.add_argument("--seed", type=int, default=20260813)
    g.add_argument("--out", default="runs/queries_raw.jsonl")
    g.add_argument("--resume", action="store_true", help="이미 만든 문항은 건너뛴다")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("split", help="논문 단위로 dev/test 로 나눈다")
    s.add_argument("--queries", default="runs/queries_raw.jsonl")
    s.add_argument("--out-dir", default="data/eval")
    s.add_argument("--seed", type=int, default=20260813)
    s.add_argument("--test-ratio", type=float, default=0.5)
    s.set_defaults(func=cmd_split)

    p = sub.add_parser("pool", help="등급 판정용 후보 풀 만들기 (로컬 검색, 비용 0원)")
    p.add_argument("--queries", required=True)
    p.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    p.add_argument("--index", default=str(config.DATA_DIR / "embeddings" / "cs2021-ft"))
    p.add_argument("--depth", type=int, default=20,
                   help="짝당 후보 수. 깊을수록 정확해지지만 판정비가 는다. "
                        "0 이면 자르지 않음 (--from-runs 에서만 뜻이 있음)")
    p.add_argument("--per-lang", type=int, default=15, help="언어별로 몇 편까지 가져와 합칠지")
    p.add_argument("--from-runs", nargs="*", default=[],
                   help="실행 결과 파일들. 주면 검색을 새로 하지 않고 그 결과의 상위 "
                        "N편을 후보로 씀")
    p.add_argument("--from-runs-depth", type=int, default=10,
                   help="--from-runs 를 쓸 때 실행 결과마다 상위 몇 편을 가져올지")
    p.add_argument("--from-runs-stage", choices=["rerank", "fused", "both"], default="rerank",
                   help="재정렬 후(reranked_ids)만 모을지, 합치기 직후도 모을지. "
                        "재정렬 전후를 nDCG 로 비교하려면 both 여야 함")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_pool)

    j = sub.add_parser("grade", help="후보마다 관련도 0~3 판정 (유료)")
    j.add_argument("--pool", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--model", default="gpt-4.1-mini",
                   help="심판 모델. 질문 생성기와 다른 계열이어야 한다")
    j.add_argument("--limit", type=int, default=None)
    j.add_argument("--max-cost", type=float, default=None,
                   help="이 금액(달러)을 넘으면 중단한다. 예산이 빠듯할 때 안전장치")
    j.set_defaults(func=cmd_grade)

    a = sub.add_parser("audit", help="정답 누수 검사 (한국어 포함, 비용 0원)")
    a.add_argument("--queries", default="data/eval/test.jsonl")
    a.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"),
                   help="문항에 _title 이 없을 때만 읽는다")
    a.add_argument("--run", default=None,
                   help="실행 결과(선택). 주면 겹침 구간별 Recall@10 을 함께 낸다")
    a.add_argument("--model", default=config.REWRITER_MODEL)
    a.add_argument("--cache", default="data/cache/ko_en_queries.jsonl")
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_audit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
