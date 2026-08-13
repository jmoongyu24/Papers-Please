"""평가셋 v2 생성 — 옛 평가셋의 설계 결함 세 가지를 고친 버전.

## 옛 평가셋(v1)의 무엇이 문제였나

**① 난이도 이름이 설계 의도와 뒤집혀 있었다.**
`generate_queries.py:35` 의 주석이 그대로 말해 준다 — `easy/mid/hard = 일상어 / 어설픈
전문어 / 정확한 전문어`. 즉 코드의 `hard` 는 **대학원생의 정확한 질문**이고 `easy` 는
**초심자의 일상어 질문**이다. 원래 의도(easy=대학원생 … hard=1~2학년)와 정반대라,
"어려울수록 잘 맞힌다"는 이상한 표가 나왔다. v2 는 이름을 의도대로 되돌린다.

**② 제목을 모델에게 주고 "어순만 바꿔라"라고 시켰다 — 이게 누수의 주범이다.**
실측(2026-08-13): 대학원생 층 영어 질문은 제목 내용어의 **중앙값 57%** 를 담고 있었고
Recall@10 이 **1.000** 이었다. 하나도 안 틀렸다.

결정적인 사실은, 그 층에서 **'이 논문만의 고유 용어'를 걷어내도 여전히 1.000** 이라는
것이다. 즉 원인은 희귀 용어가 아니라 **제목의 낱말 조합을 그대로 재현한 것**이다.
`image captioning`(71만 편 중 1,136편) 처럼 흔한 말만 써도, 조합이 제목과 같으면
검색기에게는 정답을 적어 준 것과 같다.

    → v2 는 **제목을 아예 주지 않는다.** 초록만 주고, "이 논문을 아직 못 찾은 사람"의
      자리에서 쓰게 한다. 이것이 가장 효과가 큰 한 가지 장치다.

**③ 한국어 문항은 누수 검사를 받은 적이 없다.**
`audit_dataset.py` 와 `filter_queries.py` 의 검사기가 영어 낱말만 세기 때문에, 한국어
질문의 낱말 집합은 영어 논문 용어와 **항상 공집합**이었다. "깨끗하다"가 아니라 "검사한
적이 없다"가 맞다. 실제로 제목을 통째로 번역한 문항이 통과해 있었다.

    → v2 는 **한 번의 호출로 한국어와 영어를 함께** 만든다. 그러면 한국어 문항에도
      영어판이 항상 있으므로 모든 누수 검사가 영어 위에서 돌아간다. 비용은 오히려 절반이다.

## 금지 어휘 목록을 두지 않는 이유 (v2 의 설계 결정)

한때 "등장 논문 수 50편 이하인 표현은 금지" 를 넣으려 했으나 **뺐다.** 이유 둘.

1. **효과가 작다.** 위에서 봤듯 대학원생 층은 고유 용어를 걷어내도 1.000 이었다.
   진짜 원인은 제목 구조이지 희귀 용어가 아니다. 제목을 안 주는 것으로 이미 해결된다.
2. **기준이 모호하고, 현실을 왜곡한다.** 실제 사용자는 희귀한 논문을 찾기도 한다.
   "50편 이하는 쓰면 안 된다"는 규칙은 그런 정상적인 검색을 데이터셋에서 지워 버린다.

대신 **거르지 않고 잰다.** 문항마다 제목 겹침과 희귀 용어 사용을 기록해 두고, 성능을
그 값으로 층화해 보고한다. 그러면 "누수를 뺀 값"을 나중에 어떤 기준으로든 다시 계산할 수
있고, 임계값을 유리하게 고르는 사후 해석도 막힌다.

## 난이도 축 — '표현 거리' 가 아니라 '묻는 사람이 얼마나 아는가'

옛 축은 "제목과 표현이 얼마나 다른가" 였는데, 이 축은 **글자 일치 검색기에만** 어렵다.
뜻으로 찾는 의미 검색기에는 오히려 전문 용어가 정확할수록 쉬우므로 **축이 뒤집힌다.**
v2 의 축은 검색 방식과 무관하다 — 묻는 사람의 전문성이 기준이다.

    easy        대학원생        정확한 학술 용어를 쓴다
    medium      학부 연구생      범용 전문어와 일상어가 섞인다
    hard        1~2학년         전문 용어를 못 쓰고 일상어로 두루뭉술하게 쓴다
    known_item  논문을 아는 사람  특정 논문을 지목해 찾는다 (탐색형과 섞지 않고 따로 측정)

마지막 층을 따로 두는 이유: 특정 논문을 이름으로 찾는 것은 **실재하는 검색 유형**이지만
(ISSUE #6), 주제로 논문을 발견하는 것과는 **다른 능력**이다. 옛 평가셋은 대학원생 층이
자기도 모르게 이 유형이 돼 버려서 두 능력이 한 숫자에 섞였다.

## 언어를 짝짓는 이유

옛 평가셋은 논문 100편 중 **두 언어를 모두 가진 논문이 0편**이었다. 한국어 질문은 A 논문,
영어 질문은 B 논문에 대한 것이라 "한국어가 영어보다 낫다"를 말할 수 없다 — 논문 자체가
다르기 때문이다. v2 는 같은 논문·같은 난이도를 두 언어로 만들어 다른 조건을 모두 고정한다.

번역투를 막으려고 "직역하라"가 아니라 **"같은 사람이 각 언어로 자연스럽게 물었다면"** 으로
지시하고, 뜻이 같은지는 로컬 모델 역번역으로 따로 검사한다.

실행:
  # 사전 시험 (돈을 조금만 쓴다)
  python -m evaluation.generate_queries_v2 --n-papers 5 --out runs/pilot_v2.jsonl
  # 본 생성
  python -m evaluation.generate_queries_v2 --n-papers 200 --out data/eval/v2_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from src import config
from src.retrieval.fusion import normalize_paper_id
from src.utils import read_jsonl, write_jsonl

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

SYSTEM = (
    "너는 학술 논문 검색 시스템의 평가 데이터를 만드는 전문가다. "
    "주어진 지시를 정확히 지키고, 요구된 JSON 형식으로만 답한다."
)


def build_prompt(abstract: str, difficulty: str) -> str:
    """초록만 주고 검색어를 만들게 한다. **제목은 절대 주지 않는다.**"""
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


SCHEMA = {
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


def sample_papers(corpus: str, n: int, exclude: set[str], seed: int,
                  min_abs_words: int = 80) -> list[dict]:
    """코퍼스에서 논문을 뽑는다. **기존 평가셋의 정답 논문은 제외한다.**

    제외하는 이유(ISSUE #25): 같은 논문에서 옛 질문과 새 질문이 나오면 둘이 쌍둥이가 되어,
    옛 평가셋으로 고른 설정이 새 평가셋에서도 유리해진다. 새 시험지의 뜻이 없어진다.

    초록이 너무 짧은 논문은 뺀다 — 질문을 만들 재료가 부족해 층이 구분되지 않는다.
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

    # 분야가 한쪽으로 쏠리지 않게 첫 번째 분야 기준으로 층화 추출한다
    by_cat: dict[str, list[dict]] = {}
    for p in pool:
        by_cat.setdefault(str(p["categories"]).split()[0] if p["categories"] else "?", []).append(p)
    rng = random.Random(seed)
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
    print(f"분야 층화 추출: {len(out)}편 · 분야 {len({str(p['categories']).split()[0] for p in out})}종")
    return out


def call_openai(client, model: str, prompt: str) -> tuple[dict, dict]:
    """한 번 호출해 JSON 과 토큰 사용량을 돌려준다."""
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "query", "strict": True, "schema": SCHEMA}},
    )
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # 추론 계열 모델은 일부 인자를 안 받는다. 최소 인자로 한 번 더 시도한다.
        if "unsupported" not in str(e).lower() and "unrecognized" not in str(e).lower():
            raise
        kwargs.pop("response_format", None)
        resp = client.chat.completions.create(**kwargs)

    u = resp.usage
    usage = {"in": u.prompt_tokens, "out": u.completion_tokens,
             # 추론 토큰은 눈에 안 보이지만 출력 요금으로 붙는다. 예산이 여기서 무너진다.
             "reasoning": getattr(getattr(u, "completion_tokens_details", None),
                                  "reasoning_tokens", 0) or 0}
    text = resp.choices[0].message.content or "{}"
    return json.loads(text), usage


# ── 생성 후 자동 점검 (거르지 않고 '기록'한다) ────────────────────────────
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")


def norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def longest_copy(query_en: str, abstract: str) -> int:
    """초록에서 몇 낱말까지 연속으로 그대로 옮겼는가."""
    a, q = norm(abstract).split(), norm(query_en).split()
    aset = {" ".join(a[i:i + n]) for n in range(3, 9) for i in range(len(a) - n + 1)}
    best = 0
    for n in range(3, min(9, len(q) + 1)):
        for i in range(len(q) - n + 1):
            if " ".join(q[i:i + n]) in aset:
                best = max(best, n)
    return best


def title_overlap(query_en: str, title: str) -> float:
    """제목 내용어 중 몇 %가 질문에 이미 들어 있는가 (거르지 않고 기록만 한다)."""
    from evaluation.audit_dataset import content_words
    t = content_words(title)
    return len(content_words(query_en) & t) / len(t) if t else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="평가셋 v2 생성 (제목 미제공 · 한영 동시)")
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR / "corpus-cs2021.jsonl"))
    ap.add_argument("--n-papers", type=int, default=200)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--difficulties", nargs="*", default=["easy", "medium", "hard"])
    ap.add_argument("--exclude-from", nargs="*",
                    default=["data/eval/train.jsonl", "data/eval/test.jsonl"])
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="이미 만든 문항은 건너뛴다")
    args = ap.parse_args()

    exclude = set()
    for p in args.exclude_from:
        if Path(p).exists():
            exclude |= {normalize_paper_id(r["gold_id"]) for r in read_jsonl(p)}

    papers = sample_papers(args.corpus, args.n_papers, exclude, args.seed)

    out_path = Path(args.out)
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
    print(f"생성 시작: {len(todo)}회 호출 · 모델 {args.model} "
          f"(한 번에 한국어·영어 둘 다 생성 → 문항 {len(todo)*2}개)\n", flush=True)

    t0 = time.time()
    for i, (p, d) in enumerate(todo, 1):
        try:
            got, usage = call_openai(client, args.model, build_prompt(p["abstract"], d))
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
                # 거르지 않고 기록만 한다. 나중에 원하는 기준으로 층화해 계산할 수 있게.
                "title_overlap": round(title_overlap(got.get("en", ""), p["title"]), 3),
                "abstract_copy_words": longest_copy(got.get("en", ""), p["abstract"]),
                "_title": p["title"], "_categories": p["categories"],
            })

        if i % 10 == 0 or i == len(todo):
            write_jsonl(out_path, rows)
            print(f"  {i}/{len(todo)} · {time.time()-t0:.0f}초 · "
                  f"토큰 in {total_usage['in']:,} / out {total_usage['out']:,} "
                  f"(추론 {total_usage['reasoning']:,})", flush=True)

    meta = {"_meta": {"produced_by": "evaluation.generate_queries_v2",
                      "model": args.model, "n_papers": len(papers),
                      "difficulties": args.difficulties, "seed": args.seed,
                      "time": time.strftime("%Y-%m-%d %H:%M:%S"), "usage": total_usage}}
    write_jsonl(out_path, [meta] + rows)
    print(f"\n완료: 문항 {len(rows)}개 → {out_path}")
    print(f"토큰: 입력 {total_usage['in']:,} · 출력 {total_usage['out']:,} "
          f"(그중 추론 {total_usage['reasoning']:,})")


if __name__ == "__main__":
    main()
