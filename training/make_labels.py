"""학습 라벨 생성 — 정답 논문에서 '실제로 검색되는 학술 용어'를 뽑는다.

기존 방식과 무엇이 다른가:
기존(`build_training_data.py`)은 변환기에게 후보 검색어를 여러 개 만들게 한 뒤 **arXiv로
직접 채점**해서, 정답 논문을 찾아낸 것만 라벨로 썼다. 문제가 둘이었다.
  - 후보 하나마다 arXiv 호출 1회 (지난 1회 생성에서 실측 1,147회). 벌크 수집에 API를 쓰는
    패턴이라 arXiv 정책 취지에 어긋나고, 3초 간격 때문에 느리다.
  - 라벨 확보율이 28%뿐이다. 변환기가 우연히 좋은 후보를 못 만들면 그 질문은 버려진다.

이 파일은 방향을 뒤집는다. **정답 논문을 이미 알고 있으므로, 그 논문에서 좋은 용어를 직접
뽑아낸다.** 변환기의 운에 기대지 않으므로 확보율이 100%에 가깝고, arXiv 호출이 0회다.

무엇이 '좋은 용어'인가 (실측으로 정한 기준, ISSUE #16):
검색 실패는 두 종류이고 원인이 정반대다.
  - **도달 불가** — 만든 용어가 정답 논문에 아예 없다. 그러면 절대 검색되지 않는다.
  - **묻힘** — 용어는 논문에 있지만 너무 흔하다. arXiv에서 수만 편이 함께 걸려 정답이 밀려난다.
실제 측정에서 검색에 성공한 검색어가 쓴 용어는 arXiv 310만 편 중 **중앙값 144편**에
등장했고, 묻힌 검색어의 용어는 **6,166편**에 등장했다(43배 차이).

그래서 라벨은 이 두 조건을 동시에 만족하는 구로 만든다.
  1) 정답 논문의 제목·초록에 **그대로 등장**할 것        → 도달 불가 방지
  2) 문서 빈도가 **너무 높지도 낮지도 않을** 것          → 묻힘 방지 + 유추 가능성 확보

2번의 아래쪽 한계가 왜 필요한가: 논문에 한 번만 나오는 표현(예: "through three key
components")을 쓰면 검색은 100% 되지만, 그건 그 논문의 지문(指紋)이라 **사용자 질문만 보고는
절대 유추할 수 없다.** 학습해도 못 따라 하는 목표는 라벨로 쓸모가 없다.

실행 예:
  python -m training.make_labels --queries data/eval/train.jsonl --limit 30 \\
      --out data/training/labels_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src import config
from src.retrieval.phrase_stats import PhraseFrequency, normalize
from src.utils import read_jsonl, write_jsonl

# 구의 앞뒤에 오면 의미가 없는 흔한 말들 (여기서 시작하거나 끝나는 구는 버린다)
# ※ 전치사를 빠뜨리면 "through dynamic"(1,038편), "systems through"(1,539편) 같은
#   문장 조각이 학술 용어로 뽑힌다. 실제로 첫 생성에서 그 일이 있었다.
EDGE_STOP = set("""
a an the of for and or in on at with to from by as is are be was were been being that this
these those we our their its it can may will would could should has have had do does did
not no but if then than when where which who whom while during into over under about
through across against among within without upon toward towards beyond along around
per versus between before after since until unless whether either neither onto out off
such very more most other others new novel recent present presents propose proposed
show shows shown study studies work works paper method methods approach approaches
result results using use used uses based via also both each any all one two three
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")

# 구 안 어디에든 이 말이 들어 있으면 학술 용어가 아니라 '문장 조각'이다.
# (첫 시도에서 "proved to be difficult", "enables to learn", "background since" 같은 것이
#  라벨로 뽑혔다. 드물기는 해도 사용자 질문에서 유추할 수 없어 학습 목표로 쓸모가 없다.)
NON_TERM = set("""
is are was were be been being am has have had having do does did doing
can could may might will would shall should must
proved prove proves shown show shows showed enables enable enabled allows allow allowed
presents present presented provides provide provided demonstrates demonstrate achieves achieve
obtains obtain yields yield leads lead makes make made gives give given takes take
we our us their they them it its this that these those there here
however therefore thus hence moreover furthermore also although though while since because
very more most much many less least better best worse first second third
paper study work section figure table results result experiments experiment
""".split())

# 용어 안쪽에 와도 자연스러운 연결어.
# 처음에는 for/in/with/to 도 허용했는데, 그러자 제목에서 "data in daily",
# "reasoning for visual", "optimization with limited" 같은 **문장 조각**이 용어로 뽑혔다.
# 실제 학술 용어에서 안쪽 연결어는 거의 "of" 뿐이다(age of information, bag of words).
OK_INSIDE = {"of"}

# 논문 제목을 여는 동명사·전치사. 이걸로 시작하는 구는 제목의 앞 조각이지 용어가 아니다.
# (예: "Rethinking Individual Fairness..." 에서 "rethinking individual")
TITLE_OPENER = set("""
rethinking revisiting reconsidering exploring investigating understanding examining
towards toward improving enhancing leveraging exploiting utilizing employing
evaluating benchmarking measuring assessing analyzing comparing bridging unifying
scaling accelerating optimizing advancing extending generalizing revisited
introducing designing building developing training learning teaching
""".split())


def is_term_like(phrase: str) -> bool:
    """학술 용어처럼 생겼는지 판정한다 (문장 조각을 걸러내기 위함)."""
    words = phrase.split()
    if words[0] in TITLE_OPENER:
        return False
    for w in words:
        if w in NON_TERM:
            return False
        if w in EDGE_STOP and w not in OK_INSIDE:
            return False
    # 연결어로 시작하거나 끝나면 조각이다
    return words[0] not in OK_INSIDE and words[-1] not in OK_INSIDE


# 구를 만들 때 여기서 끊는다. 문장 부호를 건너뛰면 실제로는 붙어 있지 않은 단어들이
# 하나의 구처럼 만들어진다. 실제로 겪은 문제:
#   "https://github.com/foo" → 'github com', 'com foo'
#   "...a new method. Code is..." → 'method code'(문장 경계를 넘음)
# 이런 가짜 구가 학습 라벨과 어휘 조회 양쪽에 노이즈로 들어갔다.
_SEGMENT = re.compile(r"[^A-Za-z0-9\- ]+")


def candidate_phrases(text: str, lo: int = 2, hi: int = 4) -> list[str]:
    """제목·초록에서 학술 용어가 될 만한 구를 모두 뽑는다 (2~4단어).

    **문장 부호에서 끊어** 실제로 이어져 있는 단어들만 하나의 구로 만든다.
    """
    out, seen = [], set()
    for segment in _SEGMENT.split(text):
        words = _WORD.findall(segment)
        for n in range(lo, hi + 1):
            for i in range(len(words) - n + 1):
                g = [w.lower() for w in words[i:i + n]]
                if g[0] in EDGE_STOP or g[-1] in EDGE_STOP:
                    continue
                if any(len(w) <= 1 for w in g):
                    continue
                p = " ".join(g)
                if p in seen or not is_term_like(p):
                    continue
                seen.add(p)
                out.append(p)
    return out


def pick_labels(phrases: list[str], stats: PhraseFrequency, title: str = "",
                full_text: str = "", k: int = 4,
                df_lo: int = 50, df_hi: int = 3000,
                df_target: int = 200, title_df_lo: int = 3,
                min_score: float = 0.8) -> list[tuple[str, int]]:
    """정답 논문의 구 중에서 라벨로 쓸 것을 고른다.

    고르는 기준 (실측에 맞춘 것):
      1) **문서 빈도가 목표 구간**일 것 — 검색에 성공한 검색어의 용어는 arXiv 310만 편 중
         중앙값 144편에 등장했고, 묻힌 검색어의 용어는 6,166편이었다.
      2) **제목에 있으면 크게 우대하고, 하한도 면제한다.**
      3) **초록에서 반복되면 우대** — 진짜 용어는 반복되고, 문장 조각은 한 번만 나온다.

    2번의 하한 면제가 왜 필요한가 (실패에서 배운 것):
    하한 50편은 "proved to be difficult"(22편) 같은 **문장 조각**을 막으려고 둔 것이었다.
    그런데 같은 하한이 "makeup transfer"(47편) 같은 **진짜 도메인 용어**까지 버렸다. 그 결과
    "화장 색상 바꾸기" 질문의 라벨에서 makeup 이 통째로 빠지고, 대신 "weakly supervised"
    (2,435편)·"style transfer"(1,696편) 같은 일반 기계학습 용어만 남았다. 사용자가 가장
    쉽게 떠올릴 단어이자 논문을 가장 잘 가려내는 단어가 빠진 것이다.

    즉 **'드물다'는 것만으로는 문장 조각과 전문 용어를 구분할 수 없다.** 구분자는 제목이다.
    제목은 저자가 그 논문의 핵심 용어를 고르는 자리이고, 문장 조각은 제목에 들어가지 않는다.
    그래서 제목에 있는 구는 훨씬 낮은 하한(title_df_lo)만 넘으면 통과시킨다.
    (완전히 0으로 두지 않는 이유: 문서 빈도 1~2편은 그 논문에만 있는 지문이라 사용자
     질문에서 유추할 수 없어 학습 목표로 쓸모가 없다.)
    """
    # 제목에 줄바꿈·연속 공백이 들어 있으면 "makeup transfer" 같은 구가 매칭되지 않는다.
    # (실제로 "...Controllable Makeup\n  Transfer" 때문에 핵심 용어를 놓치고 있었다.)
    title_l = " ".join(title.lower().split())
    text_l = " ".join(full_text.lower().split())

    scored = []
    for p in phrases:
        d = stats.df(p)
        if d is None or d > df_hi:
            continue
        in_title = p in title_l
        if d < (title_df_lo if in_title else df_lo):
            continue
        score = 0.0
        if in_title:
            score += 3.0                                   # 제목에 있으면 가장 강한 신호
        rep = text_l.count(p)
        if rep >= 2:
            score += 1.0 + min(rep - 2, 2) * 0.3           # 반복될수록 진짜 용어
        score += 0.3 * (len(p.split()) - 1)                # 길수록 구체적
        # 목표 빈도에서 멀수록 감점 (로그 거리)
        import math
        score -= abs(math.log10(max(d, 1)) - math.log10(df_target))
        scored.append((score, d, p))

    scored.sort(key=lambda x: (-x[0], x[1]))

    # k개를 억지로 채우지 않는다. 제목에도 없고 본문에서 반복되지도 않는 구는 점수가
    # 낮은데, 그런 것까지 끌어오면 "combination of classical", "main drawbacks",
    # "benchmarks like" 같은 무의미한 표현이 라벨에 섞인다. 그걸 학습시키면 모델이
    # 아무 일반 표현이나 지어내게 되므로, **자격을 갖춘 것만** 쓴다.
    # 다만 하나도 못 고르면 라벨 자체가 없어지므로, 그럴 때만 최고점 하나를 허용한다.
    picked: list[tuple[str, int]] = []
    for score, d, p in scored:
        if score < min_score:
            break
        if any(p in q or q in p for q, _ in picked):        # 겹치는 구는 하나만
            continue
        picked.append((p, d))
        if len(picked) >= k:
            break
    if not picked and scored:
        picked = [(scored[0][2], scored[0][1])]
    return picked


def filter_by_question_similarity(
    rows: list[dict], threshold: float = 0.40,
    model_name: str = "BAAI/bge-m3", device: str = "cuda") -> None:
    """질문에서 유추할 수 없는 용어를 라벨에서 뺀다 (rows 를 그 자리에서 고친다).

    왜 필요한가 (실측 근거):
    정답 논문에서 뽑은 용어는 **검색은 잘 되지만 학습 목표로는 못 쓸 수도** 있다.
    예를 들어 "사진에서 화장 색상을 바꾸려면?" 이라는 질문의 정답 논문(CA-GAN)에서
    뽑힌 용어에는 makeup transfer 와 함께 "aware gan", "weakly supervised" 도 들어간다.
    앞의 둘은 질문에서 떠올릴 수 있지만 뒤의 둘은 불가능하다. 그런 것까지 목표로 주면
    모델은 질문과 무관한 일반 용어를 지어내는 법을 배운다.

    어떻게 재는가:
    bge-m3 로 질문과 용어를 각각 임베딩해 코사인 유사도를 잰다. 이 모델은 다국어라
    **한국어 질문과 영어 용어를 같은 공간에서** 비교할 수 있다.

    문턱 0.40 의 근거 (90문항 실측):
      잘리는 쪽 — aware gan 0.267 · weakly supervised 0.31
      남는 쪽   — unpaired data 0.433 · relative spatial 0.49
    이 문턱으로 문항당 용어가 3.23개 → 2.92개로 줄고, 실제 arXiv 검색 성능은
    Recall@10 0.711 → 0.656, Recall@100 0.900 → 0.833 이었다. 즉 검색력을 조금 내주고
    학습 가능성을 얻는 맞바꿈이다.

    ※ 개수를 고정해 자르면 안 된다. '상위 2개만' 으로 잘라 본 결과 Recall@10 이 0.467
      까지 떨어졌다. unpaired data(0.433) 처럼 유추 가능한 용어까지 기계적으로 잘려서다.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    questions = [r["input"] for r in rows]
    terms = sorted({t for r in rows for t in r["terms"]})
    q_emb = model.encode(questions, normalize_embeddings=True, batch_size=16,
                         show_progress_bar=False)
    t_emb = model.encode(terms, normalize_embeddings=True, batch_size=64,
                         show_progress_bar=False)
    idx = {t: i for i, t in enumerate(terms)}

    n_before = sum(len(r["terms"]) for r in rows)
    for i, r in enumerate(rows):
        sims = {t: float(q_emb[i] @ t_emb[idx[t]]) for t in r["terms"]}
        kept = [t for t in r["terms"] if sims[t] >= threshold]
        if not kept:                       # 다 잘리면 가장 가까운 하나는 남긴다
            kept = [max(sims, key=sims.get)]
        keep_set = set(kept)
        r["term_df"] = [d for t, d in zip(r["terms"], r["term_df"]) if t in keep_set]
        r["term_sim"] = [round(sims[t], 3) for t in kept]
        r["terms"] = kept
        r["output"] = build_query(r["input"], kept)
    n_after = sum(len(r["terms"]) for r in rows)
    print(f"  유사도 {threshold} 미만 제거: 용어 {n_before} → {n_after}개 "
          f"(문항당 {n_before/len(rows):.2f} → {n_after/len(rows):.2f}개)")


def build_query(raw_query: str, terms: list[str], max_len: int = 300) -> str:
    """서비스와 같은 형태의 arXiv 검색어를 만든다 (hierarchical.build_arxiv_query 와 동일 규칙)."""
    parts = []
    orig = (raw_query or "").replace('"', " ").strip()
    if orig and re.search(r"[A-Za-z]", orig):     # 한글만 있는 질문은 넣어도 안 걸리므로 생략
        parts.append(f'all:"{orig}"')
    parts += [f'abs:"{t}"' for t in terms]
    q = ""
    for p in parts:
        cand = p if not q else f"{q} OR {p}"
        if len(cand) > max_len:
            break
        q = cand
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description="정답 논문에서 학습 라벨을 직접 뽑는다")
    ap.add_argument("--queries", default="data/eval/train.jsonl")
    ap.add_argument("--corpus", default="data/corpus/corpus-v1.jsonl",
                    help="정답 논문의 제목·초록을 가져올 곳")
    ap.add_argument("--df-corpus", default="data/corpus/corpus-full.jsonl",
                    help="문서 빈도를 셀 코퍼스 (arXiv 전체여야 한다)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=4, help="라벨로 쓸 용어 개수")
    ap.add_argument("--df-lo", type=int, default=50)
    ap.add_argument("--df-hi", type=int, default=3000)
    ap.add_argument("--sim-threshold", type=float, default=0.40,
                    help="질문과의 의미 유사도가 이 값 미만인 용어는 라벨에서 뺀다. "
                         "0 을 주면 이 필터를 끈다")
    ap.add_argument("--sim-device", default="cuda")
    ap.add_argument("--out", default="data/training/labels_train.jsonl")
    args = ap.parse_args()

    rows = list(read_jsonl(args.queries))
    if args.limit:
        rows = rows[: args.limit]
    gold_ids = {r["gold_id"] for r in rows}

    papers, titles = {}, {}
    for p in read_jsonl(args.corpus):
        if p["id"] in gold_ids:
            papers[p["id"]] = f"{p['title']} {p['abstract']}"
            titles[p["id"]] = p["title"]
    print(f"질문 {len(rows)}개 · 정답 논문 {len(papers)}편 확보")

    # 1) 후보 구를 모아 문서 빈도를 한 번에 센다 (코퍼스를 한 번만 훑는다)
    per_paper = {pid: candidate_phrases(txt) for pid, txt in papers.items()}
    all_phrases = sorted({p for ps in per_paper.values() for p in ps})
    print(f"후보 구 {len(all_phrases):,}개의 문서 빈도를 센다")
    stats = PhraseFrequency(args.df_corpus)
    stats.ensure(all_phrases)

    # 2) 질문마다 라벨을 고른다
    out, n_empty = [], 0
    for r in rows:
        txt = papers.get(r["gold_id"])
        if txt is None:
            continue
        pid = r["gold_id"]
        picked = pick_labels(per_paper[pid], stats,
                             title=titles.get(pid, ""), full_text=txt,
                             k=args.k, df_lo=args.df_lo, df_hi=args.df_hi)
        terms = [p for p, _ in picked]
        if not terms:
            n_empty += 1
            continue
        out.append({
            "query_id": r["query_id"], "input": r["text"], "gold_id": r["gold_id"],
            "lang": r.get("lang"), "difficulty": r.get("difficulty"),
            "terms": terms,
            "term_df": [d for _, d in picked],
            "output": build_query(r["text"], terms),
        })

    # 3) 질문에서 유추할 수 없는 용어를 뺀다 (학습 가능한 목표만 남기기 위함)
    if args.sim_threshold > 0 and out:
        print(f"\n질문과의 의미 유사도로 거른다 (bge-m3, 문턱 {args.sim_threshold})")
        filter_by_question_similarity(out, threshold=args.sim_threshold,
                                      device=args.sim_device)

    write_jsonl(args.out, out)
    print(f"\n라벨 확보 {len(out)}/{len(rows)}개 ({len(out)/len(rows):.1%}) · "
          f"조건에 맞는 구가 없어 제외 {n_empty}개")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
