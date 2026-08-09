# 확정 평가 결과 (results/)

보고서·논문에 인용하는 **확정 결과**만 여기 둔다. 커밋해서 누구나 재현·검증할 수 있게 한다.
굴러가며 나오는 임시 실험은 `runs/`에 두고 커밋하지 않는다.

## test300_*.jsonl — arXiv 실시간 평가, test 전체 300문항 (2026-08-07)

| 파일 | 변환기 | 설명 |
|---|---|---|
| `test300_passthrough.jsonl` | 없음 | **기준선.** 사용자 검색어를 그대로 arXiv에 던진다 |
| `test300_hierarchical.jsonl` | `hierarchical` | Qwen3-4B 계층 변환 (학습 전) |
| `test300_finetuned.jsonl` | `finetuned` | 위에 SFT(LoRA) 적용 |
| `test300_dpo.jsonl` | `dpo` | SFT 위에 DPO까지 적용 (최종 모델) |

측정 조건: `data/eval/test.jsonl` 300문항 전체 · arXiv 상위 **100편** 회수 · 네 모델 모두
**오류 0건**(측정 오염 없음) · 커밋 `7feba92`.

각 줄에 `retrieved_ids`(검색된 논문 ID 100개)가 순위대로 들어 있어, **arXiv를 다시 부르지
않고** 어떤 K로든 지표를 재계산할 수 있다.

### 재현 방법

```bash
# 저장된 결과만 다시 집계 (arXiv 호출 없음)
python -m evaluation.arxiv_eval --report-only results/test300_dpo.jsonl

# 네 모델 비교 + 유의성 검정 + 깊이 분석
python -m evaluation.compare_runs results/test300_passthrough.jsonl \
    results/test300_hierarchical.jsonl results/test300_finetuned.jsonl \
    results/test300_dpo.jsonl --k 10 --depth-table

# 언어별 / 난이도별
python -m evaluation.compare_runs results/test300_*.jsonl --k 10 --by lang

# 처음부터 다시 측정 (모델당 30~60분, 이어하기 지원)
python -m evaluation.arxiv_eval --queries data/eval/test.jsonl \
    --rewriter dpo --k 100 --out runs/test300_dpo.jsonl
```

### 요약 (Recall@10, 부트스트랩 95% 신뢰구간)

| 모델 | 전체 | 영어(147) | 한국어(153) |
|---|---|---|---|
| passthrough (기준선) | 0.167 [0.127, 0.210] | 0.327 | 0.013 |
| hierarchical | 0.220 [0.173, 0.267] | 0.259 | 0.183 |
| finetuned (SFT) | 0.273 [0.223, 0.323] | 0.293 | 0.255 |
| **dpo (SFT+DPO)** | **0.343** [0.290, 0.397] | 0.354 | 0.333 |

## test300_groundedf_k200.jsonl · test150_unbiased.jsonl (2026-08-10)

| 파일 | 조건 | 용도 |
|---|---|---|
| `test300_groundedf_k200.jsonl` | 어휘조회+상용구필터, 후보 깊이 **200** | 깊이 실험(100/150/200)의 원자료. 논문 번호 200개가 저장돼 있어 어떤 깊이로든 재계산 가능 |
| `test150_unbiased.jsonl` | 위와 같되 **어휘 조회에서 정답 논문 제외** | 평가 편향의 크기를 잰 원자료(ISSUE #22) |

**중요:** 두 파일을 비교하면 어휘 조회 평가의 편향(R@10 −0.073, p=0.001)을 재현할 수 있다.
`test300_groundedf_*` 계열의 수치는 **정답 논문이 조회에 포함된 낙관적 조건**임에 주의할 것.

```bash
# 깊이별 재계산 (arXiv 호출 0회)
python -m evaluation.arxiv_eval --report-only results/test300_groundedf_k200.jsonl \
    --k-values 10 30 50 100 150 200
# 재정렬 적용 (arXiv 호출 0회)
python -m evaluation.rerank_eval --run results/test150_unbiased.jsonl --depth 200
```

자세한 해석은 [../docs/ISSUE.md](../docs/ISSUE.md) #13, #21, #22, #23 참고.
