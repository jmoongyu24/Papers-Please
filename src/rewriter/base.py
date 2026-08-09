"""쿼리 변환기들이 공통으로 따르는 약속(인터페이스)과, 평가용 기본 변환기들.

모든 변환기는 `rewrite(raw_query) -> RewriteResult` 함수를 갖는다.
평가 모듈은 이 인터페이스만 알고, 변환기를 갈아 끼우며 성능을 비교한다.

여기 담긴 변환기:
- PassthroughRewriter : 변환을 아예 안 함. **"변환 전" 기준점(baseline)** — 이 프로젝트의
  핵심 주장("변환하면 검색이 좋아진다")을 검증하려면 모든 평가에 반드시 포함해야 한다.

실제 변환기들은 별도 파일에 있고 모두 같은 인터페이스를 따른다:
  hierarchical.py (Qwen3-4B 계층 변환) · baselines.py (single_step, hyde)
  finetuned.py (LoRA 학습 모델)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.schemas import RewriteResult

# 검색 방식 이름 (schemas.RewriteResult.queries 의 키)
BACKENDS = ("sparse", "dense", "arxiv")


@runtime_checkable
class Rewriter(Protocol):
    name: str

    def rewrite(self, raw_query: str) -> RewriteResult:
        ...


class PassthroughRewriter:
    """변환하지 않고 원본 검색어를 그대로 쓴다 (변환 전 기준점)."""

    name = "passthrough"

    def rewrite(self, raw_query: str) -> RewriteResult:
        return RewriteResult(
            raw_query=raw_query,
            queries={b: raw_query for b in BACKENDS},
            intent=raw_query,
        )


# 이름 -> 변환기 인스턴스를 만들어주는 등록소.
# 언어 모델을 쓰는 변환기(hierarchical/single_step/hyde)는 import 시 ollama가 필요하므로,
# 그 이름을 부를 때만 늦게(lazy) import 한다. passthrough는 의존성이 없다.
def build_rewriter(name: str) -> Rewriter:
    if name == "passthrough":
        return PassthroughRewriter()
    if name == "hierarchical":
        from src.rewriter.hierarchical import HierarchicalRewriter
        return HierarchicalRewriter()
    if name == "single_step":
        from src.rewriter.baselines import SingleStepRewriter
        return SingleStepRewriter()
    if name == "hyde":
        from src.rewriter.baselines import HydeRewriter
        return HydeRewriter()
    if name == "finetuned":
        # SFT(지도 미세조정)로 학습한 모델. 학습 전/후 비교용
        from src.rewriter.finetuned import FinetunedRewriter
        return FinetunedRewriter()
    if name == "dpo":
        # SFT 위에 DPO(선호 학습)까지 얹은 모델
        from src.rewriter.finetuned import FinetunedRewriter
        return FinetunedRewriter(adapter_path="models/qwen3-4b-query-dpo")
    if name.startswith("grounded"):
        # 어휘 조회를 얹은 변환기. "grounded:<바탕변환기>" 형태로 바탕을 고를 수 있다.
        # 예) grounded:dpo (기본) · grounded:hierarchical · grounded:passthrough
        from src.rewriter.grounded import GroundedRewriter
        base = name.split(":", 1)[1] if ":" in name else "dpo"
        return GroundedRewriter(base=base)
    raise ValueError(f"알 수 없는 변환기 이름: {name}")
