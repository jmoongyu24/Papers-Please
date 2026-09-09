"""서비스가 쓸 모델을 미리 만들어 두는 스크립트. 그래픽 메모리를 줄이려고 씀.

    python -m training.export ollama --llama-cpp ~/llama.cpp
    python -m training.export fp16

## ollama - arXiv 검색어 변환기를 Ollama 모델 세 개로

`models/query-translator-dpo` 는 LoRA 라 transformers 로 기본 모델 위에 얹어 써야 하고,
그러면 8.27GB 가 들고 검색이 끝나도 안 내려감. LoRA 를 합쳐 GGUF 로 바꾸고 Ollama 에
등록하면 검색이 도는 동안에만 올라감.

정밀도를 달리해 셋을 만듦. 설정 화면에서 고를 수 있고 값은 `config.ARXIV_REWRITER_CHOICES`
에 있음. 셋 다 가중치는 같음.

    등급   Ollama 모델            정밀도    그래픽 메모리
    초급   papers-rewriter        q4_K_M   3.24GB
    중급   papers-rewriter-q8     q8_0     4.89GB
    고급   papers-rewriter-f16    f16      8.40GB

이 변환기는 arXiv '최신 논문' 칸에만 쓰임. 추천 목록에 나가는 10편은 이 변환기를 안 쓰므로
등급과 무관하게 같음.

### 함정 두 가지 - 둘 다 오류 없이 조용히 망가짐

1. transformers 5.x 는 대화 형식을 `chat_template.jinja` 로 따로 저장하는데,
   `convert_hf_to_gguf.py` 는 `tokenizer_config.json` 안만 봄. 그냥 두면 형식이 통째로 빠짐.
2. 형식을 GGUF 에 넣어도 Ollama 0.23.1 이 안 읽음. Modelfile 에 TEMPLATE 을 직접 적어야
   함. 안 적으면 모델이 오류 없이 같은 말을 반복함.

`ollama create` 가 안전텐서 폴더를 바로 읽는 길은 막혀 있음
(`unsupported architecture "Qwen3ForCausalLM"`). 반드시 GGUF 를 거쳐야 함.

llama.cpp 는 변환 스크립트만 있으면 됨. 빌드할 필요 없음:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp
    cd llama.cpp && git sparse-checkout set --no-cone '/*.py' '/conversion/**' '/gguf-py/**'
    pip install gguf sentencepiece

## fp16 - 재정렬 모델을 float16 사본으로

원본은 float32 파일 2.2GB 라 읽어서 float16 으로 바꾸느라 매번 5.4~5.8초가 걸림. 미리 바꿔
저장해 두면 첫 검색 4.9초, 그 뒤로는 1.8초임. 검색마다 올렸다 내리므로 이 차이가 그대로
응답 시간이 됨. 점수는 128쌍을 대조해 차이가 정확히 0 이고 순위도 같아서
`MIN_RERANK_SCORE` 를 다시 잴 필요가 없음.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src import config

DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER = "models/query-translator-dpo"

# 등급별 양자화 값. None 이면 GGUF 를 그대로(f16) 등록함
GRADE_QUANT = {"basic": "q4_K_M", "medium": "q8_0", "advanced": None}

# Ollama 0.23.1 이 GGUF 안의 대화 형식을 안 읽어서 직접 적음. Qwen3-Instruct-2507 이
# 쓰는 형식 그대로임: <|im_start|>역할\n내용<|im_end|>\n 을 메시지마다 반복하고 마지막에
# 답할 자리를 엶.
MODELFILE_TEMPLATE = '''FROM {gguf}
TEMPLATE """{{{{- range $i, $m := .Messages }}}}<|im_start|>{{{{ $m.Role }}}}
{{{{ $m.Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
'''


def _newer_libstdcxx() -> str | None:
    """변환 스크립트가 쓸 새 C++ 표준 라이브러리를 찾음.

    시스템 libstdc++ 에 CXXABI_1.3.15 가 없으면 pyarrow 가 ImportError 로 깨짐.
    `src/config._fix_libstdcxx` 와 같은 문제인데, 변환은 별도 프로세스라 여기서 또 함.
    """
    found = sorted(glob.glob("/home/*/anaconda3/lib/libstdc++.so.6*"))
    return found[-1] if found else None


def cmd_ollama(args) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    convert = Path(args.llama_cpp).expanduser() / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(f"변환 스크립트가 없다: {convert}\n"
                         f"이 파일 맨 위 설명대로 llama.cpp 를 받을 것.")

    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="export-"))
    merged, gguf = work / "merged", work / "merged-f16.gguf"
    work.mkdir(parents=True, exist_ok=True)

    # 1. LoRA 를 기본 모델에 합침
    print(f"[1/4] {args.adapter} 를 {args.base} 에 합치는 중...")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    del model

    # 2. 함정 1. 대화 형식을 tokenizer_config.json 안으로 옮김
    cfg_path, jinja = merged / "tokenizer_config.json", merged / "chat_template.jinja"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "chat_template" not in cfg:
        if not jinja.exists():
            raise SystemExit(f"대화 형식을 못 찾았다: {jinja}\n"
                             f"이대로 변환하면 모델이 오류 없이 같은 말만 반복한다.")
        cfg["chat_template"] = jinja.read_text(encoding="utf-8")
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[2/4] 대화 형식을 tokenizer_config.json 에 넣었다")
    else:
        print("[2/4] 대화 형식이 이미 들어 있다")

    # 3. GGUF 로 변환
    print("[3/4] GGUF 로 변환하는 중...")
    env = dict(os.environ)
    lib = _newer_libstdcxx()
    if lib:
        env["LD_PRELOAD"] = lib
    subprocess.run([sys.executable, str(convert), str(merged),
                    "--outfile", str(gguf), "--outtype", "f16"],
                   check=True, cwd=str(convert.parent), env=env)

    # 4. 함정 2. Modelfile 에 TEMPLATE 을 직접 적어 등록. GGUF 하나로 여러 판을 만듦
    modelfile = work / "Modelfile"
    modelfile.write_text(MODELFILE_TEMPLATE.format(gguf=gguf), encoding="utf-8")
    for grade in args.grades:
        name, label, note = config.ARXIV_REWRITER_CHOICES[grade]
        quant = GRADE_QUANT[grade]
        print(f"[4/4] ollama create {name} ({label}, {quant or 'f16 그대로'})...")
        cmd = ["ollama", "create"]
        if quant:
            cmd += ["-q", quant]
        subprocess.run(cmd + [name, "-f", str(modelfile)], check=True)

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
        print(f"중간 파일 삭제: {work}")
    made = [config.ARXIV_REWRITER_CHOICES[g][0] for g in args.grades]
    print(f"\n완료: {', '.join(made)}")
    print(f"확인: ollama show --modelfile {made[0]}  (TEMPLATE 이 비어 있으면 실패)")


def cmd_fp16(args) -> None:
    import torch
    from sentence_transformers import CrossEncoder

    out = Path(args.out)
    print(f"{args.model} 을 float16 으로 저장하는 중 -> {out}")
    ce = CrossEncoder(args.model, device="cpu", model_kwargs={"dtype": torch.float16})
    ce.model.half().save_pretrained(out)
    ce.tokenizer.save_pretrained(out)
    print("완료. 다음 검색부터 재정렬 모델 적재가 5.4~5.8초에서 1.8초로 줄어든다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="서비스가 쓸 모델을 미리 만듦")
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("ollama", help="arXiv 검색어 변환기를 Ollama 4비트로 등록")
    o.add_argument("--base", default=DEFAULT_BASE)
    o.add_argument("--adapter", default=DEFAULT_ADAPTER)
    o.add_argument("--grades", nargs="+", default=list(config.ARXIV_REWRITER_CHOICES),
                   choices=list(config.ARXIV_REWRITER_CHOICES),
                   help="만들 등급. 기본은 셋 다. GGUF 변환은 한 번만 하고 등록만 반복함")
    o.add_argument("--llama-cpp", required=True,
                   help="llama.cpp 폴더 (convert_hf_to_gguf.py 가 있는 곳)")
    o.add_argument("--work-dir", default=None,
                   help="중간 파일을 둘 곳. 합친 모델 7.6GB 와 GGUF 8GB 가 잠깐 생김")
    o.add_argument("--keep", action="store_true", help="중간 파일을 지우지 않음")
    o.set_defaults(func=cmd_ollama)

    f = sub.add_parser("fp16", help="재정렬 모델을 float16 사본으로 저장")
    f.add_argument("--model", default=config.RERANKER_MODEL)
    f.add_argument("--out", default=str(config.RERANKER_FP16_DIR))
    f.set_defaults(func=cmd_fp16)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
