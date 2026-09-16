"""
서비스가 쓸 모델을 미리 만들어 두는 스크립트
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

GRADES = {
    "basic":    (config.ARXIV_REWRITER_MODEL, "q4_K_M"),
    "medium":   ("papers-rewriter-q8", "q8_0"),
    "advanced": ("papers-rewriter-f16", None),
}

MODELFILE_TEMPLATE = '''FROM {gguf}
TEMPLATE """{{{{- range $i, $m := .Messages }}}}<|im_start|>{{{{ $m.Role }}}}
{{{{ $m.Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
'''


def _newer_libstdcxx() -> str | None:
    """변환 스크립트가 쓸 C++ 표준 라이브러리를 찾음"""
    found = sorted(glob.glob("/home/*/anaconda3/lib/libstdc++.so.6*"))
    return found[-1] if found else None


def cmd_ollama(args) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    convert = Path(args.llama_cpp).expanduser() / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(f"변환 스크립트가 없다: {convert}\n"
                         f"이 파일 맨 위 설명대로 llama.cpp를 받을 것.")

    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="export-"))
    merged, gguf = work / "merged", work / "merged-f16.gguf"
    work.mkdir(parents=True, exist_ok=True)

    # 1. LoRA를 베이스 모델에 적용
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    del model

    # 2. 대화 형식을 tokenizer_config.json 안으로 옮김
    cfg_path, jinja = merged / "tokenizer_config.json", merged / "chat_template.jinja"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "chat_template" not in cfg:
        if not jinja.exists():
            raise SystemExit(f"대화 형식을 못 찾았다: {jinja}\n"
                             f"이대로 변환하면 모델이 오류 없이 같은 말만 반복한다.")
        cfg["chat_template"] = jinja.read_text(encoding="utf-8")
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. GGUF로 변환
    env = dict(os.environ)
    lib = _newer_libstdcxx()
    if lib:
        env["LD_PRELOAD"] = lib
    subprocess.run([sys.executable, str(convert), str(merged),
                    "--outfile", str(gguf), "--outtype", "f16"],
                   check=True, cwd=str(convert.parent), env=env)

    # 4. Modelfile에 TEMPLATE을 직접 적어 등록. GGUF 하나로 여러 등급을 만듦
    modelfile = work / "Modelfile"
    modelfile.write_text(MODELFILE_TEMPLATE.format(gguf=gguf), encoding="utf-8")
    for grade in args.grades:
        name, quant = GRADES[grade]
        cmd = ["ollama", "create"]
        if quant:
            cmd += ["-q", quant]
        subprocess.run(cmd + [name, "-f", str(modelfile)], check=True)

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)


def cmd_fp16(args) -> None:
    import torch
    from sentence_transformers import CrossEncoder

    out = Path(args.out)
    ce = CrossEncoder(args.model, device="cpu", model_kwargs={"dtype": torch.float16})
    ce.model.half().save_pretrained(out)
    ce.tokenizer.save_pretrained(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("ollama")
    o.add_argument("--base", default=DEFAULT_BASE)
    o.add_argument("--adapter", default=DEFAULT_ADAPTER)
    o.add_argument("--grades", nargs="+", default=list(GRADES), choices=list(GRADES))
    o.add_argument("--llama-cpp", required=True)
    o.add_argument("--work-dir", default=None)
    o.add_argument("--keep", action="store_true")
    o.set_defaults(func=cmd_ollama)

    f = sub.add_parser("fp16")
    f.add_argument("--model", default=config.RERANKER_MODEL)
    f.add_argument("--out", default=str(config.RERANKER_FP16_DIR))
    f.set_defaults(func=cmd_fp16)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
