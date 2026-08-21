#!/usr/bin/env python3
"""Đẩy adapter + artefact lên HuggingFace Hub (bonus B5, +2).

Vì sao cần đến nó. `adapters/correct/adapter_model.safetensors` nặng 123,9 MB —
32.464.896 tham số × 4 byte, fp32 vì `align_trainable_precision` ép trainable sang fp32
cho GradScaler của fp16. GitHub chặn file >100 MB, và Git LFS cũng không cứu được vì
GitHub từ chối nhận LFS object đẩy lên một **public fork**:

    batch response: can not upload new objects to public fork

Mục 6.2 của Codelabs cho đúng lối thoát này: đẩy adapter sang HuggingFace rồi để link
trong REPORT.md / LINKS.md. Hub không có giới hạn đó.

Codelabs cũng nói rõ: *"Sau push_to_hub vẫn phải upload thêm results/ và
submission/REPORT.md — chỉ có adapter thì không chấm được."* Nên script này đẩy cả ba.

    huggingface-cli login          # dán token Write, chỉ lưu trên máy bạn
    python scripts/push_to_hub.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
USER = "kyanhhh"
REPO_ID = f"{USER}/lab21-2A202601558-qwen35-triage-vi"

CARD = """---
base_model: unsloth/Qwen3.5-4B
library_name: peft
license: apache-2.0
language:
- vi
tags:
- lora
- text-classification
- vietnamese
---

# Lab 21 — LoRA triage adapter (Qwen3.5-4B)

**Nguyễn Kỳ Anh · MSSV 2A202601558** · AICB-P2T3 Ngày 21, Track 3.

Adapter LoRA fine-tune `unsloth/Qwen3.5-4B` cho tác vụ phân loại ticket CSKH tiếng Việt
thành JSON 4 trường: `intent`, `urgency`, `product`, `sentiment`.

Report đầy đủ: [`submission/REPORT.md`](./submission/REPORT.md) trong chính repo này.
Mã nguồn: <https://github.com/nguyenkyanh2003/Day21-Track3-Finetuning-Lab-2A202601558-NguyenKyAnh>

## Kết quả — và vì sao KHÔNG nên dùng adapter này

| Run | target | regression | format | latency |
|---|---|---|---|---|
| (a) base + prompt ngây thơ | 0.000 | 0.7578 | 0.000 | 3173 ms |
| **(b) base + prompt tối ưu** | **0.765** | 0.7578 | **1.000** | **986 ms** |
| (c) adapter này | 0.535 | 0.7556 | 0.990 | 1465 ms |

**Cổng hồi quy: FAILED** (`target Δ = -0.230`). Adapter thua chính base model khi base
được prompt tử tế, và còn chậm hơn 1,49×. Nó được công bố làm **artefact của một thí
nghiệm có phán quyết âm**, không phải làm model để dùng.

Nguyên nhân, đo được: adapter học được *hình dạng* JSON (format 0.99) và học được *chép*
tên sản phẩm (47/50 đúng), nhưng **không học được bộ nhãn** — 70% đầu ra `intent` nằm
ngoài từ vựng đóng 5 giá trị, so với 0% của baseline (b), vì prompt (b) liệt kê thẳng các
giá trị hợp lệ vào context. Trong 32 ca thua: `intent` sai 28, `sentiment` 23, `urgency`
21, `product` chỉ 2.

43% khoảng cách điểm chỉ là **dấu tiếng Việt**: adapter viết `hỏi_thông_tin` thay vì
`hoi_thong_tin`. Chấm lại sau khi bỏ dấu, (c) lên 0.635 còn (b) đứng yên 0.765.

## Cấu hình huấn luyện

| | |
|---|---|
| Base | `unsloth/Qwen3.5-4B` |
| Phần cứng | Colab Free Tesla T4 (14.6 GB), **fp16** (Turing không có bf16) |
| LoRA | `r=16`, `alpha=32`, 12 lớp text-linear (không gắn vào vision tower) |
| Tham số huấn luyện | 32.464.896 |
| LR | `1e-4` (10× thang full fine-tune) |
| Batch hiệu dụng | 16 (`per_device=1` × `grad_accum=16`) |
| max_steps | **15** (`EPOCHS=1` — ngân sách thời gian lab) |
| max_length | 1024 (p95 đo được: 98 token) |
| Loss mask | `assistant-only`, `supervised_fraction=0.4149` |
| Train loss cuối | 1.3454 · 476.1 s · peak VRAM 12.01 GB |

## Dùng thử

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3.5-4B", dtype="float16",
                                            device_map="auto")
model = PeftModel.from_pretrained(base, "__REPO__")
tok = AutoTokenizer.from_pretrained("__REPO__")
```

Nếu bạn thật sự cần phân loại triage tiếng Việt: **dùng base model với prompt liệt kê từ
vựng** thay vì adapter này. Nó chính xác hơn 0.230 điểm và nhanh hơn 1,49×.
""".replace("__REPO__", REPO_ID)


def main() -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("thieu huggingface_hub -> pip install -U huggingface_hub", file=sys.stderr)
        return 1

    adapter = ROOT / "adapters" / "correct"
    if not adapter.is_dir():
        print(f"khong thay {adapter} -- chay NB3 truoc", file=sys.stderr)
        return 1

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception:
        print("chua dang nhap -> chay: huggingface-cli login", file=sys.stderr)
        return 1
    print(f"dang nhap voi: {who}")

    api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
    print(f"repo: https://huggingface.co/{REPO_ID}")

    # 1. adapter (123,9 MB -- Hub khong co tran 100 MB nhu GitHub)
    print("dang day adapters/correct/ ...")
    api.upload_folder(folder_path=str(adapter), repo_id=REPO_ID, repo_type="model")

    # 2. results/ -- Codelabs: "chi co adapter thi khong cham duoc"
    print("dang day results/ ...")
    api.upload_folder(folder_path=str(ROOT / "results"), path_in_repo="results",
                      repo_id=REPO_ID, repo_type="model")

    # 3. report + reflection
    for rel in ("submission/REPORT.md", "submission/REFLECTION.md"):
        print(f"dang day {rel} ...")
        api.upload_file(path_or_fileobj=str(ROOT / rel), path_in_repo=rel,
                        repo_id=REPO_ID, repo_type="model")

    # 4. model card -- ghi de README.md ma save_pretrained sinh ra
    print("dang day model card ...")
    api.upload_file(path_or_fileobj=CARD.encode("utf-8"), path_in_repo="README.md",
                    repo_id=REPO_ID, repo_type="model")

    print()
    print("XONG. Link cho REPORT.md / LINKS.md:")
    print(f"  https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
