# Lab 21 — Links

**Họ tên**: Nguyễn Kỳ Anh · **MSSV**: 2A202601558 · **Track 3 — Fine-tune LLM bằng LoRA**

## URL nộp bài

```
https://github.com/nguyenkyanh2003/Day21-Track3-Finetuning-Lab-2A202601558-NguyenKyAnh
```

Repo **public**. `python scripts/verify.py` → **26 passed · 1 warning · 0 failures ·
"Ready to submit."** (warning duy nhất là ghi chú rằng verdict FAILED vẫn được chấm đầy đủ.)

## Artefact bắt buộc — mục 6.1

| Đường dẫn | Nội dung | |
|---|---|---|
| `submission/REPORT.md` | report 7 mục, MSSV + họ tên ở đầu, ~4.400 từ | ✅ |
| `results/mask_proof.json` | bằng chứng loss mask (NB1) | ✅ |
| `results/template_check.json` | template có giữ `<think>` không (NB1) | ✅ |
| `results/token_stats.json` | p95 → `max_length` (NB1) | ✅ |
| `results/baselines_frozen.json` | ba baseline đóng băng trước khi train (NB2) | ✅ |
| `results/baseline_preds.json` | dự đoán của (a) và (b) trên từng mẫu (NB2) | ✅ |
| `results/runs.csv` | một dòng mỗi run huấn luyện (NB3 + NB4) | ✅ |
| `results/verdict.json` | phán quyết cổng hồi quy (NB5) | ✅ |
| `results/autopsy.json` | ba cấu hình sai chấm trên tập target (NB5) | ✅ |
| `results/qualitative.json` | dự đoán fine-tune từng mẫu (NB5) | ✅ |
| `notebooks/*.py` | nguồn jupytext, không có output | ✅ |
| `colab/*.ipynb` | sinh từ `notebooks/`, output rỗng | ✅ |

Không có trong repo, đúng như mục 6.1 dặn: `.env`, `.venv/`, `gguf/`, checkpoint
optimizer, trọng số base model, ba adapter đối chứng (`attn_only` / `wrong_lr` / `qlora`).

## `adapters/correct/` — vì sao không có ở đây

`adapter_model.safetensors` nặng **123,9 MB** (32.464.896 tham số × 4 byte; fp32 vì
`align_trainable_precision` ép trainable sang fp32 cho GradScaler của fp16). Vượt trần
**100 MB/file** của GitHub.

Đã thử **Git LFS** theo mục 6.2 và bị chặn ở tầng khác:

```
batch response: can not upload new objects to public fork
                nguyenkyanh2003/Day21-Track3-Finetuning-Lab-2A202601558-NguyenKyAnh
```

GitHub không nhận LFS object đẩy lên một **public fork** — object phải nằm trong kho LFS
của repo gốc, nơi tôi không có quyền ghi. Mục 6.1 xếp adapter vào loại **nên có**, không
bắt buộc; bắt buộc là `results/` + `submission/REPORT.md`, cả hai đều đầy đủ.

Adapter còn nguyên trên máy tại `adapters/correct/` và tái lập được:

```bash
EPOCHS=1 COMPUTE_TIER=T4 python notebooks/03_train_correct.py
```

với seed 42 và `max_steps=15` như ghi trong `results/runs.csv`.

## HuggingFace Hub — đã đẩy (bonus B5, +2)

```
https://huggingface.co/kyanhhh/lab21-2A202601558-qwen35-triage-vi
```

Repo **public**, chứa: `adapter_model.safetensors` (123,9 MB) + `adapter_config.json`,
**toàn bộ `results/`**, `submission/REPORT.md`, `submission/REFLECTION.md`, và một model
card ghi rõ base model cùng phán quyết FAILED.

Đẩy bằng `python scripts/push_to_hub.py`. Đây cũng là nơi lấy `adapters/correct/`
mà GitHub không nhận.

## Kết quả tóm tắt

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0,000 | 0,7578 | 0,000 | 3173 |
| **(b) base + optimized prompt** | **0,765** | 0,7578 | **1,000** | **986** |
| (c) LoRA fine-tune | 0,535 | 0,7556 | 0,990 | 1465 |

**Verdict: FAILED** — `target Δ = −0,230`, `regression Δ = −0,0022`.
Phân tích đầy đủ ở `submission/REPORT.md` §5–§7.

## Tái lập

```bash
python scripts/verify.py       # gatekeeper
python scripts/fill_report.py  # sinh lại bảng §1-§6 từ results/
```
