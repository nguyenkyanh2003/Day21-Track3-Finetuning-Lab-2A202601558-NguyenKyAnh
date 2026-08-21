# Lab 21 — Links

**Họ tên**: Nguyễn Kỳ Anh · **MSSV**: 2A202601558 · **Track 3 — Fine-tune LLM bằng LoRA**

## Kênh nộp: GitHub

**URL nộp bài**

```
https://github.com/nguyenkyanh2003/Day21-Track3-Finetuning-Lab-2A202601558-NguyenKyAnh
```

Repo **public**. Artefact theo mục 6.1:

| Đường dẫn | Nội dung | Trạng thái |
|---|---|---|
| `submission/REPORT.md` | report 7 mục, có MSSV + họ tên ở đầu | §1–§2 đã điền bằng số NB1 |
| `results/mask_proof.json` | bằng chứng loss mask (NB1) | ✅ |
| `results/template_check.json` | template có giữ `<think>` không (NB1) | ✅ |
| `results/token_stats.json` | p95 → `max_length` (NB1) | ✅ |
| `results/baselines_frozen.json` | ba baseline đóng băng (NB2) | ⏳ cần GPU |
| `results/runs.csv` | một dòng mỗi run huấn luyện (NB3 + NB4) | ⏳ cần GPU |
| `results/verdict.json` | phán quyết cổng hồi quy (NB5) | ⏳ cần GPU |
| `results/autopsy.json` | ba cấu hình sai chấm trên tập target (NB5) | ⏳ cần GPU |
| `results/qualitative.json` | ví dụ định tính (NB5) | ⏳ cần GPU |
| `results/baseline_preds.json` | dự đoán của (a) và (b) từng mẫu (NB2) | ⏳ cần GPU |
| `adapters/correct/` | adapter chính (NB3) | ⏳ cần GPU |
| `notebooks/*.py` | nguồn jupytext, không có output | ✅ |
| `colab/*.ipynb` | sinh từ `notebooks/` bằng `scripts/build_colab.py`, output rỗng | ✅ |

Không có trong repo (đúng như mục 6.1 yêu cầu): `.env`, `.venv/`, `gguf/`,
checkpoint optimizer, trọng số base model, ba adapter đối chứng.

## Kênh HuggingFace Hub

Chưa đẩy. Cần `adapters/correct/` trước, tức cần chạy NB3 xong.

Nếu đẩy, repo sẽ là `nguyenkyanh2003/lab21-2A202601558-qwen35-triage-vi`
(base model: `unsloth/Qwen3.5-4B`, tier `T4`) — tính thưởng **B5 (+2)**, và link
sẽ được ghi vào phụ lục của `submission/REPORT.md`.

## Kiểm tra trước khi nộp

```bash
python scripts/verify.py       # gatekeeper — phải không còn dòng FAIL
python scripts/fill_report.py  # sinh lại bảng §1-§6 từ results/
```
