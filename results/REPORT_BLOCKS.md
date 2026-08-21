# Khối dán vào submission/REPORT.md

> Sinh tự động từ `results/` bằng `scripts/fill_report.py`. Số ở đây khớp artefact theo cấu tạo — đừng gõ lại bằng tay.

## §1 — Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | p95 đo được = **98** token (p50=93, p99=100, max=101) → `suggested_max_length=256` |
| `MASK_MODE` | `assistant-only` |
| Tier / model | `—` / `—` |
| max_steps | — optimizer step |
| Precision | `—` |

**Template giữ khối `<think>`?** CÓ — `reasoning preserved — safe to train on traces`

## §2 — Mask proof

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

## §3 — Ba baseline

> **Thiếu `results/baselines_frozen.json (NB2)`** — chạy notebook sinh ra nó rồi chạy lại script này.

## §4 — Giải phẫu cấu hình sai

> **Thiếu `results/runs.csv (NB3 + NB4)`** — chạy notebook sinh ra nó rồi chạy lại script này.

## §5 — Phán quyết

> **Thiếu `results/verdict.json (NB5)`** — chạy notebook sinh ra nó rồi chạy lại script này.

## §6 — Định tính (2 ca FT THUA + 2 ca FT THẮNG + 1 ca hoà)

> **Thiếu `results/qualitative.json (NB5)`** — chạy notebook sinh ra nó rồi chạy lại script này.
