# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Kỳ Anh  **MSSV**: 2A202601558  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16 GB (14.6 GB khả dụng, Colab Free)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
> Các bảng số được sinh bằng `python scripts/fill_report.py` đọc thẳng từ `results/`,
> không gõ tay — xem phần "Khai báo thay đổi" ở cuối.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (corpus mặc định của lab, không đổi) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | **1024** — p95 đo được là **98** token *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → `<n>` optimizer step *(results/runs.csv)* |

**Template có giữ khối `<think>` không?** **CÓ** — *(results/template_check.json)*

`thinking_survives()` render một message assistant chứa trace thật và tìm lại nó trong
chuỗi đầu ra. Kết quả: `open_tag_present=true`, `body_present=true`, verdict
`"reasoning preserved — safe to train on traces"`. Chuỗi render được ghi nguyên văn
trong artefact:

```
<|im_start|>user
2+2?<|im_end|>
<|im_start|>assistant
<think>
buoc 1: kiem tra. buoc 2: tra loi.
</think>

4<|im_end|>
```

Nghĩa là template Qwen3.5 **không** nuốt nội dung `<think>` trong `apply_chat_template`.
Đây là kiểm tra deck §16 bắt làm một lần cho mỗi base model, vì nếu template xoá trace
thì reasoning trong dataset *không bao giờ* tới được hàm loss và **không có gì báo lỗi**.
Với model này tôi không phải xử lý gì thêm; nhưng xem mục "quan sát về `MASK_MODE`" ở §2
— template an toàn không có nghĩa là corpus của tôi có trace để bảo vệ.

### Vì sao `max_length=1024` chứ không phải 256 như p95 gợi ý

`results/token_stats.json` đo trên đúng tokenizer sẽ dùng để train:

| n | mean | p50 | p95 | p99 | max | gợi ý |
|---|---|---|---|---|---|---|
| 250 | 93.1 | 93 | **98** | 100 | 101 | `256` |

Hàm `_round_pow2(p95)` gợi ý `max_length=256`, còn tier `T4` đặt `1024`. Tôi **giữ 1024**,
và đây là lý do — không phải vì tier bảo thế:

1. Trong lab này `max_length` **chỉ dùng để cắt**, không pad tới độ dài cố định:
   `build_example()` (`src/labkit/data.py:152`) chỉ cắt khi `len(full_ids) > max_length`.
2. Chuỗi dài nhất trong toàn corpus là **101 token**. Vậy ở cả 256 lẫn 1024, số mẫu bị
   cắt là **0/250** — hai lựa chọn tạo ra **dữ liệu huấn luyện giống hệt nhau**.
3. Tier `T4` dùng `per_device_batch=1`, nên trong một batch không có mẫu thứ hai để pad
   theo. Không có chi phí padding nào để tiết kiệm.
4. Biên còn lại là biên an toàn: nếu tôi thay corpus bằng ticket dài hơn (bonus B2),
   1024 vẫn đúng còn 256 sẽ âm thầm cắt mất câu trả lời.

Nói cách khác: con số 1024 ở đây là một **quyết định có số đo đứng sau**, và số đo cho
biết lựa chọn này không tốn gì. Nếu tier dùng `per_device_batch > 1` (như `BIGGPU`) thì
kết luận sẽ ngược lại và tôi sẽ hạ xuống 256.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (`supervised_preview`, nguyên văn từ `results/mask_proof.json`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị che (`masked_preview`):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

Đối chứng với `MASK_MODE=everything` trên **cùng một mẫu**: `supervised 94/94 (100%)`,
và cả câu hỏi của người dùng nằm trong phần được tính loss. Đó chính là con bug deck §16
mô tả — model học cách *viết lại câu hỏi* thay vì trả lời — và nó chỉ khác đúng một biến
môi trường. `verify.py` loại thẳng bài nộp có `supervised_fraction ≥ 0.95`; ở đây là
0.4149, cách ngưỡng rất xa.

### Quan sát: vì sao `MASK_MODE` là no-op trên corpus này

Đọc kỹ hai khối trên: **`<think>\n\n` nằm ở phần BỊ CHE, và `</think>` là token đầu tiên
được tính loss.** Chat template của Qwen3.5 tự mở *và đóng* một khối `<think></think>`
rỗng ngay bên trong generation prompt, còn cả 250 câu trả lời huấn luyện đều là JSON
trần. Hệ quả: bên trong vùng được giám sát **không còn khối suy luận nào để loại ra**,
nên `assistant-only`, `masked-think` và `response-only` sinh ra **mask giống hệt nhau**
(F-30 trong `SIMULATION-FINDINGS.md`).

Điều này quan trọng với bonus **B3 (reasoning-trace collapse)**: chạy hai `MASK_MODE`
trên corpus mặc định sẽ cho hai kết quả *bằng nhau* và không chứng minh được gì. Muốn
tái lập §13.5 thì bắt buộc phải có dữ liệu huấn luyện chứa trace thật.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | `<0.xx>` | `<0.xx>` | `<0.xx>` | `<0.xx>` |
| (b) base + optimized prompt | `<0.xx>` | `<0.xx>` | `<0.xx>` | `<0.xx>` |
| (c) LoRA fine-tune | `<0.xx>` | `<0.xx>` | `<0.xx>` | `<0.xx>` |

**(b) có thật sự mạnh hơn (a) không?** `<0.xx>`

**Bạn có sửa `OPTIMIZED_PROMPT` không?** **KHÔNG.** `verify.py` so `optimized_prompt_sha`
trong `results/baselines_frozen.json` với SHA của prompt đi kèm lab; giá trị phải trùng.
Tôi không đụng tới nó, không đụng tới `eval_target.jsonl` và `eval_regression.jsonl`
(checksum trong `data/checksums.json` giữ nguyên). Thứ tự chạy là NB2 **trước** NB3 nên
mốc (b) được chốt trước khi tôi nhìn thấy bất kỳ kết quả huấn luyện nào.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | `<0.xx>` | 1e-4 | `<0.xx>` | `<0.xx>` | `<0.xx>` | `<0.xx>` |
| `attn_only` | q,v | *(matched)* | `<0.xx>` | 1e-4 | `<0.xx>` | `<0.xx>` | `<0.xx>` | `<0.xx>` |
| `wrong_lr` | text-linear | 16 | `<0.xx>` | 1e-5 | `<0.xx>` | `<0.xx>` | `<0.xx>` | `<0.xx>` |
| `qlora` | text-linear | 16 | `<0.xx>` | 1e-4 | `<0.xx>` | `<0.xx>` | `<0.xx>` | `<0.xx>` |

**Xếp hạng bằng cột `target`, không bằng cột train loss.**

**Thiết kế công bằng — mỗi run đổi đúng một biến:**

| Run | Biến duy nhất đổi so với `correct` | Giữ nguyên |
|---|---|---|
| `attn_only` | **vị trí gắn adapter** (`text-linear` → `q,v`), rank được nâng bằng `matched_rank()` để giữ nguyên ngân sách tham số | LR, step, dữ liệu, precision |
| `wrong_lr` | **learning rate** (1e-4 → 1e-5, tức thang của full fine-tune) | vị trí, rank, step, dữ liệu |
| `qlora` | **độ chính xác của base** (16-bit → NF4 4-bit) | vị trí, rank, LR, step, dữ liệu |

Đây là điểm khác biệt so với lab cũ: so `q,v @ r=16` với `all-linear @ r=16` là so **ngân
sách tham số**, không phải so **vị trí**, nên không kết luận được gì về vị trí. `matched_rank()`
nâng r của `attn_only` lên để hai run có cùng số tham số huấn luyện (`verify.py` bắt buộc
lệch <5%), khi đó biến duy nhất còn lại thật sự là *nơi* adapter được gắn.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

`<0.xx>`

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`<0.xx>`

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

`<0.xx>`

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `<0.xx>`
`target Δ = <0.xx>` · `regression Δ = <0.xx>` · `valid_trace_rate = <0.xx>`

Diễn giải (≥100 từ):

`<0.xx>`

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `<0.xx>` | | | | ✅ FT thắng |
| 2 | `<0.xx>` | | | | ✅ FT thắng |
| 3 | `<0.xx>` | | | | ❌ **FT thua** |
| 4 | `<0.xx>` | | | | ❌ **FT thua** |
| 5 | `<0.xx>` | | | | |

Năm ví dụ được chọn **tự động** bằng `scripts/fill_report.py`: hai mẫu có
`Δ = score_ft − score_b` âm nhất, hai mẫu Δ dương nhất, một mẫu hoà — trên toàn bộ tập
eval, không phải do tôi nhặt tay. Cách chọn này loại trừ khả năng cherry-pick về mặt cấu
tạo, chứ không phải bằng lời hứa.

Có mẫu chung nào ở các ca FT thua không?

`<0.xx>`

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).**

`<0.xx>`

**Ba điều tôi học được:**

1. **Mask không phải chuyện lý thuyết, nó là một chuỗi bạn đọc được.** Trước lab tôi nghĩ
   "mask câu hỏi" là một cờ trong config. Khi in ra `supervised_preview`, tôi thấy token
   được giám sát đầu tiên là `</think>` chứ không phải `{` — tức chat template đã tự chèn
   và đóng một khối suy luận rỗng trước cả khi câu trả lời bắt đầu. Cùng một mẫu, đổi
   `MASK_MODE=everything`, con số nhảy từ 39/94 lên 94/94 và câu hỏi của người dùng lọt
   vào loss. Khác biệt giữa một model hoạt động và một model viết lại câu hỏi là một biến
   môi trường — và cách duy nhất để biết mình đang ở phía nào là giải mã ngược cái mask.
2. `<0.xx>`
3. `<0.xx>`

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

`<0.xx>`

---

## Khai báo thay đổi so với repo gốc

Rubric yêu cầu khai báo mọi thay đổi ảnh hưởng tới phép so sánh. Tôi đã thay đổi hai chỗ,
**cả hai đều không chạm vào tập eval, prompt, hay cách chấm điểm**:

| Thay đổi | Là gì | Vì sao |
|---|---|---|
| `notebooks/02_baselines.py` | thêm một lần `report.write_json(...)` ghi `results/baseline_preds.json` (dự đoán của (a) và (b) trên từng mẫu eval) | NB2 tính `preds_b` rồi vứt đi, chỉ giữ điểm tổng hợp. Mục 6 và rubric 3.4 lại đòi so sánh (b) với (c) **trên từng ticket**. Không lưu ở đây thì cách duy nhất lấy lại là sinh toàn bộ tập eval thêm một lượt (~17 phút T4). |
| `scripts/fill_report.py` | script mới, đọc `results/` và sinh các bảng của report | Rubric 4.3 chấm "số trong report khớp file trong results/". Sinh bảng từ artefact làm điều đó đúng theo cấu tạo thay vì theo sự cẩn thận khi chép tay. |
| `.gitattributes` *(mới)* | `data/*.jsonl -text` | Trên Windows `core.autocrlf=true` đổi LF → CRLF lúc checkout. `verify.py` băm các file này **theo byte**, nên một bản clone chưa ai đụng vào vẫn bị báo `eval sets unmodified: FAIL` — một cáo buộc sửa tập eval không có thật. Đã xác nhận: SHA của bản LF khớp `data/checksums.json` chính xác cho cả 4 file. |
| `scripts/verify.py` | thêm `sys.stdout.reconfigure(encoding="utf-8")` ở đầu `main()` | Console Windows dùng cp1252 và **crash bằng `UnicodeEncodeError`** đúng lúc in ra dòng báo `REPORT.md` còn là template (chuỗi chứa `<điền>`). Một cổng kiểm tra chết ngay khi báo lỗi thì vô dụng. Không đổi một tiêu chí kiểm tra nào. |
| `colab/Lab21_RUN_ALL.ipynb` | `REPO` trỏ về fork của tôi; thêm ô 5 chạy `fill_report.py` rồi zip + tải `results/` | Ô Setup gốc clone repo *upstream*, nên mọi thay đổi trên sẽ không có mặt khi chạy Colab. Và vì `results/` + `adapters/` nằm trong `.gitignore`, `git push` không mang kết quả về — phải tải tay. |

Không đổi: `data/*.jsonl` (checksum nguyên vẹn), `OPTIMIZED_PROMPT` (SHA nguyên vẹn),
`NAIVE_PROMPT`, hàm chấm điểm, cổng hồi quy, và cấu hình của cả bốn run huấn luyện.
`colab/*.ipynb` đã được sinh lại bằng `scripts/build_colab.py` để khớp nguồn (F-18).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse — **không khả thi trên corpus mặc định**, xem §2: cả ba
      `MASK_MODE` không-`everything` sinh ra mask giống hệt nhau (F-30). Cần làm B2 trước.
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
