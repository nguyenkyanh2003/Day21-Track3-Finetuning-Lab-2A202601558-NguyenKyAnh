# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Kỳ Anh  **MSSV**: 2A202601558  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16 GB (14,6 GB khả dụng, Colab Free)` · `precision=fp16`

> Mọi con số dưới đây khớp với file trong `results/`. Các bảng được sinh bằng
> `python scripts/fill_report.py` đọc thẳng từ artefact, không gõ tay.

**Khai báo ngay từ đầu:** tôi chạy với `EPOCHS=1` (**15 optimizer step**) thay vì mặc định 2,
vì quỹ thời gian lab còn ~2,5 giờ. `EVAL_LIMIT` **để trống** — cả hai tập eval đều đầy đủ
(50 target + 15 regression), `smoke_mode=false`. Cả bốn run huấn luyện dùng chung
`max_steps=15`, nên mọi phép so sánh được chấm vẫn hợp lệ. Ngân sách huấn luyện nhỏ này
là nguyên nhân trực tiếp của phán quyết ở §5, và tôi phân tích nó ở đó thay vì giấu đi.

Tổng thời gian chạy thật: **3381 s (56,3 phút)** — nb1 19 s · nb2 411 s · nb3 540 s ·
nb4 1603 s · nb5 808 s.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (corpus mặc định, **không đổi**) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | **1024** — p95 đo được là **98** token *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 1 epoch → **15** optimizer step *(results/runs.csv)* |
| Precision | `fp16` — T4 là Turing (sm_75), **không có bf16** |

**Template có giữ khối `<think>` không?** **CÓ** — *(results/template_check.json)*

`thinking_survives()` render một message assistant chứa trace thật rồi tìm lại nó trong
chuỗi đầu ra: `open_tag_present=true`, `body_present=true`, verdict
`"reasoning preserved — safe to train on traces"`. Chuỗi render nguyên văn:

```
<|im_start|>user
2+2?<|im_end|>
<|im_start|>assistant
<think>
buoc 1: kiem tra. buoc 2: tra loi.
</think>

4<|im_end|>
```

Template Qwen3.5 **không** nuốt nội dung `<think>` trong `apply_chat_template`. Đây là
kiểm tra deck §16 bắt làm một lần cho mỗi base model, vì nếu template xoá trace thì
reasoning trong dataset *không bao giờ* tới được hàm loss và **không có gì báo lỗi**. Với
model này tôi không phải xử lý gì thêm — nhưng xem §2 để thấy "template an toàn" không
đồng nghĩa với "corpus của tôi có trace để bảo vệ".

### Vì sao `max_length=1024` chứ không phải 256 như p95 gợi ý

| n | mean | p50 | p95 | p99 | max | gợi ý |
|---|---|---|---|---|---|---|
| 250 | 93,1 | 93 | **98** | 100 | 101 | `256` |

`_round_pow2(p95)` gợi ý `max_length=256`, tier `T4` đặt `1024`. Tôi **giữ 1024**, và lý
do là số đo chứ không phải "vì tier bảo thế":

1. Trong lab này `max_length` **chỉ dùng để cắt**, không pad tới độ dài cố định —
   `build_example()` (`src/labkit/data.py:152`) chỉ cắt khi `len(full_ids) > max_length`.
2. Chuỗi dài nhất toàn corpus là **101 token**, nên ở cả 256 lẫn 1024 số mẫu bị cắt là
   **0/250**: hai lựa chọn tạo ra **dữ liệu huấn luyện giống hệt nhau**.
3. Tier `T4` dùng `per_device_batch=1`, trong batch không có mẫu thứ hai để pad theo, nên
   không có chi phí padding nào để tiết kiệm.
4. Phần dư là biên an toàn cho corpus khác dài hơn.

Nếu tier dùng `per_device_batch > 1` (như `BIGGPU`) thì kết luận sẽ ngược lại và tôi hạ
xuống 256.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0,4149** (39/94 token) |
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

Đối chứng `MASK_MODE=everything` trên **cùng một mẫu**: `supervised 94/94 (100%)`, cả câu
hỏi người dùng nằm trong loss. Đó chính là con bug deck §16 mô tả — model học *viết lại
câu hỏi* — và nó chỉ khác đúng một biến môi trường. Ngưỡng loại của `verify.py` là
`supervised_fraction ≥ 0.95`; 0,4149 cách rất xa.

### Quan sát: vì sao `MASK_MODE` là no-op trên corpus này

Đọc kỹ hai khối trên: **`<think>\n\n` nằm ở phần BỊ CHE, và `</think>` là token đầu tiên
được tính loss.** Chat template Qwen3.5 tự mở *và đóng* một khối `<think></think>` rỗng
ngay trong generation prompt, còn cả 250 câu trả lời huấn luyện đều là JSON trần. Hệ quả:
bên trong vùng được giám sát **không còn khối suy luận nào để loại ra**, nên
`assistant-only`, `masked-think` và `response-only` sinh ra **mask giống hệt nhau**.

Điều này cũng giải thích `valid_trace_rate = 0.0` ở §5: model được huấn luyện để trả JSON
trần, và nó làm đúng thế. Và nó khiến bonus **B3** không khả thi trên corpus mặc định —
chạy hai `MASK_MODE` sẽ cho hai kết quả bằng nhau và không chứng minh được gì.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0,000 | 0,7578 | 0,000 | 3173 |
| **(b) base + optimized prompt** | **0,765** | 0,7578 | **1,000** | **986** |
| (c) LoRA fine-tune | 0,535 | 0,7556 | 0,990 | 1465 |

**(b) có thật sự mạnh hơn (a) không?** **CÓ, và cách biệt tuyệt đối**: 0,000 → 0,765.
Prompt ngây thơ không sinh ra được một JSON hợp lệ nào (`format=0,000`), nên
`target=0,000` không phải vì model không hiểu bài toán mà vì đầu ra không parse được.
Prompt tối ưu vừa đưa format lên **1,000**, vừa chạy **nhanh hơn 3,2×** (986 ms so với
3173 ms) — vì nó chặn model viết lời dẫn dài dòng trước khi trả JSON. Đây đúng là luận
điểm deck §17: baseline rẻ hơn *và* mạnh hơn.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** **KHÔNG.** `optimized_prompt_sha` trong
`results/baselines_frozen.json` là `719e74d3b6232053`, `verify.py` xác nhận trùng với
prompt đi kèm lab. Checksum `data/*.jsonl` nguyên vẹn. NB2 chạy **trước** NB3 nên mốc (b)
được chốt trước khi tôi thấy bất kỳ kết quả huấn luyện nào. `smoke_mode=false`,
`n_target=50`, `n_regression=15`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32.464.896 | 1e-4 | 1,3454 | 0,535 | 0,99 | 476,1 | 12,01 |
| `attn_only` | q,v | **283** *(matched)* | 32.456.704 | 1e-4 | **1,1302** | **0,585** | 1,00 | 423,7 | 12,02 |
| `wrong_lr` | text-linear | 16 | 32.464.896 | 1e-5 | 2,0513 | **0,000** | 0,00 | 484,1 | 12,01 |
| `qlora` | text-linear | 16 | 32.464.896 | 1e-4 | 1,5300 | 0,310 | 0,92 | 514,3 | **7,09** |

**Ngân sách công bằng (2.1):** `correct` 32.464.896 vs `attn_only` 32.456.704 → lệch
**0,025%**, ngưỡng <5% → ĐẠT. **Cùng ngân sách step (2.2):** cả bốn run `max_steps=15` → ĐẠT.

**Mỗi run đổi đúng một biến (2.3):**

| Run | Biến duy nhất đổi so với `correct` | Giữ nguyên |
|---|---|---|
| `attn_only` | **vị trí gắn adapter** (`text-linear` → `q,v`); rank nâng lên 283 bằng `matched_rank()` để giữ nguyên ngân sách tham số | LR, step, dữ liệu, precision |
| `wrong_lr` | **learning rate** (1e-4 → 1e-5, thang full fine-tune) | vị trí, rank, step, dữ liệu |
| `qlora` | **độ chính xác của base** (16-bit → NF4 4-bit) | vị trí, rank, LR, step, dữ liệu |

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà?**

Nó **THẮNG**: 0,585 so với 0,535, hơn 0,050 (+9,3% tương đối), lại còn `format=1,00` so
với 0,99 và nhanh hơn **1,66×** (882 ms so với 1465 ms). Với ngân sách tham số khớp trong
0,025% và cùng 15 step, biến duy nhất khác nhau là *nơi* adapter được gắn — nên kết quả
này nói thẳng rằng **ở ngân sách 15 step trên tác vụ này, vị trí không phải đòn bẩy theo
chiều deck §10.2 dự đoán**; dồn cùng một lượng tham số vào `q,v` với rank rất cao (283)
lại tốt hơn là rải mỏng ra 12 lớp linear với r=16.

Thứ tự theo train loss (attn_only 1,1302 > correct 1,3454 > qlora 1,53 > wrong_lr 2,0513)
lần này **trùng** với thứ tự theo target. Tôi không coi đó là bằng chứng train loss là
thang đo đúng, mà là một lần trùng may — và §4.2 cho thấy vì sao: cùng bảng đó, khoảng
cách *độ lớn* giữa hai cột lệch nhau tới mức nếu chỉ đọc loss thì kết luận sẽ sai hoàn
toàn. Deck gọi vị trí attention-only là **Lỗi #1**; số đo của tôi không ủng hộ điều đó ở
ngân sách này. Cách đọc trung thực là: 15 step quá ít để 12 lớp linear ở r=16 kịp học,
trong khi r=283 trên `q,v` có đủ năng lực trên mỗi ma trận để dịch chuyển hành vi nhanh
hơn. Đây là kết luận về **ngân sách huấn luyện**, không phải kết luận rằng deck sai.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Train loss: 2,0513 so với 1,3454 — **tệ hơn 1,52×**. Nghe như "kém hơn một chút, có lẽ
cần train thêm". Trên tập target: **0,000 so với 0,535**, và `format=0,000`, tức nó
**không sinh ra nổi một JSON hợp lệ nào trong 50 mẫu**, đồng thời chậm nhất bảng
(5188 ms — model lan man vì không học được cách dừng).

Đó chính xác là điều tôi sẽ kết luận sai: cột loss nói "kém 52%", cột năng lực nói
"hỏng hoàn toàn, 100% mẫu lỗi". Một LR nhỏ hơn 10× không làm model học chậm hơn 1,5 lần —
nó làm model **chưa kịp rời khỏi phân phối gốc** sau 15 step, nên đầu ra vẫn là văn xuôi
tiếng Việt chứ không phải JSON. Nếu chấm bằng loss tôi sẽ xếp `wrong_lr` là "hạng tư
nhưng dùng tạm được"; chấm bằng target thì nó là **thảm hoạ**. Đây là Lỗi #3 hiện ra bằng
số đo của chính tôi: chỉ số thay thế giữ đúng *thứ tự* nhưng bóp méo *độ lớn* tới mức
quyết định dựa trên nó sẽ sai.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?**

VRAM đỉnh **7,09 GB** so với 12,01 GB = **tiết kiệm 41,0%**, thật và lớn. Giá phải trả,
đo trên cùng thang đo:

* target **0,310 so với 0,535** — mất **42,1% tương đối**
* format 0,92 so với 0,99 — 4 mẫu hỏng JSON
* latency **4254 ms so với 1465 ms — chậm hơn 2,9×** (khử lượng tử NF4 tốn thời gian ở mỗi bước sinh)
* thời gian train 514,3 s so với 476,1 s — chậm hơn 8%

**Số đo của tôi ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" (deck §12).**
Đổi 41% VRAM lấy 42% năng lực tác vụ *và* 2,9× độ trễ là một cuộc trao đổi tệ ở tier T4,
nơi bf16/fp16 LoRA vẫn vừa card (12,01 GB trên 14,6 GB khả dụng). QLoRA chỉ đáng cân nhắc
khi 12 GB *không* vừa — tức khi lựa chọn thay thế là không train được gì cả.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`FAILED`**
`target Δ = -0,2300` · `regression Δ = -0,0022` · `valid_trace_rate = 0,00`

Lý do cổng đưa ra: *"target task did not beat the optimized-prompt baseline (0.535 vs
0.765, delta -0.230). A fine-tune that loses to a better prompt is not a fine-tune you
should ship."*

**Diễn giải.** Phán quyết này đúng và tôi không tìm cách nới nó. Bản fine-tune thua
baseline (b) 0,230 điểm target, đồng thời **chậm hơn 1,49×** (1465 ms so với 986 ms) và
format thấp hơn một chút (0,99 so với 1,00) — thua ở cả ba nhóm mà nó lẽ ra phải thắng.
Điểm sáng duy nhất là nhóm **regression gần như không suy chuyển**: Δ = −0,0022 trên 15
câu hỏi phổ thông, tức 15 step LoRA không gây quên thảm hoạ. Nhưng "không phá hỏng gì"
không phải lý do để triển khai. Chẩn đoán theo đúng thứ tự NB5 gợi ý: `format` = 0,99 nên
template và mask **không** phải thủ phạm (§2 đã chứng minh mask đúng); `regression` không
tụt nên không phải quên thảm hoạ; `target` có nhúc nhích (0,000 của prompt ngây thơ →
0,535) nên LR **không** sai thang. Còn lại đúng một khả năng, và §6 xác nhận nó bằng dữ
liệu: **prompt engineering đã thắng, ở ngân sách huấn luyện này**. Đó là một kết quả hợp
lệ, và deck §1 nói thẳng rằng đôi khi kết luận đúng là "bài toán này không cần fine-tune".

---

## 6. Định tính — bắt buộc có cả ca THUA

Năm ví dụ dưới đây được chọn **tự động** bằng `scripts/fill_report.py`: hai mẫu có
`Δ = score_ft − score_b` âm nhất, hai mẫu Δ dương nhất, một mẫu hoà — trên toàn bộ 50 mẫu
eval, không do tôi nhặt tay. Cách chọn này loại trừ cherry-pick về mặt cấu tạo.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Δ | Nhận xét |
|---|---|---|---|---|---|---|
| 8 | *Bảo hành bao lâu* — chuột không dây | `hoi_thong_tin · thap · tich_cuc` | **1,00** — đúng cả 4 | **0,25** — `hỏi_thông_tin · thấp · tích_cực` | −0,75 | ❌ **FT thua** |
| 34 | *Giao hàng chậm, quá hạn* — sạc dự phòng | `van_chuyen · cao · tieu_cuc` | **1,00** — đúng cả 4 | **0,25** — `giao_hang_chậm · thấp · tiêu cực` | −0,75 | ❌ **FT thua** |
| 19 | *Vỡ khi nhận* — nồi chiên không dầu | `san_pham_loi · cao · trung_tinh` | 0,50 — đoán `hoan_tien`, `tieu_cuc` | **0,75** — `ho_tran · cao · trung_tinh` | +0,25 | ✅ FT thắng |
| 20 | *Sai màu, khẩn* — bàn phím cơ | `san_pham_loi · cao · trung_tinh` | 0,50 — đoán `hoi_thong_tin`, `tieu_cuc` | **0,75** — `ho_tran · cao · trung_tinh` | +0,25 | ✅ FT thắng |
| 6 | *Đổi size, hỏi cho biết thôi* — balo laptop | `doi_tra · thap · tieu_cuc` | 0,50 | 0,50 | 0,00 | hoà |

**Có mẫu chung nào ở các ca FT thua không? Có, và nó rất rõ.**

Trong **32/50** mẫu fine-tune thua, các trường sai phân bố như sau:

| Trường | Sai trong bao nhiêu ca thua | Loại trường |
|---|---|---|
| `intent` | **28/32** | từ vựng đóng, 5 giá trị |
| `sentiment` | 23/32 | từ vựng đóng, 3 giá trị |
| `urgency` | 21/32 | từ vựng đóng, 3 giá trị |
| `product` | **2/32** | **chép từ văn bản ticket** |

Fine-tune đoán đúng `product` **47/50 (94%)** — trường duy nhất không phải chọn trong từ
vựng đóng. Nói cách khác: nó học được **hình dạng JSON** (format 0,99) và học được **chép
tên sản phẩm**, nhưng **không học được bộ nhãn**.

Đo trực tiếp tỷ lệ đầu ra nằm **ngoài từ vựng cho phép**:

| | `intent` | `urgency` | `sentiment` |
|---|---|---|---|
| (b) prompt | **0/50 (0%)** | 0/50 (0%) | 0/50 (0%) |
| (c) fine-tune | **35/50 (70%)** | 10/50 (20%) | 14/50 (28%) |

Baseline (b) **không bao giờ** ra ngoài từ vựng, vì prompt tối ưu **liệt kê thẳng các giá
trị hợp lệ** trong system prompt. Bản fine-tune sinh ra 20 giá trị `intent` khác nhau,
trong đó chỉ 2 nằm trong từ vựng — phần còn lại là những từ tiếng Việt *nghe rất hợp lý*
mà model tự bịa: `ho_tran`, `hoi_tra`, `giao_hang_chậm`, `hỏi giá`, `thiếu phụ kiện`,
`hỗ trợ kỹ thuật`.

**Và ca số 8 là ví dụ đắt nhất của cả lab:**

```
gold : {"intent": "hoi_thong_tin", "urgency": "thap",  "sentiment": "tich_cuc"}
(c)  : {"intent": "hỏi_thông_tin", "urgency": "thấp",  "sentiment": "tích_cực"}
```

Fine-tune **hiểu đúng 100% về ngữ nghĩa** — và bị 0,25 điểm vì nó viết nhãn **có dấu**.
Đây không phải lỗi suy luận, đây là lỗi **hình thức bề mặt**: 15 step không đủ để ghi đè
thói quen viết tiếng Việt có dấu của base model trên đúng những token nhãn.

Tôi lượng hoá phần đó bằng một phép chấm phụ (chỉ để chẩn đoán, **không** thay cho điểm
chính thức): nếu bỏ dấu và chuẩn hoá gạch dưới trước khi so tất cả bốn trường —

| | target chính thức | target sau khi bỏ dấu |
|---|---|---|
| (b) prompt | 0,765 | 0,765 *(không đổi — nó vốn không dùng dấu)* |
| (c) fine-tune | 0,535 | **0,635** |

**0,100 trong khoảng cách 0,230 — tức 43% — chỉ là dấu tiếng Việt.** 0,130 còn lại mới là
sai phân loại thật (những `ho_tran`, `giao_hang_chậm`). Con số (b) không nhúc nhích xác
nhận phép chẩn đoán này không hề ưu ái baseline.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** **Không nên deploy bản fine-tune này.** Nó thua baseline (b) 0,230 điểm
target, chậm hơn 1,49×, và format kém hơn — thua ở mọi nhóm trừ regression, nơi nó chỉ
hoà. Khi một prompt được viết tử tế vừa chính xác hơn vừa rẻ hơn vừa không tốn một giây
GPU nào, quyết định kỹ thuật đúng là dùng prompt đó và đóng thí nghiệm lại.

Nhưng câu hỏi đắt hơn là *vì sao* nó thua, và ở đây số đo chỉ đúng một chỗ: **không phải
LoRA sai, mà là ngân sách huấn luyện quá nhỏ so với thứ cần học.** 15 optimizer step trên
225 mẫu đủ để dạy model hình dạng JSON (format 0,99) và dạy nó chép tên sản phẩm (94%
đúng) — hai thứ có tín hiệu dày đặc trong mọi mẫu. Nó **không** đủ để ghi vào trọng số một
ràng buộc chỉ xuất hiện gián tiếp: rằng `intent` phải là một trong đúng năm chuỗi ASCII.
Prompt (b) thắng vì nó **không cần học** ràng buộc đó — nó dán thẳng danh sách vào context
lúc suy luận. 70% đầu ra `intent` của bản fine-tune nằm ngoài từ vựng, so với 0% của
prompt; và 43% khoảng cách điểm chỉ là dấu tiếng Việt.

Vậy đòn bẩy thật trong lab này là gì? Không phải rank — `attn_only` với r=283 và `correct`
với r=16 có cùng ngân sách tham số và chênh nhau vỏn vẹn 0,050. Không phải lượng tử hoá —
`qlora` chỉ làm mọi thứ tệ đi. **Learning rate là đòn bẩy phá hoại lớn nhất** (sai thang
10× → 0,000, hỏng hoàn toàn), còn **đòn bẩy xây dựng lớn nhất lại nằm ngoài mọi siêu tham
số LoRA: nó là câu hỏi "thứ tôi muốn model học có nằm trong dữ liệu đủ dày để 15 step
nhìn thấy không?"** Với ràng buộc từ vựng đóng, câu trả lời là không — và không có giá trị
`r` nào sửa được điều đó.

**Ba điều tôi học được:**

1. **Mask không phải chuyện lý thuyết, nó là một chuỗi bạn đọc được.** Token được giám sát
   đầu tiên là `</think>` chứ không phải `{` — chat template đã tự chèn và đóng một khối
   suy luận rỗng trước cả khi câu trả lời bắt đầu. Đổi `MASK_MODE=everything` trên đúng
   mẫu đó, con số nhảy từ 39/94 lên 94/94 và câu hỏi người dùng lọt vào loss. Khác biệt
   giữa một model hoạt động và một model viết lại câu hỏi là một biến môi trường, và cách
   duy nhất để biết mình ở phía nào là giải mã ngược cái mask.

2. **Chỉ số thay thế có thể giữ đúng thứ tự mà vẫn khiến bạn quyết định sai.** Train loss
   xếp bốn run đúng y thứ tự của target — nếu chỉ nhìn cột đó tôi đã tưởng mình an toàn.
   Nhưng `wrong_lr` có loss tệ hơn `correct` 1,52 lần trong khi năng lực tác vụ của nó là
   **0,000 trên 0,535**. Thứ tự đúng, độ lớn sai một trời một vực. "Xếp hạng bằng target"
   không chỉ là để tránh xếp nhầm hạng — nó là để biết khoảng cách thật giữa các hạng.

3. **Fine-tune học được cái có tín hiệu dày, không học được cái chỉ xuất hiện gián tiếp.**
   `product` — chép từ chính ticket — đúng 94%. `intent` — phải nhớ một từ vựng đóng 5
   phần tử — sai 70%. Cùng một model, cùng một adapter, cùng một bước huấn luyện. Trước
   lab tôi nghĩ fine-tune là "dạy model làm tác vụ"; giờ tôi nghĩ nó là "dịch chuyển phân
   phối đầu ra", và những gì không được dữ liệu ép mạnh thì phân phối gốc vẫn thắng — ở
   đây là thói quen viết tiếng Việt **có dấu**.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử** — theo thứ tự chi phí tăng dần, và mỗi thứ đều là
một giả thuyết §6 chỉ ra:

1. **Rẻ nhất, có thể là đủ:** giữ nguyên adapter nhưng vẫn đưa danh sách từ vựng vào system
   prompt lúc suy luận. §6 dự đoán điều này thu hẹp phần lớn khoảng cách 0,230, vì 43% của
   nó chỉ là dấu và phần lớn còn lại là nhãn ngoài từ vựng.
2. **`EPOCHS=3` (45 step) với đúng cấu hình `correct`** — kiểm giả thuyết trung tâm của tôi
   là ngân sách chứ không phải cấu hình. Nếu tỷ lệ ngoài-từ-vựng của `intent` tụt từ 70%
   xuống dưới 20%, giả thuyết đúng.
3. **Decode có ràng buộc** (constrained/grammar decoding) để đầu ra không thể nằm ngoài từ
   vựng — biến 43% khoảng cách do dấu thành 0 mà không cần train thêm giây nào.
4. **B4 — quét rank có kiểm soát** `r ∈ {8,16,64}` cố định `text-linear`, để biết chênh
   lệch 0,050 giữa `correct` và `attn_only` có nằm trong nhiễu giữa các seed hay không.
   Tôi chưa dám gọi 0,050 là một hiệu ứng thật khi chỉ có một lần chạy mỗi cấu hình.

---

## Khai báo thay đổi so với repo gốc

Rubric yêu cầu khai báo mọi thay đổi ảnh hưởng tới phép so sánh. Tôi thay đổi năm chỗ,
**không chỗ nào chạm vào tập eval, prompt, hàm chấm điểm, hay cấu hình bốn run**:

| Thay đổi | Là gì | Vì sao |
|---|---|---|
| `EPOCHS=1` | 15 step thay vì 30 | Quỹ thời gian lab. Là cần gạt được `.env.example` ghi nhận (`1-3`), áp cho **cả** NB3 lẫn NB4 nên bốn run vẫn cùng ngân sách step. Ảnh hưởng tới kết quả và tôi phân tích ở §5/§7 thay vì giấu. |
| `notebooks/02_baselines.py` | thêm `report.write_json(...)` ghi `results/baseline_preds.json` | NB2 tính `preds_a`/`preds_b` rồi vứt đi, chỉ giữ điểm tổng hợp. §6 và rubric 3.4 lại đòi so sánh (b) với (c) **trên từng ticket**. Không lưu thì cách duy nhất lấy lại là sinh toàn bộ tập eval thêm một lượt. Toàn bộ phân tích từ-vựng-đóng ở §6 chỉ tồn tại nhờ file này. |
| `notebooks/05_evaluate_and_verdict.py` | thêm khoá `ft_pred_full` vào `qualitative.json` | Khoá `ft_pred` có sẵn bị cắt còn 90 ký tự cho vừa bảng in ra màn hình; JSON cắt dở **không parse lại được**, nên khi đọc artefact về sau mọi dự đoán đều trông như lỗi định dạng. `ft_pred` giữ nguyên. |
| `scripts/fill_report.py` *(mới)* | sinh bảng §1–§6 thẳng từ `results/` | Rubric 4.3 chấm "số trong report khớp file trong results/" — sinh từ artefact làm điều đó đúng theo cấu tạo. Script cũng tự tính hai phép kiểm tra công bằng, tự phát hiện khi thứ tự theo `final_loss` khác thứ tự theo `target`, và tự chọn 5 ví dụ định tính theo Δ. |
| `.gitattributes`, `scripts/verify.py` | `data/*.jsonl -text`; `stdout.reconfigure(utf-8)` | Hai lỗi chỉ xuất hiện trên Windows. `core.autocrlf=true` đổi LF→CRLF lúc checkout, mà `verify.py` băm các file này **theo byte** → một bản clone chưa ai đụng vào bị báo `eval sets unmodified: FAIL`. Đã xác nhận SHA bản LF khớp `data/checksums.json` cho cả 4 file. Và console cp1252 crash bằng `UnicodeEncodeError` đúng lúc in dòng báo `REPORT.md` còn là template. Không đổi một tiêu chí kiểm tra nào. |

Không đổi: `data/*.jsonl` (checksum nguyên vẹn), `OPTIMIZED_PROMPT`
(`sha=719e74d3b6232053`, `verify.py` xác nhận), `NAIVE_PROMPT`, hàm chấm điểm, cổng hồi
quy, và cấu hình LoRA của cả bốn run. `colab/*.ipynb` được sinh lại bằng
`scripts/build_colab.py` để khớp nguồn.

---

## Phụ lục — thưởng

- [ ] B1 NB6 merge + hot-swap — không chạy, hết thời gian GPU
- [ ] B2 dataset miền riêng — dùng corpus mặc định
- [ ] B3 reasoning-trace collapse — **không khả thi trên corpus mặc định**. §2 chứng minh:
      template đóng khối `<think></think>` rỗng ngay trong generation prompt và cả 250 câu
      trả lời huấn luyện là JSON trần, nên `assistant-only` / `masked-think` /
      `response-only` sinh ra mask **giống hệt nhau**; `valid_trace_rate=0,00` xác nhận
      không có trace nào để bảo vệ. Muốn làm B3 phải làm B2 trước.
- [ ] B4 quét rank có kiểm soát — chưa chạy; nêu ở §7 như việc tiếp theo
- [ ] B5 HuggingFace Hub — chưa đẩy

**Ghi chú về `adapters/correct/` — vì sao nó KHÔNG có trong repo này.**
`adapter_model.safetensors` nặng **123,9 MB**: 32.464.896 tham số × **4 byte**. Bốn byte
chứ không phải hai, vì `align_trainable_precision` (bản vá F-23) ép mọi tham số trainable
sang fp32 để GradScaler của fp16 unscale được — nên adapter được lưu ở fp32.

Con số đó vượt trần **100 MB/file** của GitHub. Tôi đã thử **Git LFS** theo đúng hướng dẫn
mục 6.2, và bị chặn ở tầng khác: GitHub từ chối nhận LFS object đẩy lên một **public
fork** — `can not upload new objects to public fork`. LFS object của fork phải nằm trong
kho LFS của repo gốc, mà tôi không có quyền ghi vào đó.

Mục 6.2 cho phép đúng lối thoát này: *"hoặc đẩy adapter sang HuggingFace rồi chỉ để link
trong REPORT.md / LINKS.md"*. Mục 6.1 cũng xếp adapter vào loại **nên có**, không bắt buộc
— bắt buộc là `results/` và `submission/REPORT.md`, cả hai đều đầy đủ trong repo này.

Adapter vẫn còn nguyên trên máy tại `adapters/correct/` và tái lập được bằng
`EPOCHS=1 python notebooks/03_train_correct.py` với đúng seed 42 và `max_steps=15` ghi
trong `results/runs.csv`.
