# Reflection — Lab 21

**Nguyễn Kỳ Anh · 2A202601558 · 2026-08-21**

*Viết dựa trên đúng những gì đã xảy ra trong buổi chạy lab này; mọi con số đều lấy từ
`results/` và đối chiếu được.*

---

**1. Điều gì làm bạn ngạc nhiên nhất?**

Mẫu số 8 trong tập eval. Nhãn đúng là
`{"intent": "hoi_thong_tin", "urgency": "thap", "sentiment": "tich_cuc"}`, và bản
fine-tune trả về `{"intent": "hỏi_thông_tin", "urgency": "thấp", "sentiment": "tích_cực"}`.
Nó hiểu đúng **cả bốn trường** — và chỉ được 0,25 điểm, vì nó viết nhãn **có dấu**.

Tôi đã chuẩn bị tinh thần rằng model sẽ phân loại sai. Tôi không hề nghĩ tới khả năng nó
phân loại *đúng* rồi trượt vì hình thức bề mặt. Khi chấm lại toàn tập sau khi bỏ dấu,
điểm của fine-tune nhảy từ 0,535 lên 0,635 còn baseline (b) đứng nguyên 0,765 — tức
**43% khoảng cách thua cuộc chỉ là dấu tiếng Việt**. Nếu tôi chỉ nhìn con số 0,535 rồi
kết luận "model chưa học được bài toán", tôi đã sai về nguyên nhân, và mọi thứ tôi làm
tiếp theo để "sửa" nó cũng sẽ sai hướng.

Điều ngạc nhiên thứ hai: `attn_only` — cấu hình mà deck gọi là **Lỗi #1** — **thắng**
`correct` trên tập target (0,585 so với 0,535) khi hai bên có cùng ngân sách tham số
(lệch 0,025%) và cùng 15 step.

---

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ tôi đoán. Tôi tưởng nút thắt là GPU. Hoá ra toàn bộ pipeline NB1→NB5 chạy
**56,3 phút** mà tôi không phải đụng vào gì cả.

Thời gian thật sự bị đốt nằm ở **đường ống quanh phần tính toán**:

* `verify.py` báo `eval sets unmodified: FAIL` trên một bản clone chưa ai đụng vào. Thủ
  phạm là `core.autocrlf=true` của Git đổi LF→CRLF lúc checkout, trong khi `verify.py`
  băm các file đó **theo byte**. Nếu tôi tin cái báo lỗi đó, tôi đã đi tìm bug trong dữ
  liệu suốt nửa tiếng.
* `Lab21_RUN_ALL.ipynb` bị dán đè một đoạn transcript terminal lúc nó đang mở trong IDE —
  5,4 KB phình thành 23,3 KB, JSON đứt giữa chừng, Colab không mở nổi.
* Và cay nhất: ô tải kết quả về có một `SyntaxError`, phát hiện ra **sau khi** pipeline đã
  chạy xong 56 phút. Colab compile trọn cả ô trước khi chạy, nên một lỗi ở dòng 10 làm
  dòng 5 cũng không chạy — không có file nào được tạo dù mọi thứ trước đó đều đúng.

Bài học tôi rút ra: trong một lab kiểu này, phần "chạy được" và phần "lấy được kết quả ra"
là hai vấn đề khác nhau, và cái thứ hai không hề dễ hơn.

---

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ fine-tuning là **"dạy model làm một tác vụ"** — cứ đưa đủ ví dụ thì nó học
được cách làm.

Giờ tôi nghĩ nó là **"dịch chuyển phân phối đầu ra"**, và cái gì không được dữ liệu ép đủ
mạnh thì phân phối gốc vẫn thắng. Bằng chứng nằm ngay trong bảng của tôi: cùng một
adapter, cùng 15 step, trường `product` — chép thẳng từ ticket — đúng **47/50 (94%)**,
còn `intent` — phải nhớ một từ vựng đóng gồm đúng 5 chuỗi ASCII — có **70% đầu ra nằm
ngoài từ vựng**. Model bịa ra `ho_tran`, `giao_hang_chậm`, `hỏi giá`, `hỗ trợ kỹ thuật`:
toàn những cụm tiếng Việt nghe rất hợp lý, và không cái nào là nhãn hợp lệ.

Cái nó không học được không phải là "bài toán khó hơn". Đó là ràng buộc chỉ xuất hiện
**gián tiếp** trong dữ liệu. Baseline (b) không cần học nó — prompt liệt kê thẳng 5 giá
trị vào context, và ra **0% ngoài từ vựng**.

Hệ quả là tôi cũng không còn tin rằng "fine-tune xong thì bỏ prompt engineering đi được".

---

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng nhiều: đọc và tóm tắt bộ tài liệu lab, vá NB2 để lưu `baseline_preds.json`, vá NB5
để lưu `ft_pred_full`, viết `scripts/fill_report.py` sinh bảng report thẳng từ artefact,
chẩn đoán vụ `autocrlf`, và viết phần lớn bản nháp `REPORT.md` — kể cả file này.

Ba chỗ nó sai, và cả ba đều đáng nhớ:

* **Hai lần cùng một lỗi escape.** Code nó thêm vào NB2 có `SyntaxError` vì `\n` bị biến
  thành xuống dòng thật. Lần đó bắt được bằng cách chạy thử trước. Lần thứ hai đúng y lỗi
  đó lọt vào ô tải kết quả của Colab — và lần này chỉ lộ ra **sau** khi tôi đã chạy xong
  56 phút GPU. Nó đọc code của chính nó và không thấy; chỉ có chạy mới thấy.
* **Chẩn đoán sai ngay từ đầu.** Khi cell 2 báo 3 unit test đỏ trên Colab, nó khẳng định
  đó là ba test về precision trong `test_modeling_and_train.py`. Chạy kiểm tra thì file
  đó **41/41 xanh**. Nó suy luận từ chỗ ba test ấy bị SKIP trên máy tôi và đoán tiếp —
  nghe rất hợp lý, và sai.
* **Nó không chạy được phần quan trọng nhất.** Máy tôi có GTX 1650 4 GB và torch bản CPU,
  nên toàn bộ NB2–NB5 phải do tôi bấm chạy trên Colab. AI chuẩn bị được đường ống, không
  tạo ra được số đo.

Điểm chung: nó mạnh ở chỗ đọc tài liệu và viết code khuôn mẫu, yếu ở chỗ **khẳng định
điều nó chưa chạy**. Cách dùng đúng là bắt nó chạy thử trước khi tin.

---

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

**Đóng băng tập eval và đo một prompt được viết tử tế — trước khi train một step nào.**

Trong lab này, thứ tự NB2-trước-NB3 là cái đã cứu tôi khỏi một kết luận sai. Nếu tôi train
trước rồi mới dựng baseline, tôi đã có 0,535 trong tay và rất dễ tự thuyết phục rằng đó
là con số tốt — cho tới khi phát hiện prompt suông đạt 0,765 với chi phí bằng không và
nhanh hơn 1,49×.

Bước thứ hai, rút thẳng từ §6 của tôi: **hỏi xem thứ khách hàng muốn model học có phải là
một từ vựng đóng hay một định dạng cố định không.** Nếu đúng, prompt hoặc decode có ràng
buộc giải quyết được ngay, rẻ hơn và chắc chắn hơn fine-tune — và tôi sẽ chỉ đề xuất
fine-tune khi đã chứng minh được cả hai cách kia không đủ.
