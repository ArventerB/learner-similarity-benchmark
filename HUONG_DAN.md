# Hướng dẫn sử dụng thư mục `paper`

Thư mục này chứa toàn bộ mã, dữ liệu trung gian, kết quả và bản thảo của bài báo
**"Which Similarity Measure for Similar-Learner Retrieval?"**. Mọi con số, bảng và hình
trong bản thảo đều do mã sinh ra từ kết quả chạy thật. Không có số nào gõ tay.

## 1. Các tệp cần đọc trước

| Tệp | Nội dung |
|---|---|
| `manuscript/paper.md` | Bản thảo bài báo (tiếng Anh), đã chèn sẵn số liệu, bảng và hình |
| `manuscript/paper.docx` | Cùng bản thảo ở dạng Word, 33 trang A4, phụ lục xoay ngang |
| `results/figures/` | 8 hình của bài (PNG 300 dpi và PDF) |
| `results/tables/` | Tất cả các bảng dạng CSV và Markdown |
| `results/summary.json` | Mọi con số được trích trong bài |
| `DATASHEET.md` | Nguồn dữ liệu, giấy phép, các cột không đọc |
| `docs/FEATURE_RULE.md` | Quy tắc dựng vector năng lực |

**Không sửa trực tiếp `manuscript/paper.md`.** Tệp này được sinh tự động. Muốn sửa lời văn
thì sửa `manuscript/paper_template.md` hoặc các tệp trong `manuscript/sections/`, rồi chạy
lại lệnh ghép bài ở mục 3.

## 2. Dữ liệu

Chỉ dùng dữ liệu công khai đã ẩn danh từ nguồn. Không dùng bảng điểm, không dùng dữ liệu
của trường, không đọc thuộc tính nhân khẩu học (giới tính, tuổi, vùng, khuyết tật) của bất
kỳ nguồn nào.

| Nguồn | Miền | Giấy phép |
|---|---|---|
| ASSISTments 2009, 2015, 2017 | Toán trung học cơ sở | Không ghi trong tệp, xem trang ASSISTments |
| EdNet-KT1 (mẫu 20.000 người học) | Luyện thi tiếng Anh | CC BY-NC 4.0 |
| XES3G5M | Toán lớp 3 | MIT |
| Slepemapy | Kiến thức địa lý | Open Database License |
| PTADisc | Lập trình và các môn máy tính bậc đại học | CC BY-NC-SA 4.0 |
| OULAD | Đào tạo đại học từ xa | CC BY 4.0 |

Dữ liệu gốc nằm ở `data/raw/` (khoảng 1,6 GB tải về). Tệp `data/raw/MANIFEST.json` ghi
đường dẫn tải, dung lượng và mã băm SHA-256 của từng tệp. Khi công bố mã, **không đưa
`data/raw/` lên kho công khai**: chỉ đưa script tải và mã băm.

## 3. Chạy lại từ đầu

Cần Python 3.13 và máy nhiều nhân CPU. Không cần GPU.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python -I src/lsb/download.py
```

```bash
.venv/Scripts/python -I src/lsb/extract.py
```

```bash
.venv/Scripts/python -I src/lsb/build_views.py
```

```bash
bash run_all.sh
```

```bash
.venv/Scripts/python -I src/lsb/run_extras.py --jobs 4
```

```bash
.venv/Scripts/python -I src/lsb/run_skill_task.py --jobs 4
```

```bash
.venv/Scripts/python -I src/lsb/analysis.py
```

```bash
.venv/Scripts/python -I src/lsb/figures.py
```

```bash
.venv/Scripts/python -I src/lsb/check_claims.py
```

```bash
.venv/Scripts/python -I src/lsb/build_paper.py
```

Bốn lệnh cuối chạy trong vài phút. Sau khi sửa lời văn chỉ cần chạy lại lệnh cuối cùng.

`check_claims.py` đối chiếu từng nhận định viết tay trong bài (ví dụ "cosine có hạng tốt
nhất ở n ≤ 200") với số liệu. Nếu sau khi chạy thêm dữ liệu mà một nhận định không còn
đúng, lệnh báo lỗi và liệt kê câu cần viết lại.

## 4. Xuất sang Word

Cần cài Node.js. Lệnh dưới đây tạo lại `manuscript/paper.docx` từ `paper.md`:

```bash
cd tools/docx_export && npm install && node export_docx.js
```

Tệp Word được tạo lại từ đầu mỗi lần chạy. Những gì sửa trực tiếp trong Word sẽ **không**
quay ngược về tệp Markdown và sẽ bị ghi đè nếu chạy lại lệnh này. Nếu giảng viên góp ý
trên bản Word, hãy lưu bản đó dưới tên khác, rồi chuyển các sửa đổi về lời văn vào
`manuscript/sections/` trước khi tạo lại.

## 5. Chạy thêm lần lặp

`run_all.sh` chạy theo từng khối 20 lần lặp và bỏ qua những ô đã xong, nên có thể dừng rồi
chạy tiếp. Mỗi khối mất khoảng 60 đến 80 phút trên máy 12 nhân. Sau khi có thêm khối, chạy
lại bốn lệnh `analysis.py`, `figures.py`, `check_claims.py`, `build_paper.py`: bảng, hình và mọi con số trong
bài tự cập nhật theo số lần lặp mới.

## 6. Cấu trúc mã

| Tệp trong `src/lsb/` | Vai trò |
|---|---|
| `download.py` | Tải dữ liệu công khai, ghi mã băm |
| `extract.py` | Giải nén đúng những tệp cần dùng |
| `build_views.py` | Dựng vector năng lực, chia người học thành ba phần |
| `metrics.py` | Các độ đo: Euclid, cosine, Mahalanobis, metric học được |
| `protocol.py` | Lấy mẫu, chuẩn hóa, kNN, chấm điểm, chỉ số chẩn đoán |
| `run_experiment.py` | Chạy lưới thực nghiệm song song, tiếp tục được |
| `run_extras.py` | Độ nhạy siêu tham số và độ bền |
| `run_skill_task.py` | Nhiệm vụ thứ năm: kết cục theo từng kỹ năng |
| `analysis.py`, `analysis_extras.py`, `stats.py` | Bảng, kiểm định, cơ chế, quy tắc chọn độ đo |
| `figures.py` | Vẽ hình |
| `check_claims.py` | Đối chiếu từng nhận định viết tay trong bài với số liệu |
| `build_paper.py` | Chèn số liệu và bảng vào bản thảo |

## 7. Kết quả chính (lần chạy ngày 11/11/2026, 40 lần lặp mỗi ô, 183 ô)

Kết quả không ủng hộ giả thuyết ban đầu rằng Mahalanobis hoặc metric học được sẽ tốt hơn.

| Câu hỏi | Kết quả |
|---|---|
| Độ đo nào tốt nhất? | Cosine trên vector đã chuẩn hóa. R² trung bình 0,215 (n = 50) đến 0,265 (n = 500), so với 0,125 đến 0,216 của Euclid. Thắng ở 116/183 ô và ở 7/8 nguồn. |
| Mahalanobis có giúp không? | Không. Bản Ledoit–Wolf kém Euclid 0,145 R² trung bình, kém ở cả 8 nguồn, và càng nhiều người học càng kém (−0,091 ở n = 50, −0,192 ở n = 500). |
| Shrinkage có giúp không? | Có so với hiệp phương sai mẫu, nhưng chỉ vì nó kéo metric về phía Euclid. |
| Metric học được? | NCA, MLKR tiến gần cosine ở n = 500 nhưng không vượt. LFDA lỗi 55% số lần ở n = 50, ITML lỗi 13%. |
| Vì sao whitening hỏng? | Chỉ số signal–variance alignment lớn hơn 1 ở 100% số ô (trung vị 4,7): tín hiệu dự đoán nằm ở các hướng phương sai cao mà whitening triệt tiêu. Tương quan hạng với mức thiệt hại là −0,71. |
| Có quy tắc chọn độ đo không? | Không quy tắc nào tốt hơn "luôn dùng cosine". Chọn bằng leave-one-out trong mẫu chỉ có lợi từ khoảng 200 người học. |
| Mô hình không dựa metric? | Ridge và random forest dự đoán tốt hơn mọi độ đo kNN từ n = 100. Ở n = 50 thì ngang cosine. |
| Ảnh hưởng của giao thức? | Chia ngẫu nhiên thay vì theo thời gian thổi phồng R² khoảng 52%. Huấn luyện trên module khác làm AUC của OULAD giảm 0,043. |
| Kết cục theo từng kỹ năng? | Mahalanobis vẫn kém nhất. Ưu thế của cosine so với Euclid không còn có ý nghĩa thống kê. |

Số liệu đầy đủ nằm trong `manuscript/paper.md` và `results/tables/`.

