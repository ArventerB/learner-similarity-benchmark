# Mở rộng dữ liệu và các hướng ít người làm (10/11/2026)

Nguồn: một vòng deep-research (19 nguồn fetch, 75 khẳng định trích xuất, 25 khẳng định kiểm chứng 3 phiếu: 22 xác nhận, 3 bị bác) cộng đếm số bài bằng OpenAlex do tôi chạy trực tiếp.

Nhãn: **[đã kiểm chứng]** là kết luận có nguồn qua bỏ phiếu. **[số đếm OpenAlex]** là một nguồn, tìm theo tiêu đề và tóm tắt, nhiễu, chỉ dùng để so sánh tương đối. **[suy luận]** là nhận định của tôi, chưa kiểm chứng.

---

## 1. Dữ liệu công khai có thể thay hoặc bổ sung bảng điểm trường

### Đã kiểm chứng

| Bộ dữ liệu | Quy mô | Tín hiệu | Giấy phép | Phù hợp Mahalanobis |
|---|---|---|---|---|
| **PTADisc** (5 môn lập trình/CS, dùng trong ACKT, arXiv 2511.20009) | 29.454 sinh viên; Python 685 KC, Java 773, C 847, C++ 547, DS 767; khoảng 4,7 đến 11,4 triệu log mỗi môn | Log chấm bài online, đúng/sai theo KC | Chưa kiểm chứng | **Tốt nhất về bối cảnh**: sinh viên đại học, cùng một nhóm xuất hiện ở nhiều môn, nên dựng được vector năng lực đa môn |
| **XES3G5M** (NeurIPS 2023 D&B) | 18.066 học sinh, 7.652 câu hỏi, 865 KC, 5,55 triệu tương tác | Đúng/sai, có cây KC phân cấp (kc_routes) | MIT | Tốt cho vector theo KC; là toán lớp 3, lệch miền so với đại học |
| **EdNet** | 784.309 học sinh, 131 triệu tương tác; bản DKT2 lấy mẫu 20.000 HS, 1.781 KC | Đúng/sai | CC BY-NC 4.0 (phi thương mại, cần trích dẫn) | Tốt nhưng phải subsample nếu không có GPU mạnh |
| **Assist17** (bản DKT2) | 1.708 HS, 411 KC, 934.638 tương tác | Đúng/sai | Chưa kiểm chứng | Cỡ nhỏ, hợp để chạy thử nhanh |
| **PTADisc "Comp"** (bản DKT2) | 45.180 HS, 472 KC, 6,07 triệu tương tác | Đúng/sai | Chưa kiểm chứng | Dùng được |

Mã tiền xử lý và dữ liệu của DKT2 (ECML-PKDD 2025, giấy phép MIT): github.com/zyy-2001/DKT2. Dùng lại mã này sẽ tiết kiệm nhiều tuần.

### Cảnh báo quan trọng về dữ liệu **[đã kiểm chứng]**

Một audit trên 7 bộ dữ liệu giáo dục công khai cho thấy chỉ 3/7 đạt cả 4 tiêu chí chất lượng. UCI Student Performance sụp đổ khi chia theo nhóm (R² từ 0,242 xuống âm 0,097). Trên OULAD, chia ngẫu nhiên theo sinh viên cho AUROC 0,716, còn leave-one-module-out chỉ 0,661. Bài học: mọi thực nghiệm phải chia **student-disjoint** hoặc **module-disjoint**, nếu không kết quả bị thổi phồng và reviewer sẽ nhận ra.

### Chưa kiểm chứng trong vòng này

Quy mô, giấy phép và link của ASSISTments 2009/2012, Junyi, KDD Cup Algebra/Bridge, NIPS34 Eedi, OULAD, MOOCCubeX, xAPI-Edu-Data. Cần tự mở trang gốc xác nhận trước khi dùng. Hai khẳng định bị bác nên không dựa vào: "XES3G5M đủ lớn để thay bảng điểm đại học" (lệch miền), và "UniER là nguồn dữ liệu đã tiền xử lý sẵn".

### Mức độ phổ biến của từng bộ **[số đếm OpenAlex, 2022–2026]**

| Bộ | Số bài |
|---|---|
| OULAD | 682 |
| MOOCCube/MOOCCubeX | 83 |
| Junyi + knowledge tracing | 23 |
| XES3G5M | 21 |
| NIPS34/Eedi + KT | 10 |
| PTADisc | 3 |

PTADisc và XES3G5M ít người dùng, nghĩa là ít đối thủ trực tiếp nhưng cũng ít bài để so sánh.

---

## 2. Các bài báo liên quan thêm

Đã kiểm chứng:

- **HISE-KT** (arXiv 2511.15191, 2025). Mahalanobis trên 5 đặc trưng cặp sinh viên, hiệp phương sai ước lượng từ các cặp huấn luyện và regularize bằng shrinkage Schäfer-Strimmer, để chọn Top-S sinh viên tương tự. AUC tốt nhất trên 4 bộ (Assistment09 0,8703). Bỏ thông tin sinh viên tương tự làm AUC giảm mạnh (xuống 0,7734), nghĩa là similar-student retrieval thật sự có giá trị. Preprint, không có độ lệch chuẩn. **Đây là tiền lệ trực tiếp: Mahalanobis cố định cộng shrinkage đã có người làm.**
- **RAG-KT** (arXiv 2604.10960, 04/2026). KT có truy hồi đa góc nhìn, độ tương tự thiết kế tay (cosine, điểm đường đi trên đồ thị, độ gần IRT), không học metric. Ablation trên ASSIST09: đầy đủ AUC 85,74, peer ngẫu nhiên 83,62, bỏ truy hồi 62,37.
- **Jiang, Pardos & Wei** (arXiv 1812.10078, 2018). Goal-based course recommendation từ bảng điểm bằng RNN.
- **Hu & Rangwala** (arXiv 2001.00632, 2019). Dự đoán điểm bằng attention-based GCN; cấu trúc đồ thị môn học được học bằng attention, không phải precision matrix.
- **IC3K/KDIR 2025** (Izmir University of Economics). 1.307 sinh viên tốt nghiệp, dự đoán thành công môn tự chọn từ embedding hồ sơ, F1 chỉ khoảng 0,50 đến 0,65; dữ liệu không công khai.
- **ACKT** (arXiv 2511.20009, 2025). Cold-start KT trên PTADisc bằng cụm K-Means trên trạng thái tri thức. Một khẳng định cho rằng "ACKT không dùng metric học nên là khoảng trống" bị bác 0-3, vì vậy không dùng ACKT làm bằng chứng về khoảng trống.

Chưa tìm được bài nào đã kiểm chứng cho: Mahalanobis trong EDM clustering, và Graphical Lasso/precision matrix cho đồ thị tiên quyết môn học. Điều này không có nghĩa là không tồn tại.

---

## 3. Các niche khả thi và ít người công bố

Số đếm là **[số đếm OpenAlex]**, tìm theo tiêu đề và tóm tắt, nhiễu (cụm từ rộng cho số quá cao, cụm hẹp có thể đếm thiếu).

| # | Niche | Câu hỏi nghiên cứu | Dữ liệu | Số bài gần nhất (2022–2026) | Mức trống **[suy luận]** | Khả thi 7 tháng | Nơi gửi |
|---|---|---|---|---|---|---|---|
| N1 | **Benchmark thực nghiệm: Euclid vs cosine vs Mahalanobis (Σ mẫu, shrinkage) vs learned metric (LMNN/ITML/NCA) cho similar-learner retrieval, chia student/module-disjoint** | Metric học được có thật sự hơn Mahalanobis cố định có shrinkage và cosine, và hơn ở cỡ mẫu nào? | PTADisc, Assist17, XES3G5M, OULAD, thêm bảng điểm trường nếu có | metric learning cho học viên: 28; so sánh độ đo tương tự học viên: 15; similar-student retrieval + KT: 9 | **Cao**: gần như chưa có so sánh có hệ thống | **Cao**: chỉ cần mã và dữ liệu công khai | Hội nghị Scopus, sau đó Q2 |
| N2 | **Học metric khi ít dữ liệu: shrinkage, hạng thấp, few-shot cho cold-start learner matching** | Cần bao nhiêu sinh viên để metric học được ổn định, và shrinkage bù được bao nhiêu? | Subsample PTADisc/Assist17 ở n = 50, 100, 200, 500, và bảng điểm nhỏ của trường | cold-start/few-shot + KT: 68 (nhưng không phải metric learning); shrinkage + giáo dục: 7 | **Cao** cho phần metric; chủ đề cold-start thì đông | **Cao** | Hội nghị, Q2 |
| N3 | **Precision matrix (Graphical Lasso) khám phá đồ thị tiên quyết kỹ năng từ điểm hoặc mastery** | Ma trận precision thưa có khôi phục được quan hệ tiên quyết không? | XES3G5M có cây KC phân cấp, có thể dùng làm chuẩn tham chiếu **[suy luận]**; PTADisc; bảng điểm trường | Graphical Lasso + tiên quyết/chương trình: 2 | **Rất cao** nhưng có thể đếm thiếu | Trung bình: cần thiết kế cách đánh giá | Q2 |
| N4 | **Metric learning trên OULAD cho dự đoán bỏ học và gợi ý can thiệp** | Metric học được có cải thiện nhận diện sinh viên nguy cơ so với cosine/Euclid khi chia module-disjoint? | OULAD | OULAD + bỏ học/nguy cơ: 228 | **Thấp**: OULAD đông | Cao | Q2 |
| N5 | **Bối cảnh Việt Nam hoặc dữ liệu tiếng Việt** | Đặc thù chương trình đào tạo và điểm của sinh viên Việt Nam | Bảng điểm trường | Việt Nam + học tập/KT: 97 (đa số là phân tích thống kê, không phải metric learning) | Trung bình | Phụ thuộc xin dữ liệu | Hội nghị trong nước |
| N6 | **Metric công bằng hoặc robust trong giáo dục** | Metric học được có thiên lệch theo nhóm sinh viên không? | OULAD (có giới tính, vùng, tình trạng khuyết tật) | Số đếm 145 không đáng tin vì cụm từ rộng | Chưa rõ | Trung bình, rủi ro đạo đức | Q2 |

**Xếp hạng khả thi cho nhóm: N1 > N2 > N4 > N3 > N5 > N6.**

---

## 4. Kết luận

**Hướng chính: N1 kết hợp N2.** Một bài benchmark thực nghiệm so sánh bốn nhóm độ đo tương tự người học (Euclid, cosine, Mahalanobis cố định có shrinkage, metric học được) trên PTADisc, Assist17, XES3G5M và OULAD, báo cáo hiệu quả theo cỡ mẫu (50 đến 500 sinh viên), với chia student-disjoint hoặc module-disjoint.

Lý do:

- **Giải quyết đúng vấn đề giới hạn dữ liệu.** Bảng điểm nhỏ của trường chuyển thành một điều kiện thực nghiệm (cỡ mẫu nhỏ), không còn là điều kiện cần.
- **Khoảng trống thật.** HISE-KT đã dùng Mahalanobis cố định có shrinkage, RAG-KT dùng độ tương tự thiết kế tay. Chưa thấy ai so sánh có hệ thống với metric học được và đo theo cỡ mẫu.
- **Kết quả nào cũng viết được.** Nếu metric học được hơn: đóng góp phương pháp. Nếu không hơn sau shrinkage: đó vẫn là phát hiện có giá trị và là nội dung của một bài benchmark.
- **Không cần GPU mạnh** nếu subsample và dùng lại mã DKT2.

**Hướng dự phòng: N3.** Dùng Graphical Lasso trên mastery theo KC của XES3G5M, lấy cây KC kc_routes làm tham chiếu để chấm độ khôi phục quan hệ. Số bài trùng rất thấp theo số đếm, nhưng cần tự thiết kế cách đánh giá và rủi ro là kết quả đồ thị yếu.

## 5. Chưa kiểm chứng và rủi ro

- Số bài theo niche chỉ có một nguồn (OpenAlex), nhiễu. Cần tự tra Scopus bằng chuỗi tìm cụ thể trước khi khẳng định "chưa ai làm" trong bài.
- Giấy phép PTADisc, Assist17 và nhiều bộ khác chưa kiểm chứng. Đọc trang gốc trước khi tải.
- PTADisc là log chấm bài lập trình, không phải bảng điểm. Bài phải nói rõ phạm vi kết luận.
- Chưa có bằng chứng thực nghiệm nào cho thấy metric học được hơn Mahalanobis cố định ở cỡ mẫu nhỏ. Đó chính là câu hỏi của bài, nên cần chạy thử sớm.
- Hai bài (HISE-KT, RAG-KT) là preprint chưa bình duyệt. Nhiều nguồn có ngày công bố rất gần, chỉ dựa vào báo cáo của bước kiểm chứng.
- Chưa xác định hạn nộp và hạng của từng hội nghị hoặc tạp chí.

## Nguồn chính

- XES3G5M: https://papers.neurips.cc/paper_files/paper/2023/file/67fc628f17c2ad53621fb961c6bafcaf-Paper-Datasets_and_Benchmarks.pdf ; https://github.com/ai4ed/XES3G5M
- EdNet: https://arxiv.org/pdf/1912.03072v3.pdf ; https://github.com/riiid/ednet
- DKT2: https://arxiv.org/abs/2501.14256 ; https://github.com/zyy-2001/DKT2
- PTADisc, ACKT: https://arxiv.org/html/2511.20009v1 ; https://github.com/wahr0411/PTADisc
- Audit dữ liệu giáo dục: https://arxiv.org/pdf/2609.29625 ; https://thesai.org/Downloads/Volume17No9/Paper_58-Student_Disjoint_Cross_Module_Benchmarking.pdf
- HISE-KT: https://arxiv.org/html/2511.15191v1
- RAG-KT: https://arxiv.org/html/2604.10960
- Goal-based course recommendation: https://arxiv.org/abs/1812.10078
- Hu & Rangwala: https://arxiv.org/pdf/2001.00632
- IC3K/KDIR 2025: https://www.scitepress.org/publishedPapers/2025/136828/pdf/index.html
- OpenAlex API: https://api.openalex.org/
