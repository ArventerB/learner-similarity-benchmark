# Tổng hợp khoảng trống và xếp hạng khả thi (10/11/2026)

Nguồn: 20 bài được 5 agent đọc trực tiếp, cộng 23 khẳng định đã qua bỏ phiếu 3 phiếu ở vòng workflow trước (HISE-KT, survey metric learning, LMNN, LNML, scikit-learn, PKT, Goal-based CR, Hidasi & Czapp).

Mức đọc: **toàn văn** là đọc PDF hoặc HTML đầy đủ. **Qua tóm tắt** là đọc qua công cụ fetch có tóm tắt, không phải nguyên văn. Mọi mục **[suy luận]** là nhận định của tôi, chưa kiểm chứng. Phần về nơi nộp và hạng Q chưa tra Scimago hay danh mục hội nghị.

---

## 1. Bảng các bài đã đọc

| Bài | Mức đọc | Dữ liệu | Độ đo tương tự người học | Mã | Venue | Điểm yếu thiết kế chính |
|---|---|---|---|---|---|---|
| HISE-KT (2511.15191) | toàn văn (vòng trước) | Assist09 3.013 SV, Slepemapy 5.000, Statics2011 331, Frcsub 536 | Mahalanobis cố định, Σ từ cặp huấn luyện, shrinkage, trên 5 đặc trưng cặp | không nêu | preprint | Không so Euclid/cosine/learned; chia cố định 8:1:1; không phân tích theo cỡ mẫu |
| RAG-KT (2604.10960) | qua tóm tắt | ASSIST09, ASSIST12, DBE-KT22, Eedi | Cosine, khoảng cách đồ thị, IRT-2PL, trộn tay 4:3:3 | không | preprint | Không có std; số chính khớp seq_len khác với nêu; trọng số chỉnh tay |
| ACKT (2511.20009) | qua tóm tắt | PTADisc 5 môn, 29.454 SV | K-means trên trạng thái tri thức | không | preprint, DOI giả | Nhiều hàng baseline trùng giá trị, một giá trị mỗi ô, không std |
| ReAL (EMNLP Findings 2025) | **toàn văn** | Junyi, ASSIST09, TextLog tự thu | **Cosine** trên vector mastery DKT, top-k = 5 | github.com/karin0018/ReAL | bình duyệt | Đánh giá bằng simulator DKT cùng họ với bộ truy hồi; ablation chỉ là biểu đồ; std chỉ ở một bảng (3 lần chạy) |
| IB-GRPO (2601.14686) | toàn văn (không đọc kỹ mục phương pháp) | Junyi 835 KC, ASSIST09 | Không dùng | không | preprint | Số của ReAL và GenAL chép từ bài gốc, baseline khác chạy lại |
| GEHRL (CIKM 2023) | **toàn văn** | Junyi, ASSISTments2015, KSS | Không (chỉ GRU; KNN baseline dùng cosine giữa lộ trình) | MindSpore (gitee) | bình duyệt | Chỉ simulator; 3 seed; không kiểm định; có reward âm |
| KnowLP (AAAI 2026) | toàn văn bản arXiv v2 (trang AAAI bị chặn) | Junyi, MOOCCubeX-Computer 629 SV, ASSIST09 | Tương tự giữa **khái niệm**, không phải người học | repo ẩn danh | bình duyệt | Không std, một t-test; ablation chỉ bỏ một agent |
| Survey KG-LPR (Electronics 2026) | **toàn văn** | Tổng quan dữ liệu | Không có Mahalanobis, metric learning, precision matrix | không | bình duyệt, narrative review | Không có quy trình hệ thống; không nhắc GEHRL, KnowLP |
| DKT2 (ECML-PKDD 2025) | toàn văn | Assist17 1.708 SV, EdNet 20.000 SV, Comp 45.180 SV | Không | MIT, có `preprocess_data.py` | bình duyệt | Là bài mô hình, dùng lại dữ liệu |
| XES3G5M (NeurIPS 2023) | toàn văn (phụ lục không có trong PDF công khai) | 18.066 HS, 865 KC lá, 5,55 triệu tương tác | Không | MIT | bình duyệt | 80% câu điền khuyết, KC mất cân bằng, không có thời lượng |
| EdNet (1912.03072) | toàn văn | 784.309 HS, 188 tag câu hỏi (293 gồm bài giảng) | Không | CC BY-NC 4.0 | tài liệu | Con số tổng tương tác lệch giữa tóm tắt và thân bài (131.417.236 và 131.441.538) |
| PTADisc (NeurIPS 2023) | toàn văn | 74 môn, 1,53 triệu SV; tập chéo môn 29.454 SV | Không | CC BY-NC-SA 4.0, Hugging Face | bình duyệt | Chỉ giữ bài có gán nhãn khái niệm; chỉ 5 bộ môn chéo phổ biến nhất |
| Audit dữ liệu (2609.29625) | **toàn văn** | 7 bộ, N từ 145 đến 32.593 | Không | GitHub/Zenodo | preprint | Ngưỡng chưa hiệu chuẩn; OULAD ở bài này **không** chia theo sinh viên |
| OULAD student-disjoint (IJACSA 2026) | **toàn văn** | OULAD, CC BY 4.0 | Không (hồi quy logistic, RF, HGB) | không nêu | IJACSA | Một seed, một tác giả, có dùng công cụ OpenAI để soạn |
| Askarbekuly & Luković (2407.09500) | toàn văn | 28 bài | Không | không | preprint | Chỉ 3 CSDL, chỉ tạp chí, không có đánh giá chất lượng |
| Riedmann et al. (IJAIED 2025) | **toàn văn** | 89 bài RL | Không | không | bình duyệt | Chọn lọc bởi một tác giả ở giai đoạn đầu |
| Hu & Rangwala (2001.00632) | toàn văn | GMU, khoảng 21,7 nghìn SV mỗi kỳ test, **không công khai** | Không (attention GCN) | không | preprint | Hai kỳ test, một lần chạy, không kiểm định |
| IC3K/KDIR 2025 (Izmir) | toàn văn | 1.307 SV tốt nghiệp, **không công khai** | Không | không | bình duyệt | F1 0,50 đến 0,65; nguy cơ rò rỉ chọn mô hình; không có baseline đa số |
| ITML (ICML 2007) | toàn văn | UCI nhỏ | Metric học được | không nêu | bình duyệt | Phương sai tăng khi số ràng buộc dưới khoảng 10c²; không hướng dẫn cho n nhỏ hơn d |
| AISAR (Sustainability 2022) | toàn văn (bản La Trobe Figshare) | OULAD | Mahalanobis chỉ trong DBSCAN trên **điểm 1 chiều** | không | bình duyệt | Cụm được định nghĩa bằng ngưỡng điểm nên độ tinh khiết gần 1 là hiển nhiên; không ablation |

---

## 2. Các khoảng trống lặp lại ở nhiều bài

**G1. Không bài nào học metric người học. [mạnh nhất, đọc từ 8 bài]**
ReAL dùng cosine cố định. RAG-KT trộn tay. ACKT dùng cụm K-Means. HISE-KT dùng Mahalanobis cố định. GEHRL, KnowLP, IB-GRPO không truy hồi người học. Giới hạn: đây là kết luận trong phạm vi các bài đã đọc, không phải khảo sát cạn.

**G2. Không bài nào phân tích theo cỡ mẫu.** HISE-KT dùng chia cố định, ReAL có bộ TextLog thưa (khoảng 13 bản ghi mỗi người học) nhưng không khảo sát đường cong theo n. ACKT chỉ thử một mức 0,1%.

**G3. Vệ sinh đánh giá yếu.** GEHRL có 3 seed nhưng không kiểm định. ReAL chỉ có std ở một bảng. RAG-KT chạy 5 lần chia nhưng không nêu std. IB-GRPO chép số baseline. KnowLP chỉ có một t-test. Riedmann et al. xác nhận bức tranh chung: 54/89 bài RL không kiểm định.

**G4. Chia theo sinh viên hoặc theo môn mới chỉ được định lượng một lần.** Bài IJACSA cho AUROC 0,716 xuống 0,661 trên OULAD, một seed. Bài audit cho thấy UCI Student từ R² 0,242 xuống âm 0,097 khi chia theo nhóm.

**G5. Không có bảng điểm đại học công khai.** Mọi bài dùng bảng điểm đều dùng dữ liệu riêng (GMU, Izmir, UC Berkeley, UCLA). Một benchmark dựa trên dữ liệu công khai sẽ là bài đầu tiên cho phép tái lập.

**G6. Bằng chứng "Mahalanobis trong giáo dục" rất yếu.** AISAR chỉ dùng Mahalanobis trên điểm 1 chiều (tương đương Euclid đã co giãn) và đánh giá không đáng tin. HISE-KT là tiền lệ duy nhất có thiết kế hợp lý nhưng không có so sánh.

**G7. Survey Electronics 2026 nêu đích danh "đặc trưng người học tương quan và khó nhận diện" là vấn đề mở.** Ma trận hiệp phương sai là câu trả lời trực tiếp để trích dẫn làm động lực. Survey không có từ "small-sample", nên góc độ cỡ mẫu nhóm phải tự lập luận.

**G8. Đồ thị tiên quyết bằng precision matrix (Graphical Lasso).** Chỉ dựa trên vòng workflow trước (PKT chỉ dùng dữ liệu mô phỏng; báo cáo CMU là bài tập lớp). Chưa được đọc lại ở vòng này.

---

## 3. Xếp hạng khả thi

Thang: Cao / Trung bình / Thấp. Các nhận định về nơi nộp là **[suy luận]**.

| Hạng | Hướng | Dữ liệu | Tính xác thực, tái lập | Tính mới | Tài nguyên | Nộp |
|---|---|---|---|---|---|---|
| **1** | **Benchmark độ đo tương tự người học theo cỡ mẫu: Euclid, cosine, Mahalanobis Σ mẫu, Mahalanobis shrinkage, learned (LMNN/ITML/NCA)**, nhiệm vụ kNN dự đoán trên người học bị giữ lại | Cao: PTADisc chéo môn (CC BY-NC-SA, Hugging Face), XES3G5M (MIT), OULAD (CC BY 4.0), Assist17 qua DKT2 (MIT) | Cao: scikit-learn, metric-learn, mã DKT2 | Trung bình đến cao: lấp G1, G2, G3, G4, G6, G7 | Cao: chạy được trên CPU | Hội nghị Scopus khả thi nhất; Q2 nếu mở rộng; Q1 rủi ro cao vì không có đóng góp phương pháp mới |
| 2 | Thay cosine bằng learned Mahalanobis trong bước truy hồi của ReAL, đo E_p | Cao (Junyi, ASSIST09 công khai) | Trung bình: simulator DKT cùng họ bộ truy hồi (nguy cơ vòng tròn); ReAL dùng LLM | Cao | Thấp đến trung bình: LLM tốn tài nguyên | Q2 nếu chạy được |
| 3 | Learned metric cho cảnh báo sớm trên OULAD, chia theo module | Cao (CC BY 4.0) | Cao | Thấp: OULAD đông (khoảng 228 bài về bỏ học 2022 đến 2026, theo OpenAlex ở vòng trước) | Cao | Q2 hoặc hội nghị |
| 4 (dự phòng) | Graphical Lasso khám phá đồ thị tiên quyết kỹ năng | Chưa kiểm chứng: độ sâu cây KC của XES3G5M chưa xác minh (chưa mở `kc_routes_map.json`) | Trung bình: đồ thị vô hướng so với quan hệ tiên quyết có hướng; scikit-learn cảnh báo bất ổn khi dữ liệu tương quan cao | Cao | Cao | Q2 khả dĩ nếu có chuẩn đánh giá tốt |
| 5 | LPR bằng RL hoặc LLM (kiểu GEHRL, KnowLP, IB-GRPO) | Cao | Thấp: simulator, baseline chép số | Thấp | Thấp: không GPU | Không khuyến nghị |
| 6 | Bài chỉ dựa vào bảng điểm trường (dưới 150 SV) | Thấp: dữ liệu riêng, nhỏ | Thấp: không ai tái lập được | Trung bình | Cao | Chỉ làm case study phụ |

---

## 4. Kết luận

**Hướng chính: hạng 1.** Thiết kế đề xuất, rút ra từ các bài đã đọc **[suy luận]**:

- **Nhiệm vụ đánh giá không dùng simulator.** Dự đoán kết quả của người học bị giữ lại bằng cách lấy k láng giềng gần nhất theo từng độ đo. Trên PTADisc chéo môn: dự đoán điểm môn thứ 5 từ 4 môn trước, hoặc vector 5 chiều theo môn, hoặc vector mastery theo nhóm KC đã gộp xuống khoảng 20 đến 100 chiều. Cách này tránh vòng tròn DKT của ReAL và GEHRL.
- **Nhãn cho LMNN/NCA** lấy từ dải kết quả (đạt hoặc rớt, mức thành thạo). **Ràng buộc cho ITML** lấy từ cặp có kết quả gần nhau. Chọn tiền nghiệm A0 là đơn vị hay nghịch đảo hiệp phương sai co (shrunk) là biến thí nghiệm, vì ITML cho thấy phương sai tăng mạnh khi ít ràng buộc.
- **Giao thức:** cỡ mẫu 50, 100, 200, 500 sinh viên; lặp 100 lần; báo trung bình, std, kiểm định; chia student-disjoint, thêm module-disjoint cho OULAD; báo cáo khoảng cách giữa chia ngẫu nhiên và chia theo nhóm.
- **Bảng điểm trường** chỉ làm case study phụ, cần ẩn danh.

Lý do chọn: lấp được G1 đến G7 cùng lúc, mã và dữ liệu đều có giấy phép công khai, chạy hoàn toàn trên CPU. Kết quả nào cũng viết được: nếu metric học được hơn thì là đóng góp phương pháp, nếu không hơn sau shrinkage thì là một phát hiện thực nghiệm có giá trị.

**Hướng dự phòng: hạng 4**, với điều kiện mở `kc_routes_map.json` của XES3G5M xác minh cây KC trước khi cam kết.

## 5. Điều chỉnh so với các báo cáo trước

- **AISAR:** báo cáo trước ghi là bài có dùng Mahalanobis cho hệ đánh giá e-learning. Sau khi đọc toàn văn, đó là ví dụ cần thận trọng, không phải bằng chứng ủng hộ (Mahalanobis chỉ trên điểm 1 chiều, không ablation). Chỉ nên trích làm động lực hoặc ví dụ phản diện.
- **Giấy phép đã xác minh:** PTADisc là CC BY-NC-SA 4.0, EdNet là CC BY-NC 4.0, XES3G5M là MIT, OULAD là CC BY 4.0. Phần "PTADisc chưa kiểm chứng giấy phép" của báo cáo trước không còn đúng.
- **PTADisc đầy đủ nặng 125,5 GB.** Chỉ tải thư mục `cross_course_datasets`.
- **Bài audit:** OULAD ở bài này không chia theo sinh viên, nên không dùng con số của nó để minh họa chia student-disjoint. Dùng bài IJACSA cho việc đó (0,716 xuống 0,661, một seed).
- **Askarbekuly (20/28 dùng dữ liệu nội bộ):** con số này đến từ một hình, agent đọc văn bản không xác nhận lại được. Chỉ dùng số 12/28 dùng rating làm đích, 5/28 dùng dữ liệu công khai.

## 6. Còn chưa kiểm chứng

- RAG-KT và ACKT chỉ đọc qua tóm tắt và đều là preprint có dấu hiệu lỗi số liệu. Không dựa vào chúng ngoài việc nêu như tiền lệ.
- Trang AAAI của KnowLP bị chặn; bản đọc là arXiv v2 có thể khác bản camera-ready.
- Độ sâu cây KC của XES3G5M, và việc `kc_routes` dùng được làm chuẩn cho đồ thị tiên quyết.
- Chưa tra Scimago, Scopus hay hạn nộp của hội nghị nào. IJACSA được web search ghi là có trong Scopus nhưng chưa xác minh.
- Giả thuyết trung tâm (learned metric có hơn Mahalanobis shrinkage ở n nhỏ không) chưa có bằng chứng thực nghiệm. LMNN gốc tự nhận kết quả trên tập nhỏ lẫn lộn.
