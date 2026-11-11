# Mở rộng hướng hạng 1 để nộp tạp chí Q2 (10/11/2026)

Nguồn: vòng deep-research (19 nguồn fetch, 65 khẳng định, 25 khẳng định bỏ phiếu 3 phiếu: 23 xác nhận, 2 bị bác), cộng một agent tra trực tiếp Scimago và trang chính thức của 12 tạp chí.

Nhãn: **[đã kiểm chứng]** là có nguồn qua bỏ phiếu hoặc tra trang chính thức. **[suy luận]** là nhận định của tôi, chưa có nguồn độc lập. Hạng Q lấy từ SJR 2025 trên Scimago do agent đọc; APC và chính sách đọc từ trang nhà xuất bản.

---

## 1. Trả lời ngắn

Thiết kế hạng 1 hiện tại (4 bộ dữ liệu, một nhiệm vụ kNN, so sánh các độ đo) là **mức hội nghị**. Để lên tạp chí cần thêm bốn nhóm việc bắt buộc và một quy tắc chọn metric nhỏ để bài không chỉ là so sánh thuần:

1. **Nhiều bộ dữ liệu hơn, nhiều miền hơn.**
2. **Thống kê đúng chuẩn cho nhiều thuật toán trên nhiều bộ dữ liệu.**
3. **Giải thích cơ chế**: vì sao metric nào thắng ở n/d nào.
4. **Baseline không dựa metric**, để chứng minh việc chọn metric có ý nghĩa.
5. **Một quy tắc chọn shrinkage hoặc metric theo n/d**, kiểm tra bằng leave-one-dataset-out.

Phát hiện quan trọng về nơi nộp: **hầu hết tạp chí ứng viên đang là Q1 trên Scimago 2025**, nên "Q2" không phải mục tiêu thấp nhất. Q2 thực sự trong danh sách chỉ có JEDM (sát ranh, từng Q3 năm 2024) và Applied Intelligence (không hợp phạm vi).

---

## 2. Bằng chứng đã kiểm chứng về chuẩn của một bài benchmark ở tạp chí

| Điểm | Bằng chứng | Hệ quả cho nhóm |
|---|---|---|
| Quy mô | Feldbauer & Flexer (KAIS 59, 2019) so sánh 12 phương pháp giảm hubness cộng 2 baseline (Euclid, cosine) trên **50 bộ dữ liệu**. pyKT (NeurIPS 2022 D&B) so sánh 10 mô hình trên 7 bộ | 4 bộ dữ liệu là cận dưới. Không cần 50, nhưng cần nhiều hơn rõ rệt |
| Thống kê | Feldbauer & Flexer dùng Friedman, Nemenyi, critical difference diagram (CD = 2,81 với K = 14, N = 50, alpha = 0,05), nested 10-fold CV, tìm siêu tham số ngẫu nhiên. Benavoli et al. 2017 (JMLR 18) lập luận nên dùng phân tích Bayesian thay cho NHST | Báo cáo Friedman/CD kèm Bayesian signed-rank hoặc kiểm định phân cấp có ROPE |
| Chống rò rỉ | pyKT: đánh giá sai gây label leakage và thổi phồng hiệu năng. Musgrave et al. (ECCV 2020): nhiều bài metric learning có sai sót phương pháp luận | Giao thức student-disjoint và module-disjoint là **điểm bán chính**, không phải chi tiết kỹ thuật |
| Cơ chế | Radovanović et al. (JMLR 11, 2010): hubness là hiện tượng phân phối k-occurrence lệch mạnh khi chiều cao, gắn với intrinsic dimension | Đo intrinsic dimension và skewness của k-occurrence, không giả định hubness có ở 20 đến 100 chiều |
| Lý thuyết đi kèm | Verma & Aggarwal (SNAM 10, 2020) so sánh độ đo tương tự trên 5 bộ dữ liệu khuyến nghị và nêu quan hệ lý thuyết giữa các độ đo | Kết hợp empirical và lý thuyết làm bài dày hơn |
| Công bằng | JLA 2025 (Opoku, Pei, Xing) và IJCAI 2025 (Zong et al.) kiểm toán công bằng trên OULAD, nhưng chỉ cho mô hình dự đoán có giám sát, một khóa với một thuộc tính nhị phân | Audit nhiều module, nhiều thuộc tính cho kNN là chỗ trống, nhưng là phần phụ |

**Giới hạn thống kê quan trọng [suy luận, nhất quán với nguồn]:** với chỉ 4 bộ dữ liệu, Friedman/Nemenyi có power rất thấp. Dùng các lần chạy nhiều lần trong cùng bộ dữ liệu làm đơn vị so sánh thì vi phạm tính độc lập. Hướng đúng là tăng số bộ dữ liệu độc lập, và dùng kiểm định Bayesian phân cấp xử lý đo lặp.

**Một khẳng định bị bác:** "Cosine tốt hơn trên dữ liệu lớn, Overlap tốt hơn trên dữ liệu nhỏ" (bài SNAM) bị bác 0-3, nên không dùng làm tiền lệ cho việc chọn metric theo cỡ mẫu.

---

## 3. Các mở rộng cụ thể, đánh giá giá trị, chi phí, rủi ro **[suy luận]**

Chi phí tính theo CPU và công sức của nhóm. LMNN có độ phức tạp O(n²d), với n ≤ 500 và d ≤ 100 vẫn nhẹ.

| # | Mở rộng | Giá trị với reviewer | Chi phí | Rủi ro | Mức |
|---|---|---|---|---|---|
| a | **Thêm bộ dữ liệu và miền**: mục tiêu 8 đến 12 "dataset-view" độc lập, ít nhất 3 miền (lập trình, toán, MOOC/LMS). Ví dụ PTADisc (từng môn), XES3G5M, Assist09, Assist12, Assist17, EdNet subsample, OULAD (từng module), Junyi, Statics2011, Frcsub | Rất cao | Trung bình | Các view cùng nguồn (nhiều môn của PTADisc, nhiều module OULAD) không hoàn toàn độc lập; phải nêu rõ | **Must** |
| b | **Nhiều nhiệm vụ hạ nguồn**: dự đoán điểm, cảnh báo sớm, dự đoán kỹ năng kế tiếp, ghép cặp cold-start, phân cụm | Cao | Trung bình | Phình phạm vi; chọn 2 đến 3 nhiệm vụ | Should |
| c | **Phân tích cơ chế**: condition number của Σ, effective rank, intrinsic dimension, skewness của k-occurrence, tỷ lệ n/d | Cao | Thấp | Kết quả có thể không giải thích được; vẫn báo cáo | **Must** |
| d | **Quy tắc chọn metric hoặc shrinkage theo n/d và effective rank**, kiểm tra bằng leave-one-dataset-out | Rất cao: biến benchmark thành bài có đóng góp | Trung bình | Quy tắc có thể không tổng quát; cần ≥ 8 bộ dữ liệu để kiểm tra | **Must** |
| e | **Kiểm định chuẩn**: Friedman/Nemenyi/CD diagram, Wilcoxon, Bayesian signed-rank phân cấp, effect size | Cao | Thấp | Power thấp nếu ít bộ dữ liệu | **Must** |
| f | **Độ bền**: nhiễu, dữ liệu thiếu, mất cân bằng lớp, dịch chuyển giữa môn hoặc khóa | Trung bình đến cao | Thấp đến trung bình | Phình phạm vi | Should |
| g | **Audit công bằng** trên OULAD: nhiều module, nhiều thuộc tính (giới, khuyết tật, học vấn trước, IMD), hiệu chỉnh đa kiểm định | Trung bình | Thấp | Kết quả nhạy cảm, cần cẩn thận khi diễn giải | Nice |
| h | **Baseline không dựa metric**: logistic và GBDT trên vector, matrix factorization, kNN trên embedding DKT/AKT | Rất cao: nếu thiếu, reviewer hỏi "metric có quan trọng không?" | Trung bình | GBDT có thể thắng mọi kNN; kết quả vẫn đáng báo cáo | **Must** |
| i | **Giao thức chống rò rỉ**: student-disjoint, module-disjoint, temporal; so sánh với chia ngẫu nhiên | Cao | Thấp | Gần như không có | **Must** |
| j | **Tái lập**: mã, seed, phiên bản thư viện, Zenodo DOI, datasheet | Cao | Thấp | Không | **Must** |
| k | **Case study bảng điểm trường** | Thấp đến trung bình | Phụ thuộc xin dữ liệu | Chậm, nhỏ, riêng tư | Nice |
| l | **Demo ứng dụng**: truy hồi láng giềng cho gợi ý lộ trình hoặc can thiệp, đánh giá định tính | Trung bình | Thấp | Dễ bị coi là minh họa | Nice |

---

## 4. Nếu kết quả là "không metric nào thắng ổn định"

- **Chưa có bằng chứng đã kiểm chứng rằng một bài kết quả âm được nhận ở Q2.**
- Có tiền lệ "reality check" ở hội nghị hàng đầu: Musgrave et al. (ECCV 2020) kết luận tiến bộ của deep metric learning là "marginal at best"; pyKT cho thấy phần lớn mô hình mới chỉ hơn DKT rất ít. Nhưng cả hai là hội nghị, và cả hai **cùng đóng góp một thứ tích cực** (giao thức chuẩn, thư viện chuẩn). Khẳng định "khung kết quả âm này áp dụng được cho benchmark của nhóm" bị bác 1-2.
- **Cách trình bày để an toàn:** đừng đặt bài là "không ai thắng". Đặt là "bản đồ chọn metric": metric nào tốt ở vùng n/d nào, kèm quy tắc chọn, giao thức đánh giá và gói tái lập. Kết quả âm trở thành một phần của bản đồ.
- **Giảm rủi ro "incremental" và "chỉ so sánh thư viện scikit-learn":** (d) quy tắc chọn metric, (c) cơ chế, (h) baseline không dựa metric, và giao thức student-disjoint.

---

## 5. Chọn tạp chí

**[đã kiểm chứng]** SJR 2025 trên Scimago và trang chính thức, do agent đọc. Scopus chỉ được xác nhận gián tiếp qua việc có trên Scimago.

| Tạp chí | Hạng 2025 | Hợp benchmark? | Phí | Nhận định **[suy luận]** |
|---|---|---|---|---|
| **JEDM** | Q2 (Q3 năm 2024, sát ranh) | Có: phạm vi nêu "comparative analyses", yêu cầu tái lập, khuyến khích chia sẻ mã | **Không phí**, truy cập mở | **Khớp phạm vi nhất**, rủi ro là hạng có thể tụt về Q3 |
| **JLA** | Q1 | Nghiên cứu thực nghiệm; báo cáo dữ liệu hoặc công cụ để tái lập | **Không phí**; tác giả gửi phải dùng email cơ quan, không dùng Gmail | Q1 không phí, giới hạn khoảng 8.000 từ, 6 đến 8 tháng đến khi đăng |
| **UMUAI** | Q1 (Education), Q2 (CS Applications, HCI) | "Chủ yếu nhận bài thực nghiệm"; mô hình hóa người học thuộc phạm vi | Hybrid, APC £2.490 nếu chọn mở | Hợp chủ đề |
| **Computers and Education: AI** | Q1 | Nêu rõ nhận bài thực nghiệm | OA, USD 3.620 | Khớp chủ đề, phí cao |
| **KAIS** | Q1 (Information Systems) và Q2 | Không nêu benchmark | Hybrid, APC £2.390 | Q2 thực tế nếu đóng khung data mining |
| EAIT | Q1 | Không nêu benchmark; cần khung sư phạm | Hybrid, $3.690 | Khả thi nhưng phải có góc sư phạm |
| ESWA | Q1 | Không nêu benchmark | Hybrid, USD 3.630 | Khả thi, khung giáo dục yếu |
| IEEE Access | Q1 | Không nêu | OA, $2.160 | Nhanh nhưng reviewer giáo dục ít |
| ILE | Q1 | Tập trung tác động học tập; tỷ lệ nhận 10% | Hybrid; không phí nếu không chọn OA | Không hợp bài phương pháp thuần |
| Applied Intelligence | Q2 | Phạm vi ưu tiên phương pháp **mới**, ngược với benchmark | Hybrid, $3.290 | **Không hợp** |
| IJAIED | Q1 (dữ liệu thời Springer) | Không nêu | Từ 2026 chuyển sang Elsevier, OA, USD 3.680 | Rất chọn lọc; việc còn trong Scopus sau khi chuyển nhà xuất bản chưa xác minh |
| Smart Learning Environments | Q1 | Không nêu | OA, £1.290 | Hợp vừa phải |

Lưu ý: tôi chưa xác minh từng tạp chí hybrid có miễn phí khi không chọn truy cập mở hay không. Chỉ ILE được agent xác nhận không phí nếu không chọn OA. Không tạp chí nào nêu rõ chấp nhận kết quả âm hoặc đăng ký trước.

**Đề xuất cụ thể [suy luận]:** nhắm **JEDM làm đích Q2 chính** (khớp phạm vi, không phí), **JLA hoặc UMUAI làm đích với Q1** nếu quy tắc chọn metric (mở rộng d) cho kết quả mạnh, và hội nghị Scopus làm bản đầu. Kiểm tra lại hạng JEDM ngay trước khi nộp vì năm 2024 là Q3.

---

## 6. Kế hoạch 7 tháng với điểm dừng

Tháng tính từ 11/2026. Khối lượng và mốc là **[suy luận]**.

| Tháng | Việc | Điểm go/no-go |
|---|---|---|
| T1 | Tải và tiền xử lý 8 đến 12 dataset-view dùng lại mã DKT2; chốt vector (điểm theo môn, mastery gộp 20 đến 100 chiều); dựng giao thức student-disjoint, module-disjoint; chạy cosine, Euclid, Σ mẫu | Có ≥ 6 dataset-view chạy được, nếu không thì rút gọn phạm vi |
| T2 | Thêm Mahalanobis shrinkage, LMNN/ITML/NCA/SDML; định nghĩa nhãn và ràng buộc; thêm baseline không dựa metric (logistic, GBDT, MF) | **Cổng 1:** có hiện tượng phụ thuộc n/d nhìn thấy được không. Nếu không, chuyển sang khung "reality check" kèm quy tắc chọn |
| T3 | Chạy đầy đủ: 4 mức n, 100 lần lặp, kiểm định Friedman/CD và Bayesian; đo condition number, effective rank, intrinsic dimension, hubness | Kết quả có ổn định qua các bộ dữ liệu không |
| T4 | Xây quy tắc chọn metric theo n/d và effective rank; kiểm tra leave-one-dataset-out | **Cổng 2:** quy tắc có tổng quát hóa được không. Nếu không, quyết định khung "bản đồ chọn metric" |
| T5 | Độ bền (nhiễu, thiếu, mất cân bằng); audit công bằng OULAD nếu còn thời gian; viết bản hội nghị | Nộp bản hội nghị nếu đến hạn |
| T6 | Mở rộng bản tạp chí: lý thuyết ngắn, phân tích cơ chế, hướng dẫn thực hành, gói tái lập (Zenodo) | Bản thảo hoàn chỉnh |
| T7 | Sửa theo góp ý, kiểm tra lại hạng tạp chí, nộp; hoàn thiện báo cáo nghiệm thu | Chọn tạp chí theo hạng lúc đó |

Nếu thiếu thời gian, **Must** (a, c, d, e, h, i, j) giữ nguyên, bỏ trước g, k, l rồi b, f.

---

## 7. Chưa kiểm chứng

- **Chưa có bài benchmark về độ đo tương tự người học** trong các bài mẫu. Chỉ tìm được 3 đến 4 bài mẫu (Feldbauer & Flexer, pyKT, Musgrave, Verma & Aggarwal) thay vì 8 đến 15 bài, và nhiều bài là hội nghị. Hạng Q của KAIS và SNAM trong các bài mẫu chưa xác minh.
- **Nhiều xác minh chỉ đọc abstract** (Demšar, Benavoli, JLA 2025, pyKT dựa trên trang abstract).
- **Giả thuyết trung tâm** (metric học được có hơn Mahalanobis shrinkage ở n nhỏ không) vẫn chưa có bằng chứng thực nghiệm.
- **Mọi đánh giá chi phí và kế hoạch 7 tháng là suy luận.** Chưa tra tỷ lệ chấp nhận của từng tạp chí (chỉ ILE có 10%). Hạng Q và APC thay đổi theo năm.
- **Scopus chỉ xác nhận gián tiếp** qua Scimago. IJAIED chuyển sang Elsevier từ 2026, tính liên tục trong Scopus chưa kiểm tra.
