# Phản biện học thuật đề tài SVC2025-189 và hướng mở rộng Mahalanobis

Ngày lập: 04/11/2026. Nguồn: báo cáo tổng kết SVC2025-189, mã nguồn `project_root`, tra cứu OpenAlex, và một vòng deep-research (23 nguồn được fetch, 110 khẳng định trích xuất, 25 khẳng định được kiểm chứng chéo 3 phiếu: 21 xác nhận, 4 bị bác).

Quy ước: mục có nhãn **[đã kiểm chứng]** là kết luận có nguồn đã được đối chiếu; mục có nhãn **[khuyến nghị reviewer]** là nhận định chuyên môn, không phải kết quả tra cứu.

---

## 0. Kết luận chính

1. **Đề tài ở trạng thái hiện tại không đủ điều kiện gửi Q1/Q2.** Ba lý do: dữ liệu tương tác được sinh mô phỏng ngẫu nhiên; bài toán thực chất là rating prediction, không phải learning path; metric xếp hạng suy biến (NDCG@10 ≈ 0,99 ở mọi mô hình, kể cả Linear Regression).
2. **Hướng "learning path recommendation" đã bão hòa.** Số bài tăng gấp đôi mỗi năm từ 2023. Cộng đồng ở KDD/CIKM/AAAI/EMNLP đã chuyển sang chuẩn đánh giá bằng KT-simulator và learning effectiveness E_p, với baseline là họ RL (CSEAL, GEHRL, DLPR, GEPKSD, ReAL, KnowLP). Mô hình SVD/KNN/BPR hoàn toàn vắng mặt trong các bảng so sánh đó.
3. **Hướng Mahalanobis metric learning trên trạng thái tri thức là khoảng trống thật nhưng hẹp.** Bài SOTA ReAL (EMNLP 2025) chỉ dùng cosine để tìm người học tương tự trên vector DKT. Bài gần nhất HISE-KT (arXiv, 11/2025) dùng Mahalanobis nhưng chỉ trên 5 đặc trưng thủ công với Σ mẫu, chưa có metric learning. Tính mới chỉ đứng vững nếu ma trận M được **học** từ ràng buộc sư phạm và chứng minh cải thiện E_p trên benchmark công khai.
4. **Kết quả hiện tại đi ngược luận điểm Mahalanobis ưu việt.** KNN-Mahalanobis (RMSE 0,405) kém KNN-MSD (0,395) và KNN-cosine (0,398); SVD thuần (0,322) tốt hơn mô hình lai (0,330). Điều này nhất quán với lý thuyết: Σ ước lượng trên vector rating thưa, chiều cao, n nhỏ là ill-conditioned và không mang thông tin năng lực.

---

## PHẦN A. Khả thi khi viết bài quốc tế

### A1. Mức bão hòa và lý do desk reject

**Số lượng công bố theo năm** (OpenAlex, tìm "learning path" + recommendation trong tiêu đề/tóm tắt, tra 04/11/2026):

| Năm | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (đến 11) |
|---|---|---|---|---|---|---|---|---|
| Số bài | 25 | 44 | 52 | 58 | 103 | 195 | 478 | 440 |

Tổng 1.533 bài, trong đó 1.113 bài thuộc 2024–2026. Các nhánh con 2024–2026: kết hợp RL 165 bài, kết hợp knowledge graph 232 bài, kết hợp LLM 73 bài.

**[đã kiểm chứng] Các survey mới nhất xác nhận ba điểm yếu chí mạng:**

- **Dữ liệu nội bộ hoặc mô phỏng, không tái lập được.** Askarbekuly & Luković (arXiv 2407.09500, 2024; 1.395 bài sàng lọc, 28 bài phân tích): 20/28 bài dùng dữ liệu nội bộ không công bố, 5/28 dùng dữ liệu công khai, 1 bài sinh dữ liệu ngẫu nhiên. Các tác giả viết rằng đa số nghiên cứu "không cho phép tái lập". Dữ liệu 17.710 rating sinh ngẫu nhiên của đề tài rơi vào nhóm hiếm nhất và kém tin cậy nhất. Review IEOM 2025 phê bình trực tiếp một bài dùng hồ sơ tổng hợp: "reliance on synthetic profiles may reduce real-world applicability".
- **Dùng metric rating làm proxy cho học tập.** Cùng survey trên: rating-based relevance là metric phổ biến nhất, dưới một nửa bài tối ưu metric học tập, chỉ một phần ba đo hiệu quả sư phạm bằng outcome-based assessment; các tác giả coi đây là "hạn chế từ góc độ khoa học". Survey Electronics 15(1):238 (Li, Li & Gao, 01/2026) xếp SVD/NMF/CF vào nhóm "resource ranking rather than structured path construction", chỉ phù hợp kịch bản phụ trợ.
- **Thiếu kiểm định thống kê.** Riedmann, Schaper & Lugrin (IJAIED 35, 2025; 89 bài RL trong giáo dục): 54/89 bài không kiểm định ý nghĩa thống kê; trong 35 bài có kiểm định chỉ 18 bài vượt mọi baseline có ý nghĩa; 24 bài chỉ đánh giá bằng người học mô phỏng. Kết luận của họ: số bài tăng nhưng "chất lượng phương pháp không tăng".

**[đã kiểm chứng] Vì sao NDCG@10 ≈ 0,99 đồng loạt là dấu hiệu lỗi, không phải kết quả tốt.** Hidasi & Czapp (RecSys 2023) và Krichene & Rendle (KDD 2020, best paper) chứng minh rằng khi chỉ xếp hạng trong tập item đã được đánh giá (không có negative khó), metric bị thổi phồng nghiêm trọng, "sự khác biệt giữa hai thuật toán có thể biến mất", và thứ hạng có thể đảo ngược so với full ranking. Thư viện kiểu Surprise mặc định đánh giá theo cách này. Reviewer có kinh nghiệm sẽ nhận ra ngay từ bảng kết quả.

### A2. Điều kiện tối thiểu cho Q2/Q1

**[đã kiểm chứng] Chuẩn thực nghiệm của nhánh goal-oriented LPR (2019–2026):**

| Thành phần | Chuẩn cộng đồng | Nguồn |
|---|---|---|
| Dữ liệu | Log tương tác thật, công khai: Junyi (245 nghìn đến 525 nghìn người học, 21 đến 25 triệu bản ghi), ASSIST09 (4.092 đến 4.217 người học, 347 nghìn đến 397 nghìn bản ghi), MOOCCubeX-Computer (629 người học, 17.447 bản ghi), EdNet | ReAL Table 1; KnowLP Table 2; IB-GRPO Table 2 |
| Baseline | KNN, GRU4Rec, Contextual Bandit, DQN, Actor-Critic, RLTutor, CSEAL (KDD 2019), SRC, GEHRL (CIKM 2023), DLPR và GEPKSD (KDD 2024), gần đây GenAL và ReAL (LLM-based) | ReAL mục 5.3; KnowLP (AAAI 2026) |
| Metric | Learning effectiveness E_p = (E_e − E_s)/(E_sup − E_s) đo bằng KT-simulator (KSS/KES xây trên DKT) từ pre/post-test mô phỏng | GEHRL; ReAL Eq.; IB-GRPO Eq. (1) |
| Kiểm định | t-test p < 0,05, đánh dấu * trong bảng; mã nguồn công khai (ví dụ github.com/karin0018/ReAL) | KnowLP; ReAL |

Trong các bài này, từ "RMSE" và "NDCG" xuất hiện 0 lần, từ "simulator" xuất hiện 17 lần (GEHRL). Lưu ý: chuẩn E_p thuộc nhánh RL/goal-oriented; các nhánh KG-based hoặc ACO vẫn dùng metric khác, nên không khẳng định tuyệt đối cho toàn lĩnh vực.

**[khuyến nghị reviewer] Tối thiểu để qua vòng desk ở Q2:** một benchmark công khai cộng một tập dữ liệu thật của trường (bảng điểm nhiều môn, vài trăm sinh viên trở lên); ít nhất một baseline mỗi họ (KT-based, RL-based, KG-based, recsys hiện đại như LightGCN/SASRec); E_p hoặc HR@K/NDCG@K với full ranking; 5 seed, mean ± std, kiểm định; ablation; mã nguồn. Khảo sát Google Form chỉ dùng cho user study định tính.

---

## PHẦN B. Hướng Mahalanobis

### B1. Động lực học thuật **[khuyến nghị reviewer, có tham chiếu lý thuyết]**

Cho vector năng lực x ∈ ℝᵈ, khoảng cách Mahalanobis d_M(x,y) = √((x−y)ᵀ M (x−y)) với M nửa xác định dương; M = Σ⁻¹ là trường hợp cổ điển (Bellet, Habrard & Sebban, survey arXiv 1306.6709).

- **Tương quan giữa môn học.** Điểm Giải tích, Đại số tuyến tính, Xác suất thống kê tương quan mạnh. Euclid đếm chênh lệch ở ba môn này như ba thông tin độc lập; Mahalanobis chia theo cấu trúc hiệp phương sai nên chỉ "tính một lần" cho cụm môn tương quan.
- **Bất biến thang đo.** Điểm thi, số lần làm bài, thời gian xem video có đơn vị khác nhau; Mahalanobis tự chuẩn hóa theo phương sai từng chiều, Cosine bỏ qua độ lớn, Euclid và Manhattan phụ thuộc đơn vị.
- **Liên hệ ma trận precision với đồ thị kỹ năng.** Phần tử bằng 0 của Σ⁻¹ ứng với hai kỹ năng độc lập có điều kiện; đồ thị của precision thưa (ước lượng bằng Graphical Lasso, Friedman et al. 2008) gần với đồ thị tiên quyết. Đây là móc nối lý thuyết chưa bài nào khai thác.

**Phản biện ngược phải chuẩn bị:** Mahalanobis với Σ mẫu chỉ là Euclid sau whitening bằng PCA. **[đã kiểm chứng]** HISE-KT (Duan et al., arXiv 2511.15191, 19/11/2025) đã làm đúng điều này ở dạng thô: Mahalanobis trên 5 đặc trưng thủ công với mean/covariance mẫu. Vì vậy tính mới của đề tài **chỉ** nằm ở ma trận M được học (LMNN, ITML, NCA) từ ràng buộc sư phạm, trên KT embedding, và được chứng minh cải thiện E_p so với cosine trong pipeline kiểu ReAL.

### B2. Ba pipeline đề xuất **[khuyến nghị reviewer]**

| Pipeline | Mô tả | Đã có gì | Còn trống gì | Mục tiêu |
|---|---|---|---|---|
| **P1. KT embedding + Mahalanobis metric learning** | DKT/SAKT trên ASSIST09/Junyi cho vector trạng thái tri thức; ràng buộc cặp/bộ ba từ quỹ đạo tiến bộ (cùng/khác kết quả trên nhóm bài kế tiếp); học M bằng LMNN (Weinberger & Saul, JMLR 2009) hoặc ITML (Davis et al., ICML 2007, regularize LogDet nên M luôn xác định dương); thay cosine trong bước truy hồi của ReAL hoặc KNN-baseline của GEHRL | ReAL dùng cosine trên DKT state; HISE-KT dùng Mahalanobis Σ mẫu trên 5 đặc trưng | Chưa bài nào học M trên KT embedding với ràng buộc sư phạm và đo E_p | Q1 nếu E_p cải thiện có ý nghĩa thống kê |
| **P2. Phân cụm Mahalanobis trên hồ sơ năng lực + lộ trình nguyên mẫu** | GMM full-covariance (chính là phân cụm Mahalanobis) trên bảng điểm nhiều môn của sinh viên trường; Σ_k bằng Ledoit-Wolf shrinkage; mỗi cụm sinh lộ trình nguyên mẫu từ lịch sử; cá nhân hóa bằng khoảng cách đến tâm cụm | Có vài bài k-means Euclid vs Mahalanobis trên điểm sinh viên (tạp chí thấp) | Chưa bài nào nối cụm Mahalanobis với sinh lộ trình và đánh giá sư phạm | Q2, khả thi trong 7 tháng với dữ liệu thật của trường |
| **P3. Mahalanobis làm khoảng cách mục tiêu trong RL** | Reward shaping bằng d_M giữa trạng thái hiện tại và mục tiêu, M từ precision ước lượng bằng Graphical Lasso trên ma trận kỹ năng | CSEAL/GEHRL dùng reward Euclid hoặc hiệu mastery | Chưa có | Q1, khó nhất |

Mô hình SVD_Mahalanobis_KNN hiện tại giữ lại làm baseline, không phải trọng tâm.

### B3. Thách thức toán học và cách xử lý **[khuyến nghị reviewer, có tham chiếu]**

- **Σ kỳ dị khi d > n hoặc dữ liệu thưa.** Vector rating thưa gần như chắc chắn cho Σ kỳ dị. Xử lý theo thứ tự: (a) đổi đầu vào sang vector năng lực dày từ KT hoặc bảng điểm; (b) shrinkage Ledoit-Wolf (2004) hoặc OAS (Chen et al. 2010): Σ̂ = (1−α)S + α·tr(S)/d·I, luôn khả nghịch; (c) học M hạng thấp M = LᵀL với L ∈ ℝᵏˣᵈ, k ≪ d, điều LMNN/NCA làm tự nhiên; (d) Graphical Lasso cho precision thưa, vừa ổn định vừa diễn giải được.
- **Ill-conditioned.** Theo dõi condition number κ(Σ); nếu κ > 10⁶ dùng pseudo-inverse hoặc tăng α. Chuẩn hóa z-score trước khi ước lượng.
- **Curse of dimensionality.** Khoảng cách hội tụ về cùng giá trị khi d lớn; metric learning hạng thấp giải quyết trực tiếp. Kiểm tra bằng tỷ lệ d_max/d_min của láng giềng.
- **Chi phí.** LMNN có độ phức tạp O(n²d) mỗi vòng; với EdNet (784 nghìn học sinh) phải lấy mẫu hoặc dùng NCA mini-batch. ITML rẻ hơn khi ràng buộc ít.

---

## PHẦN C. Định vị bài báo **[khuyến nghị reviewer; hạng Q cần kiểm tra lại trên Scimago trước khi gửi]**

### Hướng 1: Hệ thống đề xuất lộ trình sư phạm

Đóng góp là tác động học tập; cần dữ liệu thật của trường hoặc user study; thuật toán có thể đơn giản.

| Nơi gửi | Hạng | Lý do |
|---|---|---|
| Computers and Education: Artificial Intelligence | Q1 | Nhận bài hệ thống có đánh giá thực nghiệm, tỷ lệ chấp nhận cao hơn C&E |
| Education and Information Technologies | Q1 | Chuộng nghiên cứu triển khai ở bối cảnh quốc gia, nhiều bài Đông Nam Á |
| IEEE Transactions on Learning Technologies | Q1 | Cần cả thuật toán lẫn đánh giá sư phạm |
| Interactive Learning Environments | Q1 | Phù hợp nếu có user study |
| EDM / AIED | CORE B / CORE A | Nhận bài ngắn, cộng đồng KT tập trung ở đây |

### Hướng 2: Thuật toán Mahalanobis / Metric Learning

Đóng góp là phương pháp; benchmark công khai; nhiều baseline.

| Nơi gửi | Hạng | Lý do |
|---|---|---|
| Expert Systems with Applications | Q1 | Chuộng "phương pháp X cho lĩnh vực Y" có thực nghiệm dày |
| Knowledge-Based Systems | Q1 | Tương tự ESWA, mạnh về metric learning và recsys |
| Applied Intelligence | Q2 | Mục tiêu thực tế cho nhóm sinh viên |
| IEEE Access | Q1/Q2 | Dễ hơn, uy tín đang giảm |
| ACM RecSys / PAKDD | CORE A / CORE B | RecSys rất khó; PAKDD khả thi hơn |

**Lộ trình thực tế cho nhóm:** hội nghị trong nước có Scopus (KSE, RIVF, SoICT) trước, sau đó Applied Intelligence hoặc Computers and Education: AI.

---

## PHẦN D. Tên bài và khung Abstract

**Ba phương án tên bài:**

1. *Learning a Mahalanobis Metric over Knowledge States for Personalized Learning Path Recommendation*
2. *Covariance-Aware Learner Similarity: Mahalanobis Metric Learning for Competency-Based Course Sequencing in Online Education*
3. *Beyond Euclidean Similarity: Precision-Matrix Metric Learning for Student Modeling and Adaptive Learning Path Generation*

**Khung Abstract 5 câu:**

1. *Context:* Personalized learning path recommendation has become central to online education, yet most systems estimate learner similarity with Euclidean or cosine distance over raw performance vectors.
2. *Problem:* Such metrics ignore the strong correlation structure among skills and courses, double-counting redundant dimensions and producing neighbors that are statistically close but pedagogically dissimilar.
3. *Method:* We propose a Mahalanobis metric learning framework that learns a positive semi-definite matrix from pedagogical constraints over knowledge-tracing embeddings, with shrinkage regularization and a low-rank parameterization to remain stable in high-dimensional, sparse settings.
4. *Setup:* We evaluate on ASSISTments 2009 and Junyi plus a real transcript dataset of [N] undergraduate students, against cosine/Euclidean retrieval, CSEAL, GEHRL, DLPR and ReAL, using learning effectiveness E_p under a DKT-based simulator across five random seeds with paired t-tests.
5. *Findings:* The learned metric improves E_p by [x]% over cosine retrieval and [y]% over the strongest RL baseline, and the sparse precision matrix recovers interpretable prerequisite relations among skills.

---

## PHẦN E. Thống kê công bố liên quan

### E1. OpenAlex (tiêu đề + tóm tắt, tra 04/11/2026)

| Chủ đề | Tổng | 2021–2026 | 2024–2026 |
|---|---|---|---|
| "learning path recommendation" (cụm chính xác) | 475 | 431 | 362 |
| "learning path" + recommendation | 1.533 | 1.326 | 1.113 |
| "knowledge tracing" | 2.339 | 1.918 | 1.444 |
| Mahalanobis + "learning path" | 1 | 0 | 0 |
| Mahalanobis + "knowledge tracing" | 0 (xem E3) | 0 | 0 |
| Mahalanobis + "e-learning" | 8 | 5 | 4 |
| Mahalanobis + "student performance" | 7 | 5 | 3 |
| Mahalanobis + students + clustering | 42 | 27 | 18 |
| "metric learning" + "knowledge tracing" | 0 | 0 | 0 |
| "metric learning" + "learning path" | 2 | 1 | 1 |
| "ASSISTments dataset" | 98 | 89 | – |
| ASSISTments + "knowledge tracing" | 205 | 170 | – |
| EdNet | 302 | 243 | 196 |
| "learning path" + recommendation + Vietnam | 4 | 3 | – |

Lưu ý: OpenAlex bao gồm preprint và tạp chí ngoài Scopus; số trên Scopus thấp hơn khoảng 30–40%.

### E2. Kiểm chứng chéo bằng nguồn khác

- **Crossref:** không hỗ trợ tìm cụm từ chính xác, mọi truy vấn trả về hàng triệu kết quả nên không dùng được để đếm.
- **Semantic Scholar:** xem bảng dưới (nếu có). Lưu ý Semantic Scholar đếm theo toàn văn/từ khóa nên số thường cao hơn OpenAlex tiêu đề+tóm tắt.
- **Scopus / Google Scholar:** cần tài khoản hoặc không có API, nhóm nên tự tra một lần trước khi viết phần related work và ghi rõ chuỗi tìm kiếm cùng ngày tra.
- **Điểm quan trọng từ vòng deep-research:** vòng kiểm chứng 3 phiếu không xác nhận được bất kỳ con số đếm nào, nên mọi số ở E1 phải ghi trong bài là "theo OpenAlex, ngày 04/11/2026", không trình bày như số liệu tuyệt đối.

Kết quả thử Semantic Scholar ngày 04/11/2026: toàn bộ truy vấn bị từ chối (HTTP 429, cần API key), kể cả khi giãn cách 6 giây. **Kết luận: các con số ở E1 hiện chỉ có một nguồn (OpenAlex) và chưa được kiểm chứng chéo.** Một bằng chứng định tính cho thấy OpenAlex đếm thiếu: HISE-KT (arXiv 2511.15191) dùng Mahalanobis trong knowledge tracing nhưng không xuất hiện trong truy vấn "Mahalanobis + knowledge tracing" (kết quả 0). Vì vậy số thật cho các nhánh hẹp nên hiểu là "rất ít, dưới 10" thay vì con số chính xác.

### E3. Các bài cụ thể gần nhất với ý tưởng Mahalanobis + năng lực người học

| Bài | Năm | Nơi đăng | Dùng Mahalanobis thế nào |
|---|---|---|---|
| HISE-KT (Duan et al.), arXiv 2511.15191 | 11/2025 | preprint | Truy hồi học sinh tương tự trong KT bằng Mahalanobis trên 5 đặc trưng thủ công, Σ mẫu, không metric learning. **Bài gần nhất, chưa có trong OpenAlex nên đếm "0" ở E1 là lỗi thời** |
| ReAL (Lv et al.) | 2025 | Findings of EMNLP 2025 | SOTA LPR, truy hồi người học tương tự trên DKT state bằng **cosine**; 0 lần nhắc Mahalanobis/metric learning |
| Recommendation of Learning Path Using an Improved ACO Based on Novel Coordinate System | 2017 | IIAI-AAI | Mahalanobis trong hệ tọa độ cho ACO; bài duy nhất khớp "learning path" + Mahalanobis |
| AISAR (Sustainability 14(17):10551) | 2022 | MDPI Sustainability | Hệ đánh giá và gợi ý e-learning có dùng Mahalanobis, 85 trích dẫn |
| Optimizing K-Means through Distance Metric Simulation for Enrollment Segmentation | 2025 | CAUCHY (Indonesia) | So sánh Euclid/Mahalanobis trên dữ liệu sinh viên, tạp chí thấp |
| Using Mahalanobis Clustering Algorithm for College Student Learning Fundamental Mathematics | 2012 | Advanced Materials Research | Fuzzy c-means với Mahalanobis, cũ |

### E4. Các survey dùng làm mốc bão hòa

- Li, Li & Gao, *Personalized LPR Based on Knowledge Graphs: A Survey*, Electronics 15(1):238, 01/2026.
- Askarbekuly & Luković, *Learning Outcomes, Assessment, and Evaluation in ERS: A Systematic Review*, arXiv 2407.09500, 2024 (28 bài).
- Riedmann, Schaper & Lugrin, *RL in Education: A Systematic Literature Review*, IJAIED 35, 2025 (89 bài).
- Frontiers in Education, 2024, review AI-driven learning paths (78 bài, 2019–2024).
- Discover Artificial Intelligence, 2025 (125 bài, 2015–giữa 2025).
- Qiu, Zuhairi & Morcos, IEEE Access 12, 2024 (chỉ giữ bài có phương pháp cụ thể hoặc mã nguồn).

---

## Việc phải làm ngay, theo thứ tự ưu tiên

1. Thay dữ liệu mô phỏng bằng ít nhất một benchmark công khai (ASSIST09 hoặc Junyi) cộng bảng điểm thật của trường.
2. Đổi đầu vào của Mahalanobis từ vector rating sang vector năng lực (DKT state hoặc điểm nhiều môn).
3. Sửa giao thức đánh giá: full ranking hoặc E_p qua KT-simulator; 5 seed; t-test.
4. Chọn pipeline P2 nếu mục tiêu là Q2 trong 7 tháng; P1 nếu nhắm Q1 và có thêm thời gian.
5. Trích dẫn và đối chiếu với HISE-KT (2025) và ReAL (2025) trong phần related work để reviewer thấy nhóm biết khoảng trống nằm ở đâu.

---

## Nguồn chính

- GEHRL: https://dl.acm.org/doi/pdf/10.1145/3583780.3614897
- ReAL (EMNLP 2025): https://aclanthology.org/2025.findings-emnlp.11.pdf ; mã: https://github.com/karin0018/ReAL
- KnowLP (AAAI 2026): https://doi.org/10.1609/AAAI.V40I17.38479
- IB-GRPO: https://arxiv.org/pdf/2601.14686
- GraphRAG dual KG LPR: https://arxiv.org/html/2506.22303v1
- HISE-KT: https://arxiv.org/abs/2511.15191
- Survey KG-based LPR, Electronics 2026: https://doi.org/10.3390/electronics15010238
- Askarbekuly & Luković 2024: https://arxiv.org/abs/2407.09500
- Riedmann et al., IJAIED 2025: https://link.springer.com/article/10.1007/s40593-025-00494-6
- Hidasi & Czapp, RecSys 2023: https://arxiv.org/abs/2307.14951
- Krichene & Rendle, KDD 2020: https://arxiv.org/abs/2010.11060
- Qiu et al., IEEE Access 2024: https://doi.org/10.1109/ACCESS.2024.3428419
- Frontiers in Education 2024: https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2024.1424386/full
- Discover AI 2025: https://link.springer.com/article/10.1007/s44163-025-00598-x
- Bellet et al., metric learning survey: https://arxiv.org/abs/1306.6709
- Weinberger & Saul, LMNN (JMLR 2009): https://jmlr.org/papers/v10/weinberger09a.html
- Davis et al., ITML: https://bigdata.oden.utexas.edu/publication/information-theoretic-metric-learning/
- Ledoit & Wolf 2004: https://www.uzh.ch/cmsssl/econ/dam/jcr:ffffffff-961c-1dd9-ffff-ffffceb83f14/wellCond.pdf
- Chen et al., OAS shrinkage: https://arxiv.org/abs/0907.4698
- Friedman et al., Graphical Lasso: https://arxiv.org/pdf/0708.3517.pdf
- AISAR, Sustainability 2022: https://www.mdpi.com/2071-1050/14/17/10551
- OpenAlex API: https://api.openalex.org/
