# PREREG_06 — Đăng ký trước nghiên cứu mutant T4 (v1.0, đóng băng 16-10-2026)

*Đóng băng trước khi chạy `code/p1_t4.py` (chạy đầy đủ 36 mutant × 121 video). SHA-256 của file này ghi vào `P1/CP1_REPORT_06.md`. Muốn đổi bất kỳ mục nào → PREREG v1.1 (ghi lý do), báo kết quả theo **cả** v1.0 và v1.1.*

## 1. Dữ liệu và đối tượng

- Corpus: 121 video (CDnet2014 53, BMC 20, LASIESTA 48), 166.589 khung, 3.170 sự kiện (định nghĩa sự kiện theo `spec/FSM_06.md` §1.6). Cache `D:\Datasets\_derived\06\corpus_flat.npz` sha256 `a8da5105…1ec1`; điểm gate `inputs/gate_scores.parquet` sha256 `0eb808e0…2dcd`.
- Điểm vận hành của mutant: T = 15, W = 4, τ_open/τ_close = 0,55/0,30, watchdog 12/6, R = 5. Chương trình gốc = bộ điều khiển **đã sửa** (`>=`).
- Tập mutant: 36 mutant, 6 lớp × 6, danh sách đóng băng từ P0 trong `code/p0_sanity.py` (sha256 `1c5e25f3…54f7`), ngữ nghĩa từng bước trong `code/mutants.py` (sha256 `05e144e2…a716`; đã kiểm khớp 0 khung lệch với `run_mutant` của P0 trên 21 video).
- Mô hình năng lượng Deep3: E = 30,4 + a·(290,8 − 30,4) mJ/khung. miss_all = tỉ lệ sự kiện bị lỡ (mọi D).

## 2. Định nghĩa (đóng băng)

- **Vi phạm I_R (chính):** model checking trạng thái tường minh (`mutants.model_check`, score đối nghịch, mọi khung hợp lệ) tìm được một lần chạy có R = 5 khung CLOSED liên tiếp. **Phụ:** vi phạm trên corpus (≥ 1 cửa sổ 5 khung CLOSED trên ≥ 1 video).
- **Lọc tương đương:** (i) mutant có trace OPEN/CLOSED trùng 100 % với chương trình gốc trên 121 video = *mutant tương đương với gốc*, loại; (ii) các mutant có trace trùng nhau 100 % gộp thành một lớp, giữ đại diện có ID nhỏ nhất. Số còn lại = n_eff; số vi phạm I_R trong n_eff = n_v.
- **Frontier đúng:** `results/frontier_correct.parquet` (sha256 `db6d7cce…0518`): 2.730 cấu hình `>=` (T ∈ 5..30, W ∈ 1..15, 7 mức τ_open ∈ {0,40 … 0,70}, τ_close = τ_open·0,30/0,55), 84 điểm Pareto theo (E, miss_all).
- **Chuẩn hoá:** e = (E − E_min)/(E_max − E_min), m = (miss − miss_min)/(miss_max − miss_min), min/max trên 2.730 cấu hình đúng (E ∈ [212,525; 267,886] mJ, miss ∈ [0,02524; 0,12744]).
- **Khoảng cách tới frontier:** khoảng cách Euclid từ (e, m) của mutant tới đường gấp khúc nối các điểm Pareto (sắp theo E). Hai phía (mutant có thể nằm trên hay dưới frontier).
- **δ = 1 bậc lưới τ** = trung vị khoảng cách chuẩn hoá giữa hai mức τ kề nhau với cùng (T, W): **δ = 0,1062** (n = 2.340 cặp; IQR 0,0819–0,1452). Báo thêm δ/2 = 0,0531 và 2δ = 0,2124.
- **Frontier-indistinguishable ở mức δ:** mutant vi phạm I_R (chính) và khoảng cách ≤ δ.
- **Phụ — một phía:** mutant vi phạm I_R và điểm của nó *không* bị một điểm Pareto nào chi phối chặt quá δ (tức khoảng cách ≤ δ **hoặc** không bị điểm Pareto nào chi phối).

## 3. Giả thuyết và luật PASS/FAIL

| ID | Giả thuyết | PASS khi |
|---|---|---|
| **H4′** | ≥ 1/3 mutant vi phạm I_R (sau lọc) là frontier-indistinguishable ở δ | tỉ lệ điểm (k/n_v) ≥ 1/3. Báo kèm k, n_v, CI 95 % bootstrap theo video (B = 1000, seed 42; frontier, chuẩn hoá và điểm mutant tính lại trong mỗi mẫu, δ giữ cố định). Với n_v < 10 không in ±, chỉ in k/n_v và CI mô tả |
| **H5** | Lỗi ship (M1a, `fsl > W`) là frontier-indistinguishable ở δ | khoảng cách điểm ≤ δ. Báo kèm tỉ lệ mẫu bootstrap có khoảng cách ≤ δ |
| **H6a** | Model checking bắt 100 % mutant vi phạm, mỗi mutant < 10 s | mọi mutant vi phạm trên corpus đều bị MC báo vi phạm, và thời gian MC tối đa < 10 s (laptop i7-1185G7, 1 luồng) |
| **H6b** | Monitor I_R bắt trong ≤ R khung | với mọi mutant vi phạm trên corpus, trên mọi video vi phạm, báo động đầu tiên ở khung thứ ≤ R tính từ đầu cửa sổ vi phạm đầu tiên |
| **H6c** | Kiểm thử corpus (20 video ngẫu nhiên, seed) bắt < 50 % | 20 video = `numpy.random.default_rng(42).choice(121, 20, replace=False)` trên danh sách video sắp theo tên; oracle chính = "có ≥ 1 sự kiện D ≥ R bị lỡ"; tỉ lệ mutant vi phạm (chính, sau lọc) bị bắt < 50 %. Phụ: oracle benchmark (|ΔmAP-proxy| ≥ 0,5 điểm hoặc |ΔE| ≥ 1 %), và tỉ lệ bắt trung bình trên 1.000 tập 20 video ngẫu nhiên |

## 4. Tri thức có trước khi đóng băng (khai minh bạch)

- P0 (15-10): chạy nhanh 10 mutant × 20 video; biết điểm (E, miss) của bản ship và bản sửa trên toàn corpus (ΔE = −3,24 %, miss 301 vs 258).
- P1 (16-10, trước khi đóng băng): đã chạy `frontier_sweep.py` (cần để tính δ) và đã chạy thử `model_check` trên 36 mutant khi kiểm engine (5 mutant MC-vi phạm: M1a, M1b, M5a, M5b, M5d; mỗi mutant < 0,1 s). **Chưa** tính khoảng cách frontier, lọc tương đương hay oracle kiểm thử của bất kỳ mutant nào.
- Hệ quả: n_v dự kiến nhỏ (≤ 5) → H4′ có độ phân giải thô (bước 1/n_v). Ghi trước: không thêm mutant sau khi thấy kết quả; mở rộng tập mutant (PLAN ≥ 40) chỉ qua PREREG v1.1 và báo song song.

## 5. Đầu ra

`results/mutants_full.csv` (một dòng/mutant), `results/p1_t4.json` (H4′/H5/H6, CI, bảng tương đương), chạy bằng `python code/p1_t4.py --seed 42`.
