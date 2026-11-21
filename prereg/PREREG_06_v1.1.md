# PREREG_06 v1.1 — Mở rộng nghiên cứu mutant T4 (đóng băng 17-10-2026)

*Bổ sung cho `PREREG_06.md` v1.0 (SHA-256 `f2d33ae2…d940`, đóng băng 16-10-2026, không đổi). Lý do v1.1: CP1_DECISION_06 #4 — n_v = 3 quá nhỏ, δ = 1 bậc τ quá thô, H6c FAIL do một lần rút 20 video. Mọi định nghĩa và giả thuyết v1.0 giữ nguyên; v1.1 chỉ **thêm** mutant, một ngưỡng δ_boot và 4 giả thuyết. Kết quả v1.0 và v1.1 báo **cạnh nhau**. SHA-256 của file này và của `code/mutants_v11.py` ghi ở `results/PREREG_06_v1.1.sha256` trước khi chạy `code/p2_t4.py`; script tự kiểm cả hai hash (và hash v1.0) rồi mới chạy.*

## 1. Dữ liệu, chương trình gốc, tập mutant

- Corpus như v1.0: 121 video, 166.589 khung, 3.170 sự kiện; cache `corpus_flat.npz` (sha256 `a8da5105…1ec1`).
- Điểm vận hành: T = 15, W = 4, τ_open/τ_close = 0,55/0,30, watchdog 12/6, R = 5, d = 0, b = 0, κ = 1.
- **Hai họ mutant, hai chương trình gốc:**
  - *Họ G1* — gốc **ORIG** = bộ điều khiển đã sửa `>=` (như v1.0). Gồm 36 mutant v1.0 (M1a…M6f) và **41 mutant mới N01…N41**.
  - *Họ CB* — gốc **ORIG_CB** = ORIG nối tiếp circuit breaker ASMAG đúng (spec/FSM_06.md §2: cửa sổ 300, θ_high 0,95, θ_low 0,60, K = 300, P = 30; kích hoạt = gate request của khung G1 gần nhất, tức kích hoạt khung trước như ASMAG; khung BYPASS thuần luôn dispatch vì d = 0). Gồm **10 mutant mới C01…C10** (C05, C06 là bậc 2, ghi rõ).
- Tổng mutant mới: **51** (≥ 40 theo CP1 #4). Danh sách, mô tả, knob và ngữ nghĩa từng bước: `code/mutants_v11.py` (đóng băng bằng hash; `Ctl11` tái tạo đúng `mutants.Ctl` cho mọi knob v1.0 — selftest trên chuỗi tổng hợp 216/216).
- Lớp nhắm (CP1 #4): reuse-cap (so sánh, reset sai sự kiện, tràn/bão hoà bộ đếm, cap theo thời gian), bộ đếm fsl (bước k, chỉ đếm một số khung), đơn vị clock/κ (refresh wall-clock κ ≠ 1, thanh ghi trễ 1 tick, quyết định trễ 1 khung), missing-frame (valid = 0: reset / không tăng / tăng đôi / OPEN ma / hoãn refresh / ép OPEN sau khung thiếu), pha refresh (±, chu kỳ T − 1, khoá theo OPEN của gate), lịch probe CB (trượt pha, độ dài, chu kỳ, thoát/vào BYPASS, probe không chạy detector, bypass nửa tốc độ, probe rơi vào khung thiếu).
- Ghi trước: mutant missing-frame (N33–N38) và C09 là **ẩn trên corpus** (valid = 1 mọi khung) nên dự kiến bị lọc là tương đương với gốc; chúng chỉ có kết quả ở phân tích thăm dò mục 5.

## 2. Định nghĩa

Giữ nguyên v1.0 (vi phạm I_R chính = model checking; phụ = corpus; lọc tương đương; frontier; chuẩn hoá; khoảng cách; δ_τ; FI; một phía), với các mở rộng sau.

- **Model checking (chính).** Họ G1: BFS trạng thái tường minh `mutants.model_check` (v1.0) / `mutants_v11.mc_g1` (v1.1), score đối nghịch 7 mức, mọi khung hợp lệ. Họ CB: `mutants_v11.mc_cb` — BFS trên khung ACTIVE, khung PROBE và đoạn BYPASS thuần (một bước vĩ mô; G1 đóng băng, đồng hồ refresh chạy), vào BYPASS và quyết định thoát ở cuối probe là đối nghịch (xấp xỉ trên của cửa sổ 300 khung và tỉ lệ probe). Nếu MC chạm trần trạng thái → verdict lấy theo corpus và đánh dấu.
- **Lọc tương đương (hợp v1.0 ∪ v1.1 = 87 mutant).** (i) trace OPEN/CLOSED trùng 100 % với **gốc của họ mình** trên 121 video → loại; (ii) mutant trùng trace nhau (bất kể họ) gộp một lớp, đại diện = ID nhỏ nhất theo thứ tự M… < N… < C…. Còn lại n_eff (báo theo họ và tổng); n_v = số vi phạm I_R (MC) trong n_eff.
- **Frontier theo họ.** Họ G1: `results/frontier_correct.parquet` (v1.0). Họ CB: cùng lưới 2.730 cấu hình (T 5..30, W 1..15, 7 mức τ) **có breaker đúng trong vòng** (numba `controller_cb`, kiểm khớp từng khung với bản Python trên 121 video tại (15, 4, 0,55) trước khi dùng) → `results/frontier_correct_cb.parquet`. Chuẩn hoá và khoảng cách như v1.0 trong lưới của từng họ. δ_τ^CB = trung vị khoảng cách hai mức τ kề nhau (cùng T, W) trên lưới CB, cách tính như δ_τ = 0,1062 của v1.0.
- **δ_boot (mới, theo họ).** B = 1000 mẫu bootstrap theo video (`numpy.random.default_rng(seed + 100)`, seed = 42); trong mỗi mẫu tính lại frontier và chuẩn hoá trên lưới của họ, rồi tính khoảng cách của **cấu hình đúng tại điểm triển khai** (15, 4, 0,55) (họ CB: kèm breaker đúng) tới frontier đó. δ_boot = phân vị 95 % (numpy, nội suy tuyến tính) của 1000 khoảng cách. Ý nghĩa: "một cấu hình đúng, chỉ do biến động lấy mẫu video, có thể cách frontier tới δ_boot".
- **FI ở mức δ:** như v1.0 — vi phạm I_R (MC) và khoảng cách điểm (toàn corpus) ≤ δ.

## 3. Giả thuyết và luật PASS/FAIL

**v1.0 (giữ nguyên, không đổi luật):** H4′, H5, H6a, H6b, H6c như `PREREG_06.md` §3. Báo (a) trên tập v1.0 — phải tái tạo đúng `results/p1_t4.json`; (b) trên tập hợp v1.0 ∪ v1.1 sau lọc (họ CB dùng δ_τ^CB thay cho δ_τ ở H4′).

**v1.1 (mới), trên V = tập mutant vi phạm I_R (MC) sau lọc, hợp hai họ:**

| ID | Giả thuyết | PASS khi |
|---|---|---|
| **H4″** | ≥ 1/3 mutant vi phạm là FI ở δ_boot | k/n_v ≥ 1/3, mỗi mutant so với δ_boot của họ mình. Báo kèm CI 95 % bootstrap theo video (B = 1000, seed 42, δ_boot cố định; frontier, chuẩn hoá, điểm mutant tính lại trong mỗi mẫu). n_v < 10 → không in ± |
| **H5″** | Lỗi ship M1a là FI ở δ_boot^G1 | khoảng cách điểm của M1a ≤ δ_boot^G1 (và M1a vi phạm MC) |
| **H6c′** | Kiểm thử corpus 20 video bắt < 50 % *trung bình* | với **mọi** m ∈ V: tỉ lệ bắt trung bình của oracle chính "có ≥ 1 sự kiện D ≥ R bị lỡ" trên 1.000 tập 20 video ngẫu nhiên (`default_rng(seed + 1)`, như v1.0) < 0,5. PASS ⇔ max_m < 0,5 |
| **H7** | Độ nhạy: ≥ 50 % mutant vi phạm cũng FI ở δ_τ/2 | tỉ lệ FI ở δ_τ/2 (họ G1: 0,0531; họ CB: δ_τ^CB/2) ≥ 0,5 |

Ghi trước về diễn giải: nếu δ_boot ≥ δ_τ thì H4″ dễ hơn H4′ (ngưỡng rộng hơn); nếu δ_boot < δ_τ thì chặt hơn. H7 là kiểm tra độ nhạy, không phải giả thuyết chính.

## 4. Tri thức có trước khi đóng băng (khai minh bạch)

- Toàn bộ kết quả v1.0 (`results/p1_t4.json`): n_eff 20, n_v 3 (M1a, M5a, M5d), khoảng cách 0,083/0,086/0,080, H6c FAIL.
- Dự đoán lý thuyết **trước khi chạy** (Mệnh đề 2.7, THEORY_06 §2.5, áp cho bypass nửa tốc độ như lưới dispatch bước 2): C04 (K = 300, P = 30, 2 | gcd) **không** vi phạm; C05 (P = 31) và C06 (K = 301) **vi phạm**. Với d = 0 và bypass mọi khung: C08 (trượt pha probe) không vi phạm.
- Đã chạy selftest `mutants_v11.py` trên **chuỗi score tổng hợp** (không phải corpus): biết knob nào đổi trace tổng hợp (N02, N07, N19, N32, N40 không đổi trên 6 chuỗi tổng hợp). Đã đếm: điều kiện vào BYPASS (trung bình gate request > 0,95 trên cửa sổ 300) xảy ra ở 83/121 video (50.595 khung) — chỉ dùng để quyết định đưa họ CB vào; chưa chạy breaker trên corpus.
- **Chưa** chạy MC, replay corpus, khoảng cách frontier, frontier CB, δ_boot hay oracle kiểm thử cho bất kỳ mutant v1.1 nào.
- Sai lệch phát hiện sau khi đóng băng (lỗi cài đặt mutant, lỗi script) → sửa, ghi "Deviation" trong `P2/CP2_REPORT_06.md` kèm kết quả trước/sau; không thêm/bớt mutant sau khi thấy kết quả.

## 5. Thăm dò (không phải giả thuyết)

- MC với ngân sách drop b = 1 mỗi run (ngữ nghĩa theo run, `mc_g1(…, b = 1)`) tại R = 6 = R\*(ORIG, b = 1) (Định lý 2′) cho mọi mutant họ G1 — để mutant missing-frame có verdict.
- Bảng theo lớp; phân bố khoảng cách; so sánh δ_boot với δ_τ.

## 6. Đầu ra

`results/mutants_v11.csv` (một dòng/mutant, hợp 87 mutant, cột họ/lớp/lớp tương đương/MC/khoảng cách/FI/oracle), `results/p2_t4.json` (v1.0 tái tạo, v1.0-trên-hợp, v1.1, δ_boot, CI), `results/frontier_correct_cb.parquet`; chạy `python code/p2_t4.py --seed 42`.
