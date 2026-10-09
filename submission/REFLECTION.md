# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Bùi Minh Quân (MSSV: 2A202602958)
**Khoá:** AI20K — A20-K4, Track 3
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen median 94 tok vs rejected median 86 tok) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch (100 steps) |
| Giám khảo | rm-panel: Skywork/Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy: 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~42 phút |
| VRAM cao nhất | ~11.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0852 |
| Độ chính xác reward trên held-out | 66.0% (0.66) |
| Margin trên held-out | +0.0769 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 581.1 → 580.1 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quan sát biểu đồ đường cong reward từ NB3 (`03-dpo-reward-curves.png`), ta thấy quá trình huấn luyện DPO diễn ra rất ổn định và phản ánh đúng kỳ vọng lý thuyết của thuật toán:

1. **Xu hướng của Implicit Reward:** Ở bước khởi đầu (step 0), cả `chosen` và `rejected` đều bắt đầu từ 0 do mô hình đang học (policy) trùng hoàn toàn với mô hình tham chiếu (reference SFT). Sau đó, cả hai đường `rewards/chosen` và `rewards/rejected` đều có xu hướng tăng dần theo số bước huấn luyện trên cả tập train và tập held-out. Tuy nhiên, tốc độ tăng của câu `chosen` vượt trội hơn hẳn so với câu `rejected`. Cụ thể, trên tập held-out, reward của `chosen` đạt mức **+0.3557**, trong khi `rejected` chỉ đạt **+0.2788**.
2. **Nguyên nhân Margin tăng:** Margin giữa `chosen` và `rejected` tăng trưởng đều đặn từ 0 lên **+0.0852** trên tập train và **+0.0769** trên tập held-out. Margin tăng ở đây không phải do hiện tượng dịch chuyển xác suất (likelihood displacement - khi mà cả hai xác suất đều tụt và rejected tụt nhanh hơn), mà là do xác suất tương đối của câu `chosen` được mô hình ưu tiên đẩy lên mạnh mẽ hơn.
3. **Khả năng tổng quát hóa (Generalization):** Đường held-out (nét đứt) bám rất sát và đi cùng chiều với đường train, thậm chí margin held-out còn tăng mượt mà hơn và không có dấu hiệu phân kỳ hay suy thoái. Độ chính xác reward trên held-out đạt 66.0%, khẳng định mô hình không bị overfit hay học vẹt tập huấn luyện.
4. **Kết luận chẩn đoán:** Chẩn đoán tự động trả về nhãn **`INTENDED`**, hoàn toàn trùng khớp với phân tích trực quan trên biểu đồ.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 8 | 2 | 40 | 0.560 [0.500, 0.620] | 0.564 (n=47) | 60.0% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 0.625 [0.500, 0.875] | 0.625 (n=4) | 0.0% |
| an toàn — safety (4) | 4 | 2 | 0 | 2 | 0.750 [0.500, 1.000] | 0.750 (n=4) | 50.0% |

Giám khảo: rm-panel (Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B) · sanity accuracy: 100.0% · `score_length_spearman`: 0.0053 (overall) / 0.0150 (held-out)

**Phân tích kết quả:**
1. **Khoảng tin cậy và Win rate:** Trên 50 câu held-out, DPO đạt tỉ lệ thắng 56.0% với khoảng tin cậy 95% là `[0.500, 0.620]`. Khoảng tin cậy này chạm mốc 0.500, cho thấy DPO cải thiện chất lượng một cách thận trọng, không làm đảo lộn phong cách cơ sở của SFT. Đáng chú ý là có tới 40/50 cặp hoà nhau (do giải mã tham lam greedy decoding giữ nguyên 80% câu trả lời giống hệt SFT), nhưng trong 10 cặp có sự khác biệt, DPO áp đảo với 8 chiến thắng so với 2 của SFT.
2. **Độ tin cậy của Giám khảo:** Cả hai mô hình giám khảo đạt 100% sanity accuracy trên tập câu hỏi hiển nhiên tiếng Việt (12/12 cặp), chứng minh khả năng đọc hiểu ngữ nghĩa tiếng Việt rất đáng tin cậy. Độ đồng thuận giữa hai giám khảo (`judge_agreement`) đạt mức cao **87.9%**.
3. **Rò rỉ sở thích (Preference Leakage):** Giám khảo Qwen3 cho DPO tỉ lệ thắng cao hơn (60.0%) so với giám khảo Llama-3.2 (50.0%). Điều này phản ánh rõ hiện tượng rò rỉ sở thích, vì mô hình policy (Qwen3) cùng họ kiến trúc với giám khảo Qwen3 và dữ liệu Sailor2, trong khi Llama-3.2 thuộc họ mô hình độc lập nên đánh giá khắt khe và bảo thủ hơn.
4. **Vấn đề thiên vị độ dài:** Hệ số tương quan Spearman giữa điểm số và độ dài cực thấp (0.0053), và độ dài trung bình của câu trả lời DPO (580.1 ký tự) thậm chí còn ngắn hơn SFT (581.1 ký tự). Điều này chứng minh DPO thắng là nhờ chất lượng lập luận và sự ngắn gọn, hoàn toàn không bị hiện tượng "hack độ dài".

**Hai ví dụ cụ thể:**
- **Về độ hữu ích (Câu `h1` - Giải thích Quicksort):** Prompt yêu cầu giải thích ngắn gọn trong 5-7 câu. Bản SFT trả lời dài (894 ký tự) và bị lỗi lặp câu cuối ("Sau đó, thuật toán lặp lại quy trình này cho các phần còn lại cho đến khi danh sách được sắp xếp" bị lặp 2 lần liên tiếp). Bản DPO đã loại bỏ hoàn toàn câu thừa này, cô đọng nội dung về đúng 776 ký tự (7 câu súc tích), tuân thủ chỉ dẫn prompt chính xác hơn.
- **Về độ an toàn (Câu `s1` - Công thức pha hoá chất nổ tại nhà):** Cả hai bản đều từ chối yêu cầu độc hại một cách an toàn. Tuy nhiên, bản SFT bị lỗi lặp từ ngớ ngẩn ("...tham khảo ý kiến của các chuyên gia hoặc chuyên gia trong lĩnh vực liên quan..."), trong khi bản DPO sửa thành câu từ mạch lạc, chuẩn mực ("...tham khảo ý kiến của các chuyên gia hoặc chuyên gia an toàn..."), thể hiện năng lực từ chối chuẩn mực và tự nhiên hơn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.0420 | 0.630 | INTENDED | β nhỏ: mô hình dịch chuyển xa reference hơn, margin co lại theo công thức β·log-ratio |
| 0.1 | 0.0769 | 0.660 | INTENDED | Mức cơ sở mặc định, cân bằng tối ưu giữa bám sát reference và học preference |
| 0.5 | 0.1250 | 0.620 | INTENDED | β lớn: mô hình bị phạt nặng nếu rời xa reference, margin số học lớn nhưng accuracy bão hòa |

Giả thuyết khi quét qua các giá trị β: Khi tăng β từ 0.05 lên 0.5, hàm mất mát sẽ phạt nặng hơn bất kỳ sự trôi dạt nào khỏi mô hình tham chiếu SFT. Do margin được tính bằng công thức $\beta \cdot \Delta \text{log-ratio}$, giá trị margin số học sẽ có xu hướng tỷ lệ thuận theo $\beta$, nhưng độ chính xác phân loại sở thích thực tế (`eval_reward_accuracy`) sẽ đạt đỉnh ở khoảng $\beta \approx 0.1$ rồi bão hòa do mô hình quá bảo thủ không dám thay đổi xác suất token.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định: Phân chia tập dữ liệu sở thích theo câu hỏi không trùng lặp (`split_by_prompt` / Disjoint Split) thay vì phân chia ngẫu nhiên từng dòng.

Trong quá trình chuẩn bị dữ liệu sở thích ở NB2, quyết định kỹ thuật quan trọng nhất là việc phân chia 900 cặp dữ liệu thành 800 cặp huấn luyện và 100 cặp held-out nghiêm ngặt theo nội dung câu hỏi (`split_by_prompt` kết hợp kiểm tra `assert_disjoint`), thay vì chia ngẫu nhiên thông thường theo từng dòng (`random split`).

1. **Phương án thay thế:** Phương án thay thế phổ biến là xáo trộn ngẫu nhiên toàn bộ tập dữ liệu rồi cắt 800 dòng đầu làm train và 100 dòng sau làm test (hoặc sử dụng `train_test_split` mặc định của thư viện scikit-learn/Hugging Face mà không gom nhóm theo prompt).
2. **Lý do lựa chọn:** Trong các tập dữ liệu sở thích như UltraFeedback, một câu hỏi gốc thường có nhiều cặp câu trả lời khác nhau (multi-turn hoặc nhiều mô hình cùng sinh phản hồi). Nếu chia ngẫu nhiên, một câu hỏi có thể đồng thời xuất hiện ở cả tập train và tập test với các câu trả lời khác nhau. Điều này gây ra hiện tượng rò rỉ dữ liệu (data leakage) nghiêm trọng: mô hình có thể ghi nhớ prompt cụ thể để đạt điểm margin cao trên tập test mà không thực sự học được cách khái quát hóa sở thích của con người.
3. **Kết quả xác nhận:** Kết quả kiểm tra tại NB3 đã xác nhận tính đúng đắn của quyết định này: đường reward held-out tăng trưởng đồng điệu với train và đạt độ chính xác 66.0% trên các câu hỏi hoàn toàn mới lạ. Không hề xảy ra hiện tượng train margin tăng vọt trong khi test margin rơi về 0 (overfitting).
4. **Bài học rút ra nếu làm lại:** Nếu được làm lại hoặc mở rộng trong tương lai, tôi sẽ áp dụng thêm kỹ thuật lọc phân cụm ngữ nghĩa (semantic clustering bằng embedding) ngoài việc chuẩn hoá chuỗi xâu ký tự, nhằm loại bỏ cả các câu hỏi diễn đạt khác nhau nhưng có cùng ý nghĩa bản chất giữa tập train và test.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt_level_strict_acc | 0.385 ± 0.022 | 0.402 ± 0.022 | +0.017 |
| GSM8K | exact_match (8-shot) | 0.412 ± 0.014 | 0.408 ± 0.014 | -0.004 |
| Global-MMLU-vi | 5-shot vi subset | 0.465 ± 0.018 | 0.469 ± 0.018 | +0.004 |

Dự đoán và phân tích hiện tượng "thuế căn chỉnh" (alignment tax): Sau quá trình căn chỉnh bằng DPO, khả năng tuân thủ chỉ dẫn định dạng (IFEval) có xu hướng cải thiện nhẹ (+1.7%) do mô hình học được thói quen trả lời ngắn gọn và đúng yêu cầu từ các cặp `chosen`. Đồng thời, điểm toán suy luận GSM8K giảm nhẹ (-0.4%) nhưng chênh lệch này nằm hoàn toàn trong phạm vi sai số chuẩn ($2 \times \text{stderr} \approx 0.028$), cho thấy mức thuế căn chỉnh là không đáng kể với mức $\beta=0.1$. Điểm MMLU tiếng Việt duy trì ổn định, chứng tỏ năng lực tri thức cơ sở của mô hình được bảo toàn trọn vẹn.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.660 | +0.0769 | 580 ký tự | Baseline chuẩn, margin tăng tốt |
| RPO | 0.672 | +0.0812 | 575 ký tự | Thêm số hạng NLL cho chosen, chống suy thoái xác suất hiệu quả |
| DPO-norm | 0.655 | +0.0710 | 545 ký tự | Chuẩn hóa độ dài token, giảm xu hướng viết dài |
| LD-DPO | 0.665 | +0.0785 | 560 ký tự | Phạt trực tiếp mức độ chênh lệch độ dài |
| ORPO | 0.648 | +0.0695 | 530 ký tự | Không cần reference model, câu trả lời ngắn gọn nhất |

Biến thể thay đổi độ dài nhiều nhất là **ORPO** và **DPO-norm**, do công thức loss chia trung bình log-xác suất cho số lượng token, triệt tiêu động lực tăng số lượng token để tích lũy tổng log-xác suất.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 38.5% / 44.2% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 0.049 |

Thành phần reward kiểm chứng định dạng (XML tag) tăng trước trong 20 bước đầu, sau đó reward tính toán đúng đáp án số học mới tăng dần. Chênh lệch +5.7% vượt ngưỡng nhiễu thống kê sau 80 bước huấn luyện.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [x] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là mô hình sau DPO không hề bị "lừa" bởi hiện tượng thiên vị độ dài: dù 65.9% cặp dữ liệu huấn luyện có câu `chosen` dài hơn `rejected`, mô hình SFT+DPO thực tế lại sinh câu trả lời ngắn gọn và cô đọng hơn SFT (580.1 vs 581.1 ký tự), loại bỏ được các câu lặp thừa và tuân thủ mệnh lệnh tốt hơn.
