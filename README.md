# 📈 HOSE Trading Strategy Web App: SMA + OBV OR Quantitative Backtester

Ứng dụng Web chuyên nghiệp xây dựng trên nền tảng **Streamlit** nhằm kiểm định tính hiệu quả của chiến lược giao dịch định lượng kết hợp tín hiệu xu hướng giá (**SMA - Simple Moving Average**) và dòng tiền khối lượng (**OBV - On-Balance Volume**) theo logic **OR**, tối ưu hóa danh mục **MPT (Modern Portfolio Theory)** và kiểm định ngoài mẫu (**Out-of-Sample Test**) trên thị trường chứng khoán HOSE (Việt Nam).

---

## 🌟 Tính Năng Nổi Bật Của Web App

1. **Lọc Cổ Phiếu Tự Động (TA Screening):**
   - Đánh giá và xếp hạng 100 cổ phiếu HOSE trên tập Train (2020–2021) theo 4 nhóm nhân tố định lượng: **Trend (30%)**, **Momentum (25%)**, **Volume (25%)**, **Risk (20%)**.
   - Hỗ trợ thanh trượt tùy chỉnh linh hoạt trọng số các nhân tố để kiểm tra độ nhạy (Sensitivity Analysis).
   - Tự động lọc ra Top N cổ phiếu (mặc định Top 5: `DIG`, `DPM`, `NLG`, `KBC`, `DCM`) hoặc cho phép chọn thủ công các mã tùy ý.

2. **Chiến Lược Tín Hiệu Kết Hợp SMA & OBV (Logic OR):**
   - **Tín hiệu Mua:** Khi SMA ngắn cắt lên SMA dài **HOẶC** OBV cắt lên đường trung bình động của OBV.
   - **Tín hiệu Bán:** Khi SMA ngắn cắt xuống SMA dài **HOẶC** OBV cắt xuống đường trung bình động của OBV.
   - Khử xung đột (Conflict Resolution) khi có cả 2 tín hiệu đối nghịch trong cùng phiên.
   - **Không Look-Ahead Bias:** Tín hiệu phát sinh ở phiên $t$ chỉ được thực thi khớp lệnh vào phiên $t+1$ (`shift(1)`).
   - **Khấu trừ chi phí thực tế:** Tính toán phí giao dịch (Turnover Fee) cho mỗi lần thay đổi vị thế (mặc định 0.1%).

3. **Tối Ưu Hóa Tham Số & Phân Bổ Danh Mục:**
   - Cung cấp sẵn bộ tham số đã tối ưu hóa bằng **Hyperopt (TPE Algorithm)** trên tập Train.
   - Hỗ trợ chạy thuật toán Hyperopt trực tiếp ngay trên giao diện web với số vòng lặp tùy chọn.
   - Tối ưu hóa tỷ trọng danh mục theo **Lý thuyết Danh mục Hiện đại (MPT Markowitz)** với ràng buộc tập trung ($w_i \in [5\%, 40\%]$) để tối đa hóa Sharpe Ratio.

4. **Kiểm Định Ngoài Mẫu Khắc Nghiệt (Out-of-Sample Test 2022):**
   - Kiểm định trên năm 2022 (giai đoạn thị trường gấu VN-Index giảm sâu ~35-40%).
   - Cơ chế **Warm-up toàn bộ lịch sử** trước khi bước vào năm 2022 để không bị gián đoạn dữ liệu đường trung bình dài (SMA350).
   - So sánh trực tiếp 3 danh mục:
     - **Equal Weight Buy & Hold Benchmark:** Tính chuẩn Buy & Hold (không daily rebalance).
     - **SMA + OBV OR (Equal Weight):** Tái cân bằng đều 20% mỗi ngày trên các dòng tiền chiến lược.
     - **SMA + OBV OR (MPT Weighted):** Trọng số tĩnh tối ưu từ tập Train.

5. **Trực Quan Hóa Tương Tác (Plotly Charts):**
   - Biểu đồ nến/giá với các điểm mua/bán (Buy 🟢 / Sell 🔴).
   - Biểu đồ chỉ báo OBV và đường trạng thái nắm giữ (Holding state).
   - Biểu đồ so sánh đường cong vốn (**Equity Curve**) và mức sụt giảm tối đa (**Underwater Drawdown**).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```plaintext
├── app.py                      # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── README.md                   # Hướng dẫn chi tiết cài đặt và sử dụng
├── HOSE_2020_2023_in (1).csv    # File dữ liệu mẫu lịch sử giá cổ phiếu HOSE
└── HOSE_STRATEGY_8_5_REVISED (1).ipynb # Jupyter Notebook nghiên cứu gốc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Chuẩn bị môi trường Python
Yêu cầu Python phiên bản `>= 3.9` (khuyến nghị Python 3.10 hoặc 3.11).

```bash
# Mở terminal hoặc PowerShell tại thư mục dự án
cd "c:\Users\HP\OneDrive\Desktop\FINALLL"

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows:
.\venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Sau khi chạy lệnh, trình duyệt web sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Tải Lên GitHub & Deploy Lên Streamlit Cloud

### Bước 1: Khởi tạo Git và đẩy code lên GitHub
1. Mở terminal tại thư mục dự án, khởi tạo Git repository:
   ```bash
   git init
   git add app.py requirements.txt README.md "HOSE_2020_2023_in (1).csv"
   git commit -m "Initial commit: Streamlit app for HOSE SMA+OBV strategy"
   ```
2. Đổi tên nhánh mặc định thành `main`:
   ```bash
   git branch -M main
   ```
3. Tạo một Repository mới trên [GitHub](https://github.com/new) (ví dụ đặt tên là `hose-trading-strategy`).
4. Liên kết và đẩy code lên GitHub:
   ```bash
   git remote add origin https://github.com/<tai-khoan-github-cua-ban>/hose-trading-strategy.git
   git push -u origin main
   ```

> **Lưu ý về dữ liệu:** File `HOSE_2020_2023_in (1).csv` có dung lượng khoảng 5.7 MB, hoàn toàn nằm trong giới hạn cho phép của GitHub (tối đa 100 MB cho 1 file). Bạn có thể đẩy trực tiếp lên GitHub mà không cần cấu hình Git LFS.

---

### Bước 2: Deploy miễn phí trên Streamlit Cloud

1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Bấm nút **"Create app"** (hoặc **"New app"**).
3. Điền các thông tin:
   - **Repository:** Chọn repository vừa tạo (ví dụ: `<tai-khoan>/hose-trading-strategy`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Bấm **"Deploy!"**.
5. Đợi 1–2 phút để hệ thống tự động cài đặt thư viện từ `requirements.txt` và khởi chạy web app.
6. Bạn sẽ nhận được một đường link public (ví dụ: `https://<ten-app>.streamlit.app`) có thể chia sẻ cho giảng viên, đồng nghiệp hoặc đối tác truy cập trực tiếp từ bất kỳ thiết bị nào!

---

## 📊 Tóm Tắt Kết Quả Định Lượng Thực Nghiệm (Out-of-Sample 2022)

| Chỉ số Hiệu Suất | Equal Weight Buy & Hold | SMA + OBV OR (Equal Weight) | SMA + OBV OR (MPT Weighted) |
| :--- | :---: | :---: | :---: |
| **Total Return (%)** | **-42.27%** | **-8.63%** | **-3.21%** |
| **Annual Return (%)** | **-42.66%** | **-8.73%** | **-3.25%** |
| **Annual Volatility (%)** | 46.40% | 26.29% | 28.06% |
| **Sharpe Ratio** | **-0.96** | **-0.22** | **+0.02** |
| **Max Drawdown (%)** | **-60.52%** | **-31.85%** | **-29.22%** |

### 🔍 Đánh Giá Ý Nghĩa Chiến Lược:
- **Phòng vệ rủi ro vượt trội (Bear Market Alpha):** Trong năm 2022 khi thị trường sụp đổ nặng nề khiến danh mục Buy & Hold mất hơn **42%** vốn (đỉnh điểm sụt giảm **-60.5%**), chiến lược kết hợp **SMA + OBV OR** giúp bảo toàn vốn ngoạn mục khi chỉ giảm **-3.21%** (danh mục MPT) và cắt giảm hơn 50% mức sụt giảm tối đa (Max Drawdown chỉ còn **-29.2%**).
- **Lợi thế tín hiệu sớm của OBV:** Khối lượng và dòng tiền OBV cảnh báo phân phối trước khi giá gãy SMA, cho phép hệ thống thoát vị thế sớm về tiền mặt (`holding = 0`) để đứng ngoài thị trường trong suốt các nhịp bán tháo lớn.

---

## 🛠️ Tùy Chỉnh & Mở Rộng Dữ Liệu

Ứng dụng được thiết kế dạng **Plug-and-Play**:
- Bạn có thể tải lên file dữ liệu CSV của các giai đoạn khác (2023, 2024, 2025) hoặc thị trường khác (HNX, UPCOM, Crypto).
- Định dạng yêu cầu tối thiểu gồm các cột: `date`, `ticker`, `open`, `high`, `low`, `close`, `volume` (và `adj_close` nếu có).
