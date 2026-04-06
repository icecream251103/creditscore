# Hệ thống Chấm điểm Tín dụng — AI Credit Scoring

Dự án xây dựng hệ thống đánh giá hồ sơ vay tín dụng tự động sử dụng Machine Learning, bao gồm:

- **Mô hình phân loại** (Logistic Regression): xét duyệt hồ sơ — *Được xét / Từ chối*
- **Mô hình hồi quy điểm tín dụng** (XGBoost): dự đoán điểm tín dụng (0–1000)
- **Mô hình hồi quy hạn mức** (XGBoost): dự đoán hạn mức tối đa được cấp (VNĐ)
- **API Flask** phục vụ dự đoán thời gian thực
- **Giao diện web hiện đại** để nhập liệu và xem kết quả

---

## Kết quả mô hình

| Mô hình | Mục tiêu | Chỉ số |
|---|---|---|
| Logistic Regression | Trạng thái hồ sơ | Accuracy = **92.7%** |
| XGBoost (GridSearchCV) | Điểm tín dụng | R² = **0.9918**, MAE = **6.58 điểm** |
| XGBoost (GridSearchCV) | Hạn mức tối đa | R² = **0.9538** |

---

## Cấu trúc dự án

```
score-nhu/
├── DuAnThucTap.ipynb       # Notebook phân tích, huấn luyện và đóng gói mô hình
├── app.py                  # Flask API backend
├── test_api.py             # Script kiểm tra API
├── static/
│   └── index.html          # Giao diện web
├── models/                 # Các mô hình đã lưu (tự sinh ra sau khi chạy notebook)
│   ├── model_trang_thai.pkl
│   ├── model_diem.pkl
│   ├── model_han_muc.pkl
│   ├── scaler.pkl
│   └── feature_orders.json
└── .venv/                  # Môi trường ảo Python
```

---

## Yêu cầu hệ thống

- Python 3.9+
- File dữ liệu: `DLKH_Banthaydoi.csv` (đặt tại `C:\Users\GIGABYTE\Downloads\`)

---

## Cài đặt

### 1. Tạo và kích hoạt môi trường ảo

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
```

### 2. Cài đặt thư viện

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn flask flask-cors requests
```

---

## Chạy dự án

### Bước 1 — Huấn luyện và đóng gói mô hình

Mở `DuAnThucTap.ipynb` trong VS Code (hoặc Jupyter) và **Run All Cells**.

Notebook sẽ:
1. Đọc và tiền xử lý dữ liệu từ `DLKH_Banthaydoi.csv`
2. Huấn luyện 3 mô hình (Logistic Regression + 2 XGBoost)
3. Tự động lưu toàn bộ model vào thư mục `models/`

> Bước này cần chạy **một lần duy nhất**. Sau đó thư mục `models/` đã có đủ file.

### Bước 2 — Khởi động API server

```bash
python app.py
```

Server chạy tại: `http://127.0.0.1:5000`

### Bước 3 — Mở giao diện web

Truy cập **http://127.0.0.1:5000** trong trình duyệt.

---

## API Reference

### `GET /api/health`

Kiểm tra trạng thái server.

**Response:**
```json
{ "status": "ok", "models_loaded": true }
```

---

### `POST /api/predict`

Đánh giá hồ sơ vay tín dụng.

**Request body (JSON):**

| Trường | Kiểu | Giá trị hợp lệ |
|---|---|---|
| `Tuổi` | int | 18 – 80 |
| `Tình trạng hôn nhân` | string | `"Không"`, `"Có"` |
| `Số người phụ thuộc` | int | ≥ 0 |
| `Thu nhập` | int | VNĐ |
| `Điểm CIC` | int | 0 – 1000 |
| `Chi phí hàng tháng` | int | VNĐ |
| `Trình độ học vấn` | string | `"Khác"`, `"Cao đẳng"`, `"Đại học"`, `"Sau đại học"` |
| `Thời gian lao động` | int | Số năm |
| `Vị trí công việc` | string | `"KD tự do"`, `"Nhân viên"`, `"Quản lý"` |
| `Thời gian làm việc tại nơi LV hiện tại` | int | Số năm |
| `Tình trạng nhà ở hiện tại` | string | `"Thuê nhà"`, `"Nhà trả góp"`, `"Sở hữu nhà ở"` |
| `Hình thức nhận lương` | string | `"Tiền mặt"`, `"Nhận lương qua NH khác"`, `"Nhận lương qua Sacombank"` |

**Ví dụ request:**
```json
{
  "Tuổi": 30,
  "Tình trạng hôn nhân": "Có",
  "Số người phụ thuộc": 1,
  "Thu nhập": 20000000,
  "Điểm CIC": 750,
  "Chi phí hàng tháng": 6000000,
  "Trình độ học vấn": "Đại học",
  "Thời gian lao động": 5,
  "Vị trí công việc": "Nhân viên",
  "Thời gian làm việc tại nơi LV hiện tại": 3,
  "Tình trạng nhà ở hiện tại": "Thuê nhà",
  "Hình thức nhận lương": "Nhận lương qua Sacombank"
}
```

**Response — Hồ sơ được duyệt:**
```json
{
  "approved": true,
  "trang_thai": "Được xét duyệt",
  "diem_tin_dung": 813.77,
  "han_muc_toi_da": 144304224.0
}
```

**Response — Hồ sơ từ chối:**
```json
{
  "approved": false,
  "trang_thai": "Từ chối",
  "diem_tin_dung": 412.5
}
```

---

## Kiểm tra API bằng script

```bash
python test_api.py
```
