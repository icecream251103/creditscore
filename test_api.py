import requests

data = {
    "Tuổi": 30,
    "Tình trạng hôn nhân": "Đã kết hôn",
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

r = requests.post("http://127.0.0.1:5000/api/predict", json=data)
print(r.status_code)
print(r.json())
