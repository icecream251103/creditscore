from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# ===== Load Models =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

model_trang_thai = joblib.load(os.path.join(MODEL_DIR, 'model_trang_thai.pkl'))
model_diem       = joblib.load(os.path.join(MODEL_DIR, 'model_diem.pkl'))
model_han_muc    = joblib.load(os.path.join(MODEL_DIR, 'model_han_muc.pkl'))
scaler           = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

with open(os.path.join(MODEL_DIR, 'feature_orders.json'), encoding='utf-8') as f:
    feature_orders = json.load(f)

feature_order_trang_thai = feature_orders['trang_thai']
feature_order_diem       = feature_orders['diem']
feature_order_han_muc    = feature_orders['han_muc']


# ===== Preprocessing =====
MARITAL_MAP   = {'Không': 0, 'Có': 1}
EDUCATION_MAP = {'Khác': 0, 'Cao đẳng': 1, 'Đại học': 2, 'Sau đại học': 3}
JOB_MAP       = {'KD tự do': 0, 'Nhân viên': 1, 'Quản lý': 2}
HOUSING_MAP   = {'Thuê nhà': 0, 'Nhà trả góp': 1, 'Sở hữu nhà ở': 2}
SALARY_MAP    = {'Tiền mặt': 0, 'Nhận lương qua NH khác': 1, 'Nhận lương qua Sacombank': 2}

NUM_COLS = [
    'Tuổi', 'Thu nhập', 'Điểm CIC',
    'Chi phí hàng tháng', 'Thời gian lao động',
    'Thời gian làm việc tại nơi LV hiện tại'
]


def preprocess(data, feature_order):
    df = pd.DataFrame([data])
    df['Tình trạng hôn nhân']       = df['Tình trạng hôn nhân'].map(MARITAL_MAP)
    df['Trình độ học vấn']          = df['Trình độ học vấn'].map(EDUCATION_MAP)
    df['Vị trí công việc']          = df['Vị trí công việc'].map(JOB_MAP)
    df['Tình trạng nhà ở hiện tại'] = df['Tình trạng nhà ở hiện tại'].map(HOUSING_MAP)
    df['Hình thức nhận lương']      = df['Hình thức nhận lương'].map(SALARY_MAP)
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    df = df.fillna(0)
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    return df[feature_order]


# ===== API Routes =====
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate required fields
        required = [
            'Tuổi', 'Tình trạng hôn nhân', 'Số người phụ thuộc',
            'Thu nhập', 'Điểm CIC', 'Chi phí hàng tháng',
            'Trình độ học vấn', 'Thời gian lao động', 'Vị trí công việc',
            'Thời gian làm việc tại nơi LV hiện tại',
            'Tình trạng nhà ở hiện tại', 'Hình thức nhận lương'
        ]
        for field in required:
            if field not in data:
                return jsonify({'error': f'Thiếu trường: {field}'}), 400

        X_status = preprocess(data, feature_order_trang_thai)
        X_diem   = preprocess(data, feature_order_diem)
        X_hm     = preprocess(data, feature_order_han_muc)

        status = int(model_trang_thai.predict(X_status)[0])
        diem   = float(model_diem.predict(X_diem)[0])

        result = {
            'trang_thai': 'Được xét duyệt' if status == 1 else 'Từ chối',
            'approved': status == 1,
            'diem_tin_dung': round(diem, 2),
        }

        if status == 1:
            han_muc = float(model_han_muc.predict(X_hm)[0])
            result['han_muc_toi_da'] = round(han_muc, 0)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': True})


if __name__ == '__main__':
    app.run(debug=False, port=5000)
