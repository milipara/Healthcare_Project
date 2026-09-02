import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import os

# ==============================================================================
# 1. 시계열 예측 모델 클래스 구조 정의 (기존 구조 유지)
# ==============================================================================
class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):
        x = x.reshape(x.size(0), -1) 
        return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ==============================================================================
# 2. 에셋 통합 로드 및 경로 자동 우회 함수
# ==============================================================================
@st.cache_resource
def load_all_assets():
    scaler_path = 'models/nutrient_scaler.pkl' if os.path.exists('models/nutrient_scaler.pkl') else 'nutrient_scaler.pkl'
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        st.error(f"🚨 중요 파일 누락: 'nutrient_scaler.pkl'을 찾을 수 없습니다.")
        st.stop()
    
    mlp = MLPModel(35)
    lstm = LSTMModel(5)
    gru = GRUModel(5)
    
    mlp_path = 'models/mlp_model.pth' if os.path.exists('models/mlp_model.pth') else 'mlp_model.pth'
    lstm_path = 'models/lstm_model.pth' if os.path.exists('models/lstm_model.pth') else 'lstm_model.pth'
    gru_path = 'models/gru_model.pth' if os.path.exists('models/gru_model.pth') else 'gru_model.pth'
    
    mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
    lstm.load_state_dict(torch.load(lstm_path, map_location='cpu'))
    gru.load_state_dict(torch.load(gru_path, map_location='cpu'))
    mlp.eval(); lstm.eval(); gru.eval()
    
    vision_path = 'food_vision/train/weights/best.pt' if os.path.exists('food_vision/train/weights/best.pt') else 'best.pt'
    vision_model = YOLO(vision_path)
    
    csv_path = 'dataset/전국통합식품영양성분정보표준데이터.csv'
    try:
        food_db = pd.read_csv(csv_path, encoding='cp949')
    except UnicodeDecodeError:
        food_db = pd.read_csv(csv_path, encoding='utf-8-sig')
    except FileNotFoundError:
        st.error(f"🚨 중요 파일 누락: '{csv_path}'를 찾을 수 없습니다.")
        st.stop()
        
    return scaler, mlp, lstm, gru, vision_model, food_db

try:
    scaler, mlp, lstm, gru, vision_model, food_db = load_all_assets()
except Exception as e:
    st.error(f"🚨 모델 로드 실패 구체적 원인: {e}")
    st.stop()

vision_to_db_map = {
    'apple': '사과', 'banana': '바나나', 'orange': '오렌지',
    'broccoli': '브로콜리', 'carrot': '당근', 'sandwich': '샌드위치',
    'pizza': '피자', 'donut': '도넛', 'cake': '케이크', 'bowl': '흰죽'
}

def safe_float(value):
    try:
        val = float(value)
        return 0.0 if np.isnan(val) else val
    except:
        return 0.0

# ==============================================================================
# 3. 메인 대시보드 UI 구성 및 데이터 수집실
# ==============================================================================
st.set_page_config(page_title="AI 영양사 대시보드", layout="wide")
st.title("🥗 멀티 타임포트 식단 사진 분석 및 당뇨·다이어트 예측 시스템")
st.markdown("매일 **아침, 점심, 저녁** 식단 사진을 업로드하면 그래프와 영양 분석 테이블이 시각적으로 실시간 갱신됩니다.")

st.sidebar.header("🗓️ 최근 7일 식단 일지 기록실")

user_input_logs = []       
photo_accumulated_logs = [] 

np.random.seed(10)
default_cals = np.random.randint(1700, 2200, size=7)
default_carbs = np.random.randint(220, 280, size=7)
default_prots = np.random.randint(65, 85, size=7)
default_fats = np.random.randint(40, 60, size=7)
default_sugars = np.random.randint(20, 40, size=7)

for i in range(7):
    day_num = 7 - i
    st.sidebar.markdown(f"---")
    st.sidebar.subheader(f"📅 D-{day_num}일 전 식단 분석 관리")
    
    photo_cal, photo_carb, photo_prot, photo_fat, photo_sugar = 0.0, 0.0, 0.0, 0.0, 0.0
    has_photo = False
    
    with st.sidebar.expander(f"📸 D-{day_num} 이미지/수동 세부 입력", expanded=(i==6)):
        meal_tabs = st.tabs(["🌅 아침", "☀️ 점심", "🌙 저녁"])
        
        for idx, meal_name in enumerate(["아침", "점심", "저녁"]):
            with meal_tabs[idx]:
                file_key = f"file_{i}_{idx}"
                img_file = st.file_uploader(f"{meal_name} 사진 업로드", type=["jpg", "jpeg", "png"], key=file_key)
                
                if img_file is not None:
                    has_photo = True
                    img = Image.open(img_file)
                    temp_img_path = f"temp_{file_key}.jpg"
                    img.save(temp_img_path)
                    
                    results = vision_model(temp_img_path, verbose=False)
                    detected_food = "흰죽" 
                    
                    for r in results:
                        if len(r.boxes) > 0:
                            first_box_cls_id = int(r.boxes[0].cls[0].item())
                            yolo_name = vision_model.names[first_box_cls_id]
                            if yolo_name in vision_to_db_map:
                                detected_food = vision_to_db_map[yolo_name]
                    
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                    
                    try:
                        food_info = food_db[food_db['식품명'].str.contains(detected_food, na=False)].iloc[0]
                        f_cal = safe_float(food_info['에너지(kcal)'])
                        f_carb = safe_float(food_info['탄수화물(g)'])
                        f_prot = safe_float(food_info['단백질(g)'])
                        f_fat = safe_float(food_info['지방(g)'])
                        f_sugar = safe_float(food_info['당류(g)'])
                        
                        st.success(f"🎯 실시간 인식: **{detected_food}**")
                        st.caption(f"🔥 {f_cal}kcal | 탄 {f_carb}g | 단 {f_prot}g")
                        
                        photo_cal += f_cal
                        photo_carb += f_carb
                        photo_prot += f_prot
                        photo_fat += f_fat
                        photo_sugar += f_sugar
                    except Exception:
                        pass
        
        st.markdown("**💡 하루 최종 영양소 수치 보정**")
        init_cal = int(safe_float(photo_cal)) if has_photo else int(default_cals[i])
        init_carb = int(safe_float(photo_carb)) if has_photo else int(default_carbs[i])
        init_prot = int(safe_float(photo_prot)) if has_photo else int(default_prots[i])
        init_fat = int(safe_float(photo_fat)) if has_photo else int(default_fats[i])
        init_sugar = int(safe_float(photo_sugar)) if has_photo else int(default_sugars[i])

        col1, col2, col3 = st.columns(3)
        with col1:
            cal = col1.number_input("칼로리總", 0, 5000, init_cal, 50, key=f"c_{i}")
        with col2:
            carb = col2.number_input("탄수總", 0, 700, init_carb, 10, key=f"cb_{i}")
        with col3:
            prot = col3.number_input("단백總", 0, 400, init_prot, 5, key=f"p_{i}")
            
        col4, col5 = st.columns(2)
        with col4:
            fat = col4.number_input("지방總", 0, 300, init_fat, 5, key=f"f_{i}")
        with col5:
            sugar = col5.number_input("당류總", 0, 300, init_sugar, 5, key=f"s_{i}")
            
    user_input_logs.append([cal, carb, prot, fat, sugar])
    photo_accumulated_logs.append([photo_cal, photo_carb, photo_prot, photo_fat, photo_sugar])

# ==============================================================================
# 4. 데이터 예측 및 매핑 테이블 세팅
# ==============================================================================
input_df = pd.DataFrame(user_input_logs, columns=['calories', 'carbohydrates', 'protein', 'fat', 'sugar'])
photo_df = pd.DataFrame(photo_accumulated_logs, columns=['p_cal', 'p_carb', 'p_prot', 'p_fat', 'p_sugar'])

input_df.index = [f"D-{7-x}" for x in range(7)]
photo_df.index = [f"D-{7-x}" for x in range(7)]

scaled_input = scaler.transform(input_df[['calories', 'carbohydrates', 'protein', 'fat', 'sugar']])
input_tensor = torch.FloatTensor(scaled_input).view(1, 7, 5)

with torch.no_grad():
    pred_mlp = mlp(input_tensor).item()
    pred_lstm = lstm(input_tensor).item()
    pred_gru = gru(input_tensor).item()

def convert_to_real_calories(scaled_val, scaler_obj):
    temp = np.zeros((1, 5))
    temp[0, 0] = scaled_val
    return scaler_obj.inverse_transform(temp)[0, 0]

real_mlp = convert_to_real_calories(pred_mlp, scaler)
real_lstm = convert_to_real_calories(pred_lstm, scaler)
real_gru = convert_to_real_calories(pred_gru, scaler)

# ==============================================================================
# 5. 메인 대시보드 화면 동적 차트 및 셀 단위 그라데이션 테이블 출력
# ==============================================================================
left_panel, right_panel = st.columns([1, 1])

with left_panel:
    st.subheader("📊 최근 7일간 종합 누적 데이터 현황")
    
    # 💡 [요청사항 반영] 가로축이 D-7에서 D-1 순서로 점차 줄어들도록 정순(ascending=True) 정렬
    chart_df = pd.DataFrame({
        '보정 칼로리(kcal)': input_df['calories'],
        '누적 칼로리(kcal)': photo_df['p_cal']
    }).sort_index(ascending=True) 
    
    st.line_chart(chart_df)
    
    st.markdown("**📋 수치 로그 디테일 테이블**")
    
    table_df = pd.DataFrame({
        '누적 칼로리(kcal)': photo_df['p_cal'].round(0).astype(int),
        '탄수화물(g)': photo_df['p_carb'].round(0).astype(int),
        '단백질(g)': photo_df['p_prot'].round(0).astype(int),
        '지방(g)': photo_df['p_fat'].round(0).astype(int),
        '당류(g)': photo_df['p_sugar'].round(0).astype(int)
    }).sort_index(ascending=True) # 💡 테이블의 가로 행 순서도 그래프와 일치하게 D-7 -> D-1 순으로 배치

    def style_cells_by_gap(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        for idx in df.index:
            ref_cal = input_df.loc[idx, 'calories']
            ref_carb = input_df.loc[idx, 'carbohydrates']
            ref_prot = input_df.loc[idx, 'protein']
            ref_fat = input_df.loc[idx, 'fat']
            ref_sugar = input_df.loc[idx, 'sugar']
            
            mappings = {
                '누적 칼로리(kcal)': (df.loc[idx, '누적 칼로리(kcal)'], ref_cal, 400.0),
                '탄수화물(g)': (df.loc[idx, '탄수화물(g)'], ref_carb, 50.0),
                '단백질(g)': (df.loc[idx, '단백질(g)'], ref_prot, 20.0),
                '지방(g)': (df.loc[idx, '지방(g)'], ref_fat, 20.0),
                '당류(g)': (df.loc[idx, '당류(g)'], ref_sugar, 15.0)
            }
            
            for col, (actual, ref, scale) in mappings.items():
                diff = actual - ref
                alpha = min(abs(diff) / scale, 0.6)
                if 0.0 < alpha < 0.08: alpha = 0.1 
                
                if diff > 0:
                    styles.loc[idx, col] = f"background-color: rgba(255, 0, 0, {alpha}); text-align: center;"
                elif diff < 0:
                    styles.loc[idx, col] = f"background-color: rgba(0, 100, 255, {alpha}); text-align: center;"
                else:
                    styles.loc[idx, col] = "background-color: transparent; text-align: center;"
        return styles

    styled_table = table_df.style.apply(style_cells_by_gap, axis=None)
    st.dataframe(styled_table, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 💡 테이블 셀 색상 시각화 가이드")
    st.markdown(
        """
        테이블의 개별 셀 색상은 사이드바의 **'하루 최종 영양소 수치 보정' 설정값**과 **'사진으로 분석된 실제 누적 수치'의 간격(차이)**을 실시간으로 계산하여 표현됩니다.
        
        * 🔴 **빨간색 셀 (과다 섭취):** 사진으로 누적된 실제 데이터가 유저가 설정한 목표 보정치보다 **높을 때** 표시됩니다.
        * 🔵 **파란색 셀 (부족 섭취):** 사진으로 누적된 실제 데이터가 유저가 설정한 목표 보정치보다 **낮을 때** 표시됩니다.
        * 🎚️ **색상의 투명도(농도):** 유저 보정 데이터와 실제 데이터의 **간격이 벌어질수록(클수록) 색상이 점점 더 진하게** 변하며, 두 수치가 정확히 일치하면 투명해집니다.
        """
    )