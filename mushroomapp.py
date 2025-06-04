import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 저장된 모델과 레이블 인코더 불러오기
try:
    with open('model.pkl', 'rb') as file:
        model = joblib.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = joblib.load(file)
except FileNotFoundError:
    st.error("모델 파일(model.pkl) 또는 레이블 인코더 파일(label_encoders.pkl)을 찾을 수 없습니다. 'train_and_save_model.py' 스크립트를 먼저 실행해주세요.")
    st.stop() # 파일이 없으면 앱 실행을 중단합니다.

# 'class' 레이블 인코더 가져오기 (예측 결과 디코딩용)
le_class = label_encoders['class']

# Streamlit 앱 제목
st.title("버섯 종류 예측기 (식용/독성)")
st.markdown("특성을 선택하여 버섯이 식용(e)인지 독성(p)인지 예측해보세요.")

# 사용자 입력 위젯
st.sidebar.header("버섯 특성 선택")

# 각 특성의 옵션 정의 (원본 데이터의 매핑 기준)
# gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
gill_color_options = {
    "black": 'k', "brown": 'n', "buff": 'b', "chocolate": 'h', "gray": 'g',
    "green": 'r', "orange": 'o', "pink": 'p', "purple": 'u', "red": 'e',
    "white": 'w', "yellow": 'y'
}
# 라디오 버튼으로 변경
selected_gill_color_display = st.sidebar.radio("아가미 색상 (gill-color)", list(gill_color_options.keys()))
selected_gill_color = gill_color_options[selected_gill_color_display]

# gill-size: broad=b,narrow=n
gill_size_options = {"broad": 'b', "narrow": 'n'}
# 라디오 버튼으로 변경
selected_gill_size_display = st.sidebar.radio("아가미 크기 (gill-size)", list(gill_size_options.keys()))
selected_gill_size = gill_size_options[selected_gill_size_display]

# spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
spore_print_color_options = {
    "black": 'k', "brown": 'n', "buff": 'b', "chocolate": 'h', "green": 'r',
    "orange": 'o', "purple": 'u', "white": 'w', "yellow": 'y'
}
# 라디오 버튼으로 변경
selected_spore_print_color_display = st.sidebar.radio("포자 자국 색상 (spore-print-color)", list(spore_print_color_options.keys()))
selected_spore_print_color = spore_print_color_options[selected_spore_print_color_display]

# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
odor_options = {
    "almond": 'a', "anise": 'l', "creosote": 'c', "fishy": 'y', "foul": 'f',
    "musty": 'm', "none": 'n', "pungent": 'p', "spicy": 's'
}
# 라디오 버튼으로 변경
selected_odor_display = st.sidebar.radio("냄새 (odor)", list(odor_options.keys()))
selected_odor = odor_options[selected_odor_display]

# 예측 버튼
if st.sidebar.button("예측하기"):
    # 입력 값을 숫자로 인코딩
    try:
        encoded_gill_color = label_encoders['gill-color'].transform([selected_gill_color])[0]
        encoded_gill_size = label_encoders['gill-size'].transform([selected_gill_size])[0]
        encoded_spore_print_color = label_encoders['spore-print-color'].transform([selected_spore_print_color])[0]
        encoded_odor = label_encoders['odor'].transform([selected_odor])[0]
    except ValueError as e:
        st.error(f"입력 값 인코딩 중 오류 발생: {e}. 'train_and_save_model.py' 스크립트가 올바른 인코더를 생성했는지 확인하세요.")
        st.stop()


    # 예측을 위한 DataFrame 생성
    input_df = pd.DataFrame([[encoded_gill_color, encoded_gill_size, encoded_spore_print_color, encoded_odor]],
                            columns=['gill-color', 'gill-size', 'spore-print-color', 'odor'])

    # 예측 수행
    prediction_encoded = model.predict(input_df)[0]

    # 예측 결과 디코딩
    predicted_class = le_class.inverse_transform([prediction_encoded])[0]

    # 결과 표시
    st.subheader("예측 결과:")
    if predicted_class == 'e':
        st.success(f"이 버섯은 **식용(e)** 입니다. 🎉")
    else:
        st.warning(f"이 버섯은 **독성(p)** 입니다. ⚠️")

    st.markdown("---")
    st.markdown("선택된 특성:")
    st.write(f"아가미 색상: **{selected_gill_color_display}** ({selected_gill_color})")
    st.write(f"아가미 크기: **{selected_gill_size_display}** ({selected_gill_size})")
    st.write(f"포자 자국 색상: **{selected_spore_print_color_display}** ({selected_spore_print_color})")
    st.write(f"냄새: **{selected_odor_display}** ({selected_odor})")