import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import itertools

# 모델과 레이블 인코더 불러오기
try:
    with open('model.pkl', 'rb') as file:
        model = joblib.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = joblib.load(file)
except FileNotFoundError:
    print("오류: 모델 파일(model.pkl) 또는 레이블 인코더 파일(label_encoders.pkl)을 찾을 수 없습니다.")
    print("'train_and_save_model.py' 스크립트를 먼저 실행하여 파일을 생성해주세요.")
    exit()

# 'class' 레이블 인코더 가져오기 (예측 결과 디코딩용)
le_class = label_encoders['class']

# 각 특성의 옵션과 매핑 정의
# (Streamlit app.py에서 사용했던 것과 동일한 매핑을 사용합니다.)
gill_color_options = {
    "black": 'k', "brown": 'n', "buff": 'b', "chocolate": 'h', "gray": 'g',
    "green": 'r', "orange": 'o', "pink": 'p', "purple": 'u', "red": 'e',
    "white": 'w', "yellow": 'y'
}

gill_size_options = {"broad": 'b', "narrow": 'n'}

spore_print_color_options = {
    "black": 'k', "brown": 'n', "buff": 'b', "chocolate": 'h', "green": 'r',
    "orange": 'o', "purple": 'u', "white": 'w', "yellow": 'y'
}

odor_options = {
    "almond": 'a', "anise": 'l', "creosote": 'c', "fishy": 'y', "foul": 'f',
    "musty": 'm', "none": 'n', "pungent": 'p', "spicy": 's'
}

# 각 특성의 가능한 원본 문자열 값 리스트 (for itertools.product)
gill_colors = list(gill_color_options.values())
gill_sizes = list(gill_size_options.values())
spore_print_colors = list(spore_print_color_options.values())
odors = list(odor_options.values())

# 모든 특성 조합 생성
all_combinations = list(itertools.product(
    gill_colors,
    gill_sizes,
    spore_print_colors,
    odors
))

print(f"총 {len(all_combinations)}가지 조합을 확인합니다.")

edible_combinations = []

# 각 조합에 대해 예측 수행
for combo in all_combinations:
    gill_color_char, gill_size_char, spore_print_color_char, odor_char = combo

    # 특성 값을 숫자로 인코딩
    try:
        encoded_gill_color = label_encoders['gill-color'].transform([gill_color_char])[0]
        encoded_gill_size = label_encoders['gill-size'].transform([gill_size_char])[0]
        encoded_spore_print_color = label_encoders['spore-print-color'].transform([spore_print_color_char])[0]
        encoded_odor = label_encoders['odor'].transform([odor_char])[0]
    except ValueError as e:
        print(f"경고: {combo} 조합 인코딩 중 오류 발생: {e}. 해당 조합은 건너뜁니다.")
        continue # 오류 발생 시 해당 조합 건너뛰기

    # 예측을 위한 DataFrame 생성
    input_df = pd.DataFrame([[encoded_gill_color, encoded_gill_size, encoded_spore_print_color, encoded_odor]],
                            columns=['gill-color', 'gill-size', 'spore-print-color', 'odor'])

    # 예측 수행
    prediction_encoded = model.predict(input_df)[0]

    # 예측 결과가 'e' (식용)인지 확인
    if le_class.inverse_transform([prediction_encoded])[0] == 'e':
        edible_combinations.append({
            "gill-color": gill_color_char,
            "gill-size": gill_size_char,
            "spore-print-color": spore_print_color_char,
            "odor": odor_char
        })

print(f"\n총 {len(edible_combinations)}가지 식용(edible) 버섯 조합을 찾았습니다.")
print("-" * 50)

# 식용 조합 출력 (더 읽기 쉽게 디코딩하여 출력)
# 각 특성의 문자열 설명을 위한 역매핑 딕셔너리 생성
gill_color_decode = {v: k for k, v in gill_color_options.items()}
gill_size_decode = {v: k for k, v in gill_size_options.items()}
spore_print_color_decode = {v: k for k, v in spore_print_color_options.items()}
odor_decode = {v: k for k, v in odor_options.items()}


if len(edible_combinations) > 0:
    for i, combo in enumerate(edible_combinations):
        print(f"조합 {i+1}:")
        print(f"  아가미 색상 (gill-color): {gill_color_decode[combo['gill-color']]} ({combo['gill-color']})")
        print(f"  아가미 크기 (gill-size): {gill_size_decode[combo['gill-size']]} ({combo['gill-size']})")
        print(f"  포자 자국 색상 (spore-print-color): {spore_print_color_decode[combo['spore-print-color']]} ({combo['spore-print-color']})")
        print(f"  냄새 (odor): {odor_decode[combo['odor']]} ({combo['odor']})")
        print("-" * 30)
else:
    print("식용 버섯 조합을 찾을 수 없습니다.")