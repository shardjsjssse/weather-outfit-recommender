# predict_clothing.py

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -----------------------
# 모델 불러오기
# -----------------------
model = load_model("lstm_model.keras")

sequence_length = 7

# -----------------------
# 스케일러 재생성 (학습과 동일 방식)
# -----------------------
data_size = 365
temps = np.random.rand(data_size) * 40 - 10

scaler = MinMaxScaler()
scaler.fit(temps.reshape(-1,1))

# -----------------------
# 날씨 입력
# -----------------------
input_temp = float(input("현재 기온을 입력하세요: "))

# 최근 7일을 동일 값으로 구성
input_sequence = np.array([input_temp] * sequence_length)
input_sequence = scaler.transform(input_sequence.reshape(-1,1))
input_sequence = input_sequence.reshape(1, sequence_length, 1)

# -----------------------
# 예측
# -----------------------
prediction = model.predict(input_sequence)
predicted_class = np.argmax(prediction)

print("예측 클래스:", predicted_class)

# -----------------------
# 이미지 로드
# -----------------------
cold_img = mpimg.imread("cold_clothing.png")
normal_img = mpimg.imread("normal_clothing.png")
hot_img = mpimg.imread("hot_clothing.png")
advice_img = mpimg.imread("advice.png")

# -----------------------
# 이미지 선택
# -----------------------
if predicted_class == 0:
    clothing_img = cold_img
    print("Cold 추천")
elif predicted_class == 1:
    clothing_img = normal_img
    print("Normal 추천")
else:
    clothing_img = hot_img
    print("Hot 추천")

# -----------------------
# 이미지 출력
# -----------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(clothing_img)
plt.title("Recommended Clothing")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(advice_img)
plt.title("Additional Advice")
plt.axis("off")

plt.tight_layout()
plt.show()
