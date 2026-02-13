# train_lstm.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------
# 1. 데이터 생성
# ---------------------------
np.random.seed(0)
data_size = 365
# temps = np.random.rand(data_size) * 40 - 10  # -10 ~ 30도
temps = 15 + 10 * np.sin(np.linspace(0, 10, data_size))
# 평균 15도, 진폭 10도, 사인파(계절성 있음), 시간 흐름, LSTM이 학습할 패턴

sequence_length = 7

X = []
y = []

# 7일 시계열 생성
for i in range(len(temps) - sequence_length):
    X.append(temps[i:i+sequence_length])
    label_temp = temps[i+sequence_length]

    if label_temp < 10:
        y.append(0)
    elif label_temp < 20:
        y.append(1)
    else:
        y.append(2)

X = np.array(X)
y = to_categorical(y, num_classes=3)

# ---------------------------
# 2. 정규화
# ---------------------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1,1)).reshape(X.shape)

# ---------------------------
# 3. 학습/테스트 분리
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LSTM 입력 형태로 변환
X_train = X_train.reshape((X_train.shape[0], sequence_length, 1))
X_test = X_test.reshape((X_test.shape[0], sequence_length, 1))

# ---------------------------
# 4. LSTM 모델 구성
# ---------------------------
model = Sequential([
    LSTM(50, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# 5. 모델 학습
# ---------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# ---------------------------
# 6. 모델 평가
# ---------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print("LSTM Accuracy:", accuracy)

# ---------------------------
# 7. 모델 저장
# ---------------------------
model.save("lstm_model.keras")

# ---------------------------
# 8. 학습 결과 그래프 저장
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("LSTM Training Result")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("lstm_training_result.png")
plt.close()

print("모델 저장 완료: lstm_model.keras")
print("그래프 저장 완료: lstm_training_result.png")
