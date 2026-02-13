# compare_models.py

import matplotlib.pyplot as plt

# --------------------------------
# 1. 각 모델 정확도 확인 후 입력
# --------------------------------
# dense_accuracy = 0.5277777910232544
dense_accuracy = 0.9861111044883728

# lstm_accuracy = 0.5416666865348816
lstm_accuracy = 0.9305555820465088

# --------------------------------
# 2. 막대 그래프 생성
# --------------------------------
models = ["Dense", "LSTM"]
accuracies = [dense_accuracy, lstm_accuracy]

plt.figure(figsize=(6,5))
bars = plt.bar(models, accuracies)

# 값 위에 숫자 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
             round(yval, 2), 
             ha='center', va='bottom')

plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.savefig("model_comparison.png")
plt.show()

print("비교 그래프 저장 완료: model_comparison.png")
