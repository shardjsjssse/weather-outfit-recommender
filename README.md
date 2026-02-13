# Weather Outfit Recommender 🌤️👕

최근 7일 온도 데이터를 기반으로 날씨를 분류하고,
해당 기온에 맞는 옷차림과 가이드 이미지를 추천하는 시스템입니다.

---

## 📌 프로젝트 개요

이 프로젝트는 시계열 데이터(최근 7일 온도)를 입력으로 받아  
오늘의 날씨를 분류하는 모델을 구현하고,

- Dense 모델
- LSTM 모델

두 가지를 비교 실험한 뒤,
최종적으로 사용자의 입력을 받아 옷차림 추천 시스템으로 확장하였습니다.

---

## 🧠 모델 구조

### 1️⃣ Dense 모델
- Dense(32, relu)
- Dense(16, relu)
- Dense(3, softmax)

### 2️⃣ LSTM 모델
- LSTM(50)
- Dropout(0.2)
- Dense(3, softmax)

입력 데이터: 최근 7일 온도 시퀀스  
출력 클래스: Cold / Normal / Hot (adivce)

이미지 출처: google gemini 3 

---

## 📊 실험 결과

### 랜덤 데이터 실험
- Dense ≈ LSTM
- 시간적 패턴이 없으면 LSTM의 장점이 크게 나타나지 않음

### 패턴 데이터 실험
- 단순한 주기 데이터에서는 Dense 모델이 더 높은 정확도를 보임
- 문제 복잡도에 따라 적절한 모델 선택이 중요함을 확인

---

## 프로젝트 구조
train_lstm.py
train_dense.py
compare_model.py
predict_clothing.py

## 👕 추천 시스템 실행 방법

```bash
pip install -r requirements.txt
python predict_clothing.py


