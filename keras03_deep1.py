import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()

model.add(Dense(10, input_dim=1))
model.add(Dense(100, input_dim=10))
model.add(Dense(1000, input_dim=100))
model.add(Dense(10000, input_dim=1000))
model.add(Dense(1000, input_dim=10000))
model.add(Dense(100, input_dim=1000))
model.add(Dense(10, input_dim=100))
model.add(Dense(1, input_dim=10))

    

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
result = model.predict(np.array([7]))
print("4의 예측값 : ", result)

