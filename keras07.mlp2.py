import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6]),([7,8,9,10,11,12])#두개 이상은 리스트로 묶어줘야된다.#(2, 6)

# x = np.array([[1,7][2,8][3,9][4,10][5,11][6,12]])
# x = x.T
x  = x.transpose() 
print(x.shape)
exit()

y = np.array([1,2,3,4,5,6]) (6,)

#2. 모델구성
model = Sequential()
model.add(Dense(200, input_shape=(2,)))

#model.add(Dense(10, input_dim=2))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

    

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=4) # 배치사이즈를 4으로 설정하여 한 번에 네 개씩 학습

#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss = ", loss)

result = model.predict(np.array([[7,13]]))
print("7의 예측값 : ", result)

