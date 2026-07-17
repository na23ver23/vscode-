import tensorflow as tf                                        #tensorflow를 tf로 줄여서 사용 
print(tf.__version__)

from keras.models import Sequential                            #keras에서 모델을 구성할 때 Sequential 모델을 사용한다.
from keras.layers import Dense                                 #keras에서 Dense 레이어를 사용한다.
import numpy as np                                             # numpy를 np로 줄여서 사용한다. 
from sklearn.model_selection import train_test_split           #사이킷런에서 train_test_split 함수를 사용하여 데이터를 학습용과 테스트용으로 나눈다.
 

#1. 데이터


x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])                                            
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])                                                      

# [실습] train과 test를 섞어서 랜덤하게 7:3을 뽑는다.
# [힌트] 사이킷런
x_train, x_test,y_train,y_test  = train_test_split(x , y, 
                                  train_size=0.7,    
                                  shuffle=True,                                    # shuffle=True로 설정하면 데이터를 섞어서 나눈다.                           
                                  random_state= 600,                               # 랜덤하게 섞이는 결과를 400로 고정 
                                  # test_size =0.3,             
                                  # shuffle=False,                                 # shuffle=False로 설정하면 데이터를 섞지 않고 순서대로 나눈다.              
                                  
                                  )

 
print(x_train , x_test)                                                       #(7,) (3,)
print(y_train , y_test)                                                       #(7,) (3,)






#2. 모델구성                                         
model = Sequential()                                           
model.add(Dense(100, input_shape=(1,))) #(None, 1 )            
model.add(Dense(50)) 
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(1))

    

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')                                           # 모델을 컴파일한다. mse=평균제곱오차 optimizer=틀린 것을 어떻게 고칠지 결정 adam=최적의 로스값  
model.fit(x_train , y_train , epochs=30, )                                                     


#4. 평가, 예측
loss = model.evaluate(x_test , y_test)
print("loss = ", loss)   


model.predict(x_test)
y_predict = model.predict([x_test])                                                   # y의 예측값은 model.예측값(x_test)로 구한다.  100번째 w를 이용해서 예측값=wx+b 

print("[y_test] 의 원값 : ", y_test)
print("[x_test] 의 예측값 : ", y_predict) 
from sklearn.metrics import r2_score, root_mean_squared_error , mean_squared_error    #sklern에서 r2_score, root_mean_squared_error , mean_squared_error 함수를 사용한다.

r2_score(y_test , y_predict)                                                          #r2_score(y_test, y_predict)으로 r2_score를 구한다.  1에 가까울수록 좋다. 0.5이하면 안좋다.
print("r2_score:",r2)                                                                 

rmse = mean_squared_error(y_test , y_predict)
print("rmse:",rmse)                                                                   #rmse(원값, 예측값)으로 rmse를 구한다.  1에 가까울수록 좋다. 0.5이하면 안좋다.

mse= mean_squared_error(y_test , y_predict)
print("mse:",mse)







################################# 그래프 그리기 ###########################################
import matplotlib.pyplot as plt 
plt.scatter(x , y)                                                                    # (x ,y) 점 찍기 
plt.plot(x_test, y_predict, color='red')                                              # 예측값을 빨간색으로 표시 plot= 선긋기 
plt.show()


