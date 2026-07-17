from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import fetch_california_housing


#1. 데이터 
datasets = fetch_california_housing()
print(datasets.DESCR)  #DESCR Description 설명 
#print(datasets)
print(datasets.feature_names)




# x = datasets.data    #x값 입력   열   
# y = datasets.target  #y값 출력   행
exit()
print(x)
print(y)

print(x.shape , y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,      
    train_size=0.75,
    random_state=333,
    shuffle=True
                                                   )
print(x_train.shape, x_test.shape) #(15480, 8) (5160, 8)
print(y_train.shape, y_test.shape) #(15480,) (5160,)


#2.모델구성 
model=Sequential()
model.add(Dense(140, input_shape=(8, )))
model.add(Dense(150))
model.add(Dense(160))
model.add(Dense(140))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100 , batch_size=200)

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)

y_predict = model.predict(x_test)
print("[y_test] 의 원값 : ", y_test)
print("[x_test] 의 예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict)
print("r2_score:", r2)

rmse = root_mean_squared_error(y_test, y_predict)
print("rmse:", rmse)

mse = mean_squared_error(y_test, y_predict)
print("mse:", mse)

"""
    


    
    
    
"""
  



