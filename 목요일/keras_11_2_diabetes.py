from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes


#1. 데이터 
datasets = load_diabetes()
#print(datasets)
#print(datasets.DESCR)
print(datasets.feature_names)


x = datasets.data
y = datasets.target

print(x)
print(y)

print(x.shape , y.shape)  # (442, 10) (442,)


x_train, x_test, y_train, y_test = train_test_split(
                                                    x, y,      
                                                    train_size=0.7,
                                                    random_state=332,
                                                    shuffle=True
                                                   )
print(x_train.shape, x_test.shape) #(442, 10) (442,)
print(y_train.shape, y_test.shape) #(331,) (111,)


#2.모델구성 
model=Sequential()
model.add(Dense(70, input_shape=(10,)))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)

y_predict = model.predict(x_test)
print("[y_test] 의 원값 : ", y_test)
print("[x_test] 의 예측값 : ", y_predict)

r2 = r2_score(y_test, y_predict)
print("r2_score:", r2)

rmse = mean_squared_error(y_test, y_predict)
print("rmse:", rmse)

mse = mean_squared_error(y_test, y_predict)
print("mse:", mse)

"""
train_size=0.95,
random_state=310,
shuffle=True   
    
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(120))
model.add(Dense(150))
model.add(Dense(160))
model.add(Dense(140))
model.add(Dense(110))
model.add(Dense(100))
model.add(Dense(1))
model.fit(x_train, y_train, epochs=100, batch_size=4)
r2_score: 0.7145278212344326    
    
"""
  



