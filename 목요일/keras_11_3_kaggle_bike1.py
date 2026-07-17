from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes
import pandas as pd






#1. 데이터
path = './_data/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)                  
test_csv =pd.read_csv(path + "test.csv", index_col=0)         
#print(train_csv)
#print(train_csv.shape)
#print(test_csv)
#print(test_csv.shape)
#print(submit_csv)
#print(submit_csv.shape)
print(train_csv.columns)
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='str')
x= train_csv.drop(['casual', 'registered', 'count'], axis=1 ) 

#print(x) #[10886 rows x 8 columns]
print('=============')
y = train_csv['count']
print(y)            
print(y.shape) #(10886,) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y,      
    train_size=0.70,
    random_state=310,
    shuffle=True
)
print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 

#2.모델 구성 

model=Sequential()
model.add(Dense(100, input_shape=(8,)))
model.add(Dense(120))
model.add(Dense(150))
model.add(Dense(160))
model.add(Dense(140))
model.add(Dense(110))
model.add(Dense(100))
model.add(Dense(1))

#3.컴파일,훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=40)

#4.평가,예측 
loss = model.evaluate(x_test , y_test)
print("loss = ", loss)   

result = model.predict([x])
print("[x] 의 예측값 : ", result)
