import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam
from sklearn.model_selection import cross_validation

dataset = datasets.load_iris()

features = dataset.data
y= dataset.target.reshape(-1,1)

encoder=OneHotEncoder()
target=encoder.fit_transform(y)

train_feature,test_feature,train_target,test_target=train_test_split(features,target,test_size=0.3)

model= Sequential()

model.add(Dense(10,input_dim=4, activation='relu'))
model.add(Dense(10,input_dim=10, activation='relu'))
model.add(Dense(10,input_dim=10, activation='relu'))
model.add(Dense(10,input_dim=10, activation='relu'))
model.add(Dense(10,input_dim=10, activation='relu'))
model.add(Dense(10,input_dim=10, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

optimizers=Adam(lr=0.05)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers,
              metrics=['accuracy'])

model.fit(train_feature,train_target,batch_size=20,epochs=1000, verbose=2)

result=model.evaluate(test_feature,test_target)

print("Resultant Error %.2f",result[0])
print("Resultant accuracy %.2f",result[1])
