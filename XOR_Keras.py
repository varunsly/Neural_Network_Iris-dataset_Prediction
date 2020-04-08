import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

#Training data is defined
training_data = np.array([[0,0],[0,1],[1,0],[1,1]],"float32")

training_target = np.array([0],[1],[1],[0], "float32")

#model is being  initiated
model= Sequential()

#Layers are being added
model.add(Dense(16,input_dim=2, activation='relu'))
model.add(Dense(16,input_dim=16, activation='relu'))
model.add(Dense(16,input_dim=16, activation='relu'))
model.add(Dense(16,input_dim=16, activation='relu'))
model.add(Dense(16,input_dim=16, activation='relu'))
model.add(Dense(16,input_dim=16, activation='relu'))
model.add(Dense(16,input_dim=16, activation='relu'))
model.add(Dense(16,input_dim=16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))


#Different Parameters are being added
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

#Model is being trained
model.fit(training_data,training_target,epochs=2000, verbose=1)


#Final model prediction
print(model.predict(training_data).round())

