# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:42:32 2020

@author: Baker
"""

from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import optimizers

import numpy as np
import pandas
import matplotlib.pyplot as plt


dataset = pandas.read_csv('abalone.data', names=['sex','1','2','3','4','5','6','7','rings'])

data = pandas.get_dummies(dataset, sparse=True)

#data['rings']=data['rings']/2

training = data.sample(frac=0.9)
testing  = data.drop(training.index)


y_training = training['rings']

x_training = training.drop(['rings'],axis=1)

y_testing = testing['rings']
x_testing = testing.drop(['rings'], axis=1)


model = Sequential()

model.add(Dense(units=32,activation='relu',input_dim = 10))
model.add(Dense(units=32,activation='sigmoid'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=1,activation='linear'))

optimum = optimizers.Nadam()

model.compile(loss=losses.mean_squared_error, optimizer=optimum, metrics=['accuracy','mean_absolute_error'])

train = []
traina = []
test = []
testa = []


'''
h = model.fit(x_training, y_training, epochs=1, verbose=0, batch_size=32, shuffle=True)
train = h.history['mean_absolute_error']
traina = h.history['accuracy']
ev = model.evaluate(x_testing, y_testing, verbose=0)
test.append(ev[2])
testa.append(ev[1])
'''

hist = model.fit(x_training, y_training, epochs=50, verbose=2, batch_size=128, shuffle=True)
loss = hist.history['loss']
train = hist.history['mean_absolute_error']
traina = hist.history['accuracy']

train=np.array(train)
test=np.array(test)
loss = np.array(loss)

plt.plot(np.arange(0,len(train),1),1/train)
#plt.plot(np.arange(0,len(test),1),1/test)
#plt.plot(np.arange(0,len(test),1),1/train-1/test)
plt.plot(np.arange(0,len(traina),1),traina)
plt.plot(np.arange(0,len(traina),1),1/loss)



plt.show()

'''
err = test-train-np.mean(test-train)
squareerr = np.power(err,2)
e=np.sqrt(sum(squareerr)/(len(test)-1))
print(e)
'''

print('max training accuracy',max(traina))
print('min training absolute error',min(train))
print('min training squared error',min(loss))

print(model.evaluate(x_testing, y_testing))

#print('max testing accuracy',min(test))

