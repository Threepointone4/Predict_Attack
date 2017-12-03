#Import required libraries 
import keras #library for neural network
import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library
#Neural network module
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
import pickle
# Importing the Keras libraries and packages
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV


# Importing the dataset
dataset = pd.read_csv('train_data.csv')
dataset2 = pd.read_csv('test_data.csv')  
ID = dataset2.iloc[:,0].values

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X_test2 = dataset2.iloc[:,1:].values



X=normalize(X,axis=0)
X_test2 =normalize(X_test2,axis=0)



y=np_utils.to_categorical(y,num_classes=3)
#y_test=np_utils.to_categorical(y_test,num_classes=3)
print("Shape of y_train",y.shape)
#print("Shape of y_test",y_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

model=Sequential()
model.add(Dense(14,input_dim=41,activation='relu'))
model.add(Dense(14,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(6,activation='relu'))

#model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
#return model

'''
model = KerasClassifier(build_fn = build)
parameters = {'batch_size': [25, 32],
              'epochs': [10,20],
             }
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_accuracy)
model.summary()
'''


#print(X_test2.shape[1])
model.fit(X,y,validation_data=(X_test,y_test),batch_size=20,epochs=20,verbose=1)




prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)

accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the dataset",accuracy )


file1 = open("predictions_ann.csv","w")
file1.close()

t = model.predict(X_test2)


print(len(ID))
#print(len(t))


print(ID[1])
g = []
for i in range(len(ID)):
	#print(i)
	temp = []
	temp.append(ID[i])
	a = t[i]
	#print(a)
	b = max(enumerate(a),key=lambda x: x[1])[0]
	temp.append((b))
	g.append(temp)

print(g)
file1 = open("predictions_ann.csv","w")


np.savetxt("predictions_ann.csv",g,fmt='%5s',delimiter=",")
