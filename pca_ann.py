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
X2 = dataset2.iloc[:,1:].values



X=normalize(X,axis=0)
X2=normalize(X2,axis=0)



y =np_utils.to_categorical(y,num_classes=3)
#y_test=np_utils.to_categorical(y_test,num_classes=3)
print("Shape of y_train",y.shape)
#print("Shape of y_test",y_test.shape)
mean_vector = []
all_samples = []
for i in range(41):
	mean_x = np.mean(X[i,:])
	all_samples.append(X[i,:])
	mean_vector.append(mean_x)


#print(mean_vector)
#print(len(mean_vector))
scatter_matrix = np.zeros((41,41))
for i in range(X.shape[1]):
    scatter_matrix += (X[i,:].reshape(41,1) - mean_vector).dot((X[i,:].reshape(41,1) - mean_vector).T)
#print('Scatter Matrix:\n', scatter_matrix)



cov_mat = np.cov(all_samples)
#print('Covariance Matrix:\n', cov_mat)




eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,41).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,41).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
    print(40 * '-')




eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
#for i in eig_pairs:
  #  print(i[0])



matrix_w = []
for i in range(30):
	matrix_w.append(np.hstack((eig_pairs[i][1].reshape(41,1))))

print('Matrix W:\n', matrix_w)

print(len(matrix_w))


matrix_w = np.array(matrix_w)

assert matrix_w.shape == (30,41), "The matrix is not 2x40 dimensional."


new_X = np.dot(X,matrix_w.reshape(41,30))


new_X = np.array(new_X)

print(len(new_X))
print(len(y))
#y=np_utils.to_categorical(y,num_classes=3)
#y_test=np_utils.to_categorical(y_test,num_classes=3)
print("Shape of y_train",y.shape)
#print("Shape of y_test",y_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X,y, test_size = 0.25, random_state = 0)

model=Sequential()
model.add(Dense(16,input_dim=10,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])



#print(X_test2.shape[1])
model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)


mean_vector = []
all_samples = []
for i in range(41):
	mean_x = np.mean(X2[i,:])
	all_samples.append(X2[i,:])
	mean_vector.append(mean_x)


#print(mean_vector)
#print(len(mean_vector))
scatter_matrix = np.zeros((41,41))
for i in range(X_test.shape[1]):
    scatter_matrix += (X2[i,:].reshape(41,1) - mean_vector).dot((X2[i,:].reshape(41,1) - mean_vector).T)
#print('Scatter Matrix:\n', scatter_matrix)



cov_mat = np.cov(all_samples)
#print('Covariance Matrix:\n', cov_mat)




eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,41).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,41).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
    print(40 * '-')




eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
#for i in eig_pairs:
  #  print(i[0])



matrix_w = []
for i in range(30):
	matrix_w.append(np.hstack((eig_pairs[i][1].reshape(41,1))))

print('Matrix W:\n', matrix_w)

print(len(matrix_w))


matrix_w = np.array(matrix_w)

assert matrix_w.shape == (30,41), "The matrix is not 2x40 dimensional."


new_X = np.dot(X2,matrix_w.reshape(41,30))


new_X = np.array(new_X)




file1 = open("predictions_pca.csv","w")
file1.close()

t = model.predict(new_X)


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
file1 = open("predictions_pca.csv","w")


np.savetxt("predictions_pca.csv",g,fmt='%5s',delimiter=",")
