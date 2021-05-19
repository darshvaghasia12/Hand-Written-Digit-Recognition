

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k

# The Data Split
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

print(X_train.shape,Y_train.shape)


#Preprocess the Data
#The image data cannot be fed directly into the model so we need to perform some operations and process the data to make it ready for our neural network.
#The dimension of the training data is (60000,28,28). The CNN model will require one more dimension so we reshape the matrix to shape (60000,28,28,1).
X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
input_shape=(28,28,1)

#Converting class vectors to binary class matrices
Y_train=keras.utils.to_categorical(Y_train,num_classes=None)
Y_test=keras.utils.to_categorical(Y_test,num_classes=None)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
#Normalizing the vectors since it is Image and the Maximum Value of the pixel is 255 So we will Simply divide it with 255
X_train/=255
X_test/=255

print('X_train.shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

#Create the Model
batch_size=128
epochs=10
num_classes=10

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

hist=model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,Y_test))
print('The Model has successfully trained')
model.save('mnist.h5')
print("Saving the model as mnist.h5")