

# importing modules 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


if __name__ == '__main__':

	# load mmist dataset
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# normalize dataset between 0 to 1
	x_train = x_train/255.0
	x_test = x_test/255.0

	# initilizing Sequential model
	model = Sequential()

	# adding LSTM layer
	model.add(LSTM(128, input_shape = (x_train.shape[1:]),
				activation = 'relu',
				return_sequences = True))

	#adding Dropout
	model.add(Dropout(0.2))

	# adding LSTM layer 2
	model.add(LSTM(128, activation = 'relu'))
	model.add(Dropout(0.2))

	# adding Dense layer
	model.add(Dense(32, activation = 'relu'))
	model.add(Dropout(0.2))

	# adding Dense layer 2
	model.add(Dense(10, activation = 'softmax'))

	# using optimizier for optiming loss in the network
	opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

	# compiling model
	model.compile(loss = 'sparse_categorical_crossentropy',
					optimizer = opt,
					metrics = ['accuracy'],
					)

	# start training and fitting model
	model.fit(x_train, y_train,
		epochs = 3,
		validation_data = (x_test, y_test))



