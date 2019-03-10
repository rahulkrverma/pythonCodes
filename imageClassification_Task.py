

import os, cv2

from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import optimizers
from keras.utils.np_utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def augmentData(dataPath, augPath):

	datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	classes = os.listdir(dataPath)
	for className in classes:
		classPath = os.path.join(dataPath, className)
		imgList = os.listdir(classPath)

		augClass = os.path.join(augPath, className)
		if not os.path.exists(augClass):
			os.mkdir(augClass)

		for img in imgList:
			imgPath = os.path.join(classPath, img)

			imgArr = load_img(imgPath)
			x = img_to_array(imgArr)
			x = x.reshape((1,) + x.shape)

			i = 0
			for batch in datagen.flow(x, batch_size = 1, save_to_dir = augClass, save_prefix = className, save_format = 'jpg'):
				i += 1
				if i > 20:
					break

def buildModel():

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Dense(num_classes, activation='softmax'))

	# sgd = optimizers.SGD(lr = 0.001, clipnorm = 0.5)
	# adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	rmse = optimizers.RMSprop(lr = 0.001)

	model.compile(loss = 'categorical_crossentropy', optimizer = rmse, metrics = ['accuracy'])
	return model

def trainModel(train_generator, validation_generator, epoch, samples_per_epoch, validation_steps, modelPath):
	
	model = buildModel()
	model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, epochs = epoch, validation_data = validation_generator, validation_steps=validation_steps)
	model.save_weights(modelPath)
	print('\n\n\tModel Saved at : \t', modelPath, '\n\n')


def testModel(modelPath, imgPath):

	model = buildModel()
	model.load_weights(modelPath)

	imgObj = load_img(imgPath)
	imgArr = img_to_array(imgObj)
	imgArr = cv2.resize(imgArr, (150, 150))
	imgArr = imgArr.reshape((1,) + imgArr.shape)
	pred = model.predict(imgArr)

	classList = ['Customer_Area', 'Gas_Station', 'Inventory', 'Office_Space', 'POS', 'Road_Signage', 'Shop_outdoor']
	label = list(pred[0]).index(max(pred[0]))
	print('\n\n\tLabel Classified : \t', classList[label], '\n\n')

if __name__ == '__main__':

	# augmentData('../data/train', '../data/trainAug')
	# augmentData('../data/test', '../data/testAug')

	epoch = 50
	batch_size = 32
	samples_per_epoch = 1000
	validation_steps = 300

	datagen = ImageDataGenerator(rescale = 1./255)
	train_generator = datagen.flow_from_directory('../data/trainAug', target_size=(150, 150), batch_size=batch_size, class_mode= 'categorical')
	validation_generator = datagen.flow_from_directory('../data/testAug', target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
	num_classes = len(train_generator.class_indices)

	trainModel(train_generator, validation_generator, epoch, samples_per_epoch, validation_steps, 'classification_scratch_softmax_50_epoch.h5')

	testModel(modelPath = 'classification_scratch_softmax_50_epoch.h5', imgPath = '')