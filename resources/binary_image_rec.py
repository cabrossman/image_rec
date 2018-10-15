from google_images_download import google_images_download   #importing the library
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import json

class BinaryRec:
	def __init__(self, height, width):
		"""
			height and width are required. rest of member function are built in object through methods
		"""
		self.height = height
		self.width = width
		self.X = None
		self.y = None
		self.y_train = None
		self.y_test = None
		self.X_train = None
		self.X_test = None
		self.model = None
		self.model_json = None

	def showImage(self,image_index = None, K = 10, print_adjusted_size = False):
		"""
			this function returns images and locations of images
			
			Args:
				image_index (int) : index of y to show
				K (int) : how many random images to show -- NOTE : will only show if image_index is None
				print_adjusted_size (bool) : whether to print original or adjusted size defined by height & width
		
			Return : None
		"""
		from random import randint
		if image_index  is None:
			for i in range(1,K):
				image_index = randint(0, len(self.y))
				if self.y.iloc[image_index,1] == 1:
					print('positive')
				else:
					print('negative')
				print(self.y.iloc[image_index,0])
				img = Image.open(self.y.iloc[image_index,0])
				if print_adjusted_size:
					img = img.resize((self.height,self.width))
				imgplot = plt.imshow(img)
				plt.show()
		else:
			if self.y.iloc[image_index,1] == 1:
				print('positive')
			else:
				print('negative')
			print(self.y.iloc[image_index,0])
			img = Image.open(self.y.iloc[image_index,0])
			if print_adjusted_size:
				img = img.resize((self.height,self.width))
			imgplot = plt.imshow(img)
			plt.show()

	def download_images(self,path_to_download_types, limit_per_category, pos_divisor, neg_divisor):
		"""
			downloads images in directory. Need to supply CSV with keyword to search for and if its poisoness or not (o or 1).
			Function will search in supplied path_to_download_types
			
			Args : 
				path_to_download_types (str) : points to path of CSV file
				limit_per_category (int) : number of images to download per category
				pos_divisor (int) : if we want positive class to have less than total limit
				neg_divisor (int) : if we want negative class to have less than total limit
		
			Return : None
		"""
		df = pd.read_csv(path_to_download_types)
		for row in df.itertuples():
		
			if row[2] == 1:
				limit = round(limit_per_category/pos_divisor)
				output_dir = 'images_positive'
			else:
				output_dir = 'images_negative'
				limit = round(limit_per_category/neg_divisor)
			
			image_dir = row[1].lower().replace(' ','_')
			
			args = {'keywords' : row[1], 'limit' : limit, 'output_directory' : output_dir,'image_directory' : image_dir, 'print_urls': True}
			response = google_images_download.googleimagesdownload()
			paths = response.download(args)

	def getFilePaths(self,root):
		"""
			assumes in reference to running file dirr. Will fetch all images in room directory
			
			Args :
				root (str) : root directory path from current working directory
			
			Return (list) : list of all full paths of images
		"""
		image_list = []
		for path, subdirs, files in os.walk(root):
			for name in files:
				image_list.append(os.path.join(path, name))
		
		return image_list

	def resizeImageArray(self,img):
		"""
			resizes an image to object's height / width
			
			Args : 
				img (str) : full path to image to resize
			
			Return : numpy array in shape (1, height, width, 3)
		"""
		print('resizing ' + img)
		im = Image.open(img)
		im = im.resize((self.height,self.width))
		return np.array(im).reshape(1,self.height,self.width,3)

	def getArraysFromPics(self):
	
		"""
			gets all images from local file, resizes, and stores in y & X member functions
			
			Args:
				None
			
			Return : None
		"""
		#get file lists
		images_positive = self.getFilePaths('images_positive')
		images_negative = self.getFilePaths('images_negative')
		
		
		image_list = []
		array_list = []
		index = 0
		for file in images_positive:
			try: 
				if index == 0:
					array_list.append(self.resizeImageArray(img = file))
					image_list.append(file)
				else:
					array_list.append(self.resizeImageArray(img = file))
					image_list.append(file)
				index = index + 1
			except:
				print('error')
			
		
		for file in images_negative:
			try: 
				array_list.append(self.resizeImageArray(img = file))
				image_list.append(file)
			except:
				print('error')
		
		self.X = np.vstack(array_list)
		#store outcome variable
		outcome = [1 if i <= index + 1 else 0 for i in range(1,len(image_list) + 1)]
		
		self.y = pd.DataFrame({'file' : image_list, 'outcome' : outcome})
		
		print('X is a numpy array with shape : ' + str(self.X.shape))
		print('y is a pandas df with two columns ; file location and outcome. here is the head')
		print(self.y.head())

	def saveData(self):
		"""
			saves member data to data folder if exists
			
			Return : None
		"""
		if self.y is not None:
			self.y['index'] = self.y.index
			self.y.to_csv('data/y.csv',index=False)
		if self.X is not None:
			np.save('data/X',self.X)
		if self.y_test is not None:
			self.y_test['index'] = self.y_test.index
			self.y_test.to_csv('data/y_test.csv',index=False)
		if self.X_test is not None:
			np.save('data/X_test',self.X_test)
		if self.y_train is not None:
			self.y_train['index'] = self.y_train.index
			self.y_train.to_csv('data/y_train.csv',index=False)
		if self.X_train is not None:
			np.save('data/X_train',self.X_train)

	def loadData(self):
		"""
			loads member data
			
			Return : None
		"""
		#load data
		self.y = pd.read_csv('data/y.csv', engine='python')
		self.y = self.y.set_index('index')
		#load data
		self.X = np.load('data/X.npy')
		#load data
		self.y_test = pd.read_csv('data/y_test.csv', engine='python')
		self.y_test = self.y_test.set_index('index')
		#load data
		self.X_test = np.load('data/X_test.npy')
		#load data
		self.y_train = pd.read_csv('data/y_train.csv', engine='python')
		self.y_train = self.y_train.set_index('index')
		#load data
		self.X_train = np.load('data/X_train.npy')

		print('X shape : ' + str(self.X.shape))
		print('y shape : ' + str(self.y.shape))
		print('X_train shape : ' + str(self.X_train.shape))
		print('y_train shape : ' + str(self.y_train.shape))
		print('X_test shape : ' + str(self.X_test.shape))
		print('y_test shape : ' + str(self.y_test.shape))

	def shuffleData(self):
		"""
			shuffels member data y & X. NOTE : needs to be preformed before saving / loading or splitting data
			
			Retrun : None
		"""
		self.y = self.y.sample(frac=1)
		
		cnt = 0
		array_list = []
		for index in list(self.y.index):
			array_list.append(self.X[index,:,:,:].reshape(1,self.height,self.width,3))
		
		self.X = np.vstack(array_list)
		print(self.y.head())

	def splitTrainTest(self,splitRatio):
		"""
			splits y & X member functions into train and test sets
			
			Args :
				splitRatio (float) : ratio to split between training and test sets
			
			Return : None
		"""
		SPLIT_NUM = int(len(self.y.index)*splitRatio)
		self.y['one'] = 1
		self.y['tempindex'] = self.y.one.cumsum()
		

		self.y_train = self.y[self.y.tempindex <= SPLIT_NUM][['file','outcome']]
		self.y_test = self.y[self.y.tempindex > SPLIT_NUM][['file','outcome']]
		self.X_train = self.X[:SPLIT_NUM,:,:,:]
		self.X_test = self.X[SPLIT_NUM:,:,:,:]
		
		self.y = self.y[['file','outcome']]
		
		print('y_train : ' + str(len(self.y_train)))
		print('x_train : ' + str(self.X_train.shape))
		
		print('y_test : ' + str(len(self.y_test)))
		print('x_test : ' + str(self.X_test.shape))

	def saveModel(self, model_dir = os.path.join(os.getcwd(), 'saved_models'), model_name = 'keras_m1'):
		"""
			Save keras model and json object
			
			Args
				model_dir (str) = directory to save model
				model_name (str) = root name (without extension) of file. Function will save a "h5" and "json" files from that name
		
			Return : None
		"""
		# Save model and weights
		if not os.path.isdir(model_dir):
			os.makedirs(model_dir)
		model_path = os.path.join(model_dir, model_name)
		self.model.save(model_path + '.h5')
		with open(model_path + '.json','w') as outfile:
			json.dump(self.model.to_json(),outfile)
		print('Saved trained model at %s ' % model_path)

	def loadModel(self, model_dir = os.path.join(os.getcwd(), 'saved_models'), model_name = 'keras_m1'):
		"""
			Load keras model and json object
			
			Args
				model_dir (str) : directory to save model
				model_name (str) : root name (without extension) of file. Function will load a "h5" and "json" files from that name
		
			Return : None
		"""
		model_path = os.path.join(model_dir, model_name)
		self.model = keras.models.load_model(model_path + '.h5')
		with open(model_path + '.json') as f:
			self.model_json = json.load(f)
		print('Loaded trained model at %s ' % model_path)

	def trainModel(self,batch_size = 128, epochs = 5, data_augmentation = False, num_predictions = 2, 
		num_classes = 1, model_dir = os.path.join(os.getcwd(), 'saved_models'), model_name = 'keras_m1',
		use_existing_model = True):
		
		"""
			Train Keras model. Note if existing model is found in model_dir and name, then it is loaded unless you specify to not use the existing model. Model will load using member functions
			
			Args
				batch_size (int) : batch size to use in training
				epochs (int) : epochs to use in training
				data_augmentation (bool) : whether to use data_augmentation
				num_predictions (int) : NA --depricated
				num_classes (int) : number of output classes
				model_dir (str) : directory to save model - used in load and save models
				model_name (str) : root name (without extension) of file. Function will load a "h5" and "json" files from that name - used in load and save models
				use_existing_model (bool) : if model exists, this indicates if we should use it or retrain another model and overwrite
		
			Return : None
		"""

		if use_existing_model and model_name in set([file.split('.')[0] for file in os.listdir(model_dir)]) :
			print('Existing models found - loading models from disk')
			print('------------------------------------------------')
			self.loadModel(model_dir = model_dir, model_name = model_name)
			x_test = self.X_test.astype('float32')/255
			y_test = np.array(self.y_test['outcome'])
			scores = self.model.evaluate(x_test, y_test, verbose=1)
			print('Test loss:', scores[0])
			print('Test accuracy:', scores[1])
			return #exit function
			
		print('training and saving new model')
		print('-----------------------------')

		model = Sequential()
		model.add(Conv2D(32, (3, 3), padding='same',input_shape=self.X_train.shape[1:]))
		model.add(Activation('relu'))
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes))
		model.add(Activation('softmax'))

		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		x_train = self.X_train.astype('float32')/255
		x_test = self.X_test.astype('float32')/255
		y_test = np.array(self.y_test['outcome'])
		y_train = np.array(self.y_train['outcome'])

		if not data_augmentation:
			print('Not using data augmentation.')
			model.fit(x_train, y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  validation_data=(x_test, y_test),
					  shuffle=False)
		else:
			print('Using real-time data augmentation.')
			# This will do preprocessing and realtime data augmentation:
			datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				zca_epsilon=1e-06,  # epsilon for ZCA whitening
				rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
				# randomly shift images horizontally (fraction of total width)
				width_shift_range=0.1,
				# randomly shift images vertically (fraction of total height)
				height_shift_range=0.1,
				shear_range=0.,  # set range for random shear
				zoom_range=0.,  # set range for random zoom
				channel_shift_range=0.,  # set range for random channel shifts
				# set mode for filling points outside the input boundaries
				fill_mode='nearest',
				cval=0.,  # value used for fill_mode = "constant"
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False,  # randomly flip images
				# set rescaling factor (applied before any other transformation)
				rescale=None,
				# set function that will be applied on each input
				preprocessing_function=None,
				# image data format, either "channels_first" or "channels_last"
				data_format=None,
				# fraction of images reserved for validation (strictly between 0 and 1)
				validation_split=0.0)

			# Compute quantities required for feature-wise normalization
			# (std, mean, and principal components if ZCA whitening is applied).
			datagen.fit(x_train)

			# Fit the model on the batches generated by datagen.flow().
			model.fit_generator(datagen.flow(x_train, y_train,
											 batch_size=batch_size),
								epochs=epochs,
								validation_data=(x_test, y_test),
								workers=4)

		#save model
		self.model = model
		self.saveModel(model_dir = model_dir, model_name = model_name)
		# Score trained model.
		scores = model.evaluate(x_test, y_test, verbose=1)
		print('Test loss:', scores[0])
		print('Test accuracy:', scores[1])

	def predict(self, X):
		"""
			use model to predict class and probability of image
			
			Args:
				X (numpy) : array of shape (height, width, 3)
			
			Return (dict) : information of outcome and probability
		"""
		p = self.model.predict(X.reshape(1,self.height,self.width,3)/255)
		predicted_outcome = 'poisoness' if p[0][0] > .5 else 'non_poisonous'
		return {'outcome' : predicted_outcome, 'prob' : p[0][0]}