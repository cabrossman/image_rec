import os
dirr = 'C:\\Users\\cabro\\OneDrive\\Documents\\jobs\\image_rec'
os.chdir(dirr)
from resources import BinaryRec
%pylab inline

#create the object
#recommender = BinaryRec(height = 32, width = 32)
recommender = BinaryRec(height = 96, width = 96)

#do we have the pics? if not download them
if 'images_positive' not in os.listdir() or 'images_negative' not in os.listdir():
	#recommender.download_images(path_to_download_types = 'resources/mushrooms_types.csv', limit_per_category = 10, pos_divisor = 2, neg_divisor = 1)
	recommender.download_images(path_to_download_types = 'resources/dog_vs_cat.csv', limit_per_category = 12, pos_divisor = 3, neg_divisor = 1)
	recommender.getArraysFromPics()
	recommender.shuffleData()
	recommender.splitTrainTest(splitRatio = .9)
	recommender.saveData()
else:
	recommender.loadData()

#show k random pics
recommender.showImage(image_index = None, K = 10, print_adjusted_size = False)

#create model if none exists
recommender.trainModel(batch_size = 128, epochs = , data_augmentation = False, num_predictions = 2, 
	num_classes = 1, model_dir = os.path.join(os.getcwd(), 'saved_models'), model_name = 'keras_m1',use_existing_model = False)

for i in range(0,5):
	index = recommender.y_test.index[i]
	X = recommender.X_test[i,:,:,:]
	response = recommender.predict(X)
	print(response)
	recommender.showImage(image_index = index, print_adjusted_size = True)