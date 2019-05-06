import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):
	"""
	Args:
		folder_path: the directory contains data files
		batch_id: training batch id (1,2,3,4,5)
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""

	###load batch using pickle###
	fl = folder_path+'/data_batch_'+str(batch_id)
	cifar = unpickle(fl)

	return cifar

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
	"""
	Args:
		folder_path: the directory contains data files
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""

	###load batch using pickle###
	fl = folder_path+'/test_batch'
	cifar = unpickle(fl)
	return cifar

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
	airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
	pass
	return

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
	"""
	Args:
		features: a numpy array with shape (10000, 3072)
	Return:
		features: a numpy array with shape (10000,32,32,3)
	"""
	features = np.reshape(features, (-1, 32, 32, 3), order='F')
	return features

#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id):
	"""
	Args:
		folder_path: directory that contains data files
		batch_id: the specific number of batch you want to explore.
		data_id: the specific number of data example you want to visualize
	Return:
		None

	Descrption:
		1)You can print out the number of images for every class.
		2)Visualize the image
		3)Print out the minimum and maximum values of pixel
	"""
	file_name = folder_path + '/data_batch_' + str(batch_id)
	dict = unpickle(file_name)
	features = dict['data']
	features = features_reshape(features)
	get_image = np.squeeze(features[data_id, :, :, :])
	print(get_image.shape)
	plt.imshow(get_image)

#Step 6: define a function that does min-max normalization on input
def normalize(x):
	"""
	Args:
		x: features, a numpy array
	Return:
		x: normalized features
	"""
	scaler = preprocessing.MinMaxScaler()
	x =  scaler.fit_transform(x)

	return x

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
	"""
	Args:
		x: a list of labels
	Return:
		a numpy array that has shape (len(x), # of classes)
	"""
	x.reshape(-1,1)
	onehot = preprocessing.OneHotEncoder(handle_unknown='ignore')
	x = onehot.fit_transform(x)

	return x

# #Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
# def preprocess_and_save(features,labels,filename):
# 	"""
# 	Args:
# 		features: numpy array
# 		labels: a list of labels
# 		filename: the file you want to save the preprocessed data
# 	"""
# 	pass

#Step 9:define a function that preprocesss all training batch data and test data.
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
	"""
	Args:
		folder_path: the directory contains your data files
	"""
	val_ratio = 0.2
	for idx in range(1, 6):
		dict = load_training_batch(folder_path, idx)
		features = dict['data']
		labels = dict['labels']

		if idx == 1:
			total_features = features
			total_labels = labels
		else:
			total_features = np.concatenate((total_features, features), axis=0)
			total_labels = np.concatenate((total_labels, labels), axis=0)

	length = len(total_labels)
	val_length = int(length * val_ratio)
	val_features = total_features[range(val_length), :]
	val_labels = np.array(total_labels[range(val_length)])
	train_features = total_features[val_length:length, :]
	train_labels = np.array(total_labels[val_length:length])

	dict = load_testing_batch(folder_path)
	test_features = dict['data']
	test_labels = np.array(dict['labels'])

	vali_features = normalize(val_features)
	# print(scaler.data_max_)
	train_features = normalize(train_features)
	test_features = normalize(test_features)

	train_labels = train_labels.reshape(-1, 1)
	vali_labels = val_labels.reshape(-1, 1)
	test_labels = test_labels.reshape(-1, 1)

	train_labels = one_hot_encoding(train_labels)
	vali_labels = one_hot_encoding(vali_labels)
	test_labels = one_hot_encoding(test_labels)
	# print(train_labels.shape)

	with open('train_data.pickle', 'wb') as f:
		pickle.dump((train_features, train_labels), f)

	with open('val_data.pickle', 'wb') as f:
		pickle.dump((vali_features, vali_labels), f)

	with open('test_data.pickle', 'wb') as f:
		pickle.dump((test_features, test_labels), f)

# #Step 10: define a function to yield mini_batch
# def mini_batch(features,labels,mini_batch_size):
# 	"""
# 	Args:
# 		features: features for one batch
# 		labels: labels for one batch
# 		mini_batch_size: the mini-batch size you want to use.
# 	Hint: Use "yield" to generate mini-batch features and labels
# 	"""
#
# #Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
# def load_preprocessed_training_batch(batch_id,mini_batch_size):
# 	"""
# 	Args:
# 		batch_id: the specific training batch you want to load
# 		mini_batch_size: the number of examples you want to process for one update
# 	Return:
# 		mini_batch(features,labels, mini_batch_size)
# 	"""
# 	file_name = ''
# 	features, labels = pass
# 	return mini_batch(features,labels,mini_batch_size)
#
# #Step 12: load preprocessed validation batch
# def load_preprocessed_validation_batch():
# 	file_name =
# 	features,labels =
# 	return features,labels
#
# #Step 13: load preprocessed test batch
# def load_preprocessed_test_batch(test_mini_batch_size):
# 	file_name =
# 	features,label =
# 	return mini_batch(features,labels,test_mini_batch_size)