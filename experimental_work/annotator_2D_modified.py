#-----------------#
# Generic Imports #
#-----------------#

import os, tempfile, csv, collections, pickle, gzip, math, sys
from collections import Counter
import pandas as pd
import numpy as np
import scipy as sc
from time import time
from random import randint
from tqdm import tqdm
from Bio import SeqIO
from ete2 import NCBITaxa
from pprint import pprint

#---------------#
# Keras Imports #
#---------------#

from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input, merge, LeakyReLU, Reshape, Concatenate, PReLU, BatchNormalization
from keras.layers import Convolution1D, Conv1D, Conv2D, SpatialDropout1D, MaxPooling1D, MaxPooling2D, MaxPool2D, LSTM, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from keras import optimizers
from keras.initializers import Constant
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, History
from keras import optimizers
import keras.utils
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#----------------#
# SciKit Imports #
#----------------#

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import cluster

#------------------------------------#
# dna2vec dense vector module import #
#------------------------------------#

sys.path.append(os.getcwd() + '/dna2vec')
from dna2vec import MultiKModel

sys.path.append(os.getcwd() + '/keras-spp')
from spp.SpatialPyramidPooling import SpatialPyramidPooling

#-----------------------------#
# Annotator Global Parameters #
#-----------------------------#

RNGSEED = 22  ## seed for random number generator for reproducibility
EMBEDDING_DIM  = 100
FILTER_SIZES = [6,7,8]
NUM_FILTERS = 200
DROP_PROB = 0.5

dataset = "./inputs"
embedding_filepath = 'dna2vec/pretrained/dna2vec-20180128-0158-k3to8-100d-5c-152610Mbp-sliding-l9Y.w2v'

ncbi = NCBITaxa()
mk_model = MultiKModel(embedding_filepath)

parent_level_mappings = {"species": "genus", 
                         "genus": "family", 
                         "family": "order", 
                         "order": "class", 
                         "class": "phylum",
                         "phylum": "superkingdom"}


#----------------------#
# Supporting Functions #
#----------------------#

def _pickle(obj, name):
        f = gzip.open(name + '.pklz','wb')
        pickle.dump(obj,f)
        f.close()

def _unpickle(path):
        f = gzip.open(path, 'r')
        obj = pickle.load(f)
        f.close()
        return obj

def get_kmers_index(sequence, word_index, kmer_len=5):
	'''
	All disjoint k-mers from the sequece are aggregated
	
	Returns: Dense 100D unit vector that represents the sequence
	'''
	assert kmer_len > 2 and kmer_len < 9
	kmers = [word_index.get(sequence[i: i + kmer_len], 0) for i in range(0, len(sequence)) if i <= len(sequence) - kmer_len]
	return kmers

def create_embedding_matrix(kmer_length = 6, predict = False):
	assert kmer_length > 2 and kmer_length < 9

	embeddings_index = {}
	word_list = []
	f = open(embedding_filepath)
	for line in f:
    		values = line.split()
		word = values[0]
		if len(word) == kmer_length:
			word_list.append(word)
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	f.close()

	word_index = dict()

        for i, word in enumerate(word_list):
		word_index[word] = i

	if predict == True:
		return word_index
	
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	
	for word, i in word_index.items():
    		embedding_vector = embeddings_index.get(word)
    		if embedding_vector is not None:
        		# words not found in embedding index will be all-zeros.
        		embedding_matrix[i] = embedding_vector

	return [embedding_matrix, word_index]

	
#----------------------#
# Classifier Functions #
#----------------------#

def generator(features, labels, num_classes, batch_size, word_index, KMER_COUNT, KMER_LEN):
	'''
	Batch generator used by classifier's fit_generator function.
	This generator will pick a random subset from (features, labels) and return it as a 2D NumPy matrix

	features: Input examples represented as a SEQLEN dimension NumPy vector
	
	labels: Output labels used by the classifier
	
	num_classes: Total number of unique classes in the dataset
	
	batch_size: Size of batch to train one itereation of the classifier
	'''
	while True:
		batch_features = np.zeros((batch_size, KMER_COUNT), dtype = np.int32)
		batch_labels = np.zeros((batch_size, 1))
		batch_labels_expanded = np.zeros((batch_size, num_classes), dtype = np.int8)

		for i in range(batch_size):
			index = randint(0, len(features) - 1)
			batch_features[i] = get_kmers_index(str(features[index].seq), word_index, KMER_LEN)
			batch_labels[i] = labels[index]

		batch_labels_expanded = keras.utils.to_categorical(batch_labels, num_classes = num_classes)
		yield batch_features, batch_labels_expanded


def dataset_load(path_to_dir, taxid, predict=False):
	fasta_records = list(SeqIO.parse(os.path.join(path_to_dir, taxid, taxid + '.fasta'), 'fasta'))
	if predict == False:
		taxid_records = pd.read_csv(os.path.join(path_to_dir, taxid, taxid + '.taxid'), memory_map = True, header = None).values
		return [fasta_records, taxid_records]
	else:
		return fasta_records


def validation_set_load(path_to_dir, dataset_name):
	fasta_records = list(SeqIO.parse(os.path.join(path_to_dir, dataset_name + '.fasta'), 'fasta'))
	taxid_records = pd.read_csv(os.path.join(path_to_dir, dataset_name + '.taxid'), memory_map = True, header = None).values
	return [fasta_records, taxid_records]

def map_to_0_index(y_train):
	num_classes_y = len(y_train)
	class_dict = dict()
	for val, key in enumerate(y_train):
		class_dict[key] = val
	return [class_dict, num_classes_y]

def map_back_to_taxid(mapped_0_index):
	return {v: k for k, v in mapped_0_index.iteritems()}

def CNNClassifier(x_train, y_train, x_test, y_test, taxid, n_gram, taxon_level, filter_size):
        #Tweak parameters: Dropout, EMBDIM, #Convolutions, Multi-task, batch_size
	KMER_LEN = n_gram
	KMER_COUNT = 100 - KMER_LEN + 1
	VOCAB_SIZE = pow(4, KMER_LEN)
	FILTER_SIZES = [filter_size - KMER_LEN, filter_size - KMER_LEN + 1, filter_size - KMER_LEN + 2]

	save_path = './tree_cnn_model_store'
	save_name = str(KMER_COUNT + KMER_LEN - 1) + "FragmentLength_" + str(n_gram) + "mer_" + taxon_level + "_" + taxid + "_" + str(filter_size)
	
	embedding_matrix, word_index = create_embedding_matrix(KMER_LEN)
	
	inpt = Input(shape=(KMER_COUNT, ), dtype='int32')
	embedding = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=KMER_COUNT,
                            trainable=True)(inpt)
	
	reshape = Reshape((KMER_COUNT, EMBEDDING_DIM, 1))(embedding)
	
	conv_0 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='normal', activation='relu')(reshape)
	conv_1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='normal', activation='relu')(conv_0)	
	conv_2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='normal', activation='relu')(conv_1)
	conv_3 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='normal', activation='relu')(conv_2)
	flatten = Flatten()(conv_3)
	reshape = Reshape((-1, 64))(flatten)

	x = Conv1D(64, 4, padding='same')(reshape)
        x = MaxPooling1D(strides = 2)(x)
	x = Activation('relu')(x)

	x = Conv1D(64, 4, padding='same')(x)
        x = MaxPooling1D(strides = 2)(x)
	x = Activation('relu')(x)

	flatten = Flatten()(x)
	dropout = Dropout(DROP_PROB)(flatten)
        
	mapped_classes, num_classes_y = map_to_0_index(np.unique(y_train))
	#_pickle(mapped_classes, os.path.join(save_path, taxon_level, taxid, save_name + "_mappedclasses"))
	
	y_train = y_train.flatten()
	y_test = y_test.flatten()

	y_train = [mapped_classes[label] for label in y_train]
	y_test = [mapped_classes[label] for label in y_test]
	
	x_train = x_train[:-1]
	x_test = x_test[:-1]
	
        out = Dense(num_classes_y, activation='softmax')(dropout)
        model_cnn = Model(inputs=inpt, outputs=out)

        model_cnn.compile(loss='categorical_crossentropy', 
			optimizer = Adam(), 
			metrics=['accuracy', 'top_k_categorical_accuracy'])

        batch_size = 256

        print model_cnn.summary()

	history = model_cnn.fit_generator(generator(x_train, y_train, num_classes_y, batch_size, word_index, KMER_COUNT, KMER_LEN),
                                steps_per_epoch = len(x_train) / batch_size,
                                nb_epoch = 100,
                                validation_data=(generator(x_test, y_test, num_classes_y, batch_size, word_index, KMER_COUNT, KMER_LEN)),
                                validation_steps = len(x_test) / batch_size)

	if not os.path.exists(os.path.join(save_path, taxon_level, taxid)):
    		os.makedirs(os.path.join(save_path, taxon_level, taxid))
        model_cnn.save(os.path.join(save_path, taxon_level, taxid, save_name + ".h5"))
	
#----------------------#
# Prediction Functions #
#----------------------#

def generate_next_batch(x_test, start_at, end_at, word_index):
	KMER_LEN = 6
        KMER_COUNT = 100 - KMER_LEN + 1

	batch_features = np.zeros((end_at - start_at, KMER_COUNT), dtype = np.int32)

	for i in range(start_at, end_at):
		batch_features[i - start_at] = get_kmers_index(str(x_test[i].seq), word_index, KMER_LEN)

	return batch_features


def predict_results(x_test, model_path):
        model_cnn = load_model(os.path.join(model_path))
	
	word_index = create_embedding_matrix(6, predict = True)

	y_pred_classes = []

	for i in tqdm(range(0, len(x_test) / 32)):
		batch_ = generate_next_batch(x_test, 32, word_index, i)
		y_proba = model_cnn.predict(batch_)
        	y_pred_classes.extend(y_proba.argmax(axis=-1))
	

def get_paths_to_models_and_mappings(path_to_models_and_mapping):
	path_to_models = []
        paths_to_model_class_mapping = []

        for outer_dir in os.listdir(path_to_models_and_mapping):
                for inner_dir in os.listdir(os.path.join(path_to_models_and_mapping, outer_dir)):
                        for filename in os.listdir(os.path.join(path_to_models_and_mapping, outer_dir, inner_dir)):
                                if filename.endswith('.h5'):
                                        path_to_models.append(os.path.join(path_to_models_and_mapping, outer_dir, inner_dir, filename))
                                else:
                                        paths_to_model_class_mapping.append(os.path.join(path_to_models_and_mapping, outer_dir, inner_dir, filename))
	return path_to_models, paths_to_model_class_mapping


def predict_results_on_tree(x_test_sample, loaded_models, loaded_class_mappings, predict_at_taxid, y_pred_classes, y_pred_probs):
	model_cnn = loaded_models[predict_at_taxid]
	y_prob = model_cnn.predict(x_test_sample)
	y_pred_class = y_prob.argmax(axis=-1)
	
	class_maps = loaded_class_mappings[predict_at_taxid] 
	inverted_dict = dict([[v,k] for k,v in class_maps.items()])
	predict_at_next_taxid = inverted_dict[y_pred_class[0]]

	y_pred_classes.append(predict_at_next_taxid)
        y_pred_probs.append(y_prob[0][y_pred_class][0])

	'''
	if ncbi.get_rank([predict_at_taxid]).values()[0] == 'family':
		y_pred_class = y_prob[0].argsort()[-5:][::-1]
		y_pred_classes.append([inverted_dict[i] for i in y_pred_class])
		y_pred_probs.append([y_prob[0][i] for i in y_pred_class])
	else:
		y_pred_classes.append(predict_at_next_taxid)
		y_pred_probs.append(y_prob[0][y_pred_class][0])
	'''
	if ncbi.get_rank([predict_at_taxid]).values()[0] == 'genus':
		return y_pred_classes[-1], y_pred_probs[-1]
	else:
		return predict_results_on_tree(x_test_sample, loaded_models, loaded_class_mappings, predict_at_next_taxid, y_pred_classes, y_pred_probs)	

def predict_result_on_tree_initializer(x_test, path_to_models_and_mapping):
	path_to_models, paths_to_model_class_mapping = get_paths_to_models_and_mappings(path_to_models_and_mapping)
        loaded_models = dict()
        loaded_class_mappings = dict()

        for i in range(len(path_to_models)):
                name_as_taxid = int(path_to_models[i].split('_')[-2])
         	loaded_models[name_as_taxid] = load_model(path_to_models[i])

        for i in range(len(paths_to_model_class_mapping)):   
                name_as_taxid = int(paths_to_model_class_mapping[i].split('_')[-3])
                loaded_class_mappings[name_as_taxid] = _unpickle(paths_to_model_class_mapping[i])
	
	start_predicting_from = 91061
        word_index = create_embedding_matrix(6, predict = True)
	
	final_predictions = []
	final_probabilities = []

	for i in tqdm(range(0, len(x_test), 32)):
                start_at = i
                if i + 32 < len(x_test):
                        end_at = i + 32
                else:
                        end_at = len(x_test)
                _batch = generate_next_batch(x_test, start_at, end_at, word_index)

		for sample_in_batch in _batch:
			y_pred_classes, y_pred_prob = predict_results_on_tree(np.reshape(sample_in_batch, (1, -1)), loaded_models, loaded_class_mappings, start_predicting_from, [], [])
			final_predictions.append(y_pred_classes)
			final_probabilities.append(y_pred_prob)
	#_pickle(final_predictions, '897_final_predictions')
	#_pickle(final_probabilities, '897_final_probabilities')
	with open("15.taxid", "a+") as fopen:
		for i in final_predictions:
			fopen.write(str(i) + "\n")

#------------------------------------------------------#
# Functions to Fine-Tune an existing Neural Network    #
#	a) Create Confusion Matrix from Validation set #
#	b) Cluster Difficult to Distinguish Classes    #
#	c) Create train and test sets for each cluster #
#	d) Train Subclass networks                     #
#------------------------------------------------------#

def calculate_eigenmap_orig(F, n_clusters = 5):
        D = np.ones(F.shape) - F
        np.fill_diagonal(D, 0)
        D = 0.5 * (D + D.T)
        similarity_mat = np.exp(-D ** 2 / (2. * 0.95 ** 2))
        spectral = cluster.SpectralClustering(n_clusters = n_clusters, eigen_solver = 'arpack', affinity = 'precomputed')

        cluster_labels = spectral.fit_predict(similarity_mat)

        cnt = dict()
        for idx, i in enumerate(cluster_labels):
                try:
                        cnt[i].append(idx)
                except KeyError:
                        cnt[i] = [idx]

        for i in cnt:
                print len(cnt[i])

        _pickle(cnt, 'cluster_dict')


def simple_prediction_on_validation_set(x_test, y_test, model_path):

        model = load_model(os.path.join(model_path))

	word_index = create_embedding_matrix(6, predict = True)

        y_pred_classes = []

        for i in tqdm(range(0, len(x_test), 32)):
		start_at = i
		if i + 32 < len(x_test):
			end_at = i + 32
		else:
			end_at = len(x_test)
                batch_ = generate_next_batch(x_test, start_at, end_at, word_index)
                y_proba = model.predict(batch_)
                y_pred_classes.extend(y_proba.argmax(axis=-1))

	# Maintain same dictionary as used during training
	mapped_classes, num_classes_y = map_to_0_index(np.unique(y_test))
	
	# Map back predicted labels from 0 index to TaxID
	mapped_to_taxid = map_back_to_taxid(mapped_classes)

	y_test = y_test.flatten()
	
	#Choose only those classes that are part of the model's output classes (For eg. 17 out of all 28 genera
	y_pred_classes = [mapped_to_taxid[label] for label in y_pred_classes]

	y_true_classes = y_test[ : len(y_pred_classes)]
        
	return map(lambda x: str(x), y_pred_classes)

def create_confusion_matrix(x_test, y_test, model_path, model_mapped_classes_dict_path):
	model_mapped_classes_dict = _unpickle(model_mapped_classes_dict_path)

	y_true, y_pred = simple_prediction_on_validation_set(x_test, y_test, model_path, model_mapped_classes_dict)

	C = confusion_matrix(y_true, y_pred)

	normalized_C = normalize(C, axis = 1, norm = 'l1')

	calculate_eigenmap_orig(normalized_C, 4)


def create_subclasses(x_train, y_train, x_test, y_test, taxid, save_to_path):
        print "Creating Training & Test Sets..."

	y_train.flatten()
	y_test.flatten()
	
	mapped_classes, num_classes_y = map_to_0_index(np.unique(y_train))
	
	y_train = [mapped_classes[label] for label in y_train]
	y_test = [mapped_classes[label] for label in y_test]

        cluster = _unpickle('cluster_dict.pklz')

        for i in cluster:
		if len(cluster[i] < 2):
			continue
                print "Number of classes in cluster " + str(i + 1) + ": " + str(len(cluster[i]))
		common_prefix = taxid + "_" + str(i)
                name_x_train = common_prefix + "_x_train"
		name_y_train = common_prefix + "_y_train"
		name_x_test = common_prefix + "_x_test"
                name_y_test = common_prefix + "_y_test"
		
		y_train_index = [idx for idx, x in enumerate(y_train) if x in cluster[i]]
		y_test_index = [idx for idx, x in enumerate(y_test) if x in cluster[i]]
		
		x_train = [x_train[idx] for idx in y_train_index]
		y_train = [y_train[idx] for idx in y_train_index]
		x_test = [x_test[idx] for idx in y_test_index]
		y_test = [y_test[idx] for idx in y_test_index]

		_pickle(x_train, save_to_path + '/' + name_x_train)
		_pickle(y_train, save_to_path + '/' + name_y_train)
		_pickle(x_test, save_to_path + '/' + name_x_test)
                _pickle(y_test, save_to_path + '/' + name_y_test)

        print "Done Creating Training & Test Sets."

def get_parent2childmap(list_of_child_taxids, parent_level="genus"):
        '''
        Returns a dictionary containing parent taxID : [child taxID] mappings.
        #---------------------------------------------------------------------#
        parent_level taxID: int
        genome taxID: int
        #---------------------------------------------------------------------#
        '''

        parent2genome = dict()
        
        for taxid in list_of_child_taxids:
            if taxid <= 0:
                continue
            key = get_desired_ranks(taxid, [parent_level])[0]
            if not parent2genome.has_key(key):
                parent2genome[key] = []
            parent2genome[key].append(taxid)
        return parent2genome

def get_kmer_characterization(taxa_tuple2kmer_dict, parent2child, parent_taxid):
	kmer_characterization = []
	child_permutations = list(itertools.combinations(parent2child[taxid], 2))
	for tuples in child_permutations:
		if taxa_tuple2kmer_dict.has_key((tuples[0], tuples[1])):
			kmer_characterization.append(taxa_tuple2kmer_dict[(tuples[0], tuples[1])])
		elif taxa_tuple2kmer_dict.has_key((tuples[1], tuples[0])):
			kmer_characterization.append(taxa_tuple2kmer_dict[(tuples[1], tuples[0])])
		else:
			kmer_characterization.append(31)
	return max(kmer_characterization)

if __name__ == '__main__':
	'''
	model = load_model("./tree_cnn_model_store/Family/213119/100FragmentLength_6mer_Family_213119.h5")
        print model.summary()
	'''
	#---------------------------------------------------------------#
        # Training a Fine-Tuned CNN for sub-classes at the lowest level #
        #---------------------------------------------------------------#
	'''
	common_path_to_model = './tree_cnn_model_store/Family/213119/'
	x_test, y_test = validation_set_load('./inputs/simple_dataset_raw/family/ml_test/213119', '213119')
	create_confusion_matrix(x_test, y_test, common_path_to_model + '100FragmentLength_6mer_Family_213119.h5', \
		common_path_to_model + '100FragmentLength_6mer_Family_213119_mappedclasses.pklz')
	
	directory_for_training = './inputs/simple_dataset_raw/family/ml_training/'
        directory_for_testing = './inputs/simple_dataset_raw/family/ml_test/'
	x_train, y_train = dataset_load(directory_for_training, '213119')
	x_test, y_test = dataset_load(directory_for_testing, '213119')	

	create_subclasses(x_train, y_train, x_test, y_test, '213119', './inputs/simple_dataset_raw/family/subclass')
	'''
	#-----------------------------------------------------------#
        # Predict TaxID for DNA Fragment at Order, Family and Genus #
        #-----------------------------------------------------------#
	'''
	x_test = dataset_load('../ISMB_Datasets/Vanilla_Case/Evaluation_Set', 'x_test', predict = True)
	predict_result_on_tree_initializer(x_test, './tree_cnn_model_store/')
	'''
	#----------------------------------#
	# Training CNN at particular level #
	#----------------------------------#
	
	taxa_list = ["genus"] 
	#"genus"]
	for taxa_rank in taxa_list:
	# Actual Training of CNN Starts Here
		directory_for_training = os.path.join("inputs_190_species","simple_dataset_raw", taxa_rank, "ml_training")
        	directory_for_testing = os.path.join("inputs_190_species","simple_dataset_raw", taxa_rank, "ml_test")
		
		# Directory is the parent TaxID
	
		directory = "1301"	
		x_train, y_train = dataset_load(directory_for_training, directory)
		x_test, y_test = dataset_load(directory_for_testing, directory)
		filter_width = 15

		CNNClassifier(x_train, y_train, x_test, y_test, directory, 6, taxa_rank, filter_width)
	
	#-----------------------------#
	# Predict on held out Species #
	#-----------------------------#
	'''
	x_test, y_test = dataset_load('./inputs_190_species/simple_dataset_raw/genus/ml_test/', '1716')
	kmer_characterization = [23, 24, 31]
	for i in kmer_characterization:
		y_pred_classses = simple_prediction_on_validation_set(x_test, y_test, \
			'./tree_cnn_model_store/genus/1716/100FragmentLength_6mer_genus_1716_' + str(i) + ".h5")
		with open(str(i) + ".taxid", "a+") as outfile:
			outfile.write("\n".join(y_pred_classses))
		with open(str(i) + ".taxid", "a+") as outfile:
			 outfile.write("\n")
	'''
	#--------------------------------------------------------------#
        # Varying Number of Convolutional filters & k-mer (word) sizes #
        #--------------------------------------------------------------#
	'''
	for filter_size in range(2, 11):
		for num_fils in range (100, 700, 100):
			x_train, y_train = dataset_load('./inputs/simple_dataset_raw/genus1/ml_training/', '204037')
			CNNClassifier(x_train, y_train, '204037', 6, filter_size, num_fils)
	'''
