#-----------------#
# Generic Imports #
#-----------------#

import os, tempfile, csv, collections, pickle, gzip, math, sys
from collections import Counter
import pandas as pd
import numpy as np
import scipy as sc
from time import time
from time import strftime
from random import randint
from tqdm import tqdm
from Bio import SeqIO
from ete2 import NCBITaxa
import subprocess
import h5py

#---------------#
# Keras Imports #
#---------------#

import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input, merge, LeakyReLU, Reshape, Concatenate, PReLU, BatchNormalization
from keras.layers import Convolution1D, Conv1D, Conv2D, SpatialDropout1D, MaxPooling1D, MaxPooling2D, MaxPool2D, LSTM, Bidirectional, TimeDistributed
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras import optimizers
from keras.initializers import Constant
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, History, LearningRateScheduler
from keras import optimizers
from keras import layers
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
FILTER_SIZES = [30,31,32]
NUM_FILTERS = 200
DROP_PROB = 0.5
bn_axis = 3

dataset = "./inputs_45_genomes"
checkpoint_save_path = "Checkpoint_"
embedding_filepath = 'dna2vec/pretrained/dna2vec-20180128-0158-k3to8-100d-5c-152610Mbp-sliding-l9Y.w2v'

mk_model = MultiKModel(embedding_filepath)

#----------------------#
# Supporting Functions #
#----------------------#

def get_kmers_index(sequence, word_index, kmer_len=5):
	'''
	All disjoint k-mers from the sequece are aggregated
	
	Returns: Dense 100D unit vector that represents the sequence
	'''
	assert kmer_len > 2 and kmer_len < 9
	kmers = [word_index.get(sequence[i: i + kmer_len], 0) for i in range(0, len(sequence)) if i <= len(sequence) - kmer_len]
	return kmers

def create_embedding_matrix(kmer_length = 3):
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

def scheduler(epoch):
	learning_rate_init = 0.001
	if epoch > 55:
		learning_rate_init = 0.0002
	if epoch > 70:
		learning_rate_init = 0.00005
	return learning_rate_init

class LossWeightsModifier(keras.callbacks.Callback):
	def __init__(self, alpha, beta, gamma):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

	# customize your behavior
	def on_epoch_end(self, epoch, logs={}):
		if epoch == 30:
			K.set_value(self.alpha, 0.1)
			K.set_value(self.beta, 0.8)
			K.set_value(self.gamma, 0.1)
		if epoch == 60:
			K.set_value(self.alpha, 0.1)
			K.set_value(self.beta, 0.2)
			K.set_value(self.gamma, 0.7)
		if epoch == 80:
			K.set_value(self.alpha, 0)
			K.set_value(self.beta, 0)
			K.set_value(self.gamma, 1)

def generator(features, labels_list, num_classes_list, batch_size, KMER_COUNT):
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

		batch_labels_coarse1 = np.zeros((batch_size, 1))
		batch_labels_coarse2 = np.zeros((batch_size, 1))
		batch_labels_fine = np.zeros((batch_size, 1))

		batch_labels_expanded_coarse1 = np.zeros((batch_size, num_classes_list[0]), dtype = np.int8)
		batch_labels_expanded_coarse2 = np.zeros((batch_size, num_classes_list[1]), dtype = np.int8)
		batch_labels_expanded_fine = np.zeros((batch_size, num_classes_list[2]), dtype = np.int8)

		for i in range(batch_size):
			index = randint(0, len(features) - 10)
			#batch_features[i] = get_kmers_index(str(features[index].seq), word_index, KMER_LEN)
			batch_features[i] = features[index]
			
			batch_labels_coarse1[i] = labels_list[0][index]
			batch_labels_coarse2[i] = labels_list[1][index]
			batch_labels_fine[i] = labels_list[2][index]

		batch_labels_expanded_coarse1 = keras.utils.to_categorical(batch_labels_coarse1, num_classes = num_classes_list[0])
		batch_labels_expanded_coarse2 = keras.utils.to_categorical(batch_labels_coarse2, num_classes = num_classes_list[1])
		batch_labels_expanded_fine = keras.utils.to_categorical(batch_labels_fine, num_classes = num_classes_list[2])

		yield batch_features, [batch_labels_expanded_coarse1, batch_labels_expanded_coarse2, batch_labels_expanded_fine]

def dataset_load(path_to_dir, taxid, predict=False):
	fasta_records = list(SeqIO.parse(os.path.join(path_to_dir, taxid, taxid + ".fasta"), 'fasta'))
	if predict == False:
		taxid_records = pd.read_csv(os.path.join(path_to_dir, taxid, taxid + '.taxid'), memory_map = True).values
		return [fasta_records, taxid_records]
	else:
		return fasta_records

def map_to_0_index(y_train):
	num_classes_y = len(y_train)
	class_dict = dict()
	for val, key in enumerate(y_train):
		class_dict[key] = val
	return [class_dict, num_classes_y]

def get_desired_ranks(taxid, desired_ranks):
        ncbi = NCBITaxa()
        lineage = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage)
        lineage2ranks = ncbi.get_rank(names)
        ranks2lineage = dict((rank,taxid) for (taxid, rank) in lineage2ranks.items())

        return [ranks2lineage.get(rank, '0') for rank in desired_ranks]

def get_level2level(list_of_lower_taxids, parent_level):
        '''
        Returns a dictionary containing child taxID: parent taxid mappings.
        '''
	child2parent = dict()
	for taxid in list_of_lower_taxids:
		key = get_desired_ranks(taxid, [parent_level])[0]
		child2parent[taxid] = key
        return child2parent

def create_class_lists(y_train, y_test):
	mapped_classes_fine, num_classes_fine = map_to_0_index(np.unique(y_train))

        y_train = y_train.flatten()
        y_test = y_test.flatten()

        y_train_fine = [mapped_classes_fine[label] for label in y_train]
        y_test_fine = [mapped_classes_fine[label] for label in y_test]

        genus2family = get_level2level(np.unique(y_train), 'family')
        y_train_coarse_2 = [genus2family[label] for label in y_train]
        mapped_classes_coarse_2, num_classes_coarse2 = map_to_0_index(np.unique(y_train_coarse_2))
        y_train_coarse_2 = [mapped_classes_coarse_2[label] for label in y_train_coarse_2]
        y_test_coarse_2 = [genus2family[label] for label in y_test]
        y_test_coarse_2 = [mapped_classes_coarse_2[label] for label in y_test_coarse_2]

	family2order = get_level2level(np.unique(y_train), 'order')
        y_train_coarse_1 = [family2order[label] for label in y_train]
        mapped_classes_coarse_1, num_classes_coarse1 = map_to_0_index(np.unique(y_train_coarse_1))
        y_train_coarse_1 = [mapped_classes_coarse_1[label] for label in y_train_coarse_1]
        y_test_coarse_1 = [family2order[label] for label in y_test]
        y_test_coarse_1 = [mapped_classes_coarse_1[label] for label in y_test_coarse_1]

	return [y_train_coarse_1, y_test_coarse_1, y_train_coarse_2, y_test_coarse_2, y_train_fine, y_test_fine, num_classes_coarse1, num_classes_coarse2, num_classes_fine]

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def Branch_CNN_Classifier(x_train, y_train, x_test, y_test, taxid, n_gram, taxon_level):
	KMER_LEN = n_gram
        KMER_COUNT = 100 - KMER_LEN + 1
        VOCAB_SIZE = pow(4, KMER_LEN)

	log_filepath = '../Organism_Prediction_DL/logs'
	weights_store_filepath = './branch_CNN_weights_store/'
	model_name = 'weights_bcnn_dynamic_bacteria_28.h5'
	model_path = os.path.join(weights_store_filepath, model_name)

	y_train_coarse_1, y_test_coarse_1, y_train_coarse_2, y_test_coarse_2, y_train_fine, y_test_fine, num_classes_coarse1, num_classes_coarse2, num_classes_fine = create_class_lists(y_train, y_test)
	
	#------------------#
	# Model Definition #
	#------------------#	

	alpha = K.variable(value=0.98, dtype="float32", name="alpha") # A1 in paper
	beta = K.variable(value=0.01, dtype="float32", name="beta") # A2 in paper
	gamma = K.variable(value=0.01, dtype="float32", name="gamma") # A3 in paper

        embedding_matrix, word_index = create_embedding_matrix(KMER_LEN)

        inpt = Input(shape=(KMER_COUNT, ), dtype='int32')
        embedding = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=KMER_COUNT,
                            trainable=False)(inpt)

        reshape = Reshape((KMER_COUNT, EMBEDDING_DIM, 1))(embedding)

	x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(reshape)
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)
	
	#-----------#
        # Block ONE #
        #-----------# 
	
	x = conv_block(x, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=1, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=1, block='c')
	
	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

	#-------------------------#
        # Coarse Branch 1 (ORDER) #
        #-------------------------#

	c_1_bch = Flatten(name='c1_flatten')(x)
	c_1_bch = Dense(64, activation='relu', name='c1_fc1')(c_1_bch)
	c_1_bch = BatchNormalization()(c_1_bch)
	c_1_bch = Dropout(0.3)(c_1_bch)
	c_1_bch = Dense(64, activation='relu', name='c1_fc2')(c_1_bch)
	c_1_bch = BatchNormalization()(c_1_bch)
	c_1_bch = Dropout(0.3)(c_1_bch)
	c_1_pred = Dense(num_classes_coarse1, activation='softmax', name='c1_predictions')(c_1_bch)

	#-----------#
        # Block TWO #
        #-----------#

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=4, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=4, block='d')
	
	#--------------------------#
        # Coarse Branch 2 (FAMILY) #
        #--------------------------#

	c_2_bch = Flatten(name='c2_flatten')(x)
	c_2_bch = Dense(64, activation='relu', name='c2_fc1')(c_2_bch)
	c_2_bch = BatchNormalization()(c_2_bch)
	c_2_bch = Dropout(0.3)(c_2_bch)
	c_2_bch = Dense(64, activation='relu', name='c2_fc2')(c_2_bch)
	c_2_bch = BatchNormalization()(c_2_bch)
	c_2_bch = Dropout(0.3)(c_2_bch)
	c_2_pred = Dense(num_classes_coarse2, activation='softmax', name='c2_predictions')(c_2_bch)

	#-------------#
        # Block THREE #
        #-------------#

	x = conv_block(x, 3, [256, 256, 1024], stage=5, block='a')
	x = identity_block(x, 3, [256, 256, 1024], stage=5, block='b')
	x = identity_block(x, 3, [256, 256, 1024], stage=5, block='c')
	x = identity_block(x, 3, [256, 256, 1024], stage=5, block='d')
	x = identity_block(x, 3, [256, 256, 1024], stage=5, block='e')
	x = identity_block(x, 3, [256, 256, 1024], stage=5, block='f')

	x = conv_block(x, 3, [512, 512, 2048], stage=6, block='a')
	x = identity_block(x, 3, [512, 512, 2048], stage=6, block='b')
	x = identity_block(x, 3, [512, 512, 2048], stage=6, block='c')

	#--------------------------#
        # Fine Predictions (GENUS) #
        #--------------------------#
	
	fine_bch = Flatten(name='flatten')(x)
	fine_bch = Dense(64, activation='relu', name='fc1')(fine_bch)
	fine_bch = BatchNormalization()(fine_bch)
	fine_bch = Dropout(0.3)(fine_bch)
	fine_bch = Dense(64, activation='relu', name='fc2')(fine_bch)
	fine_bch = BatchNormalization()(fine_bch)
	fine_bch = Dropout(0.3)(fine_bch)
	fine_pred = Dense(num_classes_fine, activation='softmax', name='fine_predictions')(fine_bch)
	
        model_bcnn = Model(inputs=inpt, outputs=[c_1_pred, c_2_pred, fine_pred])

	#---------------#
        # COMPILE & FIT #
        #---------------#

	#sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
	model_bcnn.compile(loss='categorical_crossentropy', 
              optimizer=Adam(),
              loss_weights=[alpha, beta, gamma],
              metrics=['accuracy', 'top_k_categorical_accuracy'])
	
	#change_lr = LearningRateScheduler(scheduler)
	change_lw = LossWeightsModifier(alpha, beta, gamma)
	#checkpoint = ModelCheckpoint(model_path, monitor='val_fine_predictions_acc', verbose=1, save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor = 'val_fine_predictions_loss', factor = 0.3, patience=3, min_lr=0.00005)
	cbks = [change_lw, reduce_lr]

        batch_size = 128
	batch_size_effective = batch_size * 10

        print model_bcnn.summary()

	y_train_sets = [y_train_coarse_1, y_train_coarse_2, y_train_fine]
	y_test_sets = [y_test_coarse_1, y_test_coarse_2, y_test_fine]
	num_classes_sets = [num_classes_coarse1, num_classes_coarse2, num_classes_fine]

        model_bcnn.fit_generator(generator(x_train, y_train_sets, num_classes_sets, batch_size, KMER_COUNT),
                                steps_per_epoch = len(x_train) / batch_size_effective,
                                nb_epoch = 150,
				callbacks=cbks,
                                validation_data=(generator(x_test, y_test_sets, num_classes_sets, batch_size, KMER_COUNT)),
                                validation_steps = len(x_test) / batch_size_effective)
	
	model_bcnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(), 
              metrics=['accuracy'])

	model_bcnn.save(model_path)
	
	#score = model_bcnn.evaluate(x_test, y_test_sets, verbose = 1)
	#print('score is: ', score)

def generate_next_batch(x_test, batch_size, word_index, indx):
	KMER_LEN = 6
        KMER_COUNT = 100 - KMER_LEN + 1

	batch_features = np.zeros((batch_size, KMER_COUNT), dtype = np.int32)

	for i in range(indx * batch_size, indx * batch_size + batch_size):
		batch_features[i - indx * batch_size] = get_kmers_index(str(x_test[i].seq), word_index, KMER_LEN)

	return batch_features

def predict_results(x_test, model_path, mapped_classes_coarse_1, mapped_classes_coarse_2, mapped_classes_fine):
        model_cnn = load_model(model_path)
	
	embedding_matrix, word_index = create_embedding_matrix(6)
	inverted_dict_coarse1 = dict([[v,k] for k,v in mapped_classes_coarse_1.items()])
	inverted_dict_coarse2 = dict([[v,k] for k,v in mapped_classes_coarse_2.items()])
	inverted_dict_fine = dict([[v,k] for k,v in mapped_classes_fine.items()])

	y_pred_classes = []

	for i in tqdm(range(0, 10000)):
		batch_ = generate_next_batch(x_test, 32, word_index, i)
		y_proba = model_cnn.predict(batch_)
		
		coarse1_predictions = [inverted_dict_coarse1[j] for j in y_proba[0].argmax(axis=-1)]
		coarse2_predictions = [inverted_dict_coarse2[j] for j in y_proba[1].argmax(axis=-1)]
		fine_predictions = [inverted_dict_fine[j] for j in y_proba[2].argmax(axis=-1)]

        	y_pred_classes.extend(coarse2_predictions)
        
	cnt_coarse1 = Counter()
	cnt_coarse2 = Counter()
	cnt_fine = Counter()

	for i in y_pred_classes:
		cnt_coarse2[i] += 1
	print cnt_coarse2

def call_branch_cnn(path_to_dir):
	start_time = time()
	#fasta_records_train = list(SeqIO.parse(os.path.join(path_to_dir, "x_train.fasta"), 'fasta'))
	#fasta_records_test = list(SeqIO.parse(os.path.join(path_to_dir, "x_test.fasta"), 'fasta'))
	f = h5py.File(path_to_dir + "28Genus.hdf5", "r")
        fasta_records_train = f['training/x_train']
        fasta_records_test = f['testing/x_test']
	taxid_records_train = pd.read_csv(os.path.join(path_to_dir, "y_train.taxid"), memory_map = True).values
	taxid_records_test = pd.read_csv(os.path.join(path_to_dir, "y_test.taxid"), memory_map = True).values
	Branch_CNN_Classifier(fasta_records_train, taxid_records_train, fasta_records_test, taxid_records_test, '1224', 6, 'AllLevels')
	end_time = time()
	print "Total time spent training: "
	print (end_time - start_time)	

#---------------------#
# Create HDF5 Dataset #
#---------------------#

def create_hdf5_datasets(path_to_test_train_dir):
	
	KMER_LEN = 6
        KMER_COUNT = 100 - KMER_LEN + 1

	train_len = int(subprocess.check_output(["wc", "-l", path_to_test_train_dir + "y_train.taxid"]).split(' ')[0])
	test_len = int(subprocess.check_output(["wc", "-l", path_to_test_train_dir + "y_test.taxid"]).split(' ')[0])

	f = h5py.File(path_to_test_train_dir + "28Genus.hdf5", "w")
	train_group = f.create_group("training")
	test_group = f.create_group("testing")
	
	x_train_dataset = train_group.create_dataset("x_train", (train_len, KMER_COUNT), dtype='i')

	x_test_dataset = test_group.create_dataset("x_test", (test_len, KMER_COUNT), dtype='i')
	
	embedding_matrix, word_index = create_embedding_matrix(KMER_LEN)
	idx = 0

	for fasta_records_train in tqdm(SeqIO.parse(os.path.join(path_to_test_train_dir, "x_train.fasta"), 'fasta')):
		feature = get_kmers_index(str(fasta_records_train.seq), word_index, KMER_LEN)
		x_train_dataset[idx] = feature
		idx += 1
	
	idx = 0

        for fasta_records_test in tqdm(SeqIO.parse(os.path.join(path_to_test_train_dir, "x_test.fasta"), 'fasta')):
                feature = get_kmers_index(str(fasta_records_test.seq), word_index, KMER_LEN)
                x_test_dataset[idx] = feature
                idx += 1

if __name__ == '__main__':

	#---------------------#
	# Create HDF5 Dataset #
	#---------------------#

	#create_hdf5_datasets('./inputs/simple_dataset_raw/')

	#---------------------------#
        # Load HDF5 Dataset & Train #
        #---------------------------#

	call_branch_cnn('./inputs_45_genomes/simple_dataset_raw/')

	#-------------------------------------#
        # Predict on trained Branch-CNN model #
        #-------------------------------------#
	'''
	x_test = dataset_load('./inputs/simple_dataset_raw/held_out_fasta/', '897', predict = True)
	taxid_records = pd.read_csv('./inputs/simple_dataset_raw/y_train.taxid', memory_map = True).values
	taxid_records = taxid_records.flatten()
	mapped_classes_fine, num_classes_fine = map_to_0_index(np.unique(taxid_records))

	family2order = get_level2level(np.unique(taxid_records), 'order')
        y_train_coarse_1 = [family2order[label] for label in taxid_records]
        mapped_classes_coarse_1, num_classes_coarse1 = map_to_0_index(np.unique(y_train_coarse_1))

	genus2family = get_level2level(np.unique(taxid_records), 'family')
        y_train_coarse_2 = [genus2family[label] for label in taxid_records]
        mapped_classes_coarse_2, num_classes_coarse2 = map_to_0_index(np.unique(y_train_coarse_2))

        predict_results(x_test, './branch_CNN_weights_store/weights_bcnn_dynamic_bacteria_28.h5', mapped_classes_coarse_1, mapped_classes_coarse_2, mapped_classes_fine)
	
	#----------------------------#
        # Old Classification methods #
        #----------------------------#

	directory_for_training = './inputs/simple_dataset_raw/genus/ml_training/'
	directory_for_testing = './inputs/simple_dataset_raw/genus/ml_test/'
	
	for directory in os.listdir(directory_for_training):
		x_train, y_train = dataset_load(directory_for_training, directory)
		x_test, y_test = dataset_load(directory_for_testing, directory)
		CNNClassifier(x_train, y_train, x_test, y_test, directory, 6, 'Family')
	'''
	
	#x_test = dataset_load('./inputs/simple_dataset_raw/genus/ml_test/', '897', predict = True)
	#predict_results(x_test, '100_6mer_2Classes_At_Phylum.h5')
	'''
	for filter_size in range(2, 11):
		for num_fils in range (100, 700, 100):
			x_train, y_train = dataset_load('./inputs/simple_dataset_raw/genus1/ml_training/', '204037')
			CNNClassifier(x_train, y_train, '204037', 6, filter_size, num_fils)
	'''
