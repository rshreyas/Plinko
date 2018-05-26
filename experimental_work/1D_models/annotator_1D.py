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
import subprocess
import h5py

#---------------#
# Keras Imports #
#---------------#

from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input, merge, LeakyReLU, Reshape, Concatenate, PReLU, BatchNormalization
from keras.layers import Convolution1D, Conv1D, Conv2D, SpatialDropout1D, MaxPooling1D, MaxPool1D, MaxPooling2D, MaxPool2D, LSTM, Bidirectional, TimeDistributed
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
from sklearn import cluster

#-----------------------------#
# Annotator Global Parameters #
#-----------------------------#

RNGSEED = 17  ## seed for random number generator for reproducibility
EMBEDDING_DIM  = 4
FILTER_SIZES = [15,27,30]
NUM_FILTERS = 200
DROP_PROB = 0.3
VOCAB_SIZE = 5   ## size of unique letters
SEQLEN  = 100
TRAINING_DIR = 'ml_training'
TESTING_DIR = 'ml_test'
PATH_TO_DATASETS = './inputs_190_Species/simple_dataset_raw/'

ncbi = NCBITaxa()

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

aa2int = dict({ 'A' : 0, 
		'T' : 1, 
		'G' : 2,
		'C' : 3, 
		'a' : 0, 
		't' : 1, 
		'g' : 2, 
		'c' : 3})

def seq2ids(seq):
    return np.array([aa2int.get(x,4) for x in seq])

def create_hdf5_datasets(taxonomic_level, taxid):
	
	taxid = str(taxid)
	path_to_train_dir = os.path.join(PATH_TO_DATASETS, taxonomic_level, TRAINING_DIR, taxid)
	path_to_test_dir = os.path.join(PATH_TO_DATASETS, taxonomic_level, TESTING_DIR, taxid)
	
        train_len = int(subprocess.check_output(["wc", "-l", os.path.join(path_to_train_dir, taxid + ".taxid")]).split(' ')[0])
        test_len = int(subprocess.check_output(["wc", "-l", os.path.join(path_to_test_dir, taxid + ".taxid")]).split(' ')[0])

        f = h5py.File(PATH_TO_DATASETS + taxonomic_level + "_" + taxid + ".hdf5", "w")
        train_group = f.create_group("training")
        test_group = f.create_group("testing")
	mapping_group = f.create_group("mapping")

        x_train_dataset = train_group.create_dataset("x_train", (train_len, SEQLEN), dtype='i')
	y_train_dataset = train_group.create_dataset("y_train", (train_len,), dtype = 'i')
        
	x_test_dataset = test_group.create_dataset("x_test", (test_len, SEQLEN), dtype='i')
	y_test_dataset = test_group.create_dataset("y_test", (test_len,), dtype = 'i')

	taxid_records_train = np.loadtxt(os.path.join(path_to_train_dir, taxid + ".taxid"), dtype = np.int32)
	taxid_records_test = np.loadtxt(os.path.join(path_to_test_dir, taxid + ".taxid"), dtype = np.int32)

	unique_taxids = np.unique(taxid_records_train)
	taxid_to_0_index = dict()
	
	for idx, i in enumerate(unique_taxids):
		taxid_to_0_index[i] = idx

	mapping = mapping_group.create_dataset("taxid_to_0_index", (len(unique_taxids),), dtype='i')
	mapping = unique_taxids

	idx = 0

        for fasta_records_train in tqdm(SeqIO.parse(os.path.join(path_to_train_dir, taxid + ".fasta"), 'fasta')):
                feature = seq2ids(str(fasta_records_train.seq))
                x_train_dataset[idx] = feature
		y_train_dataset[idx] = taxid_to_0_index[taxid_records_train[idx]]
                idx += 1

        idx = 0

        for fasta_records_test in tqdm(SeqIO.parse(os.path.join(path_to_test_dir, taxid + ".fasta"), 'fasta')):
                feature = seq2ids(str(fasta_records_test.seq))
		y_test_dataset[idx] = taxid_to_0_index[taxid_records_test[idx]]
                x_test_dataset[idx] = feature
                idx += 1
	
#----------------------#
# Classifier Functions #
#----------------------#

def generator(features, labels, num_classes, batch_size):
	while True:
		batch_features = np.zeros((batch_size, SEQLEN), dtype = np.int8)
		batch_labels = np.zeros((batch_size, 1))
		batch_labels_expanded = np.zeros((batch_size, num_classes), dtype = np.int8)

		for i in range(batch_size):
			index = randint(0, len(features) - 10)
			batch_features[i] = features[index]
			batch_labels[i] = labels[index]

		batch_labels_expanded = keras.utils.to_categorical(batch_labels, num_classes = num_classes)
		yield batch_features, batch_labels_expanded


def CNNClassifier(x_train, y_train, x_test, y_test, taxonomic_level, taxid, num_classes_y):
        #Tweak parameters: Dropout, EMBDIM, #Convolutions, Multi-task, batch_size
	
	inpt = Input(shape=(SEQLEN,), dtype='int8', name='input_sequence')
	x = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQLEN)(inpt)

	x = Conv1D(200, 15, padding='same')(x)
        relu_1 = Activation('relu')(x)
        x = Dropout(0.3)(relu_1)

	x = Conv1D(200, 27, padding='same')(x)
	relu_2 = Activation('relu')(x)
	x = Dropout(0.3)(relu_2)

	merge_1 = keras.layers.add([relu_1, relu_2])
	x = Conv1D(200, 30, padding='same', activation='relu')(merge_1)
        x = Dropout(0.3)(x)
        x = MaxPooling1D()(x)

	x = Flatten()(x)

	x = Dense(256, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
	
	out = Dense(num_classes_y, activation='softmax')(x)
	model_cnn = Model(inputs=inpt, outputs=out)
	model_cnn.compile(loss='categorical_crossentropy', optimizer = Adam(), metrics=['accuracy', 'top_k_categorical_accuracy'])

	#tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0, write_graph=True, write_images=True)
	#checkpoint = ModelCheckpoint(modelSavepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        batch_size = 256

        print model_cnn.summary()

	history = model_cnn.fit_generator(generator(x_train, y_train, num_classes_y, batch_size),
                                steps_per_epoch = len(x_train) / batch_size,
                                nb_epoch = 20,
                                validation_data=(generator(x_test, y_test, num_classes_y, batch_size)),
                                validation_steps = len(x_test) / batch_size)

        model_cnn.save(os.path.join(PATH_TO_DATASETS, taxonomic_level + "_" + taxid + "_model" + ".h5"))
	
def call_cnn(taxonomic_level, taxid):

	taxid = str(taxid)
        start_time = time()

        f = h5py.File(PATH_TO_DATASETS + taxonomic_level + "_" + taxid + ".hdf5", "r")
        fasta_records_train = f['training/x_train']
        fasta_records_test = f['testing/x_test']
        taxid_records_train = f['training/y_train']
        taxid_records_test = f['testing/y_test']
	mappings = f['mapping/taxid_to_0_index']	
	
	CNNClassifier(fasta_records_train, taxid_records_train, fasta_records_test, taxid_records_test, taxonomic_level, taxid, len(mappings))

        end_time = time()
        print "Total time spent training: "
        print (end_time - start_time)

if __name__ == '__main__':
	
	create_hdf5_datasets('family', 213116)
	
	#----------------------------------#
	# Training CNN at particular level #
	#----------------------------------#
	
	call_cnn('family', 213116)

	'''	
	#-----------------------------#
	# Predict on held out Species #
	#-----------------------------#
	
	x_test = dataset_load('./inputs/simple_dataset_raw/genus/ml_test/', '897', predict = True)
	predict_results(x_test, '100_6mer_2Classes_At_Phylum.h5')
	
	'''
