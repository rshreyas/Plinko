#!/usr/bin/env python
'''
 Runs the DNA2Vec + (1D, 2D, 3D) Hybrid CNN classifiers.

 This Python wrapper was written by Shreyas Ramesh <shreyas2@vt.edu>.

 The program REQUIRES Keras -- A high-level neural network API with the
 TensorFlow backed and Python 2.7. Additionally, h5py and its dependencies
 must be installed correctly on the system.
'''
# ----------------- #
#  Generic Imports  #
# ----------------- #

# FunGNet Version
__version__ = "0.1.0"

import os
import pickle
import gzip
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from ete2 import NCBITaxa
import sys
import argparse
import glob
import tensorflow as tf

from FunGNet_Classes import DataGenerator
from utils import multilevel_dataset_creator

# --------------- #
#  Keras Imports  #
# --------------- #

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input
from keras.layers import Reshape, Concatenate, Lambda
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPooling1D, MaxPool2D, MaxPool3D
from keras.optimizers import Adam
import keras.backend as K

# --------------------------- #
#  FunGNet Global Parameters  #
# --------------------------- #

EMBEDDING_DIM = 100
TRAINING_DIR = 'ml_training'
TESTING_DIR = 'ml_test'
EMBEDDING_FILENAME = 'dna2vec-20180128-0158-k3to8-100d-5c-152610Mbp-sliding-l9Y.w2v'
EMBEDDING_FILEPATH = os.path.join('dna2vec_embedding_store', EMBEDDING_FILENAME)

ncbi = NCBITaxa()


# ---------------------- #
#  Supporting Functions  #
# ---------------------- #


def _pickle(obj, name):
    """ Creates a pickled object with the name supplied.

        Args:
            obj: Object to be pickled.
            name: Pickle object save name.

        Returns:
            None

    """
    f = gzip.open(name + '.pklz', 'wb')
    pickle.dump(obj, f)
    f.close()


def _unpickle(path):
    """ Given a pickled object (Made with _pickle)
        return back the object.

        Args:
            path: Path to Pickled object

        Returns:
            Pickled object

    """
    f = gzip.open(path, 'r')
    obj = pickle.load(f)
    f.close()
    return obj


aa2int = dict({
    'A': 0,
    'T': 1,
    'G': 2,
    'C': 3
    })


def get_kmers_index(sequence, word_index, kmer_len=6):
    """ Lookup word indexes for each k-mer in the nucleotide sequence.

        Args:
            sequence: A Nucleotide Sequence.
            word_index: Dictionary of k-mer to word index.
                        This is used in the Embedding Matrix.
            kmer_len: Length of k-mers to lookup (default: 6)

        Returns:
            List of k-mer indices.

    """
    assert kmer_len > 2 and kmer_len < 9
    kmers = [word_index.get(sequence[i: i + kmer_len], 0)
             for i in range(0, len(sequence)) if i <= len(sequence) - kmer_len]
    return kmers


def create_embedding_matrix(kmer_length=6, predict=False):
    """ Function to create an embedding matrix to be used in classification.
        Uses Word2Vec (.w2v) file as source for constructing embedding
        matrix.

        Example: 5-mer
            k-mer - ATGCC
            Word Index - 32
            Row 33 in Embedding Matrix = [0.67, -1.22 ...]

        Args:
            kmer_length: Defines number of rows in embedding matrix. (default: 6)
            predict: Is the current stage inference? If so, return
                     only word indexes. (default: False)

        Returns:
            k-mer to index Dictionary
            Embedding Matrix of shape (Num_kmers, Embedding Vector Dimension)

    """

    assert kmer_length > 2 and kmer_length < 9

    embeddings_index = {}
    word_list = []
    f = open(EMBEDDING_FILEPATH)
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

    if predict is True:
        return word_index

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return [embedding_matrix, word_index]


def load_test_dataset(test_dir):
    """ Reads a FASTA file and returns a list of FASTA records.

        Args:
            test_dir: Directory holding the FASTA file to load.

        Returns:
            List of FASTA records

    """
    try:
        fasta = glob.glob(test_dir + "/*.fasta")[0]
    except:
        raise RuntimeError("No fasta file found in: " + test_dir)
    return list(SeqIO.parse(fasta, 'fasta'))


# ------------------------------ #
#  Hilbert Input Transformation  #
# ------------------------------ #


def hilbert_curve(n):
    """ Generates a Hilbert curve indexing for an (n, n) array.

        Note: 'n' must be a power of two.

        Args:
            n: Width of a square matrix.
               Creates an nxn matrix with n^2 indices

        Returns:
            nxn matrix of hilbert transformed indexes

    """

    # recursion base
    if n == 1:
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbert_curve(n//2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size*2
    d = np.flipud(np.rot90(t, -1)) + t.size*3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))


def hilbert_1d_transform(hilbert_shape):
    """ Main function that performs the following steps:
            -> Generate 2D numpy matrix consisting of rearranged indexes
            -> Transform 2D numpy matrix into a 1D tensor. Transforming into
               a 1D tensor helps because the 2D matrix rearrangement
               can be recovered from a tensor and not a 1D numpy array.

        Args:
            hilbert_shape: Width of the hilbert square matrix

        Returns:
            1D Tensor transformed from a 2D square Hilbert Index Matrix

    """
    hilbert_transform = hilbert_curve(hilbert_shape)
    reshape_res = tf.reshape(hilbert_transform, [-1])
    hilbert_transform = tf.Session().run(reshape_res)
    return hilbert_transform


# --------------------------- #
#  Dataset Creation Function  #
# --------------------------- #


def load_dataset(path_to_datasets, taxon_level, taxon_id):
    """ Loads FASTA and TAXID file pairs for Neural Network Training
        and Validation.

        Directory hierarchy Example:
            species
                -> db_creation
                    -> 1279
                        -> 1279.fasta
                        -> 1279.taxid
                    -> 1301, 1350 ...
                -> ml_test
                    -> 1279, 1301, 1350 ...
                -> ml_training
                    -> 1279, 1301, 1350 ...

            genus
                -> db_creation
                -> ml_test
                -> ml_training

        Args:
            path_to_datasets: All files must maintain directory structure as above
            taxon_level: Directory to lookup (genus, species, family etc)
            taxon_id: Directory to lookup within taxin_level directory (1279, 1301 etc)

        Returns:
            Training FASTA and TaxID records
            Validation FASTA and TaxID records
            Dictionary mapping TaxonID to 0 based index

    """
    taxon_id = str(taxon_id)

    path_to_train_dir = os.path.join(path_to_datasets, taxon_level,
                                     TRAINING_DIR, taxon_id)

    path_to_test_dir = os.path.join(path_to_datasets, taxon_level,
                                    TESTING_DIR, taxon_id)

    taxid_records_train = np.loadtxt(os.path.join(path_to_train_dir,
                                     taxon_id + ".taxid"), dtype=np.int32)
    taxid_records_test = np.loadtxt(os.path.join(path_to_test_dir,
                                    taxon_id + ".taxid"), dtype=np.int32)
    fasta_records_train = list(SeqIO.parse(os.path.join(path_to_train_dir,
                               taxon_id + ".fasta"), 'fasta'))
    fasta_records_test = list(SeqIO.parse(os.path.join(path_to_test_dir,
                              taxon_id + ".fasta"), 'fasta'))

    unique_taxids = np.unique(taxid_records_train)
    taxid_to_0_index = dict()
    for idx, i in enumerate(unique_taxids):
        taxid_to_0_index[i] = idx

    return [fasta_records_train, fasta_records_test,
            taxid_records_train, taxid_records_test, taxid_to_0_index]


# ------------- #
#  Classifiers  #
# ------------- #


def Sentence_Matrix_2D_CNN(save_path, save_name, x_train, y_train, x_test, y_test,
                           taxid_to_0_index, taxon_level, taxid, num_classes_y,
                           context_len, kmer_len, fragment_len, kmer_count, kernel_size,
                           num_kernels, drop_prob, batch_size, num_epochs,
                           use_multiprocessing, workers):
    """ Trains a CNN model based on the 2D sentence matrix structure.

        Structure: Performs Organism Classification using the
                   FunGNet architecture:
                        1. 1D nucleotide sequence to 2D sentence matrix
                           transformation using SkipGram embeddings.
                        2. Three parallel 2D kernels of width "filter_size".
                        3. Four layers of 1D kernels.

        Args:
            save_path: Path to store trained CNN models
            save_name: Model name
            x_train: Training dataset inputs
            y_train: Training dataset labels
            x_test: Validation dataset inputs
            y_test: Validation dataset labels
            taxid_to_0_index: TaxonID to 0 based index mapping
            taxon_level: Level in the taxonomy tree
            taxid: Taxon ID
            num_classes_y: Number of unique classes / labels
            context_len: Field of view for 2D CNN kernels
            kmer_len: Length of k-mer to split input sequence into
            fragment_len: Length of input sequence
            kmer_count: Number of k-mers in input fragment.
                        fragment_len - kmer_len + 1
            kernel_size: Size of kernel considering k-mer length.
                         context_len - kmer_len + 1
            num_kernels: Number of 2D CNN kernels to use
            drop_prob: Probability of dropping hidden units in neural network
            batch_size: Size of batches within each epoch
            num_epochs: Number of epoch to train network
            use_multiprocessing: Use multiple threads to generate
                                 batches of input data.
            workers: Number of threads to spawn for input batch generation

    Retruns:
        None

     """
    embedding_matrix, word_index = create_embedding_matrix(kmer_len)

    inpt = Input(shape=(kmer_count, ), dtype='int32')
    embedding = Embedding(len(word_index) + 1, EMBEDDING_DIM,
                          weights=[embedding_matrix], input_length=kmer_count,
                          trainable=True)(inpt)

    reshape = Reshape((kmer_count, EMBEDDING_DIM, 1))(embedding)

    conv_0 = Conv2D(num_kernels, kernel_size=(kernel_size, EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    conv_1 = Conv2D(num_kernels, kernel_size=(kernel_size, EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    conv_2 = Conv2D(num_kernels, kernel_size=(kernel_size, EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(kmer_count - kernel_size + 1, 1),
                          strides=(1, 1), padding='valid')(conv_0)

    maxpool_1 = MaxPool2D(pool_size=(kmer_count - kernel_size + 1, 1),
                          strides=(1, 1), padding='valid')(conv_1)

    maxpool_2 = MaxPool2D(pool_size=(kmer_count - kernel_size + 1, 1),
                          strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

    reshape_2 = Reshape((num_kernels, -1))(concatenated_tensor)

    x = Conv1D(64, 4, padding='same')(reshape_2)
    x = Activation('relu')(x)

    x = Conv1D(64, 4, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(strides=2)(x)

    x = Conv1D(128, 4, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv1D(128, 4, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(strides=2)(x)

    flatten = Flatten()(x)
    dropout = Dropout(drop_prob)(flatten)

    out = Dense(num_classes_y, activation='softmax')(dropout)
    model_cnn = Model(inputs=inpt, outputs=out)

    model_cnn.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])

    print model_cnn.summary()

    hilbert_transform = []

    params = {'kmer_len': kmer_len,
              'kmer_count': kmer_count,
              'batch_size': batch_size,
              'num_classes': num_classes_y,
              'word_index': word_index,
              'taxid_to_0_index': taxid_to_0_index,
              'hilbert_transform': hilbert_transform,
              'shuffle': True}

    training_generator = DataGenerator(x_train, y_train, **params).generate()
    validation_generator = DataGenerator(x_test, y_test, **params).generate()

    # TODO: Multithreading seems to not work in harmony with h5py. Try fixing.
    model_cnn.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=len(y_train) / batch_size,
                            validation_steps=len(y_test) / batch_size,
                            epochs=num_epochs,
                            use_multiprocessing=use_multiprocessing,
                            workers=workers)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_cnn.save(os.path.join(save_path, save_name + ".h5"))


def Embedding_Sum_1D_CNN(save_path, save_name, x_train, y_train, x_test, y_test,
                         taxid_to_0_index, taxon_level, taxid, num_classes_y,
                         context_len, kmer_len, fragment_len, kmer_count, kernel_size,
                         num_kernels, drop_prob, batch_size, num_epochs,
                         use_multiprocessing, workers):
    """ Trains a CNN model based on sum of embedding vectors as inputs

        Structure: Performs Organism Classification using the
                   FunGNet architecture:
                        1. 1D nucleotide sequence to 2D sentence matrix
                           transformation using SkipGram embeddings.
                        2. 2D matrix is summed accross all columns to obtain a
                           1D vector embedding of entire sequence.
                        3. Four layers of 1D kernels.

        Args:
            save_path: Path to store trained CNN models
            save_name: Model name
            x_train: Training dataset inputs
            y_train: Training dataset labels
            x_test: Validation dataset inputs
            y_test: Validation dataset labels
            taxid_to_0_index: TaxonID to 0 based index mapping
            taxon_level: Level in the taxonomy tree
            taxid: Taxon ID
            num_classes_y: Number of unique classes / labels
            context_len: NOT USED!
            kmer_len: Length of k-mer to split input sequence into
            fragment_len: Length of input sequence
            kmer_count: Number of k-mers in input fragment.
                        fragment_len - kmer_len + 1
            kernel_size: Size of kernel considering k-mer length.
                         context_len - kmer_len + 1
            num_kernels: Number of CNN kernels to use
            drop_prob: Probability of dropping hidden units in neural network
            batch_size: Size of batches within each epoch
            num_epochs: Number of epoch to train network
            use_multiprocessing: Use multiple threads to generate
                                 batches of input data.
            workers: Number of threads to spawn for input batch generation

    Retruns:
        None

     """
    embedding_matrix, word_index = create_embedding_matrix(kmer_len)

    inpt = Input(shape=(kmer_count, ), dtype='int32')
    embedding = Embedding(len(word_index) + 1, EMBEDDING_DIM,
                          weights=[embedding_matrix], input_length=kmer_count,
                          trainable=True)(inpt)

    embedding_sum = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(embedding)
    reshape = Reshape((-1, 1))(embedding_sum)

    x = Conv1D(num_kernels, 4, padding='same')(reshape)
    x = Activation('relu')(x)

    x = Conv1D(num_kernels, 4, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(strides=2)(x)

    x = Conv1D(num_kernels * 2, 4, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv1D(num_kernels * 2, 4, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(strides=2)(x)

    flatten = Flatten()(x)
    dropout = Dropout(drop_prob)(flatten)

    out = Dense(num_classes_y, activation='softmax')(dropout)
    model_cnn = Model(inputs=inpt, outputs=out)

    model_cnn.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])

    print model_cnn.summary()

    hilbert_transform = []

    params = {'kmer_len': kmer_len,
              'kmer_count': kmer_count,
              'batch_size': batch_size,
              'num_classes': num_classes_y,
              'word_index': word_index,
              'taxid_to_0_index': taxid_to_0_index,
              'hilbert_transform': hilbert_transform,
              'shuffle': True}

    training_generator = DataGenerator(x_train, y_train, **params).generate()
    validation_generator = DataGenerator(x_test, y_test, **params).generate()

    model_cnn.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=len(y_train) / batch_size,
                            validation_steps=len(y_test) / batch_size,
                            epochs=num_epochs,
                            use_multiprocessing=use_multiprocessing,
                            workers=workers)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_cnn.save(os.path.join(save_path, save_name + ".h5"))


def Hilbert_Transform_3D_CNN(save_path, save_name, x_train, y_train, x_test, y_test,
                             taxid_to_0_index, taxon_level, taxid, num_classes_y,
                             context_len, kmer_len, fragment_len, kmer_count,
                             kernel_size, num_kernels, drop_prob, batch_size,
                             num_epochs, hilbert_shape, hilbert_transform,
                             use_multiprocessing, workers):
    """ Trains a CNN model based on Hilbert Transformed sequence as input.

        Structure: Performs Organism Classification using the
                   FunGNet architecture:
                        1. 1D nucleotide sequence broken into k-mers.
                           Each k-mer is given an index.
                        2. 1D tensor of Hilbert Matrix is used as template to
                           replace k-mers at the respective indices.
                        3. 1D tensor is transformed a 3D tensor.
                           -> 2D Hilbert Matrix + 1D for length 100 embedding
                        4. Three parallel 3D kernels of width 3x3 are applied on
                           the transformed tensor.
                        5. Four layers of 1D kernels.

        Args:
            save_path: Path to store trained CNN models
            save_name: Model name
            x_train: Training dataset inputs
            y_train: Training dataset labels
            x_test: Validation dataset inputs
            y_test: Validation dataset labels
            taxid_to_0_index: TaxonID to 0 based index mapping
            taxon_level: Level in the taxonomy tree
            taxid: Taxon ID
            num_classes_y: Number of unique classes / labels
            context_len: NOT USED!
            kmer_len: Length of k-mer to split input sequence into
            fragment_len: Length of input sequence
            kmer_count: Number of k-mers in input fragment.
                        fragment_len - kmer_len + 1
            kernel_size: Size of kernel considering k-mer length.
                         context_len - kmer_len + 1
            num_kernels: Number of CNN kernels to use
            drop_prob: Probability of dropping hidden units in neural network
            batch_size: Size of batches within each epoch
            num_epochs: Number of epoch to train network
            hilbert_shape: Width 'n' of square hilbert matrix
            hilbert_transform: 1D transformed hilbert tensor
            use_multiprocessing: Use multiple threads to generate
                                 batches of input data
            workers: Number of threads to spawn for input batch generation

    Retruns:
        None

     """
    embedding_matrix, word_index = create_embedding_matrix(kmer_len)

    inpt = Input(shape=(kmer_count, ), dtype='int32')
    embedding = Embedding(len(word_index) + 1, EMBEDDING_DIM,
                          weights=[embedding_matrix], input_length=kmer_count,
                          trainable=True)(inpt)

    reshape = Reshape((hilbert_shape, hilbert_shape, EMBEDDING_DIM, 1))(embedding)

    conv_0 = Conv3D(num_kernels, kernel_size=(3, 3, EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    conv_1 = Conv3D(num_kernels, kernel_size=(3, 3, EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    conv_2 = Conv3D(num_kernels, kernel_size=(3, 3, EMBEDDING_DIM),
                    padding='valid', kernel_initializer='normal',
                    activation='relu')(reshape)

    maxpool_0 = MaxPool3D(pool_size=(4, 4, 1),
                          strides=(2, 2, 1), padding='valid')(conv_0)

    maxpool_1 = MaxPool3D(pool_size=(4, 4, 1),
                          strides=(2, 2, 1), padding='valid')(conv_1)

    maxpool_2 = MaxPool3D(pool_size=(4, 4, 1),
                          strides=(2, 2, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

    reshape_2 = Reshape((num_kernels, -1))(concatenated_tensor)

    x = Conv1D(64, 4, padding='same')(reshape_2)
    x = Activation('relu')(x)

    x = Conv1D(64, 4, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(strides=2)(x)

    x = Conv1D(128, 4, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv1D(128, 4, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(strides=2)(x)

    flatten = Flatten()(x)
    dropout = Dropout(drop_prob)(flatten)

    out = Dense(num_classes_y, activation='softmax')(dropout)
    model_cnn = Model(inputs=inpt, outputs=out)

    model_cnn.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])

    print model_cnn.summary()

    params = {'kmer_len': kmer_len,
              'kmer_count': kmer_count,
              'batch_size': batch_size,
              'num_classes': num_classes_y,
              'word_index': word_index,
              'taxid_to_0_index': taxid_to_0_index,
              'hilbert_transform': hilbert_transform,
              'shuffle': True}

    training_generator = DataGenerator(x_train, y_train, **params).generate()
    validation_generator = DataGenerator(x_test, y_test, **params).generate()

    # TODO: Multithreading seems to not work in harmony with h5py. Try fixing.
    model_cnn.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=len(y_train) / batch_size,
                            validation_steps=len(y_test) / batch_size,
                            epochs=num_epochs,
                            use_multiprocessing=use_multiprocessing,
                            workers=workers)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_cnn.save(os.path.join(save_path, save_name + ".h5"))


def call_cnn(dataset_path, model_save_path, args):
    """ Driver function to train CNN at a particular level in the
        taxonomy tree and at particular taxon node.

        Args:
            dataset_path: Path where the Training and Validation datasets exist.
            model_save_path: Where the models need to be stored
            args: Input parameters to the neural networks

        Returns:
            None

    """
    taxon_level = args.taxon_level
    taxon_id = args.taxon_id
    context_len = args.context_len
    kmer_len = args.kmer_len
    fragment_len = args.fragment_len
    num_kernels = args.num_kernels
    drop_prob = args.drop_prob
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hilbert_shape = args.hilbert_shape
    use_multiprocessing = args.use_multiprocessing
    workers = args.workers
    model_name = args.model_name

    # Inferred arguments passed to CNN
    taxon_id = str(taxon_id)
    kmer_count = fragment_len - kmer_len + 1
    kernel_size = context_len - kmer_len + 1
    save_path = os.path.join(model_save_path, taxon_level, taxon_id)
    save_name = (str(fragment_len) +
                 "FragmentLength_" + str(kmer_len) + "mer_" + taxon_level + "_" +
                 taxon_id + "_" + str(context_len))

    hilbert_transform = []
    if hilbert_shape > 0:
        hilbert_transform = hilbert_1d_transform(hilbert_shape)

    fasta_records_train, fasta_records_test, taxid_records_train,\
        taxid_records_test, taxid_to_0_index = \
        load_dataset(dataset_path, args.taxon_level, args.taxon_id)

    num_classes_y = len(taxid_to_0_index)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    _pickle(taxid_to_0_index, os.path.join(save_path,
                                           save_name + "_mappedclasses"))

    if model_name == 1:
        Embedding_Sum_1D_CNN(save_path, save_name, fasta_records_train,
                             taxid_records_train, fasta_records_test,
                             taxid_records_test, taxid_to_0_index, taxon_level,
                             taxon_id, num_classes_y, context_len, kmer_len,
                             fragment_len, kmer_count, kernel_size, num_kernels,
                             drop_prob, batch_size, num_epochs,
                             use_multiprocessing, workers)
    elif model_name == 2:
        Sentence_Matrix_2D_CNN(save_path, save_name, fasta_records_train,
                               taxid_records_train, fasta_records_test,
                               taxid_records_test, taxid_to_0_index,
                               taxon_level, taxon_id, num_classes_y,
                               context_len, kmer_len, fragment_len, kmer_count,
                               kernel_size, num_kernels, drop_prob, batch_size,
                               num_epochs, use_multiprocessing, workers)
    elif model_name == 3:
        if hilbert_shape > 0:
            Hilbert_Transform_3D_CNN(save_path, save_name, fasta_records_train,
                                     taxid_records_train, fasta_records_test,
                                     taxid_records_test, taxid_to_0_index,
                                     taxon_level, taxon_id, num_classes_y,
                                     context_len, kmer_len, fragment_len,
                                     kmer_count, kernel_size, num_kernels,
                                     drop_prob, batch_size, num_epochs,
                                     hilbert_shape, hilbert_transform,
                                     use_multiprocessing, workers)
        else:
            print "hilbert shape error"

    else:
        print "Model Name incorrectly entered"


def call_multilevel_cnn(dataset_path, model_save_path, args):
    """ Driver function to train CNN at a multiple levels in the
        taxonomy tree.

        This also trains all taxon nodes at that level in the tree.

        Args:
            dataset_path: Path where the Training and Validation datasets exist.
            model_save_path: Where the models need to be stored
            args: Input parameters to the neural networks

        Returns:
            None

    """
    context_len = args.context_len
    kmer_len = args.kmer_len
    fragment_len = args.fragment_len
    num_kernels = args.num_kernels
    drop_prob = args.drop_prob
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hilbert_shape = args.hilbert_shape
    use_multiprocessing = args.use_multiprocessing
    workers = args.workers
    model_name = args.model_name
    train_from_level = args.train_from_level
    train_upto_level = args.train_upto_level

    training_order = ["kingdom", "phylum", "class", "order",
                      "family", "genus", "species"]

    training_levels_subset = training_order[training_order.index(train_from_level):
                                            training_order.index(train_upto_level) + 1]

    levels_available_in_dataset = os.listdir(dataset_path)

    for level in training_levels_subset:
        if level in levels_available_in_dataset:
            print "Training at Taxon Level: {}".format(level)
            taxon_level = level
            for taxon_id in os.listdir(os.path.join(dataset_path, taxon_level, "db_creation")):
                print "Training at Taxon Node: {}".format(taxon_id)
                kmer_count = fragment_len - kmer_len + 1
                kernel_size = context_len - kmer_len + 1
                save_path = os.path.join(model_save_path, taxon_level, taxon_id)
                save_name = (str(fragment_len) +
                             "FragmentLength_" + str(kmer_len) + "mer_" +
                             taxon_level + "_" + taxon_id + "_" + str(context_len))

                hilbert_transform = []
                if hilbert_shape > 0:
                    hilbert_transform = hilbert_1d_transform(hilbert_shape)

                f_tr, f_te, t_tr, t_te, t2idx = load_dataset(dataset_path, taxon_level, taxon_id)
                fasta_records_train = f_tr
                fasta_records_test = f_te
                taxid_records_train = t_tr
                taxid_records_test = t_te
                taxid_to_0_index = t2idx

                num_classes_y = len(taxid_to_0_index)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                _pickle(taxid_to_0_index, os.path.join(save_path,
                        save_name + "_mappedclasses"))

                if model_name == 1:
                    Embedding_Sum_1D_CNN(save_path, save_name, fasta_records_train,
                                         taxid_records_train, fasta_records_test,
                                         taxid_records_test, taxid_to_0_index, taxon_level,
                                         taxon_id, num_classes_y, context_len, kmer_len,
                                         fragment_len, kmer_count, kernel_size, num_kernels,
                                         drop_prob, batch_size, num_epochs,
                                         use_multiprocessing, workers)
                elif model_name == 2:
                    Sentence_Matrix_2D_CNN(save_path, save_name, fasta_records_train,
                                           taxid_records_train, fasta_records_test,
                                           taxid_records_test, taxid_to_0_index,
                                           taxon_level, taxon_id, num_classes_y,
                                           context_len, kmer_len, fragment_len, kmer_count,
                                           kernel_size, num_kernels, drop_prob, batch_size,
                                           num_epochs, use_multiprocessing, workers)
                elif model_name == 3:
                    if hilbert_shape > 0:
                        Hilbert_Transform_3D_CNN(save_path, save_name, fasta_records_train,
                                                 taxid_records_train, fasta_records_test,
                                                 taxid_records_test, taxid_to_0_index,
                                                 taxon_level, taxon_id, num_classes_y,
                                                 context_len, kmer_len, fragment_len,
                                                 kmer_count, kernel_size, num_kernels,
                                                 drop_prob, batch_size, num_epochs,
                                                 hilbert_shape, hilbert_transform,
                                                 use_multiprocessing, workers)
                    else:
                        print "hilbert shape error"

                else:
                    print "Model Name incorrectly entered"

# --------------------- #
#  Inference Functions  #
# --------------------- #


def get_paths_to_models_and_mappings(path_to_models_and_mapping):
    """ Function that returns a list of all trained keras models
        and the mapping between the Taxon IDs and 0 based indices

        Args:
            path_to_models_and_mapping: Path where models and pickled mapping
                                        objects are stored.

        Returns:
            Path to Models and TaxID to 0 index mappings

    """
    path_to_models = []
    paths_to_model_class_mapping = []

    for outer_dir in os.listdir(path_to_models_and_mapping):
        for inner_dir in os.listdir(os.path.join(path_to_models_and_mapping,
                                    outer_dir)):
            for filename in os.listdir(os.path.join(path_to_models_and_mapping,
                                       outer_dir, inner_dir)):
                if filename.endswith('.h5'):
                    path_to_models.append(os.path.join(path_to_models_and_mapping,
                                          outer_dir, inner_dir, filename))
                else:
                    paths_to_model_class_mapping.append(os.path.join(
                                                        path_to_models_and_mapping,
                                                        outer_dir, inner_dir,
                                                        filename))
    return path_to_models, paths_to_model_class_mapping


def predict_results_on_tree(x_test_sample, predict_upto_taxon_level,
                            loaded_models, loaded_class_mappings,
                            predict_at_taxid, y_pred_classes, y_pred_probs):
    """ Recursive function to predict organism at every level in the taxonomy
        structure.

        Note: This depends on the models being present in the first place

        Args:
            x_test_sample: One sequence to be predicted
            predict_upto_taxon_level: Stopping criterion. Taxonomy node to stop at.
            loaded_models: References to models that have already been loaded.
            loaded_class_mappings: Mapping from TaxID to 0 based index
            predict_at_taxid: Next Neural Network model to load and predict on.
            y_pred_classes: Running list of predicted TaxIDs
            y_pred_probs: Running list of probabilities of each predicted label.

        Returns:
            List of predicted classes
            List of predicted class probabilities

    """
    model_cnn = loaded_models[predict_at_taxid]
    y_prob = model_cnn.predict(x_test_sample)
    y_pred_class = y_prob.argmax(axis=-1)

    class_maps = loaded_class_mappings[predict_at_taxid]
    inverted_dict = dict([[v, k] for k, v in class_maps.items()])
    predict_at_next_taxid = inverted_dict[y_pred_class[0]]

    y_pred_classes.append(predict_at_next_taxid)
    y_pred_probs.append(y_prob[0][y_pred_class][0])

    if ncbi.get_rank([predict_at_taxid]).values()[0] == predict_upto_taxon_level:
        return y_pred_classes[-1], y_pred_probs[-1]
    else:
        return predict_results_on_tree(x_test_sample,
                                       predict_upto_taxon_level,
                                       loaded_models,
                                       loaded_class_mappings,
                                       predict_at_next_taxid,
                                       y_pred_classes,
                                       y_pred_probs)


def generate_next_batch(x_test, kmer_len, fragment_len, start_at, end_at, word_index):
    """ Function that generates next batch of testing samples.

        Args:
            x_test: Loaded Testing dataset with input sequences
            kmer_len: K-mer lengths to use. This has to be the same as what
                      was used during training.
            fragment_len: Input fragment to use. This has to be the same as what
                      was used during training.
            start_at: Input index to start creating a batch at
            end_at: Input index where batch creation stops
            word_index: Dictionary mapping k-mer to word index in embedding matrix

        Returns:
            Batch of inputs

    """
    kmer_count = fragment_len - kmer_len + 1
    batch_features = np.zeros((end_at - start_at, kmer_count), dtype=np.int32)

    for i in range(start_at, end_at):
        batch_features[i - start_at] = get_kmers_index(str(x_test[i].seq),
                                                       word_index,
                                                       kmer_len)
    return batch_features


def predict_result_on_tree_initializer(model_dir, test_dir, args):
    """ Driver function to initiate prediction.

        Args:
            model_dir: Directory holding all models
            test_dir: Directory holding the testing dataset
            args: Parameters used for prediction

        Returns:
            None

    """
    predict_from_taxid = args.predict_from_taxid
    kmer_len = args.kmer_len
    fragment_len = args.fragment_len
    predict_upto_taxon_level = args.predict_upto_taxon_level

    batch_size = 32
    loaded_models = dict()
    loaded_class_mappings = dict()

    x_test = load_test_dataset(test_dir)

    path_to_models, paths_to_model_class_mapping = get_paths_to_models_and_mappings(model_dir)

    for i in range(len(path_to_models)):
        name_as_taxid = int(path_to_models[i].split('_')[-2])
        loaded_models[name_as_taxid] = load_model(path_to_models[i])

    for i in range(len(paths_to_model_class_mapping)):
        name_as_taxid = int(paths_to_model_class_mapping[i].split('_')[-3])
        loaded_class_mappings[name_as_taxid] = _unpickle(paths_to_model_class_mapping[i])

    word_index = create_embedding_matrix(kmer_len, predict=True)

    final_predictions = []
    final_probabilities = []

    for i in tqdm(range(0, len(x_test), batch_size)):
        start_at = i
        if i + batch_size < len(x_test):
            end_at = i + batch_size
        else:
            end_at = len(x_test)
        _batch = generate_next_batch(x_test, kmer_len, fragment_len,
                                     start_at, end_at, word_index)

        for sample_in_batch in _batch:
            y_pred_classes, y_pred_prob = predict_results_on_tree(
                                          np.reshape(sample_in_batch, (1, -1)),
                                          predict_upto_taxon_level,
                                          loaded_models,
                                          loaded_class_mappings,
                                          predict_from_taxid, [], [])
            final_predictions.append(y_pred_classes)
            final_probabilities.append(y_pred_prob)

    with open(os.path.join(test_dir, "FunGNet_Predicted_Labels.taxid"), "a+") as fopen:
        for i in final_predictions:
            fopen.write(str(i) + "\n")


# ------------------------------ #
#  Pythonic Interface Functions  #
# ------------------------------ #


class ArgClass:
    """ So that I don't have to duplicate argument info when
        the same thing is used in more than one mode."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def main(argv):
    # Letters taken: a, b, c, d, e, f, g, i, j, k, l, m, n, o, p, s, t, u, w
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    taxon_level = ArgClass("-e", "--taxon-level",
                           help="taxonomy level to work on",
                           default="superkingdom")

    final_taxon_level = ArgClass("-a", "--final-taxon-level",
                                 help="final taxon level at which dataset\
                                 must be created",
                                 default="phylum")

    taxon_id = ArgClass("-t", "--taxon-id",
                        help="taxonomy ID to work on",
                        type=int,
                        default=2)

    context_len = ArgClass("-f", "--context-len",
                           help="number of context nucleotides ahead the CNN kernels should train on",
                           type=int,
                           default=15)

    kmer_len = ArgClass("-k", "--kmer-len",
                        help="length of overlapping kmers to extract from raw DNA fragment",
                        type=int,
                        default=6)

    fragment_len = ArgClass("-l", "--fragment-len",
                            help="length of fragments to be drawn from fasta",
                            type=int,
                            default=100)

    num_kernels = ArgClass("-n", "--num-kernels",
                           help="number of kernels to use in 2D Convolution layers",
                           type=int,
                           default=256)

    use_multiprocessing = ArgClass("-m", "--use-multiprocessing",
                                   help="if true, use multiple threads to generate batches \
                                   of training data in fit_generator",
                                   type=bool,
                                   default=False)

    workers = ArgClass("-w", "--workers",
                       help="number of threads generating batches of training data \
                       in fit_generator",
                       type=int,
                       default=1)

    drop_prob = ArgClass("-d", "--drop-prob",
                         help="fraction of neurons to dropout",
                         type=float,
                         default=0.5)

    batch_size = ArgClass("-b", "--batch-size",
                          help="minibatch size",
                          type=int,
                          default=256)

    model_name = ArgClass("-c", "--model-name",
                             help="CNN model to use \
                             1. Sum of Embeddings + 1D CNN, \
                             2. Embedding Sentence Matrix + 2D CNN, \
                             3. Hilbert Curve Transformation + 3D CNN",
                             type=int,
                             default=2)

    hilbert_shape = ArgClass("-i", "--hilbert-shape",
                             help="hilbert curve square side length.\
                             E.g: h=16 => 16x16 matrix. \
                             Use 0 to choose FunGNet 2D CNN. \
                             Use number greater than 0 for Hilbert Transformed\
                             CNN",
                             type=int,
                             default=0)

    num_epochs = ArgClass("-o", "--num-epochs",
                          help="number of epochs to train",
                          type=int,
                          default=100)

    split_ratio = ArgClass("-s", "--split-ratio",
                           help="fraction of each contig to use in \
                           neural netork training step",
                           type=float,
                           default=0.8)

    predict_from_taxid = ArgClass("-p", "--predict-from-taxid",
                                  help="taxon ID from which prediction must begin",
                                  type=int,
                                  default=2)

    predict_upto_taxon_level = ArgClass("-u", "--predict-upto-taxon-level",
                                        help="taxonomy level to stop prediction at",
                                        type=str,
                                        default="genus")

    train_from_level = ArgClass("-g", "--train-from-level",
                                        help="train from taxon level",
                                        type=str,
                                        default="genus")

    train_upto_level = ArgClass("-j", "--train-upto-level",
                                        help="train upto taxon level",
                                        type=str,
                                        default="genus")

    subparsers = parser.add_subparsers(help="sub-commands", dest="mode")

    parser_build = subparsers.add_parser("build", help="Build FunGNet",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_build.add_argument("genomes_dataset_path", help="directory holding all \
                              whole training genomes")
    parser_build.add_argument("dataset_path", help="directory to store \
                               hierarchy of datasets \"family\", \"genus\" etc.")
    parser_build.add_argument(*fragment_len.args, **fragment_len.kwargs)
    parser_build.add_argument(*final_taxon_level.args, **final_taxon_level.kwargs)
    parser_build.add_argument(*split_ratio.args, **split_ratio.kwargs)

    parser_train = subparsers.add_parser("train", help="Train FunGNet",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train.add_argument("dataset_path", help="<directory_name> holding \"family\", \"genus\" etc")
    parser_train.add_argument("model_save_path", help="Directory to save hierarchy of FunGNet models")
    parser_train.add_argument(*taxon_level.args, **taxon_level.kwargs)
    parser_train.add_argument(*taxon_id.args, **taxon_id.kwargs)
    parser_train.add_argument(*context_len.args, **context_len.kwargs)
    parser_train.add_argument(*kmer_len.args, **kmer_len.kwargs)
    parser_train.add_argument(*fragment_len.args, **fragment_len.kwargs)
    parser_train.add_argument(*num_kernels.args, **num_kernels.kwargs)
    parser_train.add_argument(*drop_prob.args, **drop_prob.kwargs)
    parser_train.add_argument(*batch_size.args, **batch_size.kwargs)
    parser_train.add_argument(*num_epochs.args, **num_epochs.kwargs)
    parser_train.add_argument(*model_name.args, **model_name.kwargs)
    parser_train.add_argument(*hilbert_shape.args, **hilbert_shape.kwargs)
    parser_train.add_argument(*use_multiprocessing.args, **use_multiprocessing.kwargs)
    parser_train.add_argument(*workers.args, **workers.kwargs)

    multilevel_training = subparsers.add_parser("multilevel-training",
                                                help="Train FunGNet at multiple levels in the taxon tree.",
                                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    multilevel_training.add_argument("dataset_path", help="<directory_name> holding \"family\", \"genus\" etc")
    multilevel_training.add_argument("model_save_path", help="Directory to save hierarchy of FunGNet models")
    multilevel_training.add_argument(*train_from_level.args, **train_from_level.kwargs)
    multilevel_training.add_argument(*train_upto_level.args, **train_upto_level.kwargs)
    multilevel_training.add_argument(*context_len.args, **context_len.kwargs)
    multilevel_training.add_argument(*kmer_len.args, **kmer_len.kwargs)
    multilevel_training.add_argument(*fragment_len.args, **fragment_len.kwargs)
    multilevel_training.add_argument(*num_kernels.args, **num_kernels.kwargs)
    multilevel_training.add_argument(*drop_prob.args, **drop_prob.kwargs)
    multilevel_training.add_argument(*batch_size.args, **batch_size.kwargs)
    multilevel_training.add_argument(*num_epochs.args, **num_epochs.kwargs)
    multilevel_training.add_argument(*model_name.args, **model_name.kwargs)
    multilevel_training.add_argument(*hilbert_shape.args, **hilbert_shape.kwargs)
    multilevel_training.add_argument(*use_multiprocessing.args, **use_multiprocessing.kwargs)
    multilevel_training.add_argument(*workers.args, **workers.kwargs)

    parser_predict = subparsers.add_parser("predict", help="Predict source organisms, given trained FunGNet model",
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_predict.add_argument("model_dir", help="Input directory for hierarchy of FunGNet models")
    parser_predict.add_argument("test_dir", help="Input directory of fragmented test data")
    parser_predict.add_argument(*predict_from_taxid.args, **predict_from_taxid.kwargs)
    parser_predict.add_argument(*kmer_len.args, **kmer_len.kwargs)
    parser_predict.add_argument(*fragment_len.args, **fragment_len.kwargs)
    parser_predict.add_argument(*predict_upto_taxon_level.args,
                                **predict_upto_taxon_level.kwargs)

    args = parser.parse_args(argv)
    print(args)
    sys.stdout.flush()

    mode = args.mode

    if mode == "train":
        dataset_path = args.dataset_path
        model_save_path = args.model_save_path
        call_cnn(dataset_path, model_save_path, args)

    elif mode == "multilevel-training":
        dataset_path = args.dataset_path
        model_save_path = args.model_save_path
        call_multilevel_cnn(dataset_path, model_save_path, args)

    elif mode == "build":
        genomes_dataset_path = args.genomes_dataset_path
        dataset_path = args.dataset_path
        multilevel_dataset_creator.create_from_species(
                        genomes_dataset_path, dataset_path, "species",
                        args.final_taxon_level, args.fragment_len,
                        args.split_ratio, train_set=True)

    elif mode == "predict":
        model_dir = args.model_dir
        test_dir = args.test_dir
        predict_result_on_tree_initializer(model_dir, test_dir, args)


if __name__ == '__main__':
    main(sys.argv[1:])
