import numpy as np
import keras

aa2int = dict({
    'A': 0,
    'T': 1,
    'G': 2,
    'C': 3
    })


int2bp = dict()
for k, v in aa2int.items():
    int2bp[v] = k


def ids2seq(ids):
    """
    Args: ids: Sequence of integers [0 (A), 1 (T), 2 (G), 3 (C)].

    Returns: Nucleotide sequence with length equal to input sequence.

    """
    sequence = [int2bp.get(x, 'A') for x in ids]
    return ''.join(sequence)


def get_kmers_index(sequence, word_index, kmer_len=6):
    """
    Args: sequence: A Nucleotide Sequence.
           word_index: k-mer to word index (Used in Embedding Matrix) Dictionary.
           kmer_len: Length of k-mers to lookup.

    Returns: List of word indexes for every kmer in nucleotide sequence.

    """
    assert kmer_len > 2 and kmer_len < 9
    kmers = [word_index.get(sequence[i: i + kmer_len], 0)
             for i in range(0, len(sequence)) if i <= len(sequence) - kmer_len]
    return kmers


def get_hilbert_mapping(word_mapped_kmers, hilbert_1d_indexes):
    return word_mapped_kmers[hilbert_1d_indexes]


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, features, labels, hilbert_transform,
                 taxid_to_0_index=[], batch_size=256, kmer_count=95,
                 num_classes=10, shuffle=True, kmer_len=6, word_index=[]):
        'Initialization'
        self.kmer_len = kmer_len
        self.kmer_count = kmer_count
        self.batch_size = batch_size
        self.labels = labels
        self.word_index = word_index
        self.features = features
        self.hilbert_transform = hilbert_transform
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(features))
        self.taxid_to_0_index = taxid_to_0_index
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.features) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        features_temp = [self.features[k].seq for k in indexes]
        labels_temp = [self.taxid_to_0_index[self.labels[k]] for k in indexes]

        # Generate data
        X, y = self.__data_generation(features_temp, labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.features))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, features_temp, labels_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.kmer_count), dtype=int)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i in range(len(features_temp)):
            assert len(features_temp) == self.batch_size
            # Store sample
            X[i] = get_kmers_index(features_temp[i],
                                   self.word_index, self.kmer_len)
            y[i] = labels_temp[i]

        if len(self.hilbert_transform) > 0:
            for i in range(len(features_temp)):
                X[i] = get_hilbert_mapping(X[i], self.hilbert_transform)

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)

    def generate(self):
        'Generates batches of DNA strings'
        # Infinite loop
        while 1:
            # Generate batches
            imax = int(len(self.features)/self.batch_size)
            for i in range(imax):
                X, y = self.__getitem__(i)
                yield X, y
