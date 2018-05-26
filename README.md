# Convolutional Neural Networks for Organism Prediction

Author: Shreyas Ramesh - <shreyas2@vt.edu>

### The program requires
    1. Python 2.7
    2. Keras with the TensorFlow backend
    3. h5py and its dependencies


## Contents

    /data
      /all_genomes -- Contains a small set of bacterial genomes to work with
      /evaluation_set
        -> fragment-len_100_test_data.fasta -- A small file to test your models
        -> ground_truth.taxid -- Labels for the associated ".fasta" file
      /multilevel_dataset_store -- Directory to store training datasets after running 'build' mode on FunGNet

    /utils
      -> get_all_genomes_under_taxid.py -- Select subset of genomes under a TAXID
      -> multilevel_dataset_creator.py -- Functions used by 'build' mode in FunGNet.py
      -> opal_trainer.py -- Small script to automate Opal training
      -> opal_predictor.py -- Small script to automate Opal prediction
      -> taxonomy_assessment_tool.py -- Model independent evaluation script

    /dna2vec_embedding_store -- Contains a pre-trained Word2Vec embedding file
    /experimental_work -- Contains incomplete/unsucessful/under development models
    /model_store -- Directory to store trained CNN models

    -> FunGNet_Classes.py -- Contains the DataGenerator class used by the Keras 'fit_generator' function.
    -> FunGNet.py -- Main program
    -> GUIDE.txt -- Provides a set of basic commands to get started with
    -> requirements.txt -- Set of Python packages required by the program.
                     Install with: pip install -r requirements.txt


## Test Environment

  All experiments were conducted on Adam

      Hardware Specifications
        1. Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
        2. 8 Physical Cores, 2 Threads per Core
        3. 64GB Memory
        4. 2x NVIDIA (R) 1080Ti, 2x12GB Memory

      Software Specifications
        1. Operating System: Ubuntu 16.04.5
        2. Python 2.7.11
        3. NVIDIA (R) Cuda compiler driver V8.0.61


## Directions for running experiments

    Step 1: Extract tarball within /dna2vec_embedding_store into the same directory
    Step 2: Extract tarball within /data/evaluation_set/ into the same directory
 
### The four modes in which experiments can be run are:
    1. build - Build datasets
    2. train - Train a single neural network model at a specific Taxon Node (Eg. TaxID: 1301)
    3. multilevel-training - Automates training of neural nets at more than one Taxon Node. This mode comes in handy if you either want to train on TaxonIDs at a few levels (Eg. class and family), or even on all levels under say, Bacteria.
    4. predict - Given trained models and a testing dataset, the program predicts TaxIDs upto a user specified Taxonomy Level (Eg. Upto Taxon level: species)

Notes:

  1. Please ensure that fragment-len is the same in all modes (build, train predict).
     If fragment-len is not the same across modes, it can lead to unpredictable results.

  2. Please ensure kmer-len is the same in all modes.
     If fragment-len is not the same across modes, it can lead to unpredictable results.


## Examples to get you started:

### Building Datasets for Classification:
    python -W ignore FunGNet.py build --fragment-len 261 --final-taxon-level class --split-ratio 0.8 data/all_genomes/ data/multilevel_dataset_store/


### Training a Embedding Sum based 1D CNN at Genus 1301:
    python -W ignore FunGNet.py train --taxon-level genus --taxon-id 1301 --context-len 15 --kmer-len 6 --fragment-len 261 --num-kernels 256 --drop-prob 0.5 --batch-size 256 --num-epochs 100 --model-name 1 --use-multiprocessing True --workers 6 data/multilevel_dataset_store/ model_store/


### Training a Sentence Matrix based 2D CNN at Genus 1301:
    python -W ignore FunGNet.py train --taxon-level genus --taxon-id 1301 --context-len 15 --kmer-len 6 --fragment-len 261 --num-kernels 256 --drop-prob 0.5 --batch-size 256 --num-epochs 100 --model-name 2 --use-multiprocessing True --workers 6 data/multilevel_dataset_store/ model_store/


### Training a Hilbert Transformation based 3D CNN at Genus 1301:
    python -W ignore FunGNet.py train --taxon-level genus --taxon-id 1301 --context-len 15 --kmer-len 6 --fragment-len 261 --num-kernels 256 --drop-prob 0.5 --batch-size 256 --num-epochs 100 --model-name 3 --hilbert-shape 16 --use-multiprocessing True --workers 6 data/multilevel_dataset_store/ model_store/


### Multilevel Training:
    python -W ignore FunGNet.py multilevel-training --train-from-level class --train-upto-level species --context-len 15 --kmer-len 6 --fragment-len 261 --num-kernels 256 --drop-prob 0.5 --batch-size 256 --num-epochs 100 --model-name 2 --use-multiprocessing True --workers 6 data/multilevel_dataset_store/ model_store/


### Predict Using a Trained Model at Genus 1301:
Note: Please ensure that you have a dataset where each sequence is of the same length as the sequence used for training. For ex: The test sequence has to be of length 261 if the models were trained using any of the commands from the previous sections.

    python -W ignore FunGNet.py predict --predict-from-taxid 1301 --kmer-len 6 --fragment-len 261 --predict-upto-taxon-level genus model_store/ data/evaluation_set/


### Multilevel Prediction:
    python -W ignore FunGNet.py predict --predict-from-taxid 91061 --kmer-len 6 --fragment-len 261 --predict-upto-taxon-level species model_store/ data/evaluation_set/
