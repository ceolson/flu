import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from deepchem.utils.genomics import encode_fasta_sequence
from Bio import SeqIO
from IPython.display import Markdown
import scipy
import h5py

np.set_printoptions(threshold=np.inf)


# https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
BLOSUM = {
'A':   [ 4,   -1,   -2,   -2,    0,   -1,   -1,    0,   -2,   -1,   -1,   -1,   -1,   -2,   -1,    1,    0,   -3,   -2,    0,   -2,   -1,    0,   -4],
'R':   [-1,    5,    0,   -2,   -3,    1,    0,   -2,    0,   -3,   -2,    2,   -1,   -3,   -2,   -1,   -1,   -3,   -2,   -3,   -1,    0,   -1,   -4],
'N':   [-2,    0,    6,    1,   -3,    0,    0,    0,    1,   -3,   -3,    0,   -2,   -3,   -2,    1,    0,   -4,   -2,   -3,    3,    0,   -1,   -4],
'D':   [-2,   -2,    1,    6,   -3,    0,    2,   -1,   -1,   -3,   -4,   -1,   -3,   -3,   -1,    0,   -1,   -4,   -3,   -3,    4,    1,   -1,   -4],
'C':   [ 0,   -3,   -3,   -3,    9,   -3,   -4,   -3,   -3,   -1,   -1,   -3,   -1,   -2,   -3,   -1,   -1,   -2,   -2,   -1,   -3,   -3,   -2,   -4],
'Q':   [-1,    1,    0,    0,   -3,    5,    2,   -2,    0,   -3,   -2,    1,    0,   -3,   -1,    0,   -1,   -2,   -1,   -2,    0,    3,   -1,   -4],
'E':   [-1,    0,    0,    2,   -4,    2,    5,   -2,    0,   -3,   -3,    1,   -2,   -3,   -1,    0,   -1,   -3,   -2,   -2,    1,    4,   -1,   -4],
'G':   [ 0,   -2,    0,   -1,   -3,   -2,   -2,    6,   -2,   -4,   -4,   -2,   -3,   -3,   -2,    0,   -2,   -2,   -3,   -3,   -1,   -2,   -1,   -4],
'H':   [-2,    0,    1,   -1,   -3,    0,    0,   -2,    8,   -3,   -3,   -1,   -2,   -1,   -2,   -1,   -2,   -2,    2,   -3,    0,    0,   -1,   -4],
'I':   [-1,   -3,   -3,   -3,   -1,   -3,   -3,   -4,   -3,    4,    2,   -3,    1,    0,   -3,   -2,   -1,   -3,   -1,    3,   -3,   -3,   -1,   -4],
'L':   [-1,   -2,   -3,   -4,   -1,   -2,   -3,   -4,   -3,    2,    4,   -2,    2,    0,   -3,   -2,   -1,   -2,   -1,    1,   -4,   -3,   -1,   -4],
'K':   [-1,    2,    0,   -1,   -3,    1,    1,   -2,   -1,   -3,   -2,    5,   -1,   -3,   -1,    0,   -1,   -3,   -2,   -2,    0,    1,   -1,   -4],
'M':   [-1,   -1,   -2,   -3,   -1,    0,   -2,   -3,   -2,    1,    2,   -1,    5,    0,   -2,   -1,   -1,   -1,   -1,    1,   -3,   -1,   -1,   -4],
'F':   [-2,   -3,   -3,   -3,   -2,   -3,   -3,   -3,   -1,    0,    0,   -3,    0,    6,   -4,   -2,   -2,    1,    3,   -1,   -3,   -3,   -1,   -4],
'P':   [-1,   -2,   -2,   -1,   -3,   -1,   -1,   -2,   -2,   -3,   -3,   -1,   -2,   -4,    7,   -1,   -1,   -4,   -3,   -2,   -2,   -1,   -2,   -4],
'S':   [ 1,   -1,    1,    0,   -1,    0,    0,    0,   -1,   -2,   -2,    0,   -1,   -2,   -1,    4,    1,   -3,   -2,   -2,    0,    0,    0,   -4],
'T':   [ 0,   -1,    0,   -1,   -1,   -1,   -1,   -2,   -2,   -1,   -1,   -1,   -1,   -2,   -1,    1,    5,   -2,   -2,    0,   -1,   -1,    0,   -4],
'W':   [-3,   -3,   -4,   -4,   -2,   -2,   -3,   -2,   -2,   -3,   -2,   -3,   -1,    1,   -4,   -3,   -2,   11,    2,   -3,   -4,   -3,   -2,   -4],
'Y':   [-2,   -2,   -2,   -3,   -2,   -1,   -2,   -3,    2,   -1,   -1,   -2,   -1,    3,   -3,   -2,   -2,    2,    7,   -1,   -3,   -2,   -1,   -4],
'V':   [ 0,   -3,   -3,   -3,   -1,   -2,   -2,   -3,   -3,    3,    1,   -2,    1,   -1,   -2,   -2,    0,   -3,   -1,    4,   -3,   -2,   -1,   -4],
'B':   [-2,   -1,    3,    4,   -3,    0,    1,   -1,    0,   -3,   -4,    0,   -3,   -3,   -2,    0,   -1,   -4,   -3,   -3,    4,    1,   -1,   -4],
'Z':   [-1,    0,    0,    1,   -3,    3,    4,   -2,    0,   -3,   -3,    1,   -1,   -3,   -1,    0,   -1,   -3,   -2,   -2,    1,    4,   -1,   -4],
'X':   [ 0,   -1,   -1,   -1,   -2,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -2,    0,    0,   -2,   -1,   -1,   -1,   -1,   -1,   -4],
'-':   [-4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,    1]
}

CATEGORIES = {
'A': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'R': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'N': [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'D': [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'C': [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'Q': [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'E': [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'G': [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'H': [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'I': [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'L': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'K': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'M': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'F': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
'P': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], 
'S': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
'T': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], 
'W': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], 
'Y': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], 
'V': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
'X': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
'-': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
}

ORDER = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X','-']


# Read om fastas
fastas = SeqIO.parse('/projects/ml/flu/fludb_data/525916981168-ProteinFastaResults.fasta','fasta')
arr = [line for line in fastas]
arr_new = []

# Reject sequences without a subtype
for elt in arr:
    try:
        int(elt.id[1])
        arr_new.append(elt)
    except:
        pass

# Pad sequences up to max length
max_size = max(len(elt.seq) for elt in arr_new)
count = 0
for i in range(len(arr_new)):
    count += max_size-len(arr_new[i].seq)
    arr_new[i].seq = np.pad(arr_new[i].seq, (0,max_size-len(arr_new[i].seq)), 'constant', constant_values='-')

# Convert sequences to categorical or BLOSUM encoding
sequences_cat = []
sequences_blosum = []

for elt in arr_new:
    sequence_cat = []
    sequence_blosum = []
    for letter in elt.seq:
        try:
            sequence_cat.append(CATEGORIES[letter])
        except:
            sequence_cat.append(CATEGORIES['X'])        # Replace any unknown characters with 'X'
        try:
            sequence_blosum.append(BLOSUM[letter])
        except:
            sequence_blosum.append(BLOSUM['X'])         # Replace any unknown characters with 'X'
    sequences_cat.append(sequence_cat)
    sequences_blosum.append(sequence_blosum)
sequences_cat = np.array(sequences_cat)
sequences_blosum = np.array(sequences_blosum)

# Read in subtype labels
labels = []
for elt in arr_new:
    number = 0
    try:
        number = int(elt.id[1:3])
    except:
        number = int(elt.id[1])
    labels.append(number)
labels = np.array(labels)

# Shuffle sequences
permutation = np.random.permutation(range(len(labels)))
labels = labels[permutation]
sequences_blosum = sequences_blosum[permutation]
sequences_cat = sequences_cat[permutation]


train_labels = keras.utils.to_categorical(labels[:40000],19)
train_sequences_blosum = sequences_blosum[:40000]
train_sequences_categorical = sequences_cat[:40000]

valid_labels = keras.utils.to_categorical(labels[40001:65000],19)
valid_sequences_blosum = sequences_blosum[40001:65000]
valid_sequences_categorical = sequences_cat[40001:65000]

test_labels = keras.utils.to_categorical(labels[65001:],19)
test_sequences_blosum = sequences_blosum[65001:]
test_sequences_categorical = sequences_cat[65001:]


file = h5py.File('/projects/ml/flu/fludb_data/processed_data_525916981168.h5','w')
file.create_dataset('train_labels', data=train_labels)
file.create_dataset('train_sequences_blosum', data=train_sequences_blosum)
file.create_dataset('train_sequences_categorical', data=train_sequences_categorical)
file.create_dataset('valid_labels', data=valid_labels)
file.create_dataset('valid_sequences_blosum', data=valid_sequences_blosum)
file.create_dataset('valid_sequences_categorical', data=valid_sequences_categorical)
file.create_dataset('test_labels', data=test_labels)
file.create_dataset('test_sequences_blosum', data=test_sequences_blosum)
file.create_dataset('test_sequences_categorical', data=test_sequences_categorical)
file.close()

