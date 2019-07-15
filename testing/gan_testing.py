import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from deepchem.utils.genomics import encode_fasta_sequence
from Bio import SeqIO
import h5py

# http://www.tiem.utk.edu/~gross/bioed/webmodules/aminoacid.htm
FREQUENCIES = {
    'A': 0.074,
    'R': 0.042,
    'N': 0.044,
    'B': 0.044,
    'D': 0.059,
    'C': 0.033,
    'E': 0.058,
    'Q': 0.037,
    'Z': 0.037,
    'G': 0.074,
    'H': 0.029,
    'I': 0.038,
    'L': 0.076,
    'J': 0.076,
    'K': 0.072,
    'M': 0.018,
    'F': 0.040,
    'P': 0.050,
    'S': 0.081,
    'T': 0.062,
    'W': 0.013,
    'Y': 0.033,
    'V': 0.068,
    'X': 0.030,
    '[': 0.017
}
FREQUENCY_ARRAY = [0 for i in range(27)]
for key in FREQUENCIES.keys():
    idx = ord(key) - ord('A')
    FREQUENCY_ARRAY[idx] = FREQUENCIES[key]

file = h5py.File('processed_data_525916981168.h5','r')
train_labels = file.get('train_labels').value
train_sequences = file.get('train_sequences_categorical').value
valid_labels = file.get('valid_labels').value
valid_sequences = file.get('valid_sequences_categorical').value
test_labels = file.get('test_labels').value
test_sequences = file.get('test_sequences_categorical').value
file.close()
train_sequences = np.array(train_sequences)
valid_sequences = np.array(valid_sequences)
test_sequences = np.array(test_sequences)

train_sequences_h1 = []
for i in range(len(train_sequences)):
    if np.argmax(train_labels[i]) == 1:
        train_sequences_h1.append(train_sequences[i])
train_sequences_h1 = np.array(train_sequences_h1)

def bio_loss(synthetic_categorical):
    synthetic_categorical = tf.reshape(synthetic_categorical,(max_size,27))
    synthetic_categorical = tf.cast(synthetic_categorical,tf.dtypes.int64)
    synthetic_letters = tf.map_fn(lambda y: tf.math.argmax(y,output_type=tf.dtypes.int32),synthetic_categorical,dtype=tf.dtypes.int32)
    frequencies = tf.bincount(synthetic_letters,dtype=tf.dtypes.float32,minlength=27)
    fivesevensix = tf.multiply(576.,tf.ones(27))
    percentages = tf.divide(frequencies,fivesevensix)
    frequency_loss = tf.norm(tf.math.subtract(percentages,tf.constant(FREQUENCY_ARRAY,dtype=tf.dtypes.float32)))
    
#     runs = tf.constant([])
#     ptr = tf.constant(0)
#     def body(ptr,runs):
#         j = tf.add(ptr,tf.constant(1))
#         current = synthetic_letters[ptr]
        
#         def body2(j):
#             j = tf.add(j,tf.constant(1))
#             return j
                
#         j = tf.while_loop(
#             lambda j: tf.logical_and(tf.less(j,tf.shape(synthetic_letters)[0]),tf.equal(current,synthetic_letters[j])),
#             body2,
#             [j]
#         )
    
#         ctr = tf.subtract(tf.subtract(j,ptr),1)
#         runs = tf.concat(runs,ctr,0)
#         ptr = j
        
#         return runs
        
#     runs_result = tf.while_loop(
#         lambda ptr,runs: tf.less(ptr,tf.shape(synthetic_letters)[0]),
#         body,
#         [ptr,runs]
#     )[1]
    
#     consecutive_loss = tf.reduce_mean(tf.subtract(tf.exp(tf.subtract(runs_result,1)),1))

    def loss(output,target_label):
        return tf.keras.backend.binary_crossentropy(output,target_label) + 2*frequency_loss #+ consecutive_loss

    return loss

max_size = 576

#____________________________________________________________________________________________________

tf.keras.backend.clear_session()

seed = tf.keras.Input(shape=(270,))

x = tf.keras.layers.Dense(max_size,activation=tf.nn.relu)(seed)
x = tf.keras.layers.Reshape((max_size,1))(x)
x = tf.keras.layers.Conv1D(64,3,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv1D(64,3,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv1D(64,3,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv1D(27,3,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
synthetic = tf.keras.layers.Reshape((max_size,27))(x)

sequence = tf.keras.Input(shape=(max_size,27))
x = tf.keras.layers.Reshape((max_size*27,1))(sequence)
x = tf.keras.layers.Conv1D(256,24,strides=24,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.Conv1D(16,3,strides=3,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.Reshape((216*16,))(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)

generator1 = tf.keras.Model(seed,synthetic)
generator1.compile(loss='binary_crossentropy', optimizer='rmsprop')

discriminator1 = tf.keras.Model(sequence,output)
discriminator1.compile(loss='binary_crossentropy', optimizer='rmsprop')

s = generator(seed)
discriminator.trainable = False
prediction = discriminator(s)

combined1 = tf.keras.Model(seed,prediction)
combined1.compile(loss=bio_loss(combined.layers[1].layers[11].output), optimizer='rmsprop')

#__________

tf.keras.backend.clear_session()

seed = tf.keras.Input(shape=(270,))

x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(seed)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(max_size, activation=tf.nn.relu)(x)
x = tf.keras.layers.Reshape((max_size,1))(x)
synthetic = tf.keras.layers.Conv1D(27,16,activation=tf.nn.softmax,padding='same')(x)

sequence = tf.keras.Input(shape=(max_size,27))
x = tf.keras.layers.Reshape((max_size*27,1))(sequence)
x = tf.keras.layers.Conv1D(256,24,strides=24,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.Conv1D(16,3,strides=3,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.Reshape((216*16,))(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)

generator2 = tf.keras.Model(seed,synthetic)
generator2.compile(loss='binary_crossentropy', optimizer='rmsprop')

discriminator2 = tf.keras.Model(sequence,output)
discriminator2.compile(loss='binary_crossentropy', optimizer='rmsprop')

s = generator(seed)
discriminator.trainable = False
prediction = discriminator(s)

combined2 = tf.keras.Model(seed,prediction)
combined2.compile(loss=bio_loss(combined.layers[1].layers[11].output), optimizer='rmsprop')

#__________

tf.keras.backend.clear_session()

seed = tf.keras.Input(shape=(270,))

x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(seed)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(max_size, activation=tf.nn.relu)(x)
x = tf.keras.layers.Reshape((max_size,1))(x)
synthetic = tf.keras.layers.Conv1D(27,16,activation=tf.nn.softmax,padding='same')(x)

sequence = tf.keras.Input(shape=(max_size,27))
x = tf.keras.layers.Reshape((max_size*27,1))(sequence)
x = tf.keras.layers.Conv1D(256,24,strides=24,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.Conv1D(16,3,strides=3,activation=tf.nn.relu,padding='same')(x)
x = tf.keras.layers.Reshape((216*16,))(x)
x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)

generator3 = tf.keras.Model(seed,synthetic)
generator3.compile(loss='binary_crossentropy', optimizer='rmsprop')

discriminator3 = tf.keras.Model(sequence,output)
discriminator3.compile(loss='binary_crossentropy', optimizer='rmsprop')

s = generator(seed)
discriminator.trainable = False
prediction = discriminator(s)

combined3 = tf.keras.Model(seed,prediction)
combined3.compile(loss=bio_loss(combined.layers[1].layers[11].output), optimizer='rmsprop')

#____________________________________________________________________________________________________

ground_real = np.array([np.array([1,0]) for i in range(18000)])

synthetic_labels = np.array([(0,0.9+0.1*np.random.rand()) for i in range(9000)])
real_labels = np.array([np.array([0.9+0.1*np.random.rand(),0]) for i in range(9000)])
# flips1 = [np.random.randint(0,900) for i in range(20)]
# flips2 = [np.random.randint(0,10) for i in range(20)]
# for i in range(20):
#     synthetic_labels[flips1[i]][flips2[i]],real_labels[flips1[i]][flips2[i]] = real_labels[flips1[i]][flips2[i]],synthetic_labels[flips1[i]][flips2[i]]

epochs = 20
for epoch in range(epochs):
    rand = np.random.normal(0,1,(18000,270))
    
    combined1.fit(rand,ground_real,batch_size=1)
    combined2.fit(rand,ground_real,batch_size=1)
    combined3.fit(rand,ground_real,batch_size=1)
    
    rand = np.random.normal(0,1,(18000,270))
    
    sequences = np.concatenate((train_sequences_h1[:9000].reshape(9000,576,27),generator.predict(rand)[:9000]),axis=0)
    labels = np.concatenate((real_labels,synthetic_labels),axis=0)
    
    discriminator1.fit(sequences,labels,batch_size=1)
    discriminator2.fit(sequences,labels,batch_size=1)
    discriminator3.fit(sequences,labels,batch_size=1)


    
predictions = [generator1.predict(np.random.normal(0,1,(1,270)))[0],generator2.predict(np.random.normal(0,1,(1,270)))[0],generator3.predict(np.random.normal(0,1,(1,270)))[0]]

sequence_strings = ['','','']

for i in range(len(predictions[0])):
    residues = [chr(np.argmax(prediction[j][i])+ord('A')) for j in range(3)]
    for j in range(3):
	    if residues[j] == 'J':
		residues[j] = 'L'
	    if residues[j] == 'B':
		residues[j] = 'N'
	    if residues[j] == 'Z':
		residues[j] = 'Q'
	    sequence_strings[j] += residues[j]
print(sequence_strings)
