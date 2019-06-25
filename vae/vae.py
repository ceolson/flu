import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from deepchem.utils.genomics import encode_fasta_sequence
from Bio import SeqIO
import h5py

ORDER = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','<EOM>','<SOM>']
EOM_TENSOR = tf.constant([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]])
SOM_TENSOR = tf.constant([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]])
EOM_VECTOR = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]).astype('float32')
SOM_VECTOR = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]).astype('float32')
	
### Limit GPU memory used by tf
print("Limit GPU memory")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
config.log_device_placement = False

sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)
# ~ sess = tf_debug.LocalCLIDebugWrapperSession(sess)

### Prep data

# REAL DATA COMMENTED OUT __________________________________

# ~ file = h5py.File('/projects/ml/flu/processed_data_525916981168.h5','r')
# ~ train_labels = file.get('train_labels').value
# ~ train_sequences = file.get('train_sequences_categorical').value
# ~ valid_labels = file.get('valid_labels').value
# ~ valid_sequences = file.get('valid_sequences_categorical').value
# ~ test_labels = file.get('test_labels').value
# ~ test_sequences = file.get('test_sequences_categorical').value
# ~ file.close()

# ~ train_sequences = np.array(train_sequences)
# ~ valid_sequences = np.array(valid_sequences)
# ~ test_sequences = np.array(test_sequences)



# ~ train_sequences_h1 = []
# ~ for i in range(len(train_sequences)):
    # ~ if np.argmax(train_labels[i]) == 1:
        # ~ train_sequences_h1.append(train_sequences[i])
# ~ train_sequences_h1 = np.array(train_sequences_h1)

# FAKE DATA (ONE MUTATED RESIDUE IN A FIXED SEQUENCE) HERE

data_file = h5py.File('/home/ceolson0/Documents/test_fastas3.h5','r')
train_sequences_h1 = data_file.get('sequences_cat').value
data_file.close()

train_sequences_h1_offset = []
for i in range(len(train_sequences_h1)):
	seq = []
	for j in range(len(train_sequences_h1[0]) - 1):
		seq.append(train_sequences_h1[i][j+1])
	seq.append(EOM_VECTOR)
	train_sequences_h1_offset.append(seq)
train_sequences_h1_offset = np.array(train_sequences_h1_offset)


max_size = len(train_sequences_h1[0])
encode_length = len(train_sequences_h1[0][0])
batch_size=500

latent_dim = 100

def sample_from_latents(x):
	means = x[:,:latent_dim]
	log_vars = x[:,latent_dim:]
	base = tf.keras.backend.random_normal(shape=[latent_dim,])
	return means + tf.exp(log_vars) * base
	
def custom_loss(latent_h_seeds,latent_c_seeds):
	means_h = latent_h_seeds[:,:latent_dim]
	log_vars_h = latent_h_seeds[:,latent_dim:]
	
	means_c = latent_c_seeds[:,:latent_dim]
	log_vars_c = latent_c_seeds[:,latent_dim:]
	
	kl_h = tf.reduce_mean(tf.square(means_h) + tf.exp(log_vars_h) - log_vars_h - 1.) * 0.25
	kl_c = tf.reduce_mean(tf.square(means_c) + tf.exp(log_vars_c) - log_vars_c - 1.) * 0.25
	kl = kl_h + kl_c
	
	def loss(outs,labels):
		return kl + tf.keras.backend.categorical_crossentropy(outs,labels)
	
	return loss
	

encoder_input = tf.keras.Input(shape=[None,encode_length])
o,h,c = tf.keras.layers.LSTM(latent_dim,return_state=True)(encoder_input)

x = tf.keras.layers.Flatten()(h)
latent_h_seeds = tf.keras.layers.Dense(latent_dim*2)(x)
latent_h = tf.keras.layers.Lambda(sample_from_latents)(latent_h_seeds)

x = tf.keras.layers.Flatten()(c)
latent_c_seeds = tf.keras.layers.Dense(latent_dim*2)(x)
latent_c = tf.keras.layers.Lambda(sample_from_latents)(latent_c_seeds)

decoder_input = tf.keras.Input(shape=[None,encode_length])
out,_,_ = tf.keras.layers.LSTM(latent_dim,return_sequences=True,return_state=True)(decoder_input,initial_state=[latent_h,latent_c])
out = tf.keras.layers.Dense(encode_length,activation='softmax')(out)

model = tf.keras.Model([encoder_input,decoder_input],out)
model.compile(optimizer='rmsprop',loss=custom_loss(latent_h_seeds,latent_c_seeds))
model.fit([train_sequences_h1,train_sequences_h1],train_sequences_h1_offset,epochs=50)

encoder = tf.keras.Model(encoder_input,[latent_h,latent_c])
decoder = tf.keras.Model([decoder_input,latent_h,latent_c],out)

def reconstruct(sequence):
	state_h,state_c = encoder.predict(sequence)[0]
	r = []
	h,c = state_h,state_c
	new_sequence = np.zeros([1,1,encode_length])
	new_sequence[0,0,-1] = 1.
	while (np.argmax(r[-1]) != np.argmax(EOM_VECTOR) and len(r) < 1000):
		output,h,c = decoder.predict([sequence] + h + c)
		predicted_character = ORDER[np.argmax(output[0,-1,:])]
		r.append(predicted_character)
		new_sequence = np.zeros([1,1,encode_length])
		new_sequence[0,0,np.argmax(output[0,-1,:])] = 1.
	return r

print(train_sequences_h1[0:1])
print(reconstruct(train_sequences_h1[0:1]))		

