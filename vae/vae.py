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
print('Limit GPU memory')
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
train_sequences_h1_unprocessed = data_file.get('sequences_cat').value
data_file.close()

# ~ train_sequences_h1_offset = []
# ~ for i in range(len(train_sequences_h1)):
	# ~ seq = []
	# ~ for j in range(len(train_sequences_h1[0]) - 1):
		# ~ seq.append(train_sequences_h1[i][j+1])
	# ~ seq.append(EOM_VECTOR)
	# ~ train_sequences_h1_offset.append(seq)
# ~ train_sequences_h1_offset = np.array(train_sequences_h1_offset)

train_sequences_h1 = []
for i in range(len(train_sequences_h1_unprocessed)):
	seq = train_sequences_h1_unprocessed[i]
	seq = np.concatenate((SOM_VECTOR.reshape([1,22]),seq),axis=0)
	seq = np.concatenate((seq,EOM_VECTOR.reshape([1,22])),axis=0)
	train_sequences_h1.append(seq)
train_sequences_h1 = np.array(train_sequences_h1)




max_size = len(train_sequences_h1[0])
encode_length = len(train_sequences_h1[0][0])
batch_size=1
latent_dim = 100

print(np.shape(train_sequences_h1))

def sample_from_latents(x):
	means = x[:,:latent_dim]
	log_vars = x[:,latent_dim:]
	base = tf.keras.backend.random_normal(shape=[latent_dim,])
	return means + tf.exp(log_vars) * base
	
def dense(matrix,bias,in_dim,out_dim,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		W = tf.get_variable(matrix,trainable=True,shape=[in_dim,out_dim])
		b = tf.get_variable(bias,trainable=True,shape=[out_dim,])

	
	return tf.matmul(in_tensor,W) + b	
	
def encoder(sequence,lstm):	
	out,h,c = lstm(sequence)
	return [h,c]
	
def decoder(state,so_far,lstm):
	out = lstm(so_far,initial_state=state)
	logits = dense('decoder.dense.matrix','decoder.dense.bias',latent_dim,encode_length,out)
	return logits
	
encoder_lstm = tf.keras.layers.CuDNNLSTM(latent_dim*2,return_state=True)
decoder_lstm = tf.keras.layers.CuDNNLSTM(latent_dim)

sequence_in = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
so_far_reconstructed = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
correct_labels = tf.placeholder(shape=[batch_size,encode_length],dtype=tf.dtypes.float32)

latent_seeds_h,latent_seeds_c = encoder(sequence_in,encoder_lstm)
latent_h = sample_from_latents(latent_seeds_h)
latent_c = sample_from_latents(latent_seeds_c)
latent = [latent_h,latent_c]

logits = decoder(latent,so_far_reconstructed,decoder_lstm)
predicted_character = tf.nn.softmax(logits)

accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels,logits)

def compute_kl_loss(latent_h_seeds,latent_c_seeds):
	means_h = latent_h_seeds[:,:latent_dim]
	log_vars_h = latent_h_seeds[:,latent_dim:]
	
	means_c = latent_c_seeds[:,:latent_dim]
	log_vars_c = latent_c_seeds[:,latent_dim:]
	
	kl_h = tf.reduce_mean(tf.square(means_h) + tf.exp(log_vars_h) - log_vars_h - 1.) * 0.25
	kl_c = tf.reduce_mean(tf.square(means_c) + tf.exp(log_vars_c) - log_vars_c - 1.) * 0.25
	kl = kl_h + kl_c
	return kl

kl_loss = compute_kl_loss(latent_seeds_h,latent_seeds_c)

loss = tf.reduce_mean(accuracy_loss + 0.*kl_loss)

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())


saver = tf.train.Saver(tf.all_variables())


saver.restore(sess,'/home/ceolson0/Documents/models/vae')

print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


epochs = 10000
for epoch in range(epochs):
	stop_point = np.random.randint(2,max_size-1)
	batch_sequences = np.random.permutation(train_sequences_h1)[:batch_size,:1+stop_point]
	_,l = sess.run([train,loss],feed_dict={sequence_in:batch_sequences[:,:-1],so_far_reconstructed:batch_sequences[:,:-2],correct_labels:batch_sequences[:,-1]})
	if epoch%10==0: print('epoch',epoch,'loss',l)



saver.save(sess,'/home/ceolson0/Documents/models/vae')

def rec(sequence):
	new_sequence = np.zeros([1,1,encode_length])
	new_sequence[0,0,-1] = 1.
	
	new_sequence = sequence[:,:100,:]
	
	while (np.argmax(new_sequence[0,-1]) != np.argmax(EOM_VECTOR) and np.shape(new_sequence)[1] < 1000):
		character = sess.run(predicted_character,feed_dict={sequence_in:sequence,so_far_reconstructed:new_sequence})
		new_sequence = np.concatenate((new_sequence,character.reshape(1,1,encode_length)),axis=1)
	
	reconstructed_string = ''
	for i in range(np.shape(new_sequence)[1]):
		character = ORDER[np.argmax(new_sequence[0,i])]
		reconstructed_string += character
	
	return reconstructed_string

test = train_sequences_h1[100:101]
truth_string = ''
for residue in test[0]:
	character = ORDER[np.argmax(residue)]
	truth_string += character
recon_string = rec(test)

print(truth_string)
print(recon_string)


