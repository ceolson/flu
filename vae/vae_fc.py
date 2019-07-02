import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from deepchem.utils.genomics import encode_fasta_sequence
from Bio import SeqIO
import h5py

ORDER = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X','[']
def convert_to_string(prediction):
	string = ''
	for i in range(len(prediction)):
		prediction[i][-2] = 0
		index = np.argmax(prediction[i])
		residue = ORDER[index]
		string += residue
	return string
	
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

file = h5py.File('/projects/ml/flu/fludb_data/processed_data_525916981168.h5','r')

train_labels_dataset = file['train_labels']
train_labels = train_labels_dataset[()]

train_sequences_dataset = file['train_sequences_categorical']
train_sequences = train_sequences_dataset[()]

valid_labels_dataset = file['valid_labels']
valid_labels = valid_labels_dataset[()]

valid_sequences_dataset = file['valid_sequences_categorical']
valid_sequences = valid_sequences_dataset[()]

test_labels_dataset = file['test_labels']
test_labels = test_labels_dataset[()]

test_sequences_dataset = file['test_sequences_categorical']
test_sequences = test_sequences_dataset[()]

file.close()

train_sequences = np.array(train_sequences)
valid_sequences = np.array(valid_sequences)
test_sequences = np.array(test_sequences)


# FAKE DATA (ONE MUTATED RESIDUE IN A FIXED SEQUENCE) HERE

# ~ data_file = h5py.File('/home/ceolson0/Documents/test_fastas2.h5','r')
# ~ train_sequences = data_file.get('sequences_cat').value
# ~ data_file.close()

# ~ train_sequences_offset = []
# ~ for i in range(len(train_sequences)):
	# ~ seq = []
	# ~ for j in range(len(train_sequences[0]) - 1):
		# ~ seq.append(train_sequences[i][j+1])
	# ~ seq.append(EOM_VECTOR)
	# ~ train_sequences_offset.append(seq)
# ~ train_sequences_offset = np.array(train_sequences_offset)

# ~ train_sequences = []
# ~ for i in range(len(train_sequences_unprocessed)):
	# ~ seq = train_sequences_unprocessed[i]
	# ~ seq = np.concatenate((SOM_VECTOR.reshape([1,22]),seq),axis=0)
	# ~ seq = np.concatenate((seq,EOM_VECTOR.reshape([1,22])),axis=0)
	# ~ train_sequences.append(seq)
# ~ train_sequences = np.array(train_sequences)




max_size = len(train_sequences[0])
encode_length = len(train_sequences[0][0])
batch_size=100
latent_dim = 100
num_classes = 19

print(np.shape(train_sequences))

def sample_from_latents(x):
	means = x[:,:latent_dim]
	log_vars = x[:,latent_dim:]
	base = tf.keras.backend.random_normal(shape=[latent_dim,])
	return means + tf.exp(log_vars) * base
	
def dense(matrix,bias,collection,in_dim,out_dim,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		W = tf.get_variable(matrix,trainable=True,collections=[collection,tf.GraphKeys.GLOBAL_VARIABLES],shape=[in_dim,out_dim])
		b = tf.get_variable(bias,trainable=True,collections=[collection,tf.GraphKeys.GLOBAL_VARIABLES],shape=[out_dim,])

	
	return tf.matmul(in_tensor,W) + b	
	
def batchnorm(sequence,offset_name,scale_name,collection):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        offset = tf.get_variable(offset_name,trainable=True,collections=[collection,tf.GraphKeys.GLOBAL_VARIABLES],initializer=tf.zeros(tf.shape(sequence)[1:]))
        scale = tf.get_variable(scale_name,trainable=True,collections=[collection,tf.GraphKeys.GLOBAL_VARIABLES],initializer=tf.ones(tf.shape(sequence)[1:]))
    
    means,variances = tf.nn.moments(sequence,axes=[0])
    normalized = tf.nn.batch_normalization(sequence,means,variances,offset,scale,tf.constant(0.001))
    
    return normalized
    
def conv(filter_name,bias_name,model,filter_shape,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		filt = tf.get_variable(filter_name,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=filter_shape)
		bias = tf.get_variable(bias_name,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[max_size,filter_shape[-1]])
		
	out = tf.nn.conv1d(in_tensor,filters=filt,padding='SAME',stride=1)
	out = tf.add(out,bias)
	return out
	
def residual_block(filter_name1,bias_name1,filter_name2,bias_name2,model,in_dim,out_dim,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		filter1 = tf.get_variable(filter_name1,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[5,in_dim,64])
		bias1 = tf.get_variable(bias_name1,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[max_size,64])
		filter2 = tf.get_variable(filter_name2,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[5,64,out_dim])
		bias2 = tf.get_variable(bias_name2,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[max_size,out_dim])

		x = in_tensor
		x = tf.nn.leaky_relu(x)
		
		x = tf.nn.conv1d(x,filters=filter1,padding='SAME',stride=1)
		x = tf.add(x,bias1)
		
		x = tf.nn.leaky_relu(x)
		x = tf.nn.conv1d(x,filters=filter2,padding='SAME',stride=1)
		x = tf.add(x,bias2)

	return x+0.3*in_tensor
	
def encoder(sequence,training=True):
	x = tf.reshape(sequence,[batch_size,max_size*encode_length])
	x = dense('encoder.dense1.matrix','encoder.dense1.bias','encoder',max_size*encode_length,512,x)
	x = tf.nn.leaky_relu(x)
	if training: x = batchnorm(x,'encoder.batchnorm1.offset','encoder.batchnorm1.scale','encoder')
	
	x = dense('encoder.dense2.matrix','encoder.dense2.bias','encoder',512,512,x)
	x = tf.nn.leaky_relu(x)
	if training: x = batchnorm(x,'encoder.batchnorm2.offset','encoder.batchnorm2.scale','encoder')
	
	x = dense('encoder.dense3.matrix','encoder.dense3.bias','encoder',512,256,x)
	x = tf.nn.leaky_relu(x)
	if training: x = batchnorm(x,'encoder.batchnorm3.offset','encoder.batchnorm3.scale','encoder')
	
	x = dense('encoder.dense4.matrix','encoder.dense4.bias','encoder',256,latent_dim*2,x)
	x = tf.nn.leaky_relu(x)
	if training: x = batchnorm(x,'encoder.batchnorm4.offset','encoder.batchnorm4.scale','encoder')

	return x
	
def decoder(state,training=True):
	x = tf.reshape(state,[batch_size,-1])
	x = dense('decoder.dense1.matrix','decoder.dense1.bias','decoder',latent_dim,512,x)
	x = tf.nn.leaky_relu(x)
	if training: x = batchnorm(x,'decoder.batchnorm1.offset','decoder.batchnorm1.scale','decoder')

	x = dense('decoder.dense2.matrix','decoder.dense2.bias','decoder',512,512,x)
	x = tf.nn.leaky_relu(x)
	if training: x = batchnorm(x,'decoder.batchnorm2.offset','decoder.batchnorm2.scale','decoder')

	
	x = dense('decoder.dense3.matrix','decoder.dense3.bias','decoder',512,256,x)
	x = tf.nn.leaky_relu(x)
	if training: x = batchnorm(x,'decoder.batchnorm3.offset','decoder.batchnorm3.scale','decoder')

	x = dense('decoder.dense4.matrix','decoder.dense4.bias','decoder',256,max_size*encode_length,x)
	x = tf.reshape(x,[batch_size,max_size,encode_length])
	if training: x = batchnorm(x,'decoder.batchnorm4.offset','decoder.batchnorm4.scale','decoder')

	return x
	
def predictor(sequence):
	x = conv('predictor.conv1.filter','predictor.conv1.bias','predictor',(5,encode_length,64),sequence)
	x = tf.nn.leaky_relu(x)
	
	x = residual_block('predictor.res1.filter1','predictor.res1.bias1','predictor.res1.filter2','predictor.res1.bias1','predictor',64,64,x)
	x = residual_block('predictor.res2.filter1','predictor.res2.bias1','predictor.res2.filter2','predictor.res2.bias1','predictor',64,64,x)
	x = residual_block('predictor.res3.filter1','predictor.res3.bias1','predictor.res3.filter2','predictor.res3.bias1','predictor',64,64,x)
	x = residual_block('predictor.res4.filter1','predictor.res4.bias1','predictor.res4.filter2','predictor.res4.bias1','predictor',64,64,x)
	x = residual_block('predictor.res5.filter1','predictor.res5.bias1','predictor.res5.filter2','predictor.res5.bias1','predictor',64,64,x)
	
	x = tf.reshape(x,(batch_size,max_size*64))
	
	output = dense('predictor.dense1.matrix','predictor.dense1.bias','predictor',max_size*64,num_classes,x)
	return output

sequence_in = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
correct_labels = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
beta = tf.placeholder(dtype=tf.dtypes.float32)

latent_seeds = encoder(sequence_in)
latent = sample_from_latents(latent_seeds)


logits = decoder(latent)
predicted_character = tf.nn.softmax(logits)

accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels,logits)

def compute_kl_loss(latent_seeds):
	means = latent_seeds[:,:latent_dim]
	log_vars = latent_seeds[:,latent_dim:]
	
	kl = tf.reduce_mean(tf.square(means) + tf.exp(log_vars) - log_vars - 1.) * 0.5

	return kl

kl_loss = compute_kl_loss(latent_seeds)

loss = tf.reduce_mean(accuracy_loss + beta * kl_loss)

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)


input_sequence_predictor = tf.placeholder(shape=[None,max_size,encode_length],dtype=tf.dtypes.float32)
label_predictor = tf.placeholder(shape=[None,num_classes],dtype=tf.dtypes.float32)
prediction_logits_predictor = predictor(input_sequence_predictor)
prediction_predictor = tf.nn.softmax(prediction_logits_predictor)
loss_predictor = tf.nn.softmax_cross_entropy_with_logits_v2(label_predictor,prediction_logits_predictor)
loss_predictor = tf.reduce_mean(loss_predictor)

optimizer_predictor = tf.train.GradientDescentOptimizer(0.01)
train_predictor = optimizer.minimize(loss_predictor,var_list=tf.get_collection('predictor'))

init = tf.initializers.variables(tf.get_collection('predictor'))

saver = tf.train.Saver(tf.get_collection('predictor'))


with tf.variable_scope('',reuse=tf.AUTO_REUSE):
	n_input = tf.get_variable('n_input',trainable=True,shape=[batch_size,latent_dim])
	produced_tuner = decoder(n_input)
	predicted_subtype_tuner = predictor(produced_tuner)
	target_tuner = tf.stack([[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] for i in range(batch_size)],axis=0)
	loss_backtoback_tuner = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(target_tuner,predicted_subtype_tuner))
	tune = tf.train.GradientDescentOptimizer(0.001).minimize(loss_backtoback_tuner,var_list=[tf.get_variable('n_input')])

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.all_variables())

print('#######################################')
print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

print('Training variational autoencoder')
epochs = 1000
for epoch in range(epochs):
	b = np.tanh((epoch-epochs*0.4)/(epochs*0.1))*0.5+0.5
	batch_sequences = np.random.permutation(train_sequences)[:batch_size]
	_,l = sess.run([train,loss],feed_dict={sequence_in:batch_sequences,correct_labels:batch_sequences,beta:b})
	if epoch%10==0: 
		print('epoch',epoch,'loss',l)
		prediction = sess.run(decoder(tf.random_normal([batch_size,latent_dim]),training=False))[0]
		sequence = []
		sequence_string = ''
		for i in range(len(prediction)):
			index = np.argmax(prediction[i])
			residue = ORDER[index]
			sequence.append(residue)
			sequence_string += residue
		print(sequence_string)
	if epoch%1000==0:
		saver.save(sess,'/home/ceolson0/Documents/flu/models/vae_fc_mini/')


print('Training predictor')
epochs = 1500
for epoch in range(epochs):
	batch = np.random.permutation(range(len(train_sequences)))[:batch_size]
	sequence_batch = train_sequences[batch].astype('float32')
	label_batch = train_labels[batch].astype('float32')
	_,l = sess.run([train_predictor,loss_predictor],feed_dict={input_sequence_predictor:sequence_batch,label_predictor:label_batch})
	if epoch%100 == 0:
		print('Epoch', epoch)
		print('loss:', l)

print('Tuning')
for i in range(100):
	print('='*(i+1)+'_'*(99-i)+'\r')
	sess.run(tune)

tuned = sess.run(produced)[0]
print(convert_to_string(tuned))

