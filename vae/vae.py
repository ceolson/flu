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

max_size = len(train_sequences_h1[0])
encode_length = len(train_sequences_h1[0][0])
batch_size=500

latent_dim = 100

### Set up models

# Layers

def residual_block(filter_name1,filter_name2,model,in_dim,out_dim,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		filter1 = tf.get_variable(filter_name1,collections=[model],trainable=True,shape=[16,encode_length,in_dim,5])
		filter2 = tf.get_variable(filter_name2,collections=[model],trainable=True,shape=[16,encode_length,5,out_dim])

		x = in_tensor
		x = tf.nn.relu(x)
		
		x = tf.nn.conv2d(x,filter=filter1,padding='SAME',strides=[1,1,1,1])
		
		x = tf.nn.relu(x)
		x = tf.nn.conv2d(x,filter=filter2,padding='SAME',strides=[1,1,1,1])

	return x+0.3*in_tensor
	
def dense(matrix,bias,model,in_dim,out_dim,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		W = tf.get_variable(matrix,collections=[model],trainable=True,shape=[in_dim,out_dim])
		b = tf.get_variable(bias,collections=[model],trainable=True,shape=[out_dim,])

	
	return tf.matmul(in_tensor,W) + b


def conv(filter_name,model,filter_shape,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		filt = tf.get_variable(filter_name,collections=[model],trainable=True,shape=filter_shape)

	return tf.nn.conv2d(in_tensor,filter=filt,padding='SAME',strides=[1,1,1,1])
	
def lstm(model,
		 input_matrix_vector, input_matrix_state, input_bias,
		 output_matrix_vector, output_matrix_state, output_bias,
		 forget_matrix_vector, forget_matrix_state, forget_bias,
		 state_matrix_vector, state_matrix_state, state_bias,
		 vector_dim, state_dim, in_tensor, hidden_state, state):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		W_i = tf.get_variable(input_matrix_vector,trainable=True,shape=[vector_dim,state_dim],collections=[model])
		U_i = tf.get_variable(input_matrix_state,trainable=True,shape=[state_dim,state_dim],collections=[model])
		b_i = tf.get_variable(input_bias,trainable=True,shape=[state_dim,],collections=[model])
		
		W_o = tf.get_variable(output_matrix_vector,trainable=True,shape=[vector_dim,state_dim],collections=[model])
		U_o = tf.get_variable(output_matrix_state,trainable=True,shape=[state_dim,state_dim],collections=[model])
		b_o = tf.get_variable(output_bias,trainable=True,shape=[state_dim,],collections=[model])
		
		W_f = tf.get_variable(forget_matrix_vector,trainable=True,shape=[vector_dim,state_dim],collections=[model])
		U_f = tf.get_variable(forget_matrix_state,trainable=True,shape=[state_dim,state_dim],collections=[model])
		b_f = tf.get_variable(forget_bias,trainable=True,shape=[state_dim,],collections=[model])
		
		W_s = tf.get_variable(state_matrix_vector,trainable=True,shape=[vector_dim,state_dim],collections=[model])
		U_s = tf.get_variable(state_matrix_state,trainable=True,shape=[state_dim,state_dim],collections=[model])
		b_s = tf.get_variable(state_bias,trainable=True,shape=[state_dim,],collections=[model])
	
		
	forget = tf.matmul(in_tensor,W_f) + tf.matmul(hidden_state,U_f) + b_f
	forget = tf.nn.sigmoid(forget)
	
	inpt = tf.matmul(in_tensor,W_i) + tf.matmul(hidden_state,U_i) + b_i
	inpt = tf.nn.sigmoid(inpt)
	
	output_vector = tf.matmul(in_tensor,W_o) + tf.matmul(hidden_state,U_o) + b_o
	output_vector = tf.nn.sigmoid(output_vector)
	
	new_state = tf.multiply(forget,hidden_state) + tf.nn.tanh(tf.matmul(in_tensor,W_s) + tf.matmul(state,U_s) + b_s)
	new_hidden_state = tf.multiply(output_vector,new_state)
	
	return (new_hidden_state,new_state)

def encoder(sequence):
	def body(i,state):
		output_vector,new_state = lstm('encoder',
									   'encoder.lstm1.input_matrix_vector','encoder.lstm1.input_matrix_state','encoder.lstm1.input_bias',
									   'encoder.lstm1.output_matrix_vector','encoder.lstm1.output_matrix_state','encoder.lstm1.output_bias',
									   'encoder.lstm1.forget_matrix_vector','encoder.lstm1.forget_matrix_state','encoder.lstm1.forget_bias',
									   'encoder.lstm1.state_matrix_vector','encoder.lstm1.state_matrix_state','encoder.lstm1.state_bias',
									   encode_length, latent_dim*2, sequence[:,i,:], hidden_state, state)
						   
		output_vector,new_state = lstm('encoder',
									   'encoder.lstm2.input_matrix_vector','encoder.lstm2.input_matrix_state','encoder.lstm2.input_bias',
									   'encoder.lstm2.output_matrix_vector','encoder.lstm2.output_matrix_state','encoder.lstm2.output_bias',
									   'encoder.lstm2.forget_matrix_vector','encoder.lstm2.forget_matrix_state','encoder.lstm2.forget_bias',
									   'encoder.lstm2.state_matrix_vector','encoder.lstm2.state_matrix_state','encoder.lstm2.state_bias',
									   latent_dim*2, latent_dim*2, output_vector, output_vector, new_state)
						   
		output_vector,new_state = lstm('encoder',
									   'encoder.lstm3.input_matrix_vector','encoder.lstm3.input_matrix_state','encoder.lstm3.input_bias',
									   'encoder.lstm3.output_matrix_vector','encoder.lstm3.output_matrix_state','encoder.lstm3.output_bias',
									   'encoder.lstm3.forget_matrix_vector','encoder.lstm3.forget_matrix_state','encoder.lstm3.forget_bias',
									   'encoder.lstm3.state_matrix_vector','encoder.lstm3.state_matrix_state','encoder.lstm3.state_bias',
									   latent_dim*2, latent_dim*2, output_vector, output_vector, new_state)
						   
		i = i + 1
		state = new_state
		return (i,new_state)
	
	i = 0
	state = tf.zeros([batch_size,latent_dim*2])
	hidden_state = tf.zeros([batch_size,latent_dim*2])
	
	_,state = tf.while_loop(
		lambda i,state: tf.less(i,tf.shape(sequence)[1]),
		body,
		[i,state]
	)
	
	return state

def decoder(latent,state_so_far):

	output_vector,new_state = lstm('decoder',
								   'decoder.lstm1.input_matrix_vector','decoder.lstm1.input_matrix_state','decoder.lstm1.input_bias',
								   'decoder.lstm1.output_matrix_vector','decoder.lstm1.output_matrix_state','decoder.lstm1.output_bias',
								   'decoder.lstm1.forget_matrix_vector','decoder.lstm1.forget_matrix_state','decoder.lstm1.forget_bias',
								   'decoder.lstm1.state_matrix_vector','decoder.lstm1.state_matrix_state','decoder.lstm1.state_bias',
								   latent_dim, latent_dim, state_so_far, state_so_far, latent)
					   
	output_vector,new_state = lstm('decoder',
								   'decoder.lstm2.input_matrix_vector','decoder.lstm2.input_matrix_state','decoder.lstm2.input_bias',
								   'decoder.lstm2.output_matrix_vector','decoder.lstm2.output_matrix_state','decoder.lstm2.output_bias',
								   'decoder.lstm2.forget_matrix_vector','decoder.lstm2.forget_matrix_state','decoder.lstm2.forget_bias',
								   'decoder.lstm2.state_matrix_vector','decoder.lstm2.state_matrix_state','decoder.lstm2.state_bias',
								   latent_dim, latent_dim, output_vector, output_vector, new_state)
					   
	output_vector,new_state = lstm('decoder',
								   'decoder.lstm3.input_matrix_vector','decoder.lstm3.input_matrix_state','decoder.lstm3.input_bias',
								   'decoder.lstm3.output_matrix_vector','decoder.lstm3.output_matrix_state','decoder.lstm3.output_bias',
								   'decoder.lstm3.forget_matrix_vector','decoder.lstm3.forget_matrix_state','decoder.lstm3.forget_bias',
								   'decoder.lstm3.state_matrix_vector','decoder.lstm3.state_matrix_state','decoder.lstm3.state_bias',
								   latent_dim, latent_dim, output_vector, output_vector, new_state)
						   
	output_vector = dense('decoder.dense1.matrix','decoder.dense1.bias','decoder',latent_dim,encode_length,output_vector)				   
	output_vector = tf.reshape(output_vector,[batch_size,1,encode_length])

	return (output_vector,new_state)

training_set = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
correct_next = tf.placeholder(shape=[batch_size,encode_length],dtype=tf.dtypes.float32)

encodings = encoder(training_set)
print("encodings shape",encodings)

means = tf.reshape(encodings[:,:latent_dim],[-1])
log_variances = tf.reshape(encodings[:,latent_dim:],[-1])

distribution = tf.contrib.distributions.MultivariateNormalDiag(means,tf.exp(log_variances))

latent = distribution.sample()
latent = tf.reshape(latent,[batch_size,latent_dim])

reconstruction,state = decoder(latent,latent)
reconstruction = tf.reshape(reconstruction,[batch_size,encode_length])

accuracy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(reconstruction,correct_next))

kl_loss = 0.5 * tf.reduce_sum(tf.square(means) + tf.exp(log_variances) - log_variances - 1.)

loss = accuracy_loss + kl_loss

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss,var_list=tf.get_collection('encoder')+tf.get_collection('decoder'))

init = tf.initializers.variables(tf.get_collection('encoder')+tf.get_collection('decoder'))

writer = tf.summary.FileWriter('/home/ceolson0/Documents/tensorboard',sess.graph)

# ~ saver = tf.train.Saver(tf.get_collection('discriminator')+tf.get_collection('generator'))

def predict(input_sequence):
	latent_testing = sess.run(latent,feed_dict={training_set:input_sequence})
	running = latent
	current = [[SOM_VECTOR] for i in range(batch_size)]
	string = ''
	while np.argmax(current[-1]) != np.argmax(EOM_VECTOR) and len(current) < 1000:
		r,s = sess.run([reconstruction,state],feed_dict={training_set:np.append(input_sequence,current,axis=1)})
		character = ORDER[np.argmax(r[0])]
		string += character
		current[0].append(r[0])
		for i in range(1,batch_size):
			current[i].append(SOM_VECTOR)
			
		running = s
	return string


sess.run(init)
sess.run(tf.global_variables_initializer())

epochs = 500

for epoch in range(epochs):
	print("vae1, epoch",epoch)
	
	length = np.random.choice(range(max_size))
	batch_sequences = np.random.permutation(train_sequences_h1)[:batch_size].astype('float32')
	so_fars = batch_sequences[:,:length,:]
	correct_nexts = batch_sequences[:,length,:].reshape(batch_size,encode_length)

	_,l,r = sess.run([train,loss,reconstruction],feed_dict={training_set:so_fars,correct_next:correct_nexts})
	print(l)
	print(np.argmax(correct_nexts[0]),np.argmax(r[0]))
	# ~ if epoch % 20 == 0:
		# ~ test_seq = np.random.permutation(train_sequences_h1)[:batch_size].astype('float32')
		# ~ test_seq_string = ''
		# ~ for encoding in test_seq[0]:
			# ~ test_seq_string += ORDER[np.argmax(encoding)]
			
		# ~ recon = predict(test_seq)
		# ~ print(test_seq_string)
		# ~ print(recon)
		
	# ~ prediction = sess.run(decoder(tf.random_normal([batch_size,latent_dim])))[0]
	# ~ sequence = []
	# ~ sequence_string = ''
	# ~ for i in range(len(prediction)):
		# ~ index = np.argmax(prediction[i])
		# ~ residue = ORDER[index]
		# ~ sequence.append(residue)
		# ~ sequence_string += residue
	# ~ print(sequence_string)

sess.close()
