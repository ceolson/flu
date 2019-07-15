import deepchem as dc
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from deepchem.utils.genomics import encode_fasta_sequence
from Bio import SeqIO
import h5py

ORDER = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
	
### Limit GPU memory used by tf
print("Limit GPU memory")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
config.log_device_placement = False

sess = tf.Session(config=config)
# ~ sess = tf_debug.LocalCLIDebugWrapperSession(sess)

### Prep data

file = h5py.File('/projects/ml/flu/fludb_data/processed_data_525916981168.h5','r')
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

max_size = len(train_sequences[0])
encode_length = len(train_sequences[0][0])
num_classes = len(train_labels[0])
batch_size=500

def dense(matrix,bias,model,in_dim,out_dim,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		W = tf.get_variable(matrix,collections=[model],trainable=True,shape=[in_dim,out_dim])
		b = tf.get_variable(bias,collections=[model],trainable=True,shape=[out_dim,])

	
	return tf.matmul(in_tensor,W) + b

def conv(filter_name,bias_name,model,filter_shape,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		filt = tf.get_variable(filter_name,collections=[model],trainable=True,shape=filter_shape)
		bias = tf.get_variable(bias_name,collections=[model],trainable=True,shape=[max_size,filter_shape[-1]])
		
	out = tf.nn.conv1d(in_tensor,filters=filt,padding='SAME',stride=1)
	out = tf.add(out,bias)
	return out
	
def residual_block(filter_name1,bias_name1,filter_name2,bias_name2,model,in_dim,out_dim,in_tensor):
	with tf.variable_scope('',reuse=tf.AUTO_REUSE):
		filter1 = tf.get_variable(filter_name1,collections=[model],trainable=True,shape=[5,in_dim,64])
		bias1 = tf.get_variable(bias_name1,collections=[model],trainable=True,shape=[max_size,64])
		filter2 = tf.get_variable(filter_name2,collections=[model],trainable=True,shape=[5,64,out_dim])
		bias2 = tf.get_variable(bias_name2,collections=[model],trainable=True,shape=[max_size,out_dim])

		x = in_tensor
		x = tf.nn.leaky_relu(x)
		
		x = tf.nn.conv1d(x,filters=filter1,padding='SAME',stride=1)
		x = tf.add(x,bias1)
		
		x = tf.nn.leaky_relu(x)
		x = tf.nn.conv1d(x,filters=filter2,padding='SAME',stride=1)
		x = tf.add(x,bias2)

	return x+0.3*in_tensor
	
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
	
input_sequence = tf.placeholder(shape=[None,max_size,encode_length],dtype=tf.dtypes.float32)
label = tf.placeholder(shape=[None,num_classes],dtype=tf.dtypes.float32)
prediction_logits = predictor(input_sequence)
prediction = tf.nn.softmax(prediction_logits)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(label,prediction_logits)
loss = tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss,var_list=tf.get_collection('predictor'))

init = tf.initializers.variables(tf.get_collection('predictor'))

saver = tf.train.Saver(tf.get_collection('predictor'))

sess.run(init)
sess.run(tf.global_variables_initializer())

print('##########################')
epochs = 1500
for epoch in range(epochs):
	batch = np.random.permutation(range(len(train_sequences)))[:batch_size]
	sequence_batch = train_sequences[batch].astype('float32')
	label_batch = train_labels[batch].astype('float32')
	_,l = sess.run([train,loss],feed_dict={input_sequence:sequence_batch,label:label_batch})
	if epoch%100 == 0:
		print('Epoch', epoch)
		print('loss:', l)


saver.save(sess,'/home/ceolson0/Documents')
saver.restore(sess,'/home/ceolson0/Documents')


wrong_count = 0
for i in range(len(test_sequences)):
	if i%500 == 0:
		try:
			test_prediction = sess.run(prediction,feed_dict={input_sequence:test_sequences[i:i+500]})
			for j in range(i,i+500):
				if np.argmax(test_prediction[j-i]) != np.argmax(test_labels[j]):
					wrong_count += 1
			print('Accuracy:', (i+1-wrong_count)/(i+1))
		except:
			pass

sess.close()
