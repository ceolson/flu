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

data_file = h5py.File('/home/ceolson0/Documents/test_fastas2.h5','r')
train_sequences_h1 = data_file.get('sequences_cat').value
data_file.close()

max_size = len(train_sequences_h1[0])
encode_length = len(train_sequences_h1[0][0])
batch_size=50
### Set up models

# Layers

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
	
# Generator

def generator(seed):
	seed = tf.reshape(seed,(batch_size,100))
	
	seed2 = dense('generator.dense1.matrix','generator.dense1.bias','generator',100,max_size*64,seed)
	seed2 = tf.nn.leaky_relu(seed2)
	seed2 = tf.reshape(seed2,[batch_size,max_size,64])
	
	x = residual_block('generator.res1.filter1','generator.res1.bias1','generator.res1.filter2','generator.res1.bias2','generator',64,64,seed2)
	x = residual_block('generator.res2.filter1','generator.res2.bias1','generator.res2.filter2','generator.res2.bias2','generator',64,64,x)
	x = residual_block('generator.res3.filter1','generator.res3.bias1','generator.res3.filter2','generator.res3.bias2','generator',64,64,x)
	x = residual_block('generator.res4.filter1','generator.res4.bias1','generator.res4.filter2','generator.res4.bias2','generator',64,64,x)
	x = residual_block('generator.res5.filter1','generator.res5.bias1','generator.res5.filter2','generator.res5.bias2','generator',64,64,x)

	x = conv('generator.conv1.filter','generator.conv1.bias','generator',(5,64,encode_length),x)
	
	synthetic = tf.nn.softmax(x)
	return synthetic

# Discriminator

def discriminator(sequence):
	x = conv('discriminator.conv1.filter','discriminator.conv1.bias','discriminator',(5,encode_length,64),sequence)
	x = tf.nn.leaky_relu(x)
	
	x = residual_block('discriminator.res1.filter1','discriminator.res1.bias1','discriminator.res1.filter2','discriminator.res1.bias1','discriminator',64,64,x)
	x = residual_block('discriminator.res2.filter1','discriminator.res2.bias1','discriminator.res2.filter2','discriminator.res2.bias1','discriminator',64,64,x)
	x = residual_block('discriminator.res3.filter1','discriminator.res3.bias1','discriminator.res3.filter2','discriminator.res3.bias1','discriminator',64,64,x)
	x = residual_block('discriminator.res4.filter1','discriminator.res4.bias1','discriminator.res4.filter2','discriminator.res4.bias1','discriminator',64,64,x)
	x = residual_block('discriminator.res5.filter1','discriminator.res5.bias1','discriminator.res5.filter2','discriminator.res5.bias1','discriminator',64,64,x)
	
	x = tf.reshape(x,(batch_size,max_size*64))
	
	output = dense('discriminator.dense1.matrix','discriminator.dense1.bias','discriminator',max_size*64,1,x)
	return output
	
### Constructing the loss function

real_images = tf.placeholder(float,name='real_images')
noise = tf.placeholder(float,name='noise')

fake_images = generator(noise)
fake_images = tf.identity(fake_images,name='fake_images')

# Sampling images in the encoded space between the fake ones and the real ones

interpolation_coeffs = tf.random_uniform(shape=(batch_size,1,1))
sampled_images = tf.add(real_images,tf.multiply(tf.subtract(fake_images,real_images),interpolation_coeffs),name='sampled_images')

# Gradient penalty
gradients = tf.gradients(discriminator(sampled_images),sampled_images,name='gradients')[0]
norms = tf.norm(gradients,axis=[1,2])
score = tf.reduce_mean(tf.square(tf.subtract(norms,1.)),name='gradient_penalty')

# Loss based on discriminator's predictions

pred_real = tf.reshape(discriminator(real_images),[-1])
pred_real = tf.identity(pred_real,name='pred_real')

pred_fake = tf.reshape(discriminator(fake_images),[-1])
pred_fake = tf.identity(pred_fake,name='pred_fake')

diff = tf.reduce_mean(tf.subtract(pred_fake,pred_real))

# Discriminator wants fake sequences to be labeled 0, real to be labeled 1
disc_loss = tf.add(diff,tf.multiply(tf.constant(10.),score),name='disc_loss')

# Generator wants fake sequences to be labeled 1
gen_loss = - tf.reduce_mean(pred_fake,name='gen_loss')

# For tracking using tensorboard

a = tf.summary.scalar('discriminator_difference', diff)
b = tf.summary.scalar('generator_difference',gen_loss)
merged = tf.summary.merge_all()

# Optimizers
disc_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

train_discriminator = disc_optimizer.minimize(disc_loss,var_list=tf.get_collection('discriminator'),name='train_discriminator')
grads_discriminator = disc_optimizer.compute_gradients(disc_loss,var_list=tf.get_collection('discriminator'))

train_generator = gen_optimizer.minimize(gen_loss,var_list=tf.get_collection('generator'),name='train_generator')
grads_generator = gen_optimizer.compute_gradients(gen_loss,var_list=tf.get_collection('generator'))

init = tf.initializers.variables(tf.get_collection('discriminator')+tf.get_collection('generator'))

writer = tf.summary.FileWriter('/home/ceolson0/Documents/tensorboard',sess.graph)
saver = tf.train.Saver(tf.get_collection('discriminator')+tf.get_collection('generator'))
### Training

sess.run(init)
sess.run(tf.global_variables_initializer())
print('############')
print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
epochs=10000

for epoch in range(epochs):
	print('\ngan17, epoch ',epoch)

	
	# Train discriminator
	for i in range(5):
		real = np.random.permutation(train_sequences_h1)[:batch_size].astype(np.float32)
		noise_input = np.random.normal(0,1,(batch_size,100))
		_,d_loss,grads = sess.run([train_discriminator,diff,grads_discriminator],feed_dict={real_images:real,noise:noise_input})
		print("Training discriminator",d_loss)
			
	# Train generator

	real = np.random.permutation(train_sequences_h1)[:batch_size].astype(np.float32)
	noise_input = np.random.normal(0,1,(batch_size,100))
	_,g_loss,grads = sess.run([train_generator,gen_loss,grads_generator],feed_dict={noise:noise_input})
	print("Training generator",g_loss)

	print("Generator loss: ",g_loss)
	print("Discriminator loss: ",d_loss)
		
	# Print a sample string	
	prediction = sess.run(generator(tf.random_normal((50,100))))[0]
	sequence = []
	sequence_string = ''
	for i in range(len(prediction)):
		index = np.argmax(prediction[i])
		residue = ORDER[index]
		sequence.append(residue)
		sequence_string += residue
	print(sequence_string)
		
	
# save_path = saver.save(sess,'/home/ceolson0/gan17.ckpt')

sess.close()

# ~ ### Classifier

# ~ c_sequence = tf.keras.Input(shape=(max_size,encode_length))
# ~ c_sequence2 = tf.keras.layers.Reshape((max_size,encode_length,1))(c_sequence)

# ~ x = tf.keras.layers.Conv2D(64,(5,encode_length),activation=tf.nn.leaky_relu,padding='same')(c_sequence2)
# ~ x = tf.keras.layers.Conv2D(64,(5,encode_length),activation=tf.nn.leaky_relu,padding='same')(x)
# ~ x = tf.keras.layers.Conv2D(64,(5,encode_length),activation=tf.nn.leaky_relu)(x)

