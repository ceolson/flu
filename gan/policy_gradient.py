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
batch_size=50
minibatch_size=10
latent_dim=22

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
    
def batchnorm(sequence,offset_name,scale_name,model):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        offset = tf.get_variable(offset_name,collections=[model],trainable=True,initializer=tf.zeros(tf.shape(sequence)[1:]))
        scale = tf.get_variable(scale_name,collections=[model],trainable=True,initializer=tf.ones(tf.shape(sequence)[1:]))
    
    means,variances = tf.nn.moments(sequence,axes=[0])
    normalized = tf.nn.batch_normalization(sequence,means,variances,offset,scale,tf.constant(0.001))
    
    return normalized
    
def layernorm(sequence):
    means,variances = tf.nn.moments(sequence,axes=[1,2])
    means = tf.reshape(means,[batch_size,1,1])
    variances = tf.reshape(variances,[batch_size,1,1])
    return tf.divide(tf.subtract(sequence,means),variances)
    
# Generator
generator_lstm = tf.keras.layers.CuDNNLSTM(latent_dim,return_state=True,return_sequences=True)

def rollout(so_far,noise=[tf.random_normal((batch_size,latent_dim)),tf.random_normal((batch_size,latent_dim))]):
    new = so_far
    for i in range(max_size -  tf.shape(so_far)[1]):
        character,h,c = generator_lstm(new,initial_state=noise)
        character = character[-1]
        new = tf.concatenate(new,character,axis=1)
    return new

def generate_next(so_far,noise=[tf.random_normal((batch_size,latent_dim)),tf.random_normal((batch_size,latent_dim))]):
    temp,h,c = generator_lstm(so_far,initial_state=noise)
    return temp[:,-1,:]
    
def generate_full(noise=[tf.random_normal((batch_size,latent_dim)),tf.random_normal((batch_size,latent_dim))]):
    seqs = tf.stack([SOM_TENSOR for i in range(batch_size)],axis=0)
    state = noise
    for i in range(max_size):
        temp,h,c = generator_lstm(seqs,initial_state=state)
        seqs = tf.concat((seqs,tf.reshape(temp,(int(batch_size/2),1,encode_length))),axis=1)
        state = [h,c]
    return seqs
        

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
    
    output = dense('discriminator.dense1.matrix','discriminator.dense1.bias','discriminator',max_size*64,2,x)
    return output
    
generator_training_samples = tf.placeholder(float,shape=[batch_size,None,encode_length])
generator_training_labels = tf.placeholder(float,shape=[batch_size,encode_length])
print(generate_next(generator_training_samples))
generator_training_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(generate_next(generator_training_samples),generator_training_labels))
train_generator = tf.train.GradientDescentOptimizer(0.001).minimize(generator_training_loss)

discriminator_training_samples = tf.placeholder(float,shape=[batch_size,max_size,encode_length])
discriminator_training_labels = tf.placeholder(float,shape=[batch_size,2])
discriminator_training_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(discriminator(discriminator_training_samples),discriminator_training_labels))
train_discriminator = tf.train.GradientDescentOptimizer(0.001).minimize(discriminator_training_loss)

gradients = tf.gradients(tf.log(generate_next(generator_training_samples)),generator_lstm.weights)

sess.run(tf.global_variables_initializer())

print('Pretrain generator')
for epoch in range(100):
    stop_point = np.random.randint(2,max_size-1)
    batch_sequences = np.random.permutation(train_sequences_h1)[:batch_size,:1+stop_point]
    _,l = sess.run([train_generator,generator_training_loss],feed_dict={generator_training_samples:batch_sequences[:,:-1],generator_training_labels:batch_sequences[:,-1,:]})
    print(epoch,l)

print('Pretrain discriminator')
for epoch in range(100):
    fakes = sess.run(generate_full())[:int(int(batch_size/2))]
    reals = np.random.permutation(train_sequences_h1)[:int(int(batch_size/2))]
    print(np.shape(fakes),np.shape(reals))
    discriminator_input = np.concatenate((fakes,reals),axis=0)
    discriminator_labels = np.array([[0.,1.] for i in range(int(int(batch_size/2)))]+[[1.,0.] for i in range(int(int(batch_size/2)))])
    _,l = sess.run([train_discriminator,discriminator_training_loss],feed_dict={discriminator_training_samples:discriminator_input,discriminator_training_labels:discriminator_labels})
    print(epoch,l)


print('Training')
for epoch in range(100):
    fake = sess.run(generate_full())
    for i in range(max_size):
        if i == max_size - 1:
            full_sequence = fake
        else:
            full_sequence = sess.run(rollout(tf.constant(fake[:,:i])))
        cost = sess.run(discriminator(tf.constant(full_sequence)))[:,0]
        grads = sess.run(gradients,feed_dict={generator_training_samples:np.random.permutation(train_sequences_h1)[:batch_size,:i]})
        updates = grads * cost
        grad_var_pairs = zip(updates,generator_lstm.weights)
    for subepoch in range(100):
        fakes = sess.run(generate_full())
        reals = np.random.permutation(train_sequences_h1)[:int(batch_size/2)]
        discriminator_input = np.concatenate((fakes,reals),axis=0)
        discriminator_labels = np.array([[0.,1.] for i in range(int(batch_size/2))]+[[1.,0.] for i in range(int(batch_size/2))])
        _,l = sess.run([train_discriminator,discriminator_training_loss],feed_dict={discriminator_training_samples:discriminator_input,discriminator_training_labels:discriminator_labels})
    print(epoch,l)
        
    
save_path = saver.save(sess,'/home/ceolson0/Documents/flu/models/saves/nonorm/')

sess.close()

# ~ ### Classifier

# ~ c_sequence = tf.keras.Input(shape=(max_size,encode_length))
# ~ c_sequence2 = tf.keras.layers.Reshape((max_size,encode_length,1))(c_sequence)

# ~ x = tf.keras.layers.Conv2D(64,(5,encode_length),activation=tf.nn.leaky_relu,padding='same')(c_sequence2)
# ~ x = tf.keras.layers.Conv2D(64,(5,encode_length),activation=tf.nn.leaky_relu,padding='same')(x)
# ~ x = tf.keras.layers.Conv2D(64,(5,encode_length),activation=tf.nn.leaky_relu)(x)
