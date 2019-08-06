import deepchem as dc
import numpy as np
import tensorflow as tf
from Bio import SeqIO
import h5py
import scipy
import argparse
from tensorflow.python import debug as tf_debug

import constants as cst
import layers as lyr



### Limit GPU memory used by tf
print('Limit GPU memory')
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
config.log_device_placement = False

sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

### Arg parser
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, help='data to train on, one of "all", "h1", "h2", "h3", ..., "h18", or "aligned" (others are not aligned)', default='all')
parser.add_argument('--encoding', type=str, help='data encoding, either "categorical" or "blosum"', default='categorical')
parser.add_argument('--model', type=str, help='model to use, one of "gan", "vae_fc", "vae_conv", or "vae_lstm"', default='vae_fc')
parser.add_argument('--beta', type=float, help='if using a VAE, the coefficient for the KL loss', default=5)
parser.add_argument('--tuner', type=str, help='what to tune for, a combination of "subtype", "head_stem", or "design" (comma separated)', default='design')
parser.add_argument('--design', type=str, help='if using design tuner, list of strings "[position]-[residue]-[weight]" (weight is optional), e.g. "15-R-1.0,223-C-5.0"', default='1-M')
parser.add_argument('--subtype', type=int, help='if using subtype tuner, which subtype you want', default=1)
parser.add_argument('--head_stem', type=str, help='if using head-stem tuner, a string of "[head subtype],[stem subtype]"', default='1,1')
parser.add_argument('--train_model_epochs', type=int, help='how many epochs to train the generative model', default=0)
parser.add_argument('--train_predictor_epochs', type=int, help='how many epochs to train the predictor model', default=0)
parser.add_argument('--tune_epochs', type=int, help='how many epochs to tune', default=0)
parser.add_argument('--batch_size', type=int, help='batch size for training everything', default=100)
parser.add_argument('--latent_dimension', type=int, help='latent dimension for everything', default=100)
parser.add_argument('--restore_model', help='saved file to restore model from')
parser.add_argument('--restore_predictor', help='saved file to restore predictor from')
parser.add_argument('--save_model', help='where to save model to', default='/home/ceolson0/Documents/flu/saves/models/default_model/')
parser.add_argument('--save_predictor', help='where to save predictor to', default='/home/ceolson0/Documents/flu/saves/predictors/default_predictor/')
parser.add_argument('--num_outputs', help='how many samples to print out', default=1)
parser.add_argument('--random_seed', type=int, help='random seed to make execution deterministic, default is random')
parser.add_argument('--return_latents', type=int, help='1 if you want to print the latent variable with the sequence')
parser.add_argument('--channels', type=int, help='number of channels in convolution hidden layers', default=16)

args = parser.parse_args()

if args.random_seed:
    tf.set_random_seed(args.random_seed)

def design_parser(string):
    design = {}
    weights = {}
    places = string.split(',')
    for place in places:
        arr = place.split('-')
        if len(arr) == 2:
            location,residue = arr
            weight = 1.
        if len(arr) == 3:
            location,residue,weight = arr
            weight = float(weight)
        location = int(location)
        design[location] = residue
        weights[location] = weight
    return design,weights
    
def headstem_parser(string):
    head,stem = map(int,string.split(','))
    return head,stem

tuner = args.tuner.split(',')

print(             '+-------------------------------------------------------+')
print('|  RUN INFORMATION                                      |')
print('|  ===================================================  |')
print('|  Model type: {: <41s}|'.format(args.model))
print('|  Latent dimension: {: <35}|'.format(args.latent_dimension))
if args.model == 'vae_conv': 
    print('|  Channels: {: <43}|'.format(args.channels))
else:
    print('|  Channels: not applicable                             |')      
print('|  Data encoding: {: <38s}|'.format(args.encoding))
print('|  Tuner: {: <46s}|'.format(args.tuner))
print('+-------------------------------------------------------+')

# Array to convert from categorical to residue letter
if args.encoding == 'categorical':
    ORDER = cst.ORDER_CATEGORICAL
    CATEGORIES = cst.CATEGORIES
elif args.encoding == 'blosum':
    ORDER = cst.ORDER_BLOSUM
    CATEGORIES = cst.BLOSUM

### Collect data
f = h5py.File('/projects/ml/flu/fludb_data/processed_data_525916981168.h5','r')

train_labels_dataset = f['train_labels']
train_labels = train_labels_dataset[()]

valid_labels_dataset = f['valid_labels']
valid_labels = valid_labels_dataset[()]

test_labels_dataset = f['test_labels']
test_labels = test_labels_dataset[()]

if args.encoding == 'categorical':
    train_sequences_dataset = f['train_sequences_categorical']
    train_sequences = train_sequences_dataset[()]

    valid_sequences_dataset = f['valid_sequences_categorical']
    valid_sequences = valid_sequences_dataset[()]

    test_sequences_dataset = f['test_sequences_categorical']
    test_sequences = test_sequences_dataset[()]
    
elif args.encoding == 'blosum':
    train_sequences_dataset = f['train_sequences_blosum']
    train_sequences = train_sequences_dataset[()]

    valid_sequences_dataset = f['valid_sequences_blosum']
    valid_sequences = valid_sequences_dataset[()]

    test_sequences_dataset = f['test_sequences_blosum']
    test_sequences = test_sequences_dataset[()]

f.close()

if args.data != 'all' and args.data != 'aligned':
    subtype = int(args.data[1:])
    
    temp = []
    for i in range(len(train_sequences)):
        if np.argmax(train_labels[i]) == subtype:
            temp.append(train_sequences[i])
    train_sequences = np.array(temp)
    
    temp = []
    for i in range(len(valid_sequences)):
        if np.argmax(valid_labels[i]) == subtype:
            temp.append(valid_sequences[i])
    valid_sequences = np.array(temp)
    
    temp = []
    for i in range(len(test_sequences)):
        if np.argmax(test_labels[i]) == subtype:
            temp.append(test_sequences[i])
    test_sequences = np.array(temp)

# Add end of message and start of message characters if using RNN
if args.model == 'vae_lstm':
    train_sequences_unprocessed = np.array(train_sequences)
    valid_sequences = np.array(valid_sequences)
    test_sequences = np.array(test_sequences)

    train_sequences = []
    for i in range(len(train_sequences_unprocessed)):
        seq = train_sequences_unprocessed[i]
        seq_new = []
        for res in seq:
            seq_new.append(np.concatenate((res,[0.,0.]),axis=0))
        seq_new = np.array(seq_new)
        seq_new = np.concatenate((cst.SOM_VECTOR.reshape([1,24]),seq_new),axis=0)
        seq_new = np.concatenate((seq_new,cst.EOM_VECTOR.reshape([1,24])),axis=0)
        train_sequences.append(seq)
    train_sequences = np.array(train_sequences)


head_size = len(cst.HEAD)
stem_size = len(cst.STEM)
max_size = len(train_sequences[0])
encode_length = len(train_sequences[0][0])
batch_size = args.batch_size
latent_dim = args.latent_dimension
num_classes = len(train_labels[0])


### Define models

if args.model == 'vae_fc':
    def encoder(sequence,training=tf.constant(True)):
        num = tf.shape(sequence)[0]
        
        x = tf.reshape(sequence,[num,max_size*encode_length])
        x = lyr.dense('encoder.dense1.matrix','encoder.dense1.bias','encoder',max_size*encode_length,512,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'encoder.batchnorm1.offset','encoder.batchnorm1.scale','encoder.batchnorm1.total_means','encoder.batchnorm1.total_variances','encoder.num_means','encoder',(512,),training=training)
        
        x = lyr.dense('encoder.dense2.matrix','encoder.dense2.bias','encoder',512,512,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'encoder.batchnorm2.offset','encoder.batchnorm2.scale','encoder.batchnorm2.total_means','encoder.batchnorm2.total_variances','encoder.num_means','encoder',(512,),training=training)
        
        x = lyr.dense('encoder.dense3.matrix','encoder.dense3.bias','encoder',512,256,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'encoder.batchnorm3.offset','encoder.batchnorm3.scale','encoder.batchnorm3.total_means','encoder.batchnorm3.total_variances','encoder.num_means','encoder',(256,),training=training)
        
        x = lyr.dense('encoder.dense4.matrix','encoder.dense4.bias','encoder',256,latent_dim*2,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'encoder.batchnorm4.offset','encoder.batchnorm4.scale','encoder.batchnorm4.total_means','encoder.batchnorm4.total_variances','encoder.num_means','encoder',(latent_dim*2,),training=training)
        
        return x
	
    def decoder(state,training=tf.constant(True)):
        num = tf.shape(state)[0]
        
        x = tf.reshape(state,[num,-1])
        x = lyr.dense('decoder.dense1.matrix','decoder.dense1.bias','decoder',latent_dim,512,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'decoder.batchnorm1.offset','decoder.batchnorm1.scale','decoder.batchnorm1.total_means','decoder.batchnorm1.total_variances','decoder.num_means','decoder',(512,),training=training)
        
        x = lyr.dense('decoder.dense2.matrix','decoder.dense2.bias','decoder',512,512,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'decoder.batchnorm2.offset','decoder.batchnorm2.scale','decoder.batchnorm2.total_means','decoder.batchnorm2.total_variances','decoder.num_means','decoder',(512,),training=training)
        
        x = lyr.dense('decoder.dense3.matrix','decoder.dense3.bias','decoder',512,256,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'decoder.batchnorm3.offset','decoder.batchnorm3.scale','decoder.batchnorm3.total_means','decoder.batchnorm3.total_variances','decoder.num_means','decoder',(256,),training=training)
        
        x = lyr.dense('decoder.dense4.matrix','decoder.dense4.bias','decoder',256,max_size*encode_length,x)
        x = lyr.batchnorm(x,'decoder.batchnorm4.offset','decoder.batchnorm4.scale','decoder.batchnorm4.total_means','decoder.batchnorm4.total_variances','decoder.num_means','decoder',(max_size*encode_length,),training=training)
        
        x = tf.reshape(x,[num,max_size,encode_length])
        
        return x
        
elif args.model == 'vae_conv':
    def encoder(sequence,training=True):
        num = tf.shape(sequence)[0]
        
        x = lyr.conv('encoder.conv1.filter','encoder.conv1.bias','encoder',(5,encode_length,args.channels),sequence,max_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('encoder.res1.filter1','encoder.res1.bias1','encoder.res1.filter2','encoder.res1.bias1','encoder',args.channels,args.channels,x,max_size,channels=args.channels)
        
        x = tf.reshape(x,(num,max_size*args.channels)) 
        
        x = lyr.dense('encoder.dense1.matrix','encoder.dense1.bias','encoder',max_size*args.channels,2*latent_dim,x)
        output = tf.nn.leaky_relu(x)
        
        return output
        
    def decoder(sequence,training=True):
        num = tf.shape(sequence)[0]
        
        x = lyr.dense('decoder.dense1.matrix','decoder.dense1.bias','decoder',latent_dim,max_size,sequence)
        x = tf.nn.leaky_relu(x)
        
        x = tf.reshape(x,(num,max_size,1))
        
        x = lyr.conv('decoder.conv1.filter','decoder.conv1.bias','decoder',(5,1,args.channels),x,max_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('decoder.res1.filter1','decoder.res1.bias1','decoder.res1.filter2','decoder.res1.bias1','decoder',args.channels,args.channels,x,max_size,channels=args.channels)
    
        x = lyr.conv('decoder.conv2.filter','decoder.conv2.bias','decoder',(5,args.channels,encode_length),x,max_size)
        output = tf.nn.leaky_relu(x)

        return output

elif args.model == 'vae_lstm':
    encoder_lstm = tf.keras.layers.CuDNNLSTM(latent_dim*2,return_state=True)
    decoder_lstm = tf.keras.layers.CuDNNLSTM(latent_dim)
    
    def encoder(sequence):
        out,h,c = encoder_lstm(sequence)
        return [h,c]
    
    def decoder(state,so_far):
        out = decoder_lstm(so_far,initial_state=state)
        logits = lyr.dense('decoder.dense.matrix','decoder.dense.bias','decoder',latent_dim,encode_length,out)
        return logits

elif args.model == 'gan':
    def generator(seed,training=True):
        num = tf.shape(sequence)[0]
        
        seed = tf.reshape(seed,(num,100))
        
        seed2 = lyr.dense('generator.dense1.matrix','generator.dense1.bias','generator',100,max_size*64,seed)
        seed2 = tf.nn.leaky_relu(seed2)
        seed2 = tf.reshape(seed2,[num,max_size,64])

        x = lyr.residual_block('generator.res1.filter1','generator.res1.bias1','generator.res1.filter2','generator.res1.bias2','generator',64,64,seed2,max_size)
        x = lyr.residual_block('generator.res2.filter1','generator.res2.bias1','generator.res2.filter2','generator.res2.bias2','generator',64,64,x,max_size)
        x = lyr.residual_block('generator.res3.filter1','generator.res3.bias1','generator.res3.filter2','generator.res3.bias2','generator',64,64,x,max_size)
        x = lyr.residual_block('generator.res4.filter1','generator.res4.bias1','generator.res4.filter2','generator.res4.bias2','generator',64,64,x,max_size)
        x = lyr.residual_block('generator.res5.filter1','generator.res5.bias1','generator.res5.filter2','generator.res5.bias2','generator',64,64,x,max_size)


        x = lyr.conv('generator.conv1.filter','generator.conv1.bias','generator',(5,64,encode_length),x,max_size)
        x = tf.nn.softmax(x)
        return x

    def discriminator(sequence):
        num = tf.shape(sequence)[0]
        
        x = lyr.conv('discriminator.conv1.filter','discriminator.conv1.bias','discriminator',(5,encode_length,64),sequence,max_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('discriminator.res1.filter1','discriminator.res1.bias1','discriminator.res1.filter2','discriminator.res1.bias1','discriminator',64,64,x,max_size)
        x = lyr.layernorm(x,num)
        x = lyr.residual_block('discriminator.res2.filter1','discriminator.res2.bias1','discriminator.res2.filter2','discriminator.res2.bias1','discriminator',64,64,x,max_size)
        x = lyr.layernorm(x,num)
        x = lyr.residual_block('discriminator.res3.filter1','discriminator.res3.bias1','discriminator.res3.filter2','discriminator.res3.bias1','discriminator',64,64,x,max_size)
        x = lyr.layernorm(x,num)
        x = lyr.residual_block('discriminator.res4.filter1','discriminator.res4.bias1','discriminator.res4.filter2','discriminator.res4.bias1','discriminator',64,64,x,max_size)
        x = lyr.layernorm(x,num)
        x = lyr.residual_block('discriminator.res5.filter1','discriminator.res5.bias1','discriminator.res5.filter2','discriminator.res5.bias1','discriminator',64,64,x,max_size)
        x = lyr.layernorm(x,num)
        
        x = tf.reshape(x,(num,max_size*64))
        
        output = lyr.dense('discriminator.dense1.matrix','discriminator.dense1.bias','discriminator',max_size*64,1,x)
        return output

### Define predictors
        
if 'head_stem' in tuner:
    def predictor_head(sequence):
        num = tf.shape(sequence)[0]
        
        x = lyr.conv('predictor_head.conv1.filter','predictor_head.conv1.bias','predictor_head',(5,encode_length,16),sequence,head_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('predictor_head.res1.filter1','predictor_head.res1.bias1','predictor_head.res1.filter2','predictor_head.res1.bias1','predictor_head',16,16,x,head_size,channels=16)
        
        x = tf.reshape(x,(num,head_size*16))
        
        output = lyr.dense('predictor_head.dense1.matrix','predictor_head.dense1.bias','predictor_head',head_size*16,num_classes,x)
        return output
        
    def predictor_stem(sequence):
        num = tf.shape(sequence)[0]
        
        x = lyr.conv('predictor_stem.conv1.filter','predictor_stem.conv1.bias','predictor_stem',(5,encode_length,16),sequence,stem_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('predictor_stem.res1.filter1','predictor_stem.res1.bias1','predictor_stem.res1.filter2','predictor_stem.res1.bias1','predictor_stem',16,16,x,stem_size,channels=16)
        
        x = tf.reshape(x,(num,stem_size*16))
        
        output = lyr.dense('predictor_stem.dense1.matrix','predictor_stem.dense1.bias','predictor_stem',stem_size*16,num_classes,x)
        return output    

elif 'subtype' in tuner:
    def predictor(sequence):
        num = tf.shape(sequence)[0]
        
        x = lyr.conv('predictor.conv1.filter','predictor.conv1.bias','predictor',(5,encode_length,16),sequence,max_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('predictor.res1.filter1','predictor.res1.bias1','predictor.res1.filter2','predictor.res1.bias1','predictor',16,16,x,max_size,channels=16)
        
        x = tf.reshape(x,(num,max_size*16))
        
        output = lyr.dense('predictor.dense1.matrix','predictor.dense1.bias','predictor',max_size*16,num_classes,x)
        return output

### Set up graph

training_switch = tf.placeholder(dtype=tf.dtypes.bool)

def sample_from_latents(x):
    means = x[:,:latent_dim]
    log_vars = x[:,latent_dim:]
    base = tf.keras.backend.random_normal(shape=[latent_dim,])
    return means + tf.exp(log_vars) * base

if args.model in ['vae_fc','vae_conv']:
    sequence_in = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)
    correct_labels = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)
    training = tf.placeholder(dtype=tf.dtypes.bool)
    
    output_for_printing = decoder(tf.random_normal((1,latent_dim)),training=tf.constant(False))[0]

    correct_labels_softmax = tf.nn.softmax(correct_labels)
    
    beta = tf.placeholder(dtype=tf.dtypes.float32)
    
    latent_seeds = encoder(sequence_in,training=training)
    latent = sample_from_latents(latent_seeds)

    
    logits = decoder(latent,training=training)
    predicted_character = tf.nn.softmax(logits)
    
    if args.encoding == 'categorical':
        accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels,logits)
    elif args.encoding == 'blosum':
        accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels_softmax,logits)

    def compute_kl_loss(latent_seeds):
        means = latent_seeds[:,:latent_dim]
        log_vars = latent_seeds[:,latent_dim:]
        
        kl = tf.reduce_mean(tf.square(means) + tf.exp(log_vars) - log_vars - 1.) * 0.5

        return kl

    kl_loss = compute_kl_loss(latent_seeds)

    loss = tf.reduce_mean(accuracy_loss + beta * kl_loss)

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    
elif args.model == 'vae_lstm':
    sequence_in = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)
    so_far_reconstructed = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)
    correct_labels = tf.placeholder(shape=[None,encode_length],dtype=tf.dtypes.float32)
    
    correct_labels_softmax = tf.nn.softmax(correct_labels)
    
    beta = tf.placeholder(float)

    latent_seeds_h,latent_seeds_c = encoder(sequence_in,encoder_lstm)
    latent_h = sample_from_latents(latent_seeds_h)
    latent_c = sample_from_latents(latent_seeds_c)
    latent = [latent_h,latent_c]

    logits = decoder(latent,so_far_reconstructed,decoder_lstm)
    predicted_character = tf.nn.softmax(logits)

    if args.encoding == 'categorical':
        accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels,logits)
    elif args.encoding == 'blosum':
        accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels_softmax,logits)

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

    loss = tf.reduce_mean(accuracy_loss + beta*kl_loss)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss,var_list=encoder_lstm.weights+decoder_lstm.weights)

elif args.model == 'gan':
    real_images = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)
    noise = tf.placeholder(float,name='noise')

    fake_images = generator(noise)
    fake_images = tf.identity(fake_images,name='fake_images')

    # Sampling images in the encoded space between the fake ones and the real ones

    interpolation_coeffs = tf.random_uniform(shape=(tf.shape(real_images)[0],1,1))
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

    disc_loss = tf.add(diff,tf.multiply(tf.constant(10.),score),name='disc_loss')

    gen_loss = - tf.reduce_mean(pred_fake,name='gen_loss')

    # Optimizers
    disc_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

    train_discriminator = disc_optimizer.minimize(disc_loss,var_list=tf.get_collection('discriminator'),name='train_discriminator')
    grads_discriminator = disc_optimizer.compute_gradients(disc_loss,var_list=tf.get_collection('discriminator'))

    train_generator = gen_optimizer.minimize(gen_loss,var_list=tf.get_collection('generator'),name='train_generator')
    grads_generator = gen_optimizer.compute_gradients(gen_loss,var_list=tf.get_collection('generator'))

loss_backtoback_tuner = 0.
    
if 'design' in tuner:
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[1,latent_dim])

    design,design_weights = design_parser(args.design)
    
    if args.model == 'gan':
        produced_tuner = generator(n_input)
    else:
        produced_tuner = decoder(n_input,training=tf.constant(False))
    
    target_tuner = {}
    if args.encoding == 'blosum':
        for key in design.keys():
            target_tuner[key] = tf.nn.softmax(CATEGORIES[design[key]])
    else:
        for key in design.keys():
            target_tuner[key] = CATEGORIES[design[key]]

    for key in design.keys():
        temp = tf.nn.softmax_cross_entropy_with_logits_v2(target_tuner[key],produced_tuner[0,key-1])
        loss_backtoback_tuner += design_weights[key] * temp
    
if 'head_stem' in tuner:
    input_sequence_head = tf.placeholder(shape=[None,head_size,encode_length],dtype=tf.dtypes.float32)
    input_sequence_stem = tf.placeholder(shape=[None,stem_size,encode_length],dtype=tf.dtypes.float32)

    label = tf.placeholder(shape=[None,num_classes],dtype=tf.dtypes.float32)

    prediction_logits_head = predictor_head(input_sequence_head)
    prediction_head = tf.nn.softmax(prediction_logits_head)

    prediction_logits_stem = predictor_stem(input_sequence_stem)
    prediction_stem = tf.nn.softmax(prediction_logits_stem)

    loss_head = tf.nn.softmax_cross_entropy_with_logits_v2(label,prediction_logits_head)
    loss_head = tf.reduce_mean(loss_head)

    loss_stem = tf.nn.softmax_cross_entropy_with_logits_v2(label,prediction_logits_stem)
    loss_stem = tf.reduce_mean(loss_stem)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_head = optimizer.minimize(loss_head,var_list=tf.get_collection('predictor_head'))
    train_stem = optimizer.minimize(loss_stem,var_list=tf.get_collection('predictor_stem'))
    
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[1,latent_dim])
        
    if args.model == 'gan':
        produced_tuner = tf.nn.softmax(generator(n_input))
    else:
        produced_tuner = tf.nn.softmax(decoder(n_input,training=tf.constant(False)))
    
    predicted_head_subtype_tuner = predictor_head(produced_tuner[:,132:277])
    predicted_stem_subtype_tuner = predictor_stem(tf.concat([produced_tuner[:,:132],produced_tuner[:,277:]],axis=1))
    predicted_head_tuner = tf.nn.softmax(predicted_head_subtype_tuner)
    predicted_stem_tuner = tf.nn.softmax(predicted_stem_subtype_tuner)

    target_head_tuner = tf.constant(cst.TYPES[headstem_parser(args.head_stem)[0]])
    target_stem_tuner = tf.constant(cst.TYPES[headstem_parser(args.head_stem)[1]])
    loss_head_tuner = tf.nn.softmax_cross_entropy_with_logits_v2(target_head_tuner,predicted_head_subtype_tuner[0])
    loss_stem_tuner = tf.nn.softmax_cross_entropy_with_logits_v2(target_stem_tuner,predicted_stem_subtype_tuner[0])

    loss_backtoback_tuner += tf.reduce_mean(loss_head_tuner + loss_stem_tuner)
    
if 'subtype' in tuner:
    input_sequence_predictor = tf.placeholder(shape=[None,max_size,encode_length],dtype=tf.dtypes.float32)
    label_predictor = tf.placeholder(shape=[None,num_classes],dtype=tf.dtypes.float32)
    prediction_logits_predictor = predictor(input_sequence_predictor)
    prediction_predictor = tf.nn.softmax(prediction_logits_predictor)
    loss_predictor = tf.nn.softmax_cross_entropy_with_logits_v2(label_predictor,prediction_logits_predictor)
    loss_predictor = tf.reduce_mean(loss_predictor)

    optimizer_predictor = tf.train.GradientDescentOptimizer(0.01)
    train_predictor = optimizer_predictor.minimize(loss_predictor,var_list=tf.get_collection('predictor'))
    
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[1,latent_dim])
        
    if args.model == 'gan':
        produced_tuner = tf.nn.softmax(generator(n_input))
    else:
        produced_tuner = tf.nn.softmax(decoder(n_input,training=tf.constant(False)))
        
    predicted_subtype_tuner = predictor(produced_tuner)
    predicted_tuner = tf.nn.softmax(predicted_subtype_tuner)
    target_tuner = tf.constant(cst.TYPES[args.subtype])
    loss_backtoback_tuner += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(target_tuner,predicted_subtype_tuner[0]))

tune = tf.train.AdamOptimizer().minimize(loss_backtoback_tuner,var_list=tf.get_collection('tuning_var'))

### Run

# Only for LSTM
def rec(sequence):
    new_sequence = np.zeros([1,1,encode_length])
    new_sequence[0,0,-1] = 1.
    
    new_sequence = sequence[:,:100,:]
    
    while (np.argmax(new_sequence[0,-1]) != np.argmax(EOM_VECTOR) and np.shape(new_sequence)[1] < 1000):
        character = sess.run(predicted_character,feed_dict={sequence_in:sequence,so_far_reconstructed:new_sequence})
        new_sequence = np.concatenate((new_sequence,character.reshape(1,1,encode_length)),axis=1)
    
    return cst.convert_to_string(new_sequence[0])
    
sess.run(tf.global_variables_initializer())

if args.model in ['vae_fc','vae_conv']:
    saver_model = tf.train.Saver(tf.get_collection('encoder') + tf.get_collection('decoder'))
elif args.model == 'vae_lstm':
    saver_model = tf.train.Saver(encoder_lstm.weights + decoder_lstm.weights)
elif args.model == 'gan':
    saver_model = tf.train.Saver(tf.get_collection('generator') + tf.get_collection('discriminator'))

if args.restore_model:
    saver_model.restore(sess,args.restore_model)

if 'subtype' in tuner and 'head_stem' in tuner:
    saver_predictor = tf.train.Saver(tf.get_collection('predictor') + tf.get_collection('predictor_head') + tf.get_collection('predictor_stem'))
elif 'head_stem' in tuner:
    saver_predictor = tf.train.Saver(tf.get_collection('predictor_head') + tf.get_collection('predictor_stem'))
elif 'subtype' in tuner:
    saver_predictor = tf.train.Saver(tf.get_collection('predictor'))
    
if args.restore_predictor:
    saver_predictor.restore(sess,args.restore_predictor)

print('#######################################')

print('Training model')
epochs = args.train_model_epochs
num_batches = int(np.floor(len(train_sequences)/batch_size))

if args.model in ['vae_fc','vae_conv']:
    print('initial')
    prediction = sess.run(output_for_printing)
    print(cst.convert_to_string(prediction,ORDER))
    
    for epoch in range(epochs):
        shuffled_sequences = np.random.permutation(train_sequences)
        for batch in range(num_batches):
            batch_sequences = shuffled_sequences[batch*batch_size:(batch+1)*batch_size]
            prev_iters = epoch*num_batches + batch
            total_iters = epochs*num_batches
            if args.restore_model:
                b = args.beta
            else:
                b = args.beta*(np.tanh((prev_iters-total_iters*0.4)/(total_iters*0.1))*0.5+0.5)
            _,l = sess.run([train,loss],feed_dict={sequence_in:batch_sequences,correct_labels:batch_sequences,beta:b,training:True})
            
        print('epoch',epoch,'loss',l)
        prediction = sess.run(output_for_printing)
        print(cst.convert_to_string(prediction,ORDER))
        saver_model.save(sess,args.save_model)

elif args.model == 'vae_lstm':
    for epoch in range(epochs):
        shuffled_sequences = np.random.permutation(train_sequences)
        for batch in range(num_batches):
            stop_point = np.random.randint(2,max_size-1)
            batch_sequences = shuffled_sequences[batch*batch_size:(batch+1)*batch_size,:1+stop_point]
            prev_iters = epoch*num_batches + batch
            total_iters = epochs*num_batches
            if args.restore_model:
                b = args.beta
            else:
                b = args.beta*(np.tanh((prev_iters-total_iters*0.4)/(total_iters*0.1))*0.5+0.5)
            _,l = sess.run([train,loss],feed_dict={sequence_in:batch_sequences[:,:-1],so_far_reconstructed:batch_sequences[:,:-2],correct_labels:batch_sequences[:,-1],beta:b})

        print('epoch',epoch,'loss',l)
        i = np.random.randint(0,max_size-1)
        test = train_sequences[i:i+1]
        print(rec(test))
        saver_model.save(sess,args.save_model)

elif args.model == 'gan':
    for epoch in range(epochs):
        print('\nepoch ',epoch)
        for batch in range(num_batches):
            batch_sequences = shuffled_sequences[batch*batch_size:(batch+1)*batch_size]
        # Train discriminator
            d_loss_delta = np.infty
            current_loss = np.infty
            for i in range(5):
                noise_input = np.random.normal(0,1,(batch_size,100))
                _,d_loss,grads = sess.run([train_discriminator,diff,grads_discriminator],feed_dict={real_images:batch_sequences,noise:noise_input})
                print('Training discriminator',d_loss)
                d_loss_delta = current_loss - d_loss
                current_loss = d_loss
                
        # Train generator
        for batch in range(num_batches):
            noise_input = np.random.normal(0,1,(batch_size,100))
            _,g_loss,grads = sess.run([train_generator,gen_loss,grads_generator],feed_dict={noise:noise_input})
            print('Training generator',g_loss)

        print('Generator loss: ',g_loss)
        print('Discriminator loss: ',d_loss)
            
        # Print a sample string    
        prediction = sess.run(tf.nn.softmax(generator(tf.random_normal((1,100)),training=tf.constant(False))))[0]
        print(cst.convert_to_string(prediction, ORDER))
        saver_model.save(sess, args.save_model)

print('\n')        
print('Training predictor')

epochs = args.train_predictor_epochs

if 'head_stem' in tuner:
    for epoch in range(epochs):
        shuffle = np.random.permutation(range(len(train_sequences)))
        for i in range(num_batches):
            batch = shuffle[i*batch_size:(i+1)*batch_size]
            sequence_batch = train_sequences[batch].astype('float32')
            sequence_batch_head = sequence_batch[:,cst.HEAD]
            sequence_batch_stem = sequence_batch[:,cst.STEM]
            label_batch = train_labels[batch].astype('float32')
            _,_,lh,ls,ph,ps = sess.run([train_head,train_stem,loss_head,loss_stem,prediction_head,prediction_stem],
                                       feed_dict={input_sequence_head:sequence_batch_head,input_sequence_stem:sequence_batch_stem,label:label_batch})
        print('Epoch', epoch)
        print('loss:', lh,ls)
        saver_predictor.save(sess, args.save_predictor)

elif 'subtype' in tuner:
    for epoch in range(epochs):
        shuffle = np.random.permutation(range(len(train_sequences)))
        for i in range(num_batches):
            batch = shuffle[i*batch_size:(i+1)*batch_size]
            sequence_batch = train_sequences[batch].astype('float32')
            label_batch = train_labels[batch].astype('float32')
            _,l = sess.run([train_predictor,loss_predictor],feed_dict={input_sequence_predictor:sequence_batch,label_predictor:label_batch})
        print('Epoch', epoch)
        print('loss:', l)
        saver_predictor.save(sess, args.save_predictor)
    fails = 0
    for _ in range(100):
        nums = np.random.permutation(range(len(test_sequences)))
        batch = test_sequences[nums[:batch_size]]
        preds = sess.run(prediction_predictor,feed_dict={input_sequence_predictor:batch})
        for i in range(len(preds)):
            if np.argmax(preds[i]) != np.argmax(test_labels[nums[i]]):
                fails += 1
    print((100*batch_size-fails)/(100*batch_size))

print('\n')
print('Tuning')

results = []
latents = []
subtypes = []

for i in range(int(args.num_outputs)):
    print('output number {}'.format(i))
    epochs = args.tune_epochs

    # ~ if args.tuner == 'head_stem':
        # ~ for i in range(epochs):
            # ~ _,l,p1,p2 = sess.run([tune,loss_backtoback_tuner,predicted_head_tuner,predicted_stem_tuner])
            # ~ if i%int(epochs/10)==0:
                # ~ print('Epoch',i,'loss',l)
                # ~ print(p1[0])
                # ~ print(p2[0])
                
    # ~ elif args.tuner == 'design':
        # ~ for i in range(epochs):
            # ~ _,l = sess.run([tune,loss_backtoback_tuner])
            # ~ if i%int(epochs/10)==0:
                # ~ print('Epoch',i,'loss',l)

    # ~ elif args.tuner == 'subtype':
        # ~ for i in range(epochs):
            # ~ _,l,p = sess.run([tune,loss_backtoback_tuner,predicted_tuner])
            # ~ if i%int(epochs/10)==0:
                # ~ print('Epoch',i,'loss',l)
                # ~ print(p[0])
                
    for i in range(epochs):
        _,l = sess.run([tune,loss_backtoback_tuner])
        if i%int(epochs/10)==0:
            print('Epoch',i,'loss',l)
            if 'subtype' in tuner:
                p = sess.run(predicted_tuner)
                print(p[0])

    tuned = sess.run(tf.nn.softmax(produced_tuner))
    lat = sess.run(n_input[0])
    results.append(cst.convert_to_string(tuned[0],ORDER))
    latents.append(lat)
    try:
        subtype = sess.run(prediction_predictor,feed_dict={input_sequence_predictor:tuned})[0]
        subtypes.append(np.argmax(subtype))
    except:
        print('no subtype predictor available')
    
    sess.run(n_input.initializer)

for i in range(int(args.num_outputs)):
    print('>sample{}'.format(i))
    print(results[i])
    if args.return_latents:
        print(latents[i])
print(subtypes)

sess.close()
