import numpy as np
import tensorflow as tf
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
# ~ sess = tf_debug.LocalCLIDebugWrapperSession(sess)
tf.keras.backend.set_session(sess)

### Arg parser
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, help='data to train on, one of "all", "h1", "h2", "h3", ..., "h18", or "aligned" (others are not aligned)', default='all')
parser.add_argument('--encoding', type=str, help='data encoding, either "categorical" or "blosum"', default='categorical')
parser.add_argument('--model', type=str, help='model to use, one of "gan", "vae_fc", "vae_conv", or "vae_lstm"', default='vae_fc')
parser.add_argument('--beta', type=float, help='if using a VAE, the coefficient for the KL loss', default=5)
parser.add_argument('--tuner', type=str, help='what to tune for, a combination of "subtype", "head_stem", or "design" (comma separated)')
parser.add_argument('--design', type=str, help='if using design tuner, list of strings "[position]-[residue]-[weight]" (weight is optional), e.g. "15-R-1.0,223-C-5.0"')
parser.add_argument('--subtype', type=int, help='if using subtype tuner, which subtype you want')
parser.add_argument('--head_stem', type=str, help='if using head-stem tuner, a string of "[head subtype],[stem subtype]"')
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
parser.add_argument('--reconstruct', type=str, help='a sequence that you want to encode and decode')
parser.add_argument('--print_from_latents', type=str, help='a comma-separated list of latent variable arrays that you want to decode')

args = parser.parse_args()

if args.random_seed:
    tf.set_random_seed(args.random_seed)

# Turn strings of [location]-[residue]-[weight] into dictionaries
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
    
def print_from_latents_parser(string):
    latent_strings = string.split('],[')
    latents = []
    for s in latent_strings:
        latent_array_strings = s.split(',')
        latent_array = map(lambda x: float(x.strip('[]')),latent_array_strings)
        latent_array = np.array(list(latent_array),dtype=np.float32)
        latents.append(latent_array)
    return latents

if args.tuner:
    tuner = args.tuner.split(',')
else:
    tuner = []

print('+-------------------------------------------------------+')
print('|  RUN INFORMATION                                      |')
print('|  ===================================================  |')
print('|  Model type: {: <41s}|'.format(args.model))
print('|  Latent dimension: {: <35}|'.format(args.latent_dimension))
if args.model == 'vae_conv': 
    print('|  Channels: {: <43}|'.format(args.channels))
else:
    print('|  Channels: not applicable                             |')
if args.model in ['vae_conv','vae_fc','vae_lstm']:
    print('|  Beta: {: <47}|'.format(args.beta))
else:
    print('|  Beta: not applicable                                 |')
print('|  Data encoding: {: <38s}|'.format(args.encoding))
if args.tuner: 
    print('|  Tuner: {: <46s}|'.format(args.tuner))
else: 
    print('|  Tuner: none                                          |')
print('|  Model epochs: {: <39}|'.format(args.train_model_epochs))
print('|  Predictor epochs: {: <35}|'.format(args.train_predictor_epochs))
print('|  Tune epochs: {: <40}|'.format(args.tune_epochs))
print('+-------------------------------------------------------+')

print('Subtype, Head/Stem, Design')
print(args.subtype, args.head_stem, args.design)
print()
print('Restore locations')
print(args.restore_model,args.restore_predictor)
print()
print('Save locations')
print(args.save_model,args.save_predictor)
print()

# Which array to convert from categorical to residue letter
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

# Select one subtype of data if requested
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
        x = lyr.batchnorm(x,'encoder.batchnorm1.offset','encoder.batchnorm1.scale','encoder.batchnorm1.average_means','encoder.batchnorm1.average_variances','encoder.num_means','encoder',(512,),training=training)
        
        x = lyr.dense('encoder.dense2.matrix','encoder.dense2.bias','encoder',512,512,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'encoder.batchnorm2.offset','encoder.batchnorm2.scale','encoder.batchnorm2.average_means','encoder.batchnorm2.average_variances','encoder.num_means','encoder',(512,),training=training)
        
        x = lyr.dense('encoder.dense3.matrix','encoder.dense3.bias','encoder',512,256,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'encoder.batchnorm3.offset','encoder.batchnorm3.scale','encoder.batchnorm3.average_means','encoder.batchnorm3.average_variances','encoder.num_means','encoder',(256,),training=training)
        
        x = lyr.dense('encoder.dense4.matrix','encoder.dense4.bias','encoder',256,latent_dim*2,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'encoder.batchnorm4.offset','encoder.batchnorm4.scale','encoder.batchnorm4.average_means','encoder.batchnorm4.average_variances','encoder.num_means','encoder',(latent_dim*2,),training=training)
        
        return x
	
    def decoder(state,training=tf.constant(True)):
        num = tf.shape(state)[0]
        
        x = tf.reshape(state,[num,-1])
        x = lyr.dense('decoder.dense1.matrix','decoder.dense1.bias','decoder',latent_dim,512,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'decoder.batchnorm1.offset','decoder.batchnorm1.scale','decoder.batchnorm1.average_means','decoder.batchnorm1.average_variances','decoder.num_means','decoder',(512,),training=training)
        
        x = lyr.dense('decoder.dense2.matrix','decoder.dense2.bias','decoder',512,512,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'decoder.batchnorm2.offset','decoder.batchnorm2.scale','decoder.batchnorm2.average_means','decoder.batchnorm2.average_variances','decoder.num_means','decoder',(512,),training=training)
        
        x = lyr.dense('decoder.dense3.matrix','decoder.dense3.bias','decoder',512,256,x)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'decoder.batchnorm3.offset','decoder.batchnorm3.scale','decoder.batchnorm3.average_means','decoder.batchnorm3.average_variances','decoder.num_means','decoder',(256,),training=training)

        x = lyr.dense('decoder.dense4.matrix','decoder.dense4.bias','decoder',256,max_size*encode_length,x)
        x = lyr.batchnorm(x,'decoder.batchnorm4.offset','decoder.batchnorm4.scale','decoder.batchnorm4.average_means','decoder.batchnorm4.average_variances','decoder.num_means','decoder',(max_size*encode_length,),training=training)
        
        x = tf.reshape(x,[num,max_size,encode_length])
        return x
        
elif args.model == 'vae_conv':
    def encoder(sequence,training=True):
        num = tf.shape(sequence)[0]
        
        x = lyr.conv('encoder.conv1.filter','encoder.conv1.bias','encoder',(5,encode_length,args.channels),sequence,max_size)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'encoder.batchnorm1.offset','encoder.batchnorm1.scale','encoder.batchnorm1.average_means','encoder.batchnorm1.average_variances','encoder.num_means','encoder',(max_size,args.channels),training=training)
        
        x = lyr.residual_block('encoder.res1.filter1','encoder.res1.bias1','encoder.res1.filter2','encoder.res1.bias1','encoder',args.channels,args.channels,x,max_size,channels=args.channels)
        x = lyr.batchnorm(x,'encoder.batchnorm2.offset','encoder.batchnorm2.scale','encoder.batchnorm2.average_means','encoder.batchnorm2.average_variances','encoder.num_means','encoder',(max_size,args.channels),training=training)
        
        x = tf.reshape(x,(num,max_size*args.channels))
        
        x = lyr.dense('encoder.dense1.matrix','encoder.dense1.bias','encoder',max_size*args.channels,2*latent_dim,x)
        x = tf.nn.leaky_relu(x)
        output = lyr.batchnorm(x,'encoder.batchnorm3.offset','encoder.batchnorm3.scale','encoder.batchnorm3.average_means','encoder.batchnorm3.average_variances','encoder.num_means','encoder',(2*latent_dim),training=training)
        
        return output
        
    def decoder(sequence,training=True):
        num = tf.shape(sequence)[0]
        
        x = lyr.dense('decoder.dense1.matrix','decoder.dense1.bias','decoder',latent_dim,max_size,sequence)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'decoder.batchnorm1.offset','decoder.batchnorm1.scale','decoder.batchnorm1.average_means','decoder.batchnorm1.average_variances','decoder.num_means','decoder',(max_size,),training=training)
        
        x = tf.reshape(x,(num,max_size,1))
        
        x = lyr.conv('decoder.conv1.filter','decoder.conv1.bias','decoder',(5,1,args.channels),x,max_size)
        x = tf.nn.leaky_relu(x)
        x = lyr.batchnorm(x,'decoder.batchnorm2.offset','decoder.batchnorm2.scale','decoder.batchnorm2.average_means','decoder.batchnorm2.average_variances','decoder.num_means','decoder',(max_size,args.channels),training=training)
        
        x = lyr.residual_block('decoder.res1.filter1','decoder.res1.bias1','decoder.res1.filter2','decoder.res1.bias1','decoder',args.channels,args.channels,x,max_size,channels=args.channels)
        x = lyr.batchnorm(x,'decoder.batchnorm3.offset','decoder.batchnorm3.scale','decoder.batchnorm3.average_means','decoder.batchnorm3.average_variances','decoder.num_means','decoder',(max_size,args.channels),training=training)
    
        x = lyr.conv('decoder.conv2.filter','decoder.conv2.bias','decoder',(5,args.channels,encode_length),x,max_size)
        x = tf.nn.leaky_relu(x)
        output = lyr.batchnorm(x,'decoder.batchnorm4.offset','decoder.batchnorm4.scale','decoder.batchnorm4.average_means','decoder.batchnorm4.average_variances','decoder.num_means','decoder',(max_size,encode_length),training=training)

        return output

elif args.model == 'vae_lstm':
    encoder_lstm = tf.keras.layers.CuDNNLSTM(latent_dim*2,return_state=True)
    decoder_lstm = tf.keras.layers.CuDNNLSTM(latent_dim)
    
    def encoder(sequence,training=True):
        out,h,c = encoder_lstm(sequence)
        return [h,c]
    
    def decoder(state,so_far,training=True):
        out = decoder_lstm(so_far,initial_state=state)
        logits = lyr.dense('decoder.dense.matrix','decoder.dense.bias','decoder',latent_dim,encode_length,out)
        return logits

elif args.model == 'gan':
    def generator(seed,training=tf.constant(True)):
        num = tf.shape(seed)[0]
        
        seed = tf.reshape(seed,(num,100))
        
        seed2 = lyr.dense('generator.dense1.matrix','generator.dense1.bias','generator',100,max_size*64,seed)
        seed2 = tf.nn.leaky_relu(seed2)
        seed2 = lyr.batchnorm(seed2,'generator.batchnorm1.offset','generator.batchnorm1.scale','generator.batchnorm1.average_means','generator.batchnorm1.average_variances','generator.num_means','generator',(max_size*64,),training=training)
        
        seed2 = tf.reshape(seed2,[num,max_size,64])

        x = lyr.residual_block('generator.res1.filter1','generator.res1.bias1','generator.res1.filter2','generator.res1.bias2','generator',64,64,seed2,max_size)
        x = lyr.batchnorm(x,'generator.batchnorm2.offset','generator.batchnorm2.scale','generator.batchnorm2.average_means','generator.batchnorm2.average_variances','generator.num_means','generator',(max_size,64),training=training)
        
        x = lyr.residual_block('generator.res2.filter1','generator.res2.bias1','generator.res2.filter2','generator.res2.bias2','generator',64,64,x,max_size)
        x = lyr.batchnorm(x,'generator.batchnorm3.offset','generator.batchnorm3.scale','generator.batchnorm3.average_means','generator.batchnorm3.average_variances','generator.num_means','generator',(max_size,64),training=training)
        
        x = lyr.residual_block('generator.res3.filter1','generator.res3.bias1','generator.res3.filter2','generator.res3.bias2','generator',64,64,x,max_size)
        x = lyr.batchnorm(x,'generator.batchnorm4.offset','generator.batchnorm4.scale','generator.batchnorm4.average_means','generator.batchnorm4.average_variances','generator.num_means','generator',(max_size,64),training=training)

        x = lyr.conv('generator.conv1.filter','generator.conv1.bias','generator',(5,64,encode_length),x,max_size)
        x = tf.nn.softmax(x)
        return x

    def discriminator(sequence,training=tf.constant(True)):
        num = tf.shape(sequence)[0]
        
        x = lyr.conv('discriminator.conv1.filter','discriminator.conv1.bias','discriminator',(5,encode_length,64),sequence,max_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('discriminator.res1.filter1','discriminator.res1.bias1','discriminator.res1.filter2','discriminator.res1.bias1','discriminator',64,64,x,max_size)
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

# Helper function to take means and log std deviations and output normally distributed variables.
def sample_from_latents(x):
    means = x[:,:latent_dim]
    log_vars = x[:,latent_dim:]
    base = tf.random_normal(shape=[latent_dim,])
    return means + tf.exp(log_vars) * base

if args.model in ['vae_fc','vae_conv']:
    sequence_in = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)       # Training sequence
    correct_labels = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)        # Goal reconstruction (should be training sequence again)
    
    training = tf.placeholder(dtype=tf.dtypes.bool)     # Whether this is training time or evaluation time
    
    output_for_printing = tf.nn.softmax(decoder(tf.random_normal((1,latent_dim)),training=tf.constant(False)))[0]      # Convenience tensor for printing
    
    beta = tf.placeholder(dtype=tf.dtypes.float32)      # Coefficient for KL loss
    
    latent_seeds = encoder(sequence_in,training=training)
    latent = sample_from_latents(latent_seeds)
    
    logits = decoder(latent,training=training)
    reconstruction = tf.nn.softmax(logits)
    
    if args.encoding == 'categorical':
        accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels,logits)
    elif args.encoding == 'blosum':         # Treat BLOSUM rows as logits
        correct_labels_softmax = tf.nn.softmax(correct_labels)
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
    sequence_in = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)       # Training sequence
    so_far_reconstructed = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)         # Training sequence up to a certain point
    correct_labels = tf.placeholder(shape=[None,encode_length],dtype=tf.dtypes.float32)        # What the next character should be
    
    beta = tf.placeholder(float)        # Coefficient for KL loss

    latent_seeds_h,latent_seeds_c = encoder(sequence_in)
    latent_h = sample_from_latents(latent_seeds_h)
    latent_c = sample_from_latents(latent_seeds_c)
    latent = [latent_h,latent_c]

    logits = decoder(latent,so_far_reconstructed)
    predicted_character = tf.nn.softmax(logits)

    if args.encoding == 'categorical':
        accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels,logits)
    elif args.encoding == 'blosum':         # Treat BLOSUM rows as logits
        correct_labels_softmax = tf.nn.softmax(correct_labels)
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
    real_images = tf.placeholder(shape=[None,None,encode_length],dtype=tf.dtypes.float32)       # Real samples
    noise = tf.placeholder(float,name='noise')      # Noise for generator
    training = tf.placeholder(tf.dtypes.bool)       # Whether this is training time

    fake_images = generator(noise)
    fake_images = tf.identity(fake_images,name='fake_images')
    
    output_for_printing = tf.nn.softmax(generator(noise,training=tf.constant(False)))[0]

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


loss_backtoback_tuner = 0.      # Loss tensor that all tuning objectives will add on to
    
if 'design' in tuner:
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[1,latent_dim])       # Latent variable to be tuned

    design,design_weights = design_parser(args.design)
    
    if args.model == 'gan':
        produced_tuner = generator(n_input,training=tf.constant(False))
    else:
        produced_tuner = decoder(n_input,training=tf.constant(False))
    
    # Construct dictionary of target residues
    target_tuner = {}
    if args.encoding == 'blosum':
        for key in design.keys():
            target_tuner[key] = tf.nn.softmax(CATEGORIES[design[key]])
    else:
        for key in design.keys():
            target_tuner[key] = CATEGORIES[design[key]]
    
    # Sum up all the cross entropies
    for key in design.keys():
        temp = tf.nn.softmax_cross_entropy_with_logits_v2(target_tuner[key],produced_tuner[0,key-1])
        loss_backtoback_tuner += design_weights[key] * temp
    
if 'head_stem' in tuner:
    input_sequence_head = tf.placeholder(shape=[None,head_size,encode_length],dtype=tf.dtypes.float32)      # Head domain of a training sequence
    input_sequence_stem = tf.placeholder(shape=[None,stem_size,encode_length],dtype=tf.dtypes.float32)      # Stem domain of a training sequence

    label = tf.placeholder(shape=[None,num_classes],dtype=tf.dtypes.float32)        # True label of training sequence

    prediction_logits_head = predictor_head(input_sequence_head)
    prediction_head = tf.nn.softmax(prediction_logits_head)

    prediction_logits_stem = predictor_stem(input_sequence_stem)
    prediction_stem = tf.nn.softmax(prediction_logits_stem)

    loss_head = tf.nn.softmax_cross_entropy_with_logits_v2(label,prediction_logits_head)
    loss_head = tf.reduce_mean(loss_head)

    loss_stem = tf.nn.softmax_cross_entropy_with_logits_v2(label,prediction_logits_stem)
    loss_stem = tf.reduce_mean(loss_stem)

    # Training the head-stem predictor
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_head = optimizer.minimize(loss_head,var_list=tf.get_collection('predictor_head'))
    train_stem = optimizer.minimize(loss_stem,var_list=tf.get_collection('predictor_stem'))
    
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[1,latent_dim])       # Latent variable to be tuned 
        
    if args.model == 'gan':
        produced_tuner = tf.nn.softmax(generator(n_input,training=tf.constant(False)))
    else:
        produced_tuner = tf.nn.softmax(decoder(n_input,training=tf.constant(False)))
    
    # Make predictions for head and stem
    predicted_head_subtype_tuner = predictor_head(produced_tuner[:,cst.HEAD_START:cst.HEAD_STOP])
    predicted_stem_subtype_tuner = predictor_stem(tf.concat([produced_tuner[:,:cst.HEAD_START],produced_tuner[:,cst.HEAD_STOP:]],axis=1))
    predicted_head_tuner = tf.nn.softmax(predicted_head_subtype_tuner)
    predicted_stem_tuner = tf.nn.softmax(predicted_stem_subtype_tuner)

    # Construct target subtype vectors
    target_head_tuner = tf.constant(cst.TYPES[headstem_parser(args.head_stem)[0]])
    target_stem_tuner = tf.constant(cst.TYPES[headstem_parser(args.head_stem)[1]])
    
    loss_head_tuner = tf.nn.softmax_cross_entropy_with_logits_v2(target_head_tuner,predicted_head_subtype_tuner[0])
    loss_stem_tuner = tf.nn.softmax_cross_entropy_with_logits_v2(target_stem_tuner,predicted_stem_subtype_tuner[0])

    loss_backtoback_tuner += tf.reduce_mean(loss_head_tuner + loss_stem_tuner)
    
if 'subtype' in tuner:
    input_sequence_predictor = tf.placeholder(shape=[None,max_size,encode_length],dtype=tf.dtypes.float32)      # Training sequence
    label_predictor = tf.placeholder(shape=[None,num_classes],dtype=tf.dtypes.float32)          # True label of training sequence
    
    prediction_logits_predictor = predictor(input_sequence_predictor)
    prediction_predictor = tf.nn.softmax(prediction_logits_predictor)
    
    loss_predictor = tf.nn.softmax_cross_entropy_with_logits_v2(label_predictor,prediction_logits_predictor)
    loss_predictor = tf.reduce_mean(loss_predictor)
    
    # Training the subtype predictor
    optimizer_predictor = tf.train.GradientDescentOptimizer(0.01)       
    train_predictor = optimizer_predictor.minimize(loss_predictor,var_list=tf.get_collection('predictor'))
    
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[1,latent_dim])       # Latent variable to be tuned 
        
    if args.model == 'gan':
        produced_tuner = tf.nn.softmax(generator(n_input,training=tf.constant(False)))
    else:
        produced_tuner = tf.nn.softmax(decoder(n_input,training=tf.constant(False)))
        
    # Make predictions    
    predicted_subtype_tuner = predictor(produced_tuner)
    predicted_tuner = tf.nn.softmax(predicted_subtype_tuner)
    
    # Construct target subtype vector 
    target_tuner = tf.constant(cst.TYPES[args.subtype])
    
    loss_backtoback_tuner += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(target_tuner,predicted_subtype_tuner[0]))

# Unless tuner is empty, set up tuning
if 'subtype' in tuner or 'head_stem' in tuner or 'design' in tuner:
    tuner_optimizer = tf.train.AdamOptimizer()
    tune = tuner_optimizer.minimize(loss_backtoback_tuner,var_list=tf.get_collection('tuning_var'))
    grads_tuner = tuner_optimizer.compute_gradients(loss_backtoback_tuner,var_list=tf.get_collection('tuning_var'))
    
### Run

# Only for LSTM, way to generate full sequence
# Keep predicting next character until getting an end of message character
def rec(sequence):
    new_sequence = np.zeros([1,1,encode_length])
    new_sequence[0,0,-1] = 1.
    
    while (np.argmax(new_sequence[0,-1]) != np.argmax(cst.EOM_VECTOR) and np.shape(new_sequence)[1] < 1000):
        character = sess.run(predicted_character,feed_dict={sequence_in:sequence,so_far_reconstructed:new_sequence})
        new_sequence = np.concatenate((new_sequence,character.reshape(1,1,encode_length)),axis=1)
    
    return cst.convert_to_string(new_sequence[0],ORDER)

sess.run(tf.global_variables_initializer())

# Set up saving and restoring

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
num_batches = int(np.floor(len(train_sequences)/batch_size)) # How many batches to get through the whole training set

if args.model in ['vae_fc','vae_conv']:
    # Print sample before any training
    print('initial')
    prediction = sess.run(output_for_printing)
    print(cst.convert_to_string(prediction,ORDER))
    
    # Train
    for epoch in range(epochs):
        shuffled_sequences = np.random.permutation(train_sequences)
        for batch in range(num_batches):
            batch_sequences = shuffled_sequences[batch*batch_size:(batch+1)*batch_size]     # Take a batch-worth of the shuffled sequences
            
            # Implement simulated annealing for the KL loss if training a vae for the first time (not restoring an old model)
            prev_iters = epoch*num_batches + batch
            total_iters = epochs*num_batches
            if args.restore_model:
                b = args.beta
            else:
                b = args.beta*(np.tanh((prev_iters-total_iters*0.4)/(total_iters*0.1))*0.5+0.5)
                
            _,l = sess.run([train,loss],feed_dict={sequence_in:batch_sequences,correct_labels:batch_sequences,beta:b,training:True})
            
        print('epoch',epoch+1,'loss',l)
        prediction = sess.run(output_for_printing)      # Sample
        print(cst.convert_to_string(prediction,ORDER))
        saver_model.save(sess,args.save_model)

elif args.model == 'vae_lstm':
    for epoch in range(epochs):
        shuffled_sequences = np.random.permutation(train_sequences)
        for batch in range(num_batches):
            stop_point = np.random.randint(2,max_size-1)        # Which character to make the model predict
            batch_sequences = shuffled_sequences[batch*batch_size:(batch+1)*batch_size,:1+stop_point]     # Take a batch-worth of the shuffled sequences, cut off at the cutoff point
            
            # Implement simulated annealing for the KL loss if training a vae for the first time (not restoring an old model)
            prev_iters = epoch*num_batches + batch
            total_iters = epochs*num_batches
            if args.restore_model:
                b = args.beta
            else:
                b = args.beta*(np.tanh((prev_iters-total_iters*0.4)/(total_iters*0.1))*0.5+0.5)
                
            _,l = sess.run([train,loss],feed_dict={sequence_in:batch_sequences[:,:-1],so_far_reconstructed:batch_sequences[:,:-2],correct_labels:batch_sequences[:,-1],beta:b})

        print('epoch',epoch+1,'loss',l)
        
        # Try a sample, starting at at random point
        i = np.random.randint(0,max_size-1)
        test = train_sequences[i:i+1]
        print(rec(test))
        saver_model.save(sess,args.save_model)

elif args.model == 'gan':
    for epoch in range(epochs):
        shuffled_sequences = np.random.permutation(train_sequences)
        print('\nepoch ',epoch+1)
        
        # Train discriminator
        for batch in range(num_batches):
            batch_sequences = shuffled_sequences[batch*batch_size:(batch+1)*batch_size]         # Take a batch-worth of the shuffled sequences
        
            # Track changes in loss in case you want to train until convergence (not used currently)
            d_loss_delta = np.infty
            current_loss = np.infty
            
            # Train discriminator 5 times per batch
            for i in range(5):
                noise_input = np.random.normal(0,1,(batch_size,100))
                _,d_loss = sess.run([train_discriminator,diff],feed_dict={real_images:batch_sequences,noise:noise_input})
                
                d_loss_delta = current_loss - d_loss
                current_loss = d_loss

                
        # Train generator
        for batch in range(num_batches):
            noise_input = np.random.normal(0,1,(batch_size,100))
            _,g_loss = sess.run([train_generator,gen_loss],feed_dict={noise:noise_input})

        print('Generator loss: ',g_loss)
        print('Discriminator loss: ',d_loss)
            
        # Print a sample string    
        prediction = sess.run(output_for_printing)
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
            sequence_batch = train_sequences[batch].astype('float32')       # Training sequences
            sequence_batch_head = sequence_batch[:,cst.HEAD]            # Training sequences head domain
            sequence_batch_stem = sequence_batch[:,cst.STEM]            # Training sequences stem domain
            label_batch = train_labels[batch].astype('float32')         # True subtype
            
            _,_,lh,ls,ph,ps = sess.run([train_head,train_stem,loss_head,loss_stem,prediction_head,prediction_stem],
                                       feed_dict={input_sequence_head:sequence_batch_head,input_sequence_stem:sequence_batch_stem,label:label_batch})
        
        print('Epoch', epoch+1)
        print('loss:', lh,ls)
        saver_predictor.save(sess, args.save_predictor)

elif 'subtype' in tuner:
    for epoch in range(epochs):
        shuffle = np.random.permutation(range(len(train_sequences)))
        for i in range(num_batches):
            batch = shuffle[i*batch_size:(i+1)*batch_size]
            sequence_batch = train_sequences[batch].astype('float32')       # Training sequences
            label_batch = train_labels[batch].astype('float32')             # True subtypes
            
            _,l = sess.run([train_predictor,loss_predictor],feed_dict={input_sequence_predictor:sequence_batch,label_predictor:label_batch})
        
        print('Epoch', epoch+1)
        print('loss:', l)
        saver_predictor.save(sess, args.save_predictor)
    
    # Do a quick measurement of test accuracy
    fails = 0
    for _ in range(100):
        nums = np.random.permutation(range(len(test_sequences)))
        batch = test_sequences[nums[:batch_size]]
        preds = sess.run(prediction_predictor,feed_dict={input_sequence_predictor:batch})       # Model predictions
        for i in range(len(preds)):
            # Test if incorrect
            if np.argmax(preds[i]) != np.argmax(test_labels[nums[i]]):
                fails += 1
    print((100*batch_size-fails)/(100*batch_size))

print('\n')
print('Tuning')

results = []        # Will hold a sequence for each output
latents = []        # Will hold latent variables for each output
subtypes = []       # Will hold a subtype prediction for each output

# Do a round of tuning for each output
if args.tuner:
    for i in range(int(args.num_outputs)):
        print('output number {}'.format(i))
        epochs = args.tune_epochs
                    
        for i in range(epochs):
            _,l = sess.run([tune,loss_backtoback_tuner])
            if i%int(epochs/10)==0:
                print('Epoch',i,'loss',l)
                
                # If the tuner is just subtype, it's nice to see what subtype is being predicted
                if 'subtype' in tuner:
                    p = sess.run(predicted_tuner)
                    print(p[0])
                
        tuned = sess.run(tf.nn.softmax(produced_tuner))     # Get tuned sequence
        lat = sess.run(n_input[0])                          # Get tuned latent variables
        results.append(cst.convert_to_string(tuned[0],ORDER))
        latents.append(lat)
        
        # If we have a subtype tuner, use it to predict subtypes (can be a nice summary)
        try:
            subtype = sess.run(predicted_tuner,feed_dict={input_sequence_predictor:tuned})[0]
            subtypes.append(np.argmax(subtype))
        except:
            print('no subtype predictor available')
        
        # Re-initialize tuning variable to something random to tune again
        sess.run(n_input.initializer)
        
# Print outputs of tuning
for i in range(len(results)):
    print('>sample{}'.format(i))
    print(results[i])
    if args.return_latents:
        print(latents[i])
print(subtypes)

# For the option to pass a sequence through the VAE
print('\n')
print('Reconstruction')
if args.reconstruct:
    print(args.reconstruct)
    encoded_original = cst.convert_to_encoding(args.reconstruct,CATEGORIES)
    encoded_original = np.expand_dims(encoded_original,axis=0)      # Need batch size of 1
    try:
        encoded_reconstructed = sess.run(reconstruction[0],feed_dict={sequence_in:encoded_original,training:False})
        print(cst.convert_to_string(encoded_reconstructed,ORDER))
    except NameError:           # if not a vae_conv or vae_fc, reconstruction won't exist (and shouldn't)
        print('model not compatible with reconstruction')

# For the option to print sequences from latent_variables
print('\n')
print('Printing from latents')
if args.print_from_latents:
    latents = print_from_latents_parser(args.print_from_latents)
    for latent in latents:
        if args.model in ['vae_fc','vae_conv']:
            result = sess.run(decoder(np.reshape(latent,(1,latent_dim)),training=tf.constant(False)))[0]
            print(cst.convert_to_string(result,ORDER))
        elif args.model == 'gan':
            result = sess.run(generator(np.reshape(latent,(1,latent_dim)),training=tf.constant(False)))[0]
            print(cst.convert_to_string(result,ORDER))
            
sess.close()
