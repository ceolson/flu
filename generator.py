import deepchem as dc
import numpy as np
import tensorflow as tf
from Bio import SeqIO
import h5py
import scipy
import argparse

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

parser.add_argument('-d', '--data', type=str, help='data to train on, one of "all", "h1", "h2", "h3", ..., "h18", or "aligned" (others are not aligned)', default='all')
parser.add_argument('-e', '--encoding', type=str, help='data encoding, either "categorical" or "blosum"', default='categorical')
parser.add_argument('-m', '--model', type=str, help='model to use, one of "gan", "vae_fc", or "vae_lstm', default='vae_fc')
parser.add_argument('-b', '--beta', type=float, help='if using a VAE, the value for beta', default=1)
parser.add_argument('-t', '--tuner', type=str, help='what to tune for, one of "subtype", "head-stem", or "design"', default='design')
parser.add_argument('--design', type=dict, help='if using design tuner, dictionary of where you want residues, e.g. "{15: \'R\', 334: \'C\'}"', default={1:'M'})
parser.add_argument('--subtype', type=int, help='if using subtype tuner, which subtype you want', default=1)
parser.add_argument('--head_stem', type=tuple, help='if using head-stem tuner, a tuple of (head_subtype, stem_subtype)', default=(1,1))
parser.add_argument('--train_model_epochs', type=int, default=0)
parser.add_argument('--train_predictor_epochs', type=int, default=0)
parser.add_argument('--tune_epochs', type=int, default=0)
parser.add_argument('--test_with_valid', action='store_true')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--latent_dimension', type=int, default=100)
parser.add_argument('--restore', help='saved file to restore from')
parser.add_argument('--save', help='file to save to', default='/home/ceolson0/Documents/flu/saves/generic/')

args = parser.parse_args()

if args.encoding == 'categorical':
    ORDER = cst.ORDER_CATEGORICAL
elif args.encoding == 'blosum':
    ORDER = cst.ORDER_BLOSUM

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

if args.data != 'all':
    subtype = int(args.data[1:])
    
    temp = []
    for i in range(len(train_sequences)):
        if train_labels[i] == subtype:
            temp.append(train_sequences[i])
    train_sequences = np.array(temp)
    
    temp = []
    for i in range(len(valid_sequences)):
        if valid_labels[i] == subtype:
            temp.append(valid_sequences[i])
    valid_sequences = np.array(temp)
    
    temp = []
    for i in range(len(test_sequences)):
        if test_labels[i] == subtype:
            temp.append(test_sequences[i])
    test_sequences = np.array(temp)

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
    
if args.model == 'vae_fc':
    def encoder(sequence,training=True):
        x = tf.reshape(sequence,[batch_size,max_size*encode_length])
        x = lyr.dense('encoder.lyr.dense1.matrix','encoder.lyr.dense1.bias','encoder',max_size*encode_length,512,x)
        x = tf.nn.leaky_relu(x)
        if training: x = lyr.batchnorm(x,'encoder.batchnorm1.offset','encoder.batchnorm1.scale','encoder')
        
        x = lyr.dense('encoder.lyr.dense2.matrix','encoder.lyr.dense2.bias','encoder',512,512,x)
        x = tf.nn.leaky_relu(x)
        if training: x = lyr.batchnorm(x,'encoder.batchnorm2.offset','encoder.batchnorm2.scale','encoder')
        
        x = lyr.dense('encoder.lyr.dense3.matrix','encoder.lyr.dense3.bias','encoder',512,256,x)
        x = tf.nn.leaky_relu(x)
        if training: x = lyr.batchnorm(x,'encoder.batchnorm3.offset','encoder.batchnorm3.scale','encoder')
        
        x = lyr.dense('encoder.lyr.dense4.matrix','encoder.lyr.dense4.bias','encoder',256,latent_dim*2,x)
        x = tf.nn.leaky_relu(x)
        if training: x = lyr.batchnorm(x,'encoder.batchnorm4.offset','encoder.batchnorm4.scale','encoder')

        return x
	
    def decoder(state,training=True):
        x = tf.reshape(state,[batch_size,-1])
        x = lyr.dense('decoder.lyr.dense1.matrix','decoder.lyr.dense1.bias','decoder',latent_dim,512,x)
        x = tf.nn.leaky_relu(x)
        if training: x = lyr.batchnorm(x,'decoder.batchnorm1.offset','decoder.batchnorm1.scale','decoder')

        x = lyr.dense('decoder.lyr.dense2.matrix','decoder.lyr.dense2.bias','decoder',512,512,x)
        x = tf.nn.leaky_relu(x)
        if training: x = lyr.batchnorm(x,'decoder.batchnorm2.offset','decoder.batchnorm2.scale','decoder')

        
        x = lyr.dense('decoder.lyr.dense3.matrix','decoder.lyr.dense3.bias','decoder',512,256,x)
        x = tf.nn.leaky_relu(x)
        if training: x = lyr.batchnorm(x,'decoder.batchnorm3.offset','decoder.batchnorm3.scale','decoder')

        x = lyr.dense('decoder.lyr.dense4.matrix','decoder.lyr.dense4.bias','decoder',256,max_size*encode_length,x)
        x = tf.reshape(x,[batch_size,max_size,encode_length])
        if training: x = lyr.batchnorm(x,'decoder.batchnorm4.offset','decoder.batchnorm4.scale','decoder')

        return x

elif args.model == 'vae_lstm':
    encoder_lstm = tf.keras.layers.CuDNNLSTM(latent_dim*2,return_state=True)
    decoder_lstm = tf.keras.layers.CuDNNLSTM(latent_dim)
    
    def encoder(sequence):
        out,h,c = encoder_lstm(sequence)
        return [h,c]
    
    def decoder(state,so_far):
        out = decoder_lstm(so_far,initial_state=state)
        logits = lyr.dense('decoder.lyr.dense.matrix','decoder.lyr.dense.bias','decoder',latent_dim,encode_length,out)
        return logits

elif args.model == 'gan':
    def generator(seed,training=True):
        seed = tf.reshape(seed,(batch_size,100))
        
        seed2 = lyr.dense('generator.lyr.dense1.matrix','generator.lyr.dense1.bias','generator',100,max_size*16,seed)
        seed2 = tf.nn.leaky_relu(seed2)
        seed2 = tf.reshape(seed2,[batch_size,max_size,16])

        x = lyr.residual_block('generator.res1.filter1','generator.res1.bias1','generator.res1.filter2','generator.res1.bias2','generator',16,16,seed2)
        x = lyr.residual_block('generator.res2.filter1','generator.res2.bias1','generator.res2.filter2','generator.res2.bias2','generator',16,16,x)
        x = lyr.residual_block('generator.res3.filter1','generator.res3.bias1','generator.res3.filter2','generator.res3.bias2','generator',16,16,x)
        x = lyr.residual_block('generator.res4.filter1','generator.res4.bias1','generator.res4.filter2','generator.res4.bias2','generator',16,16,x)
        x = lyr.residual_block('generator.res5.filter1','generator.res5.bias1','generator.res5.filter2','generator.res5.bias2','generator',16,16,x)


        x = lyr.conv('generator.lyr.conv1.filter','generator.lyr.conv1.bias','generator',(5,16,encode_length),x,max_size)
        x = tf.nn.softmax(x)
        return x

    def discriminator(sequence):
        x = lyr.conv('discriminator.lyr.conv1.filter','discriminator.lyr.conv1.bias','discriminator',(5,encode_length,16),sequence,max_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('discriminator.res1.filter1','discriminator.res1.bias1','discriminator.res1.filter2','discriminator.res1.bias1','discriminator',16,16,x)
        x = lyr.layernorm(x)
        x = lyr.residual_block('discriminator.res2.filter1','discriminator.res2.bias1','discriminator.res2.filter2','discriminator.res2.bias1','discriminator',16,16,x)
        x = lyr.layernorm(x)
        x = lyr.residual_block('discriminator.res3.filter1','discriminator.res3.bias1','discriminator.res3.filter2','discriminator.res3.bias1','discriminator',16,16,x)
        x = lyr.layernorm(x)
        x = lyr.residual_block('discriminator.res4.filter1','discriminator.res4.bias1','discriminator.res4.filter2','discriminator.res4.bias1','discriminator',16,16,x)
        x = lyr.layernorm(x)
        x = lyr.residual_block('discriminator.res5.filter1','discriminator.res5.bias1','discriminator.res5.filter2','discriminator.res5.bias1','discriminator',16,16,x)
        x = lyr.layernorm(x)
        
        x = tf.reshape(x,(batch_size,max_size*16))
        
        output = lyr.dense('discriminator.lyr.dense1.matrix','discriminator.lyr.dense1.bias','discriminator',max_size*16,1,x)
        return output
        
if args.tuner == 'head_stem':
    def predictor_head(sequence):
        x = lyr.conv('predictor_head.lyr.conv1.filter','predictor_head.lyr.conv1.bias','predictor_head',(5,encode_length,64),sequence,head_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('predictor_head.res1.filter1','predictor_head.res1.bias1','predictor_head.res1.filter2','predictor_head.res1.bias1','predictor_head',64,64,x,head_size)
        x = lyr.residual_block('predictor_head.res2.filter1','predictor_head.res2.bias1','predictor_head.res2.filter2','predictor_head.res2.bias1','predictor_head',64,64,x,head_size)
        x = lyr.residual_block('predictor_head.res3.filter1','predictor_head.res3.bias1','predictor_head.res3.filter2','predictor_head.res3.bias1','predictor_head',64,64,x,head_size)
        x = lyr.residual_block('predictor_head.res4.filter1','predictor_head.res4.bias1','predictor_head.res4.filter2','predictor_head.res4.bias1','predictor_head',64,64,x,head_size)
        x = lyr.residual_block('predictor_head.res5.filter1','predictor_head.res5.bias1','predictor_head.res5.filter2','predictor_head.res5.bias1','predictor_head',64,64,x,head_size)
        
        x = tf.reshape(x,(batch_size,head_size*64))
        
        output = lyr.dense('predictor_head.lyr.dense1.matrix','predictor_head.lyr.dense1.bias','predictor_head',head_size*64,num_classes,x)
        return output
        
    def predictor_stem(sequence):
        x = lyr.conv('predictor_stem.lyr.conv1.filter','predictor_stem.lyr.conv1.bias','predictor_stem',(5,encode_length,64),sequence,stem_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('predictor_stem.res1.filter1','predictor_stem.res1.bias1','predictor_stem.res1.filter2','predictor_stem.res1.bias1','predictor_stem',64,64,x,stem_size)
        x = lyr.residual_block('predictor_stem.res2.filter1','predictor_stem.res2.bias1','predictor_stem.res2.filter2','predictor_stem.res2.bias1','predictor_stem',64,64,x,stem_size)
        x = lyr.residual_block('predictor_stem.res3.filter1','predictor_stem.res3.bias1','predictor_stem.res3.filter2','predictor_stem.res3.bias1','predictor_stem',64,64,x,stem_size)
        x = lyr.residual_block('predictor_stem.res4.filter1','predictor_stem.res4.bias1','predictor_stem.res4.filter2','predictor_stem.res4.bias1','predictor_stem',64,64,x,stem_size)
        x = lyr.residual_block('predictor_stem.res5.filter1','predictor_stem.res5.bias1','predictor_stem.res5.filter2','predictor_stem.res5.bias1','predictor_stem',64,64,x,stem_size)
        
        x = tf.reshape(x,(batch_size,stem_size*64))
        
        output = lyr.dense('predictor_stem.lyr.dense1.matrix','predictor_stem.lyr.dense1.bias','predictor_stem',stem_size*64,num_classes,x)
        return output    

elif args.tuner == 'subtype':
    def predictor(sequence):
        x = lyr.conv('predictor.lyr.conv1.filter','predictor.lyr.conv1.bias','predictor',(5,encode_length,64),sequence,max_size)
        x = tf.nn.leaky_relu(x)
        
        x = lyr.residual_block('predictor.res1.filter1','predictor.res1.bias1','predictor.res1.filter2','predictor.res1.bias1','predictor',64,64,x)
        x = lyr.residual_block('predictor.res2.filter1','predictor.res2.bias1','predictor.res2.filter2','predictor.res2.bias1','predictor',64,64,x)
        x = lyr.residual_block('predictor.res3.filter1','predictor.res3.bias1','predictor.res3.filter2','predictor.res3.bias1','predictor',64,64,x)
        x = lyr.residual_block('predictor.res4.filter1','predictor.res4.bias1','predictor.res4.filter2','predictor.res4.bias1','predictor',64,64,x)
        x = lyr.residual_block('predictor.res5.filter1','predictor.res5.bias1','predictor.res5.filter2','predictor.res5.bias1','predictor',64,64,x)
        
        x = tf.reshape(x,(batch_size,max_size*64))
        
        output = lyr.dense('predictor.lyr.dense1.matrix','predictor.lyr.dense1.bias','predictor',max_size*64,num_classes,x)
        return output

### Set up graph
def sample_from_latents(x):
    means = x[:,:latent_dim]
    log_vars = x[:,latent_dim:]
    base = tf.keras.backend.random_normal(shape=[latent_dim,])
    return means + tf.exp(log_vars) * base

if args.model == 'vae_fc':
    sequence_in = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
    correct_labels = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
    correct_labels_softmaxed = tf.nn.softmax(correct_labels)
    beta = tf.placeholder(dtype=tf.dtypes.float32)

    latent_seeds = encoder(sequence_in)
    latent = sample_from_latents(latent_seeds)


    logits = decoder(latent)
    predicted_character = tf.nn.softmax(logits)

    accuracy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(correct_labels_softmaxed,logits)

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
    sequence_in = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
    so_far_reconstructed = tf.placeholder(shape=[batch_size,None,encode_length],dtype=tf.dtypes.float32)
    correct_labels = tf.placeholder(shape=[batch_size,encode_length],dtype=tf.dtypes.float32)
    beta = tf.placeholder(float)

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

    loss = tf.reduce_mean(accuracy_loss + beta*kl_loss)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

elif args.model == 'gan':
    real_images = sequence_in
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

    disc_loss = tf.add(diff,tf.multiply(tf.constant(10.),score),name='disc_loss')

    gen_loss = - tf.reduce_mean(pred_fake,name='gen_loss')

    # Optimizers
    disc_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

    train_discriminator = disc_optimizer.minimize(disc_loss,var_list=tf.get_collection('discriminator'),name='train_discriminator')
    grads_discriminator = disc_optimizer.compute_gradients(disc_loss,var_list=tf.get_collection('discriminator'))

    train_generator = gen_optimizer.minimize(gen_loss,var_list=tf.get_collection('generator'),name='train_generator')
    grads_generator = gen_optimizer.compute_gradients(gen_loss,var_list=tf.get_collection('generator'))
    
if args.tuner == 'design':
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[batch_size,latent_dim])

    DESIGN = args.design
    designed_indices = list(map(lambda x: x-1,list(DESIGN.keys())))
        
    produced_tuner = decoder(n_input)

    target_tuner = [cst.CATEGORIES[DESIGN[key]] for key in DESIGN.keys()]

    loss_backtoback_tuner = 0
    for i in range(len(DESIGN.keys())):
        temp = tf.nn.softmax_cross_entropy_with_logits_v2(target_tuner[i],produced_tuner[0,designed_indices[i]])
        loss_backtoback_tuner += temp

    tune = tf.train.AdadeltaOptimizer().minimize(loss_backtoback_tuner,var_list=tf.get_collection('tuning_var'))
    
if args.tuner == 'head_stem':
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
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[batch_size,latent_dim])
        
    produced_tuner = tf.nn.softmax(decoder(n_input))
    predicted_head_subtype_tuner = predictor_head(produced_tuner[:,132:277])
    predicted_stem_subtype_tuner = predictor_stem(tf.concat([produced_tuner[:,:132],produced_tuner[:,277:]],axis=1))
    predicted_head_tuner = tf.nn.softmax(predicted_head_subtype_tuner)
    predicted_stem_tuner = tf.nn.softmax(predicted_stem_subtype_tuner)

    target_head_tuner = tf.stack([tf.constant(cst.TYPES[args.head_stem[0]]) for i in range(batch_size)],axis=0)
    target_stem_tuner = tf.stack([tf.constant(cst.TYPES[args.head_stem[1]]) for i in range(batch_size)],axis=0)
    loss_head_tuner = tf.nn.softmax_cross_entropy_with_logits_v2(target_head_tuner,predicted_head_subtype_tuner)
    loss_stem_tuner = tf.nn.softmax_cross_entropy_with_logits_v2(target_stem_tuner,predicted_stem_subtype_tuner)


    loss_backtoback_tuner = tf.reduce_mean(loss_head_tuner + loss_stem_tuner)
    tune = tf.train.GradientDescentOptimizer(0.01).minimize(loss_backtoback_tuner,var_list=tf.get_collection('tuning_var'))
    
if args.tuner == 'subtype':
    input_sequence_predictor = tf.placeholder(shape=[None,max_size,encode_length],dtype=tf.dtypes.float32)
    label_predictor = tf.placeholder(shape=[None,num_classes],dtype=tf.dtypes.float32)
    prediction_logits_predictor = predictor(input_sequence_predictor)
    prediction_predictor = tf.nn.softmax(prediction_logits_predictor)
    loss_predictor = tf.nn.softmax_cross_entropy_with_logits_v2(label_predictor,prediction_logits_predictor)
    loss_predictor = tf.reduce_mean(loss_predictor)

    optimizer_predictor = tf.train.GradientDescentOptimizer(0.01)
    train_predictor = optimizer.minimize(loss_predictor,var_list=tf.get_collection('predictor'))
    
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        n_input = tf.get_variable('n_input',trainable=True,collections=['tuning_var',tf.GraphKeys.GLOBAL_VARIABLES],shape=[batch_size,latent_dim])
        
    produced_tuner = tf.nn.softmax(decoder(n_input))
    predicted_subtype_tuner = predictor(produced_tuner)
    predicted_tuner = tf.nn.softmax(predicted_subtype_tuner)
    target_tuner = tf.stack([tf.constant(cst.TYPES[args.subtype]) for i in range(batch_size)],axis=0)
    loss_backtoback_tuner = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(target_tuner,predicted_subtype_tuner))
    tune = tf.train.GradientDescentOptimizer(0.01).minimize(loss_backtoback_tuner,var_list=tf.get_collection('tuning_var'))

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
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

if args.restore:
    saver.restore(sess,args.restore)

print('#######################################')

print('Training model')
epochs = args.train_model_epochs

if args.model == 'vae_fc':
    for epoch in range(epochs):
        b = 5*(np.tanh((epoch-epochs*0.4)/(epochs*0.1))*0.5+0.5)
        batch_sequences = np.random.permutation(train_sequences)[:batch_size]
        _,l = sess.run([train,loss],feed_dict={sequence_in:batch_sequences,correct_labels:batch_sequences,beta:b})
        if epoch%int(epochs/100)==0: 
            print('epoch',epoch,'loss',l)
            prediction = sess.run(tf.nn.softmax(decoder(tf.random_normal([batch_size,latent_dim]),training=False)))[0]
            print(cst.convert_to_string(prediction,ORDER))
        if epoch%1000==0:
            saver.save(sess,args.save)

elif args.model == 'vae_lstm':
    for epoch in range(epochs):
        b = 5*(np.tanh((epoch-epochs*0.4)/(epochs*0.1))*0.5+0.5)
        stop_point = np.random.randint(2,max_size-1)
        batch_sequences = np.random.permutation(train_sequences)[:batch_size,:1+stop_point]
        _,l = sess.run([train,loss],feed_dict={sequence_in:batch_sequences[:,:-1],so_far_reconstructed:batch_sequences[:,:-2],correct_labels:batch_sequences[:,-1],beta:b})
        if epoch%int(epochs/100)==0: 
            print('epoch',epoch,'loss',l)
            i = np.random.randint(0,max_size-1)
            test = train_sequences[i:i+1]
            print(rec(test))
            saver.save(sess,args.save)

elif args.model == 'gan':
    for epoch in range(epochs):
        print('\nepoch ',epoch)

        # Train discriminator
        d_loss_delta = np.infty
        current_loss = np.infty
        for i in range(5):
            real = np.random.permutation(train_sequences)[:batch_size].astype(np.float16)
            noise_input = np.random.normal(0,1,(batch_size,100))
            _,d_loss,grads = sess.run([train_discriminator,diff,grads_discriminator],feed_dict={real_images:real,noise:noise_input})
            print('Training discriminator',d_loss)
            d_loss_delta = current_loss - d_loss
            current_loss = d_loss
                
        # Train generator

        real = np.random.permutation(train_sequences)[:batch_size].astype(np.float16)
        noise_input = np.random.normal(0,1,(batch_size,100))
        _,g_loss,grads = sess.run([train_generator,gen_loss,grads_generator],feed_dict={noise:noise_input})
        print('Training generator',g_loss)

        print('Generator loss: ',g_loss)
        print('Discriminator loss: ',d_loss)
            
        # Print a sample string    
        prediction = sess.run(generator(tf.random_normal((batch_size,100)),training=False))[0]
        print(cst.convert_to_string(prediction, ORDER))
        saver.save(sess, args.save)

print('\n')        
print('Training predictor')

epochs = args.train_predictor_epochs

if args.tuner == 'head_stem':
    for epoch in range(epochs):
        batch = np.random.permutation(range(len(train_sequences)))[:batch_size]
        sequence_batch = train_sequences[batch].astype('float32')
        sequence_batch_head = sequence_batch[:,cst,HEAD]
        sequence_batch_stem = sequence_batch[:,cst.STEM]
        label_batch = train_labels[batch].astype('float32')
        _,_,lh,ls,ph,ps = sess.run([train_head,train_stem,loss_head,loss_stem,prediction_head,prediction_stem],feed_dict={input_sequence_head:sequence_batch_head,input_sequence_stem:sequence_batch_stem,label:label_batch})
        if epoch%int(epochs/10) == 0:
            print('Epoch', epoch)
            print('loss:', lh,ls)
        saver.save(sess, args.save)

elif args.tuner == 'subtype':
    for epoch in range(epochs):
        batch = np.random.permutation(range(len(train_sequences)))[:batch_size]
        sequence_batch = train_sequences[batch].astype('float32')
        label_batch = train_labels[batch].astype('float32')
        _,l = sess.run([train_predictor,loss_predictor],feed_dict={input_sequence_predictor:sequence_batch,label_predictor:label_batch})
        if epoch%int(epochs/10) == 0:
            print('Epoch', epoch)
            print('loss:', l)
        saver.save(sess, args.save)

print('\n')
print('Tuning')

epochs = args.tune_epochs

if args.tuner == 'head_stem':
    for i in range(epochs):
        _,l,p1,p2 = sess.run([tune,loss_backtoback_tuner,predicted_head_tuner,predicted_stem_tuner])
        if i%int(epochs/10)==0:
            print('Epoch',i,'loss',l)
            print(p1[0])
            print(p2[0])
            
elif args.tuner == 'design':
    for i in range(epochs):
        _,l = sess.run([tune,loss_backtoback_tuner])
        if i%int(epochs/10)==0:
            print('Epoch',i,'loss',l)

elif args.tuner == 'subtype':
    for i in range(epochs):
        _,l,p = sess.run([tune,loss_backtoback_tuner,predicted_tuner])
        if i%int(epochs/10)==0:
            print('Epoch',i,'loss',l)
            print(p[0])
            
tuned = np.random.permutation(sess.run(produced_tuner))
for x in tuned:
    print(cst.convert_to_string(x, ORDER))
