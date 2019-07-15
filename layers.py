import tensorflow as tf
    
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
    
def layernorm(sequence):
    means,variances = tf.nn.moments(sequence,axes=[1,2])
    means = tf.reshape(means,[batch_size,1,1])
    variances = tf.reshape(variances,[batch_size,1,1])
    return tf.divide(tf.subtract(sequence,means),variances)
    
def conv(filter_name,bias_name,model,filter_shape,in_tensor,size):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        filt = tf.get_variable(filter_name,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=filter_shape)
        bias = tf.get_variable(bias_name,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[size,filter_shape[-1]])
        
    out = tf.nn.conv1d(in_tensor,filters=filt,padding='SAME',stride=1)
    out = tf.add(out,bias)
    return out
    
def residual_block(filter_name1,bias_name1,filter_name2,bias_name2,model,in_dim,out_dim,in_tensor,size):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        filter1 = tf.get_variable(filter_name1,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[5,in_dim,64])
        bias1 = tf.get_variable(bias_name1,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[size,64])
        filter2 = tf.get_variable(filter_name2,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[5,64,out_dim])
        bias2 = tf.get_variable(bias_name2,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[size,out_dim])

        x = in_tensor
        x = tf.nn.leaky_relu(x)
        
        x = tf.nn.conv1d(x,filters=filter1,padding='SAME',stride=1)
        x = tf.add(x,bias1)
        
        x = tf.nn.leaky_relu(x)
        x = tf.nn.conv1d(x,filters=filter2,padding='SAME',stride=1)
        x = tf.add(x,bias2)

    return x+0.3*in_tensor
