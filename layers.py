import tensorflow as tf
    
def dense(matrix,bias,collection,in_dim,out_dim,in_tensor):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        W = tf.get_variable(matrix,trainable=True,collections=[collection,tf.GraphKeys.GLOBAL_VARIABLES],shape=[in_dim,out_dim])
        b = tf.get_variable(bias,trainable=True,collections=[collection,tf.GraphKeys.GLOBAL_VARIABLES],shape=[out_dim,])

    
    return tf.matmul(in_tensor,W) + b    
    
def batchnorm(sequence,offset_name,scale_name,total_means_name,total_variances_name,num_name,collection,shape,training):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        offset = tf.get_variable(offset_name,trainable=True,collections=[collection,tf.GraphKeys.GLOBAL_VARIABLES],initializer=tf.zeros(shape))
        scale = tf.get_variable(scale_name,trainable=True,collections=[collection,tf.GraphKeys.GLOBAL_VARIABLES],initializer=tf.ones(shape))
        
        total_means = tf.get_variable(total_means_name,trainable=False,initializer=tf.zeros(shape))
        total_variances = tf.get_variable(total_variances_name,trainable=False,initializer=tf.zeros(shape))
        num = tf.get_variable(num_name,trainable=False,initializer=0.)
    
    def if_training():
        means,variances = tf.nn.moments(sequence,axes=[0])
        total_means.assign_add(means)
        total_variances.assign_add(variances)
        num.assign_add(1.)
        return means,variances
        
    def if_not_training():
        means = tf.div_no_nan(total_means,num,name='avg_means')
        variances = tf.div_no_nan(total_variances,num,name='avg_vars')
        return means,variances
    
    means,variances = tf.cond(training,if_training,if_not_training)

    normalized = tf.nn.batch_normalization(sequence,means,variances,offset,scale,tf.constant(0.001))
    
    return normalized
    
def layernorm(sequence,batch_size):
    means,variances = tf.nn.moments(sequence,axes=[1,2])
    means = tf.reshape(means,[batch_size,1,1])
    variances = tf.reshape(variances,[batch_size,1,1])
    return tf.divide(tf.subtract(sequence,means),variances)
    
def conv(filter_name,bias_name,model,filter_shape,in_tensor,size,padding='SAME',stride=1):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        filt = tf.get_variable(filter_name,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=filter_shape)
        bias = tf.get_variable(bias_name,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[size,filter_shape[-1]])
        
    out = tf.nn.conv1d(in_tensor,filters=filt,padding=padding,stride=stride)
    out = tf.add(out,bias)
    return out
    
def residual_block(filter_name1,bias_name1,filter_name2,bias_name2,model,in_dim,out_dim,in_tensor,size,channels=64):
    with tf.variable_scope('',reuse=tf.AUTO_REUSE):
        filter1 = tf.get_variable(filter_name1,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[5,in_dim,channels])
        bias1 = tf.get_variable(bias_name1,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[size,channels])
        filter2 = tf.get_variable(filter_name2,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[5,channels,out_dim])
        bias2 = tf.get_variable(bias_name2,collections=[model,tf.GraphKeys.GLOBAL_VARIABLES],trainable=True,shape=[size,out_dim])

        x = in_tensor
        x = tf.nn.leaky_relu(x)
        
        x = tf.nn.conv1d(x,filters=filter1,padding='SAME',stride=1)
        x = tf.add(x,bias1)
        
        x = tf.nn.leaky_relu(x)
        x = tf.nn.conv1d(x,filters=filter2,padding='SAME',stride=1)
        x = tf.add(x,bias2)

    return x+0.3*in_tensor
