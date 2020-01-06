import tensorflow as tf
import numpy as np


def temporal_module(x_input,in_channel,name,depth_size,kernel_size,stride,is_training):
    
    #x_input shape:(4,28,28,512)
    input_shape = x_input.get_shape().as_list()
    print("temporal_module input shape is :")
    print(input_shape)
    bottom = tf.reshape(x_input, (-1, input_shape[0],input_shape[1],input_shape[2],input_shape[3]))
    #bottom = tf.reshape(x_input,[1,input_shape[0],input_shape[1],input_shape[2],input_shape[3]])
    
    with tf.variable_scope(name, reuse=False) as scope:
        w = tf.get_variable("weights", shape=[depth_size,kernel_size, kernel_size, in_channel,
                                        in_channel], initializer=tf.random_normal_initializer(stddev=0.02))
        # b = tf.get_variable(
        #     "bias", shape=[out_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(bottom, w, strides=[
            1, stride, stride, stride,1], padding="SAME")
        #conv = tf.nn.bias_add(conv, b)
        conv = tf.layers.batch_normalization(conv,center=True,scale=True,training=is_training, name=name)
        conv = tf.nn.relu(conv)
        conv = tf.reshape(conv,(input_shape[0],input_shape[1],input_shape[2],input_shape[3]))

        print("temporal_module out shape is :")
        print(conv.get_shape().as_list()) 
        #x_input shape:(4,28,28,512)
        return conv


def TemporalXception(x_input, is_training):

    # 正确的话输入应该是（4,1,1,2048）
      
     input_shape = x_input.get_shape().as_list()
     print("TemporalXception: input tensor shape is\n")
     print(input_shape)

     #shape is:【4，2048】
     x_input = tf.squeeze(x_input, [1, 2]) 
     #reshape
     bottom = tf.reshape(x_input,[1,4,2048]) #(1,4,2048)

     with tf.variable_scope("Xception", reuse=False) as scope:

        conv = tf.layers.batch_normalization(bottom,center=True,scale=True,training=is_training,name="Xception_bn_0")
        
        w = tf.get_variable("weights", shape=[1, 2048,
                                        2048], initializer=tf.random_normal_initializer(stddev=0.02))

        conv = tf.nn.conv1d(conv, w, stride=1, padding="SAME")
        conv = tf.layers.SeparableConv1D(filters=2048,kernel_size=3,strides=1,padding="same")(conv)
        conv = tf.layers.batch_normalization(conv, center=True,scale=True,training=is_training,name="Xception_bn_1")
        conv = tf.nn.relu(conv)

        # >>> net = tl.layers.Input([8, 50, 64], name='input')
        # >>> separableconv1d = tl.layers.Conv1d(n_filter=32, filter_size=3, strides=2, padding='SAME', act=tf.nn.relu, name='separable_1d')(net)
        # >>> print(separableconv1d)
        # >>> output shape : (8, 25, 32)
        #debug

        input_shape = conv.get_shape().as_list()
        print("TemporalXception: relu out tensor shape is\n")
        print(input_shape)
        
        conv = tf.layers.MaxPooling1D(pool_size=4,strides=1,padding="VALID")(conv)

        #B = tf.nn.pool(conv, [5], 'MAX', 'SAME', strides = [5])
        #conv = tf.nn.pool(conv,[])
        # conv = tf.reshape(conv,[1,2,2,2048])
        # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                       padding="VALID", name=name)

        conv = tf.reshape(conv,(1,2048))
        print("TemporalXception: final out tensor shape is\n")
        input_shape = conv.get_shape().as_list()
        print(input_shape)
        return conv

def conv_layer(bottom, kernel_size, in_channel, out_channel, stride, name):
    with tf.variable_scope(name, reuse=False) as scope:
        w = tf.get_variable("weights", shape=[kernel_size, kernel_size, in_channel,
                                        out_channel], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(
            "bias", shape=[out_channel], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(bottom, w, strides=[
            1, stride, stride, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, b)
        return conv


def fc_layer(bottom, in_dims, out_dims, name):
    bottom = tf.reshape(bottom, shape=[-1, bottom.get_shape().as_list()[-1]])
    with tf.variable_scope(name, reuse=False) as scope:
        w = tf.get_variable("weights", shape=[
            in_dims, out_dims], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(
            "bias", shape=[out_dims], initializer=tf.constant_initializer(0.0))
        print()
        fc = tf.nn.bias_add(tf.matmul(bottom, w), b)
        return fc


def bn(inputTensor, is_training, name):
    # _BATCH_NORM_DECAY = 0.99
    # _BATCH_NORM_EPSILON = 1E-12
    return tf.layers.batch_normalization(inputTensor, training=is_training, name = name)


def avgpool(bottom, kernel_size=2, stride=2, name="avg"):
    return tf.nn.avg_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
                          padding="VALID", name=name)


def maxpool(bottom, kernel_size=2, stride=2, name="max"):
    return tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
                          padding="VALID", name=name)


def res_block_3_layer(bottom, channel_list, name, change_dimension=False, block_stride=1, is_training=True):
    with tf.variable_scope(name) as scope:
        if change_dimension:
            short_cut_conv = conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[2], block_stride,
                                        "shortcut")
            block_conv_input = bn(short_cut_conv, is_training, name="shortcut")
        else:
            block_conv_input = bottom
            
        block_conv1 = conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[0], block_stride,
                                 "a")
        block_conv1 = bn(block_conv1, is_training, name="a")
        block_conv1 = tf.nn.relu(block_conv1)
        block_conv2 = conv_layer(block_conv1, 3, channel_list[0], channel_list[1], 1, "b")
        block_conv2 = bn(block_conv2, is_training, name="b")
        block_conv2 = tf.nn.relu(block_conv2)
        block_conv3 = conv_layer(block_conv2, 1, channel_list[1], channel_list[2], 1, "c")
        block_conv3 = bn(block_conv3, is_training, name="c")

        block_res = tf.add(block_conv_input, block_conv3)
        relu = tf.nn.relu(block_res)
        return relu
