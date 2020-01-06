import tensorflow as tf
from util.ops import res_block_3_layer, bn, conv_layer, maxpool, avgpool, fc_layer,temporal_module,TemporalXception
import numpy as np


class ResNet(object):
    def __init__(self, resnet_npy_path=None, trainable=True):
        if resnet_npy_path is not None:
            #npy内容由一个字典组成，字典中的每一个健值对应一层网络参数（权重和偏置）
            self.data_dict = np.load(resnet_npy_path, encoding='latin1', allow_pickle=True).item()
        else:
            self.data_dict = None
        #print(self.data_dict)
        self.var_dict = {}
        self.trainable = trainable
        self.is_training = True

    def build(self, rgb, label_num,last_layer_type="softmax"):

        assert rgb.get_shape().as_list()[1:] == [224, 224, 3]
        # 把图像每两个变成一个超图  输入通道是6
        rgb = tf.reshape(rgb,[4,224,224,6])
        input_shape = rgb.get_shape().as_list()
        print("resnet input shape is :")
        print(input_shape)
        
        self.conv1 = conv_layer(rgb, 7, 6, 64, 2, "scale1")
        #self.conv1 = conv_layer(rgb, 7, 3, 64, 2, "scale1")
        self.conv1 = bn(self.conv1, is_training=self.is_training, name="scale1")
        self.conv1 = tf.nn.relu(self.conv1)
        self.conv1 = maxpool(self.conv1, 3, 2, "pool1")

        with tf.variable_scope("scale2"):
            self.block1_1 = res_block_3_layer(self.conv1, [64, 64, 256], "block1", change_dimension=True,
                                              block_stride=1, is_training=self.is_training)
            self.block1_2 = res_block_3_layer(self.block1_1, [64, 64, 256], "block2", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            self.block1_3 = res_block_3_layer(self.block1_2, [64, 64, 256], "block3", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)

        with tf.variable_scope("scale3"):
            self.block2_1 = res_block_3_layer(self.block1_3, [128, 128, 512], "block1", change_dimension=True,
                                              block_stride=2, is_training=self.is_training)
            self.block2_2 = res_block_3_layer(self.block2_1, [128, 128, 512], "block2", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            self.block2_3 = res_block_3_layer(self.block2_2, [128, 128, 512], "block3", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            self.block2_4 = res_block_3_layer(self.block2_3, [128, 128, 512], "block4", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)

            #########add by liwx##############################################
            self.block2_4 = temporal_module(self.block2_4,512,"temporal_0",3,1,1,is_training=self.is_training)


        with tf.variable_scope("scale4"):
            self.block3_1 = res_block_3_layer(self.block2_4, [256, 256, 1024], "block1", change_dimension=True,
                                              block_stride=2, is_training=self.is_training)
            self.block3_2 = res_block_3_layer(self.block3_1, [256, 256, 1024], "block2", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            self.block3_3 = res_block_3_layer(self.block3_2, [256, 256, 1024], "block3", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            self.block3_4 = res_block_3_layer(self.block3_3, [256, 256, 1024], "block4", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            self.block3_5 = res_block_3_layer(self.block3_4, [256, 256, 1024], "block5", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            self.block3_6 = res_block_3_layer(self.block3_5, [256, 256, 1024], "block6", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            ###########add by liwx#########################################
            self.block3_6 = temporal_module(self.block3_6,1024,"temporal_1",3,1,1,is_training=self.is_training)


        with tf.variable_scope("scale5"):
            self.block4_1 = res_block_3_layer(self.block3_6, [512, 512, 2048], "block1", change_dimension=True,
                                              block_stride=2, is_training=self.is_training)
            self.block4_2 = res_block_3_layer(self.block4_1, [512, 512, 2048], "block2", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)
            self.block4_3 = res_block_3_layer(self.block4_2, [512, 512, 2048], "block3", change_dimension=False,
                                              block_stride=1, is_training=self.is_training)

        with tf.variable_scope("fc"):

            self.pool2 = avgpool(self.block4_3, 7, 1, "pool2")
            #self.fc1 = fc_layer(self.pool2, 2048, 2048, "fc1")
            #self.fc1 = tf.nn.relu(tf.nn.dropout(self.fc1, keep_prob=kp))
            self.fc1 = TemporalXception(self.pool2, is_training=self.is_training)

            self.fc2 = fc_layer(self.fc1, 2048, label_num, "fc2")

        if last_layer_type == "sigmoid":
            self.prob = tf.nn.sigmoid(self.fc2)
        elif last_layer_type == "softmax":
            self.prob = tf.nn.softmax(self.fc2)
        elif last_layer_type == "no":
            self.prob = self.fc2
        return self.prob

    
    def load_weights(self, sess):
        print("loading model")
        var = tf.global_variables()
        for item in var:
            #print(item.name)
            # if str(item.name).startswith("scale1/weights"):
                
            #     #print("match!!!!!!!!!!!!!!!!!!!!!!!!!")
            #      #[7,7,3,64] ->> [7,7,6,64]
            #     self.data_dict[item.name]=np.repeat(self.data_dict[item.name],2,axis=2)
            #     #print( self.data_dict[item.name].shape)

            if item.name in self.data_dict:

                if str(item.name).startswith("scale1/weights"):
                
                #print("match!!!!!!!!!!!!!!!!!!!!!!!!!")
                 #[7,7,3,64] ->> [7,7,6,64]
                    self.data_dict[item.name]=np.repeat(self.data_dict[item.name],2,axis=2)
                #print( self.data_dict[item.name].shape)
                
                #print("loading parameters " + str(item))
                sess.run(item.assign(self.data_dict[item.name]))


    def save_weights(self, path):
        print("saving weights")
        var = tf.global_variables()
        for item in var:
            self.var_dict[item.name] = item.eval()
        print("saving weights in npy model")
        np.save(path, self.var_dict)
