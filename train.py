import tensorflow as tf
import numpy as np
from model.ResNet import ResNet
#from util.readtxt import read_data
import cv2
import os
from opts import parser
import datasets_video
import input_data
import transforms
slim = tf.contrib.slim
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def adjust_learning_rate(epoch, learning_rate, lr_steps):
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = learning_rate * decay
    return lr     
     
def train(sess, train_loader, predictions, train_op, params):
    print('************************* Training ***********************')
    if params['epoch'] % 5 == 0:
        print('********************* get hard negatives ******************')
        # sample_numbers指定负样本数量，若不指定则正负样本数量保持一致
        train_loader.select_harder_negatives(sess, params, sample_numbers=1300)

    true_top1 = AverageMeter()
    fake_top1 = AverageMeter()
    losses = AverageMeter()
    for i, data in enumerate(train_loader.run()):
        if data is None:
            break
       # print("input data shape is\n:")
        #print(len(data[0]))
        output, loss = sess.run([predictions, train_op], feed_dict={params['x_']: data[0],params['y_']: data[1], params['lr_']: params['lr']})
        acc = np.mean((np.equal(output, data[1])).astype(int))
        fake_acc = accuracy(output, data[1])

        true_top1.update(acc)
        fake_top1.update(fake_acc)
        losses.update(loss, len(data[1]))
        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}]\t'
                      'lr: {lr:.5f}\t'
                      'Loss: {loss:.5f}\t'
                      'Fake_top1: {fake_top1:.5f}\t'
                      'True_top1: {top1:.5f}'.format(params['epoch'], i, len(train_loader),
                                                     lr=params['lr'], loss=losses.avg,
                                                     fake_top1=fake_top1.avg, top1=true_top1.avg))
            print(output)

def valid(sess, val_loader, predictions, loss_, params):
    print('************************ Validation *********************')
    true_top1 = AverageMeter()
    fake_top1 = AverageMeter()
    losses = AverageMeter()
    for i, data in enumerate(val_loader.run()):
        if data is None:
            break
        output, loss = sess.run([predictions, loss_], feed_dict={params['x_']: data[0], params['y_']: data[1]})
        acc = np.mean((np.equal(output, data[1])).astype(int))
        fake_acc = accuracy(output, data[1])
        true_top1.update(acc)
        fake_top1.update(fake_acc)
        losses.update(loss, len(data[1]))
        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}]\t'
                      'lr: {lr:.5f}\t'
                      'Loss: {loss:.5f}\t'
                      'Fake_top1: {fake_top1:.5f}\t'
                      'True_top1: {top1:.5f}'.format(params['epoch'], i, len(val_loader),
                                              lr=params['lr'], loss=losses.avg,
                                              fake_top1=fake_top1.avg, top1=true_top1.avg))
            print(output)
    return fake_top1.avg, true_top1.avg

def accuracy(output, target):
    target = np.array(target)
    result = output + target
    pos = len(result[np.where(result == 2)]) * 1.0
    neg = len(result[np.where(result == 0)]) * 0.1
    fake_acc = (pos + neg) / len(output)
    return fake_acc


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


best_acc = 0
best_true_acc = 0
# Epoch = 10
args = parser.parse_args()

 # Get train and val dataset
categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
num_classes = len(categories)

train_transform = transforms.Compose([
        transforms.GroupMultiScaleCrop(args.image_size, [1, .875, .75]),
        transforms.GroupRandomHorizontalFlip(),
        transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
    ])

train_loader = input_data.DataLoader(workers=4,
                                     root_path=args.root_path,
                                     file_list=args.train_list,
                                     input_size=args.image_size,
                                     num_segment=args.num_segment,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     prefix=prefix,
                                     transform=train_transform
                                    )

val_transform = transforms.Compose([
        transforms.GroupScale(args.image_size),
        transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
    ])

val_loader = input_data.DataLoader(workers=4,
                                   root_path=args.root_path,
                                   file_list=args.val_list,
                                   input_size=args.image_size,
                                   num_segment=args.num_segment,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   prefix=prefix,
                                   transform=val_transform
                                   )

# # the path of training set
# path_train = "/home/enbo/share/tensorflow-tutorial/stnet-tensorflow/ResNet50-tensorflow-master/resnet50/dataset/train"
# # the path of testing set
# path_test = "/home/enbo/share/tensorflow-tutorial/stnet-tensorflow/ResNet50-tensorflow-master/resnet50/dataset/test"
# # the path of pretrained resnet50's weights
# pretrained_weights = "./pretrained_weights/weights_resnet.npy"
# # the path of trained resnet50's weights
saved_weights = "./saved_weights/"
# # the path of saved model
saved_model = "./saved_model/"

# checkpoint_file = "./pretrained_weights/resnet_v2_50.ckpt"

#train_data, train_label = read_data(path_train)
#test_data, test_label = read_data(path_test)

#print("train data shape: {}, label shape: {}".format(train_data.shape, train_label.shape))
#print("test data shape: {}, label shape: {}".format(test_data.shape, test_label.shape))

x_ = tf.placeholder(tf.float32, [None, args.image_size, args.image_size, 3])
y_ = tf.placeholder(tf.int32, [None])

#初始化网络
resnet = ResNet(resnet_npy_path=args.checkpoint_file)
resnet.build(x_, label_num=num_classes, last_layer_type="no")

#得分值
logits = resnet.prob
#把每一类的得分值变成概率
predictions = slim.softmax(logits, scope='trn_pred')
predictions = tf.cast(tf.argmax(predictions, 1), tf.int32)
# predict = tf.nn.softmax(res_logits)
# one_hot_labels = slim.one_hot_encoding(y, num_classes)
one_hot_labels = slim.one_hot_encoding(y_, num_classes)

loss_ = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
lr_ = tf.placeholder(tf.float32, name='learning_rate')

# with tf.name_scope('loss'):
#     #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=res_logits))
#     loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=res_logits)
#     #打印所有需要训练的变量
#     var_net = tf.trainable_variables()
#     # l2loss = 0
#     # for var in var_net:
#     #     print(var)
#     #     shape = var.get_shape()
#     #     l2loss += tf.nn.l2_loss(var)
#     #loss = cross_entropy + 2e-4 * l2loss

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_)

train_op = slim.learning.create_train_op(loss_, optimizer)

saver = tf.train.Saver()

#global_step = tf.Variable(0, trainable=False)
#starter_learning_rate = 0.001
#每200轮迭代，学习率乘以0.96
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 200, 0.96, staircase=True)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=starter_learning_rate)

#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#train_op = tf.group([optimizer, update_ops])
#train_op = slim.learning.create_train_op(loss, optimizer)
#accuracy 返回的是 1 或者 0 1：代表预测对
#accuracy = tf.cast(tf.equal(tf.argmax(predict, axis=1), tf.argmax(y, axis=1)), tf.int32)
#accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, axis=1), tf.argmax(y, axis=1)), tf.float32))
graph = tf.get_default_graph()
# 配置session运行过程中的参数
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7 #申请百分之70的显存
config.gpu_options.allow_growth = True  # 动态申请显存

with tf.Session(config=config, graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    #把预训练的参数放进变量中
    resnet.load_weights(sess)
   # saver.restore(sess, checkpoint_file)
    
    for epoch in range(args.epochs):

        lr = adjust_learning_rate(epoch, args.learning_rate, args.lr_steps)
        params = {'x_': x_, 'y_': y_, 'lr_': lr_, 'lr': lr, 'epoch': epoch, 'loss_': loss_}

        train(sess, train_loader, predictions, train_op, params)
        fake_acc, true_acc = valid(sess, val_loader, predictions, loss_, params)
        print('\nCurrent accuracy is: {:.5f}'.format(true_acc))
        print('Best accuracy is: {:.5f}\n'.format(best_true_acc))

        if fake_acc > best_acc:
            best_true_acc = true_acc
            best_acc = fake_acc
            saver.save(sess, os.path.join(saved_model, 'resnet_stnet.ckpt'))

            print('Saved the best model done!\n')
