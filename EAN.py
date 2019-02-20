#coding:utf-8
import os
import numpy as np
import tensorflow as tf
import csv
from text_cnn import TextCNN
from alexnet1 import AlexNet1
from datagenerator import ImageDataGenerator
from only_textual_datagenerator import TextualDataGenerator
from datetime import datetime
import pickle
from tensorflow.contrib.data import Iterator
"""
Configuration Part.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
# Path to the textfiles for the trainings and validation set
option="/storage/liujinhuan/liujinhuan/new_ijk_data/"
train_file = option+'train_ijk_shuffled_811.txt'
val_file = option+'valid_ijk_shuffled_811.txt'
test_file = option+'test_ijk_shuffled_811.txt'
# Learning params
# learning_rate = 0.1
num_epochs = 30
batch_size = 600
att_hidden=2**7
n_hidden=2**9
max_l=43
dropout_rate = 0.5
num_classes = 2
filters=[2, 3, 4, 5]
hidden_units = [100, 2]
img_w = 300
feature_maps = hidden_units[0]
# How often we want to write the tf.summary data to disk
display_step = 3

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "tmp2/finetune_alexnet/tensorboard"
checkpoint_path = "tmp2/tune_alexnet/checkpoints"

def get_idx_from_sent(sent, word_idx_map, max_l):
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    return x

def make_idx_data(word_idx_map, max_l):
    train_i_idx, train_j_idx, train_k_idx = [], [], []
    valid_i_idx, valid_j_idx, valid_k_idx = [], [], []
    test_i_idx, test_j_idx, test_k_idx = [], [], []

    print('loading text data')
    print('now():' + str(datetime.now()))
    train_i,train_i_label, train_j,train_j_label, train_k,train_k_label = pickle.load(
        open("/storage/liujinhuan/liujinhuan/new_ijk_data/AUC_new_dataset_train_811.pkl", "rb"))
    valid_i,valid_i_label, valid_j,valid_j_label, valid_k,valid_k_label = pickle.load(
        open("/storage/liujinhuan/liujinhuan/new_ijk_data/AUC_new_dataset_valid_811.pkl", "rb"))
    print("valid_i.shape",len(valid_i[0]))
    test_i,test_i_label, test_j,test_j_label, test_k,test_k_label = pickle.load(
        open("/storage/liujinhuan/liujinhuan/new_ijk_data/AUC_new_dataset_test_811.pkl", "rb"))
    print('text data loaded')
    print('now():' + str(datetime.now()))

    for i in range(len(train_i)):
        train_sent_i_idx = get_idx_from_sent(train_i[i], word_idx_map, max_l)
        train_sent_j_idx = get_idx_from_sent(train_j[i], word_idx_map, max_l)
        train_sent_k_idx = get_idx_from_sent(train_k[i], word_idx_map, max_l)
        train_i_idx.append(train_sent_i_idx)
        train_j_idx.append(train_sent_j_idx)
        train_k_idx.append(train_sent_k_idx)

    for i in range(len(valid_i)):
        valid_sent_i_idx = get_idx_from_sent(valid_i[i], word_idx_map, max_l)
        valid_sent_j_idx = get_idx_from_sent(valid_j[i], word_idx_map, max_l)
        valid_sent_k_idx = get_idx_from_sent(valid_k[i], word_idx_map, max_l)
        valid_i_idx.append(valid_sent_i_idx)
        valid_j_idx.append(valid_sent_j_idx)
        valid_k_idx.append(valid_sent_k_idx)

    for i in range(len(test_i)):
        test_sent_i_idx = get_idx_from_sent(test_i[i], word_idx_map, max_l)
        test_sent_j_idx = get_idx_from_sent(test_j[i], word_idx_map, max_l)
        test_sent_k_idx = get_idx_from_sent(test_k[i], word_idx_map, max_l)
        test_i_idx.append(test_sent_i_idx)
        test_j_idx.append(test_sent_j_idx)
        test_k_idx.append(test_sent_k_idx)

    train_i_idx = np.array(train_i_idx, dtype="int")
    train_j_idx = np.array(train_j_idx, dtype="int")
    train_k_idx = np.array(train_k_idx, dtype="int")
    valid_i_idx = np.array(valid_i_idx, dtype="int")
    valid_j_idx = np.array(valid_j_idx, dtype="int")
    valid_k_idx = np.array(valid_k_idx, dtype="int")
    test_i_idx = np.array(test_i_idx, dtype="int")
    test_j_idx = np.array(test_j_idx, dtype="int")
    test_k_idx = np.array(test_k_idx, dtype="int")

    return [train_i_idx,train_i_label,  train_j_idx,train_j_label, train_k_idx, train_k_label,valid_i_idx,valid_i_label, valid_j_idx,valid_j_label, valid_k_idx,valid_k_label, test_i_idx,test_i_label, test_j_idx,
            test_j_label,test_k_idx,test_k_label]


print ("loading w2v data...")
text_x = pickle.load(open("/storage/liujinhuan/liujinhuan/cloth.binary.p", "rb"))
text_revs, text_W, text_W2, text_word_idx_map, text_vocab = text_x[0], text_x[1], text_x[2], text_x[3], text_x[4]
# print(text_W.shape)
datasets = make_idx_data(text_word_idx_map, max_l)
train_text_i,train_i_label, train_text_j,train_j_label, train_text_k, train_k_label = datasets[0], datasets[1], datasets[2],datasets[3], datasets[4], datasets[5]
# print("train_text_i.shape",train_text_i.shape[0],train_text_i.shape[1])
valid_text_i,valid_i_label, valid_text_j,valid_j_label, valid_text_k,valid_k_label = datasets[6], datasets[7], datasets[8],datasets[9], datasets[10], datasets[11]
test_text_i, test_i_label,test_text_j,test_j_label, test_text_k ,test_k_label= datasets[12], datasets[13], datasets[14],datasets[15], datasets[16], datasets[17]
"""
Main Part of the finetuning Script.
"""
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=False)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)
    test_data = ImageDataGenerator(test_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)
    tra_text_i = TextualDataGenerator(train_text_i,train_i_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)
    tra_text_j = TextualDataGenerator(train_text_j,train_j_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)
    tra_text_k = TextualDataGenerator(train_text_k,train_k_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)
    val_text_i = TextualDataGenerator(valid_text_i,valid_i_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)
    val_text_j = TextualDataGenerator(valid_text_j,valid_j_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)
    val_text_k = TextualDataGenerator(valid_text_k,valid_k_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)
    tes_text_i = TextualDataGenerator(test_text_i,test_i_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)
    tes_text_j = TextualDataGenerator(test_text_j,test_j_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)
    tes_text_k = TextualDataGenerator(test_text_k,test_k_label,
                                    batch_size_text=int(batch_size/3),
                                    num_classes=num_classes)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    i_text_ite = Iterator.from_structure(tra_text_i.data.output_types,
                                         tra_text_i.data.output_shapes)
    j_text_ite = Iterator.from_structure(tra_text_j.data.output_types,
                                         tra_text_j.data.output_shapes)
    k_text_ite = Iterator.from_structure(tra_text_k.data.output_types,
                                         tra_text_k.data.output_shapes)

    next_batch = iterator.get_next()
    next_batchi = i_text_ite.get_next()
    next_batchj = j_text_ite.get_next()
    next_batchk = k_text_ite.get_next()


# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)
test_init_op = iterator.make_initializer(test_data.data)
tra_text_i_init_op = i_text_ite.make_initializer(tra_text_i.data)
tra_text_j_init_op = j_text_ite.make_initializer(tra_text_j.data)
tra_text_k_init_op = k_text_ite.make_initializer(tra_text_k.data)
val_text_i_init_op = i_text_ite.make_initializer(val_text_i.data)
val_text_j_init_op =j_text_ite.make_initializer(val_text_j.data)
val_text_k_init_op = k_text_ite.make_initializer(val_text_k.data)
tes_text_i_init_op = i_text_ite.make_initializer(tes_text_i.data)
tes_text_j_init_op = j_text_ite.make_initializer(tes_text_j.data)
tes_text_k_init_op = k_text_ite.make_initializer(tes_text_k.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
text_x_i = tf.placeholder(tf.int32,[int(batch_size/3),train_text_i.shape[1]])
text_x_j = tf.placeholder(tf.int32,[int(batch_size/3),train_text_j.shape[1]])
text_x_k = tf.placeholder(tf.int32,[int(batch_size/3),train_text_k.shape[1]])

keep_prob = tf.placeholder(tf.float32)
outfile = "record2/finetune2_textual_visual"+ "batch_size_" + str(batch_size) + "hidden_"+str(n_hidden)+str(datetime.now().strftime('%H-%M-%S') )+".txt"
csvfile = "record2/finetune2_textual_visual"+ "batch_size_" + str(batch_size) + "hidden_"+str(n_hidden)+str(datetime.now().strftime('%H-%M-%S') )+".csv"
with tf.variable_scope("model1") as scope:
    model1 = AlexNet1(x, keep_prob, 0, batch_size)
with tf.variable_scope("model2") as scope:
    model2 = AlexNet1(x, keep_prob,1, batch_size)
with tf.variable_scope("model2",reuse=True) as scope:
    model3 = AlexNet1(x, keep_prob, 2, batch_size)

with tf.variable_scope("model_texti") as scope:
    model_texti = TextCNN(embedding_weights=text_W,input=text_x_i,filter_sizes=filters,embedding_size=img_w,sequence_length=train_text_i.shape[1],dropout_keep_prob=keep_prob)
with tf.variable_scope("model_textj") as scope:
    model_textj = TextCNN(embedding_weights=text_W,input=text_x_j,filter_sizes=filters,embedding_size=img_w,sequence_length=train_text_i.shape[1],dropout_keep_prob=keep_prob)
with tf.variable_scope("model_textj",reuse=True) as scope:
    model_textk = TextCNN(embedding_weights=text_W,input=text_x_k,filter_sizes=filters,embedding_size=img_w,sequence_length=train_text_i.shape[1],dropout_keep_prob=keep_prob)

#record by txt
file = open(outfile, "w+")
best_validation_auc_score = 0.0
count = 0
for _learning_rate in [0.01]:
    for _lamda in [0.1]:
        count = count+1
        learning_rate = _learning_rate
        lamda=_lamda
        # if lamda==0.1:
        #     lamda=0.5
        print("lamda: {} n_hidden:{} learning_rate:{}\n".format(lamda,n_hidden,learning_rate))
        file.write("lamda: {} n_hidden:{} learning_rate:{}\n".format(lamda,n_hidden,learning_rate))
        file.write("begin time:{}".format(datetime.now()))
        # Initialize model
        # Link variable to model output
        output11 = model1.res
        output12=model2.res
        output13=model3.res
        text_output11 = model_texti.h_drop
        text_output12 = model_textj.h_drop
        text_output13 = model_textk.h_drop
        reg1,reg2,reg3=model1.reg,model2.reg,model3.reg
        text_reg1,text_reg2,text_reg3=model_texti.reg,model_textj.reg,model_textk.reg
        initializer1=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)

        W1 = tf.get_variable("W1"+str(count), (4096, n_hidden), dtype=tf.float32,trainable=True, )
        W2 = tf.get_variable("W2"+str(count), (4096, n_hidden), dtype=tf.float32,trainable=True, )
        text_W1 = tf.get_variable("text_W1"+str(count), shape=(400,n_hidden), dtype=tf.float32,trainable=True,)
        text_W2 = tf.get_variable("text_W2"+str(count), shape=(400,n_hidden),dtype=tf.float32,trainable=True,)
        b1 = tf.get_variable("b1"+str(count), (n_hidden,),dtype=tf.float32, trainable=True, )
        b2 = tf.get_variable("b2"+str(count), (n_hidden,),dtype=tf.float32, trainable=True, )
        text_b1 = tf.get_variable("tb1"+str(count), shape=(n_hidden,),dtype=tf.float32,trainable=True, )
        text_b2 = tf.get_variable("tb2"+str(count), shape=(n_hidden,),dtype=tf.float32,trainable=True, )
        output1 = tf.sigmoid(tf.matmul(output11, W1) + b1)
        output2 = tf.sigmoid(tf.matmul(output12, W2) + b2)
        output3 = tf.sigmoid(tf.matmul(output13, W2) + b2)
        text_output1 = tf.sigmoid(tf.matmul(text_output11, text_W1) + text_b1)
        text_output2 = tf.sigmoid(tf.matmul(text_output12, text_W2) + text_b2)
        text_output3 = tf.sigmoid(tf.matmul(text_output13, text_W2) + text_b2)
        fea_size = n_hidden

        text_att_n = []
        text_btt_n = []
        btt_n = []
        att_n = []

        for k in range(0,fea_size):
            # with tf.variable_scope('atten_initia', reuse = None):
            text_f_n = tf.one_hot(k,fea_size)
            text_tile_f_n = tf.tile(text_f_n,[int(batch_size/3)])
            text_reshape_f_n = tf.reshape(text_tile_f_n,[int(batch_size/3),fea_size])
            Wi = tf.get_variable("Wi"+str(count)+str(k), (fea_size, att_hidden),initializer=initializer1,dtype=tf.float32,trainable=True, )
            text_Wi = tf.get_variable("text_Wi"+str(count)+str(k), (fea_size, att_hidden),initializer=initializer1,dtype=tf.float32,trainable=True, )
            b = tf.get_variable("b_att"+str(count)+str(k), (att_hidden,),initializer = tf.constant_initializer(0.1), dtype=tf.float32,trainable=True, )
            text_b = tf.get_variable("text_b_att"+str(count)+str(k), (att_hidden,),initializer = tf.constant_initializer(0.1),dtype=tf.float32,trainable=True, )
            W_all = tf.get_variable("W_all"+str(count)+str(k), (att_hidden,1),initializer=initializer1,dtype=tf.float32,trainable=True, )
            text_W_all = tf.get_variable("text_W_all"+str(count)+str(k), (att_hidden,1), initializer=initializer1,dtype=tf.float32,trainable=True, )
            c = tf.get_variable("c_att"+str(count)+str(k), (fea_size,),initializer = tf.constant_initializer(0.1),dtype=tf.float32,trainable=True, )
            text_c = tf.get_variable("text_c_att"+str(count)+str(k), (fea_size,),initializer = tf.constant_initializer(0.1),dtype=tf.float32,trainable=True, )

            a_n = tf.matmul((tf.nn.relu6(tf.matmul(output1*output2*text_reshape_f_n, Wi)+b)), W_all) + c[k]
            b_n = tf.matmul((tf.nn.relu6(tf.matmul(output1*output3*text_reshape_f_n, Wi)+ b)), W_all) + c[k]
            text_a_n =  tf.matmul(tf.nn.relu6(tf.matmul(text_output1*text_output2*text_reshape_f_n, text_Wi)+text_b), text_W_all)+text_c[k]
            text_b_n =  tf.matmul(tf.nn.relu6(tf.matmul(text_output1*text_output3*text_reshape_f_n, text_Wi)+text_b), text_W_all)+text_c[k]
            text_att_n.append(text_a_n)
            text_btt_n.append(text_b_n)
            att_n.append(a_n)
            btt_n.append(b_n)
        text_att_ = text_att_n[0]
        text_btt_ = text_btt_n[0]
        att_ = att_n[0]
        btt_ = btt_n[0]
        for i in range(1, fea_size):
            text_att_ = tf.concat([text_att_, text_att_n[i]],1)
            text_btt_ = tf.concat([text_btt_, text_btt_n[i]],1)
            att_ = tf.concat([att_, att_n[i]],1)
            btt_ = tf.concat([btt_, btt_n[i]],1)
        text_att_nor = tf.nn.softmax(tf.exp(text_att_))
        text_btt_nor = tf.nn.softmax(tf.exp(text_btt_))
        att_nor = tf.nn.softmax(tf.exp(att_))
        btt_nor = tf.nn.softmax(tf.exp(btt_))
        score1=tf.reduce_sum(tf.subtract(text_att_nor*text_output1*text_output2,text_btt_nor*text_output1*text_output3)+tf.subtract(att_nor*output1*output2,btt_nor*output1*output3),1)

        reg=text_reg1+text_reg2+text_reg3+reg1+reg2+reg3+tf.reduce_mean(W1**2)+tf.reduce_mean(W2**2)*2\
            +tf.reduce_mean(text_W1**2)+tf.reduce_mean(text_W2**2)*2\
            +tf.reduce_mean(text_b1**2)+tf.reduce_mean(text_b2**2)*2\
            +tf.reduce_mean(text_Wi**2)+tf.reduce_mean(text_W_all**2)\
            +tf.reduce_mean(Wi**2)+tf.reduce_mean(W_all**2)\
            +tf.reduce_mean(text_b**2)+tf.reduce_mean(text_c**2)\
            +tf.reduce_mean(b**2)+tf.reduce_mean(c**2)

        score=tf.sigmoid(score1)
        # List of trainable variables of the layers we want to train
        var_list= [v for v in tf.trainable_variables()]
        # Op for calculating the loss
        with tf.name_scope("cross_ent"):
            L_sqr=reg
            L_sup=tf.reduce_mean(score1)
            L_rec=lamda*L_sqr
            loss =L_rec-L_sup

        # Train op
        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            gradients = tf.gradients(loss, var_list)
            gradients = list(zip(gradients, var_list))

            # Create optimizer and apply gradient descent to the trainable variables
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(grads_and_vars=gradients)

        # Add gradients to summary
        # for gradient, var in gradients:
        #     tf.summary.histogram(var.name + '/gradient', gradient)

        # Add the variables we train to the summary
        # for var in var_list:
        #     tf.summary.histogram(var.name, var)

        # Add the loss to summary
        # tf.summary.scalar('cross_entropy', loss)

        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            res=tf.expand_dims(score,1)
            val = tf.ones((int(batch_size/3), 1), tf.float32)/2
            res=tf.concat([res,val],1)
            res1=tf.argmax(res, 1)
            y = tf.placeholder(tf.float32, [(int(batch_size/3)),2])
            res2=tf.argmax(y,1)
            correct_pred = tf.equal(res1, res2)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)
        #
        # # Merge all summaries together
        merged_summary = tf.summary.merge_all()
        #
        # # Initialize the FileWriter
        writer = tf.summary.FileWriter(filewriter_path)

        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver()

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
        val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
        test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))
        # Start Tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Add the model graph to TensorBoard
            # writer.add_graph(sess.graph)
            # Load the pretrained weights into the non-trainable layer
            model1.load_initial_weights(sess,"model1")
            model2.load_initial_weights(sess,"model2")
            model3.load_initial_weights(sess,"model2")
            # model4.load_initial_weights(sess,"model_text")
            # To continue training from one of your checkpoints
            # saver.restore(sess, "tmp2/tune_alexnet/checkpoints/model_epoch00.ckpt")
            # print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
            #                                                   filewriter_path))

            # Loop over number of epochs
            for epoch in range(num_epochs):
                print("{} Epoch number: {}".format(datetime.now(), epoch+1))
                file.write(str(epoch+1))
                # Initialize iterator with the training dataset
                sess.run(training_init_op)
                sess.run(tra_text_i_init_op)
                sess.run(tra_text_j_init_op)
                sess.run(tra_text_k_init_op)
                L_rec_sum=0.0
                L_sup_sum=0.0
                L_sqr_sum=0.0
                loss_sum=0.0

                for step in range(train_batches_per_epoch):
                    # get next batch of data
                    img_batch, label_batch = sess.run(next_batch)
                    text_batch_i, labeli_batch = sess.run(next_batchi)
                    text_batch_j, labelj_batch = sess.run(next_batchj)
                    text_batch_k, labelk_batch = sess.run(next_batchk)
                    # And run the training op
                    [_,each_L_rec_sum,each_L_sup_sum,each_L_sqr_sum,each_loss_sum]= sess.run([train_op,L_rec,L_sup,L_sqr,loss],
                                                                                                            feed_dict={x: img_batch,
                                                                                                                       y:label_batch[0:batch_size:3],
                                                                                                                       text_x_i: text_batch_i,
                                                                                                                       text_x_j: text_batch_j,
                                                                                                                       text_x_k: text_batch_k,
                                                                                                                       keep_prob: dropout_rate})
                    loss_sum+=each_loss_sum
                    L_sup_sum+=each_L_sup_sum
                    L_sqr_sum += each_L_sqr_sum
                    L_rec_sum += each_L_rec_sum

                print("loss_sum: {} L_sup_sum: {} L_sqr_sum: {} L_rec_sum:{}".format(loss_sum, L_sup_sum,L_sqr_sum,L_rec_sum))
                file.write("loss_sum: {} L_sup_sum: {} L_sqr_sum: {} L_rec_sum:{}\n".format(loss_sum, L_sup_sum,L_sqr_sum,L_rec_sum))
                print("{} Start Train".format(datetime.now()))
                sess.run(training_init_op)
                sess.run(tra_text_i_init_op)
                sess.run(tra_text_j_init_op)
                sess.run(tra_text_k_init_op)
                test_acc = 0.
                test_count = 0
                for train_epoch in range(train_batches_per_epoch):
                    img_batch, label_batch = sess.run(next_batch)
                    text_batch_i, labeli_train_batch = sess.run(next_batchi)
                    text_batch_j, labelj_train_batch = sess.run(next_batchj)
                    text_batch_k, labelk_train_batch = sess.run(next_batchk)
                    acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                        y: label_batch[0:batch_size:3],
                                                        text_x_i:text_batch_i,
                                                        text_x_j:text_batch_j,
                                                        text_x_k:text_batch_k,
                                                        keep_prob:1.})
                    # print(curres)
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{} Train Accuracy = {:.4f}".format(datetime.now(),
                                                          test_acc))
                file.write("Train Accuracy = {:.4f}\n".format(
                    test_acc))

                sess.run(validation_init_op)
                sess.run(val_text_i_init_op)
                sess.run(val_text_j_init_op)
                sess.run(val_text_k_init_op)
                val_acc = 0.
                val_count = 0
                for valid_epoch in range(val_batches_per_epoch):

                    img_batch, label_batch = sess.run(next_batch)
                    text_batch_i, labeli_val_batch = sess.run(next_batchi)
                    text_batch_j, labelj_val_batch = sess.run(next_batchj)
                    text_batch_k, labelk_val_batch = sess.run(next_batchk)
                    acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                        y: label_batch[0:batch_size:3],
                                                        text_x_i:text_batch_i,
                                                        text_x_j:text_batch_j,
                                                        text_x_k:text_batch_k,
                                                        keep_prob:1.})
                   # print(curres)
                    val_acc += acc
                    val_count += 1
                val_acc /= val_count
                print("{}   Validation Accuracy = {:.4f}".format(datetime.now(),
                                                                 val_acc))
                file.write("    Validation Accuracy = {:.4f}\n".format(
                                                                val_acc))
                # print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                # checkpoint_name = os.path.join(checkpoint_path,
                #                                'model_epoch'+str(epoch+1)+'.ckpt')
                # save_path = saver.save(sess, checkpoint_name)
                # print("{} Model checkpoint saved at {}".format(datetime.now(),
                #                                                checkpoint_name))
                sess.run(test_init_op)
                sess.run(tes_text_i_init_op)
                sess.run(tes_text_j_init_op)
                sess.run(tes_text_k_init_op)

                if val_acc > best_validation_auc_score:
                    best_validation_auc_score = val_acc
                    test_acc = 0.
                    test_count = 0
                    for test_epoch in range(test_batches_per_epoch):
                        img_batch, label_batch = sess.run(next_batch)
                        text_batch_i, labeli_test_batch = sess.run(next_batchi)
                        text_batch_j, labelj_test_batch = sess.run(next_batchj)
                        text_batch_k, labelk_test_batch = sess.run(next_batchk)
                        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                            y: label_batch[0:batch_size:3],
                                                            text_x_i:text_batch_i,
                                                            text_x_j:text_batch_j,
                                                            text_x_k:text_batch_k,
                                                            keep_prob:1.})
                        test_acc += acc
                        test_count += 1
                    test_acc /= test_count
                    print("{}               Test Accuracy = {:.4f}".format(datetime.now(),
                                                                           test_acc))
                    file.write("                Test Accuracy = {:.4f}\n".format(
                        test_acc))
                    file.flush()

file.write("end time: {}".format(datetime.now()))
file.close()
