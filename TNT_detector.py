import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal
import func
import sys
import time
from collections import Counter
import shutil

shutil.rmtree('checkpoint_dir', ignore_errors=True)
class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
         
    def __exit__(self, type, value, traceback):
        print ("nn time: {:.3f} sec".format(time.time() - self._startTime))



# parameters
file_name='TNT_dataset.txt'
learning_rate = 0.001
epochs = 100
batch_size = 100
garm_cnt=1
win_len=100
win_num=10
q=1
w_d=10000# kHz

#get signal from dataset
xx0,yy0,sig_len,fi_num=func.get_batch(file_name,garm_cnt)
xx,yy,xx_ac,yy_ac=func.let_2_mass(xx0,yy0)

print(sig_len,fi_num)

out_y=len(yy[1])
s_1000=int(sig_len/1000)
print(out_y)

x = tf.placeholder(tf.float32, [None, sig_len ])
x_shaped = tf.reshape(x, [-1, sig_len , 1])

y = tf.placeholder(tf.float32, [None, out_y])
print('begin')

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    
    conv_filt_shape = [filter_shape, num_input_channels,
                      num_filters]

    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    out_layer = tf.nn.conv1d(input_data, weights, 1, padding='SAME')
    
    out_layer += bias

    out_layer = tf.nn.leaky_relu(out_layer)

    ksize = pool_shape
    window_shape=[]
    window_shape.append(pool_shape)
    strides = []
    strides.append(ksize)
    out_layer = tf.nn.pool(out_layer, window_shape, pooling_type="MAX",padding='SAME',strides=strides)

    return out_layer,weights 

#network
layer1,weights = create_new_conv_layer(x_shaped, 1, win_num, win_len, 10, name='layer1')

layer2,weights2 = create_new_conv_layer(layer1, win_num, win_num*2, 20, 10, name='layer2')

#layer3,weights3 = create_new_conv_layer(layer2, win_num*2, win_num*4, 5, 10, name='layer3')

#layer4 = create_new_conv_layer(layer3, win_num*4, win_num*8, 3, 10, name='layer4')

flattened = tf.reshape(layer2, [-1, win_num*2*s_1000*10])

wd1 = tf.Variable(tf.truncated_normal([win_num*2*s_1000*10, out_y], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([out_y], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1

y_ = tf.nn.softmax(dense_layer1)

#learning
loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=dense_layer1)

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

# lists
list_ep=[]
mas_n=[]
mas_m=[]
mas_tr=[]
mas=[]
train_mas=[]
f = open('nr-1_sl-100.txt', 'w')
yo=0
with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(xx) / batch_size)
    print(total_batch)
    train_mas.append(0)
    # learn
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch-1):
            batch_x=xx[(i*batch_size):((i+1)*batch_size)]
            batch_y=yy[(i*batch_size):((i+1)*batch_size)]
            _, l= sess.run([optimiser, dense_layer1], 
                            feed_dict={x:batch_x, y: batch_y})
        # accuracy 
        yo1=sess.run([accuracy], feed_dict={x: xx_ac[0:100], y: yy_ac[0:100]})
        yo=yo+yo1[0]
        # save best version
        if yo1[0]>max(train_mas):
            train_mas.append(yo1[0])
            saver = tf.train.Saver()
            saver.save(sess, "checkpoint_dir/model"+str(epoch)+".ckpt")
        print(epoch)
        print(yo1[0])
        f.write(str(yo1[0])+'\t'+str(epoch)+'\n')

    f.close()
    ckpt = tf.train.get_checkpoint_state("checkpoint_dir")
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    snr=[]
    snr_match=[]
    snr_match_avr=[]
    snr_e=[]
    snr_m=[]
    snr_mmp=[]
    p=0
    n=5
    Q=[]
    x1m=[0 for f in range(n)]
    y1m=[0 for f in range(n)]
    x1=[0 for f in range(n)]
    y1=[0 for f in range(n)]
    x1n=[0 for f in range(n)]
    y1n=[0 for f in range(n)]

    match_many=100
    for_match_many=[]
    ampl_avr=[]
    # SMF
    for i in range(match_many):
        garm=[]
        sig=[]
        sdvig=random.random()
        delta_w=[sdvig*12+867, sdvig*27+852, sdvig*20+843, 
                sdvig*21+837, sdvig*14+840, sdvig*14+833]
        delta_a=5
        ampl=[(random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1)]
        gamma=0.0005
        delta_gamma=random.random()*gamma+0.000001
        delta_fi=random.random()*2*np.pi
        for t in range(sig_len):
            sg=0
            for gn in range(len(delta_w)):
                sg+=ampl[gn]*np.sin(np.pi*2*delta_w[5-gn]/w_d*t+delta_fi)*np.exp(-(delta_gamma)*t)
            garm.append(sg)
        for_match_many.append(garm)
        ampl_avr.append(sum(ampl)/6)
    maz=n
    for i in range(n):
        print('lost cicle:\t',n-i,'number of avr:\t',maz)
        xx_snr_p,yy_snr_p,p_m,p,q,p_e=func.explose_det(for_match_many,ampl_avr,'tnt',w_d,200*(i)/n+10,sig_len,1000,1)
        Q.append(q)
        p_n=sess.run([accuracy], feed_dict={x: xx_snr_p, y: yy_snr_p})

        mas_n.append(p_n)
        mas_m.append(p_m)
        mas.append(p)
        mas_tr.append(p_e)
    plt.plot(mas_n, 'b')
    plt.plot(mas_m)
    plt.plot(mas)
    plt.plot(mas_tr)
    plt.show()
print("\nEnd!")

