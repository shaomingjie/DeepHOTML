####   DeepHOTML for One-Bit MIMO Detection Problem  ###############
# This program performs training for the DeepHOTML in the following paper
# ``Binary MIMO Detection via Homotopy Optimization and Its Deep Adaptation'' by Mingjie Shao and Wing-Kin Ma
# If you have any questions, please contact mjshao@link.cuhk.edu.hk
# Nov. 27, 2020
####################################################################

# Remark: Note that in the Section III(D), we discussed a numerical issue and its fix by overestimating the noise power.
# In this file, we set sigma_0 = 0.5. This works well for most cases. However, the numerical issue may still occur occasionally.
# If you find the displayed training BER = 1 in the training phase, this indicates that the numerical issue happens. We provide two ways of dealing with this situation:
# (1) One can use the saved parameter before the numerical issue happens. Do NOT use the saved parameters after the issue happens.
# example, if you the BER = 1 happens at training iteration 5000, then it is suggested to use the last saved parameter before the 5000 iteration.
# (2) One can increase the value of sigma_0 and retrain the network.

# if one uses tensorflow 2.0 or above, use the following command
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# otherwise use the following command
#import tensorflow as tf
import numpy as np
import scipy.io as spio 
import os
import time as tm
import math
import sys
import pickle as pkl
import numpy.linalg as NormFuc
import time
import matplotlib.pyplot as plt


from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# choose the GPU. One can change according to the device
os.environ['CUDA_VISIBLE_DEVICES']='0,1'



def generate_data_Onebit_MIMO( B,K,N,snr_low,snr_high):

    H_real = np.random.randn(B,N,K).astype('float64')/np.sqrt(2)
    H_imag = np.random.randn(B, N, K).astype('float64')/np.sqrt(2)
    H1 = np.concatenate( [H_real,  -H_imag],axis=2)
    H2 = np.concatenate( [H_imag, H_real],axis=2)
    H =  np.concatenate( [H1,H2],axis=1 )

    X_real = np.sign(np.random.rand(B, K ).astype('float64')-0.5)
    X_imag = np.sign(np.random.rand(B, K).astype('float64') - 0.5)
    X =  np.concatenate([ X_real, X_imag],axis=1)


    sigma = np.zeros([B ], dtype=np.float64) 
    sigma_C = np.zeros([B ], dtype=np.float64) 
    sigma1 = np.zeros([B ], dtype=np.float64) 
    Ysign = np.zeros([B,2*N ], dtype=np.float64)
    Homega =np.zeros([B,2*K,2*N ], dtype=np.float64)
    SNR = np.zeros([B], dtype=np.float64)
    sigma_0 =0.5

    for i in range(B):
        Hslide = H[i,:, :]
        x = X[ i  ,: ]
        SNR[i] = np.random.uniform(low=snr_low, high=snr_high)
        sigma_C [i] =  np.sqrt(2 * K) * np.sqrt(1 / SNR[i] )   # complex noise standard deviation
        sigma[ i ] = sigma_C[i] / np.sqrt(2)                  # real noise stamdard deviation
        w = sigma[ i  ] * np.random.randn(2*N).astype('float64')
        Ysign[i  ,:] =  np.sign(Hslide.dot(x)+ w)
        sigma1[i] = sigma[i] + sigma_0
        Homega[i, :,:] = Hslide.T.dot(np.diag(Ysign[i  ,:]/sigma1[ i  ]))

    return H, Ysign ,X ,Homega,sigma


# if one uses tensorflow 2.0 or above, use the following command
dist = tf.compat.v1.distributions.Normal(loc=np.float64(0), scale=np.float64(1))
# otherwise use the following command
# dist = tf.contrib.distributions.Normal(loc=np.float64(0), scale=np.float64(1))

def PhiFunc(x):
    return tf.divide(dist.prob(x), dist.cdf(x))


def affine_layer(x, input_size, output_size, Layer_num):
    with tf.variable_scope(Layer_num):
        weigth = tf.get_variable(name="weigth", shape=[input_size, output_size],  initializer=tf.random_normal_initializer(stddev=0.1), dtype=np.float64)
        bias = tf.get_variable(name="bias", shape=[output_size], initializer=tf.random_normal_initializer(stddev=0.1), dtype=np.float64)
    y = tf.matmul(x, weigth) + bias
    return y


def affine_layer_diagwbias0(x1, input_size, Layer_num):
    with tf.variable_scope(Layer_num):
        weigth1 = tf.get_variable(name="weigth1", shape=[input_size], initializer=tf.random_normal_initializer(stddev=0.1), dtype=np.float64)  # initializer=tf.ones_initializer()
        bias = tf.get_variable(name="bias", shape=[input_size], initializer=tf.zeros_initializer( ), dtype=np.float64)  # initializer=tf.random_normal_initializer(stddev=0)
    y = tf.matmul(x1, tf.matrix_diag(weigth1)) + bias
    return y


# result analysis
if not os.path.exists('Onebit_DeepHOTML/'):
    os.makedirs('Onebit_DeepHOTML/')
directory = './Onebit_DeepHOTML/'


#parameters
K = 16  # no. of users
N = 64 # no. of antennas
# Note that these are complex-valued dimension.

LayerTied = 10 # no. of layers in DeepHOTML

snrdb_low = 5 # training SNR lower bound
snrdb_high = 22 # training SNR upper bound
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)

train_iter = 10000  # no. of training iterations
train_batch_size = 500 # training batch size
epoch = 100
snrdb_low_test = 0
snrdb_high_test = 25
num_snr = 6
test_batch_size = train_batch_size
test_iter = 1000


# learning rate parameters
startingLearningRate = 0.001
decay_factor = 0.9
decay_step_size = 500

Y = tf.placeholder(tf.float64, shape=[ None,2*N], name='Y')
H = tf.placeholder(tf.float64, shape=[None,2*N,2*K ], name='H')
sigma = tf.placeholder(tf.float64, shape=[None  ], name='sigma')
Homega = tf.placeholder(tf.float64, shape=[None,2*K,2*N ], name='Homega')
X_true = tf.placeholder(tf.float64, shape=[None,2*K ], name='X_true')

# training parameters
beta = []
alpha = [] 
gamma= [] 

for i in range(LayerTied ):
    alpha.append(tf.Variable(np.float64(0.5), name=('alpha_' + np.str(i))))
    beta.append(tf.Variable(np.float64(0.01), name=('beta_' + np.str(i))))
    gamma.append(tf.Variable(np.float64(0.001), name=('gamma_' + np.str(i))))
 

Xup = tf.maximum(tf.minimum(affine_layer(Y,2 * N, 2 * K, 'Iniaff1'), 1), -1)

Z = Xup
V = np.zeros([train_batch_size, 2 * K], dtype=np.float64)

for j in range(LayerTied):

    Xpre = Xup
    utav = PhiFunc(affine_layer_diagwbias0(tf.squeeze(tf.matmul(tf.transpose( Homega,perm=[0,2,1]), tf.expand_dims(Z, 2)), 2), 2 *N,  'affineinner' + np.str(j)))
    grad = tf.squeeze(tf.matmul( Homega,tf.expand_dims(utav,2)), 2)
    Xup = tf.maximum(tf.minimum( Z - beta[j] * grad   + gamma[j]* V  , 1), -1) 
    Z = Xup + alpha[j]* tf.subtract(Xpre, Xup)
    V = Xup 
 
X_est = tf.identity( Xup, name="X_est")
X_estsign = tf.identity( tf.sign(Xup), name="X_estsign")

loss2 =  tf.reduce_mean( tf.square(X_est - X_true) ) # (1/.)*\|x- \hat{x}\|_2^2
losssign = tf.reduce_mean( tf.square(X_estsign - X_true) ) # (1/.)*\|x-sign(\hat{x})\|_2^2
HX_est = tf.squeeze( tf.matmul(H,tf.expand_dims(X_est,2) ),2)
HX_true= tf.squeeze( tf.matmul(H,tf.expand_dims(X_true,2) ),2)
BER = tf.reduce_mean(tf.cast(tf.not_equal(X_true,X_estsign), tf.float64)) # (1/.)* [ x \= sign(\hat{x})]
 
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss2)

saver = tf.train.Saver()
minimum_loss = np.float64(500) 
Strainavg_loss = [] 
Strainlosssign = []
StrainHXavg_loss = []
StrainBER = []

Stestavg_loss = []
Stestlosssign = []  
StestHXavg_loss = []
StestBER = []

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    st = time.time()
    print('Onebit_DeepHOTML')


    for i in range(train_iter+1):

        if i % epoch == 0:
            testbatch_H, testbatch_Y , testbatch_X, testbatch_Homega, testbatch_sigma   = generate_data_Onebit_MIMO( test_batch_size, K, N, snr_low, snr_high)
            testavg_loss, testBER = sess.run([loss2, BER], feed_dict={ H: testbatch_H,Y: testbatch_Y,  X_true: testbatch_X, Homega: testbatch_Homega, sigma:testbatch_sigma  })

            if i >= 9000:
                saver.save(sess, directory + 'my-test-model' + str(int(i / epoch)))
                print('Parameters saving')
                with tf.variable_scope('', reuse=True):
                    W1_tf = tf.get_variable(name='Iniaff1/weigth', shape=[2 * N, 2 * K], dtype=np.float64)
                    b1_tf = tf.get_variable(name="Iniaff1/bias", shape=[2 * K], dtype=np.float64)

                    InnerW1_tf = []
                    Innerb1_tf = []
                    Inner3W2_tf = []
                    Inner3W3_tf = []

                    for j in range(LayerTied):
                        InnerW1_tf.append(
                            sess.run(tf.get_variable(name='affineinner' + np.str(j) + '/weigth1', shape=[2 * N], dtype=np.float64)))
                        Innerb1_tf.append(
                            sess.run(tf.get_variable(name='affineinner' + np.str(j) + '/bias', shape=[2 * N], dtype=np.float64)))

                    spio.savemat(
                        directory + 'OnebitNetwork_' + str(N) + 'by' + str(K) + 'SNR' + str(snrdb_low) + '_' + str(
                            snrdb_high) + 'Layer' + str(LayerTied) + str(int(i / epoch)) + '.mat', {
                            'Iniaff_W1': sess.run(W1_tf),
                            'Iniaff_b1': sess.run(b1_tf),

                            'beta': sess.run(beta),
                            'gamma': sess.run(gamma),
                            'alpha': sess.run(alpha),
                            'InnerW1': InnerW1_tf,
                            'Innerb1': Innerb1_tf,
                        },
                        format='5'
                        )

                print('Parameters saved')

            elif testavg_loss  <= minimum_loss:
                minimum_loss = testavg_loss
                saver.save(sess, directory + 'my-test-model'+ str(int(i/epoch)))
                print('Parameters saving')
                with tf.variable_scope('',reuse=True):
                    W1_tf= tf.get_variable(name='Iniaff1/weigth', shape=[ 2 * N, 2 * K], dtype=np.float64)
                    b1_tf = tf.get_variable(name="Iniaff1/bias", shape=[   2 * K], dtype=np.float64)

                    InnerW1_tf =[]
                    Innerb1_tf = [] 
                    Inner3W2_tf = []
                    Inner3W3_tf =[]
 
                    for j in range(LayerTied): 
                        InnerW1_tf.append(sess.run(tf.get_variable(name='affineinner' + np.str(j)+'/weigth1', shape=[2 * N ], dtype=np.float64)))
                        Innerb1_tf.append(sess.run(tf.get_variable(name='affineinner' + np.str(j) + '/bias', shape=[2 * N], dtype=np.float64)))

                    spio.savemat(directory +'OnebitNetwork_'+str(N)+'by'+ str(K) +'SNR'+str(snrdb_low)+'_'+str(snrdb_high)+ 'Layer'+str(LayerTied) + str(int(i/epoch))+'.mat', {
                        'Iniaff_W1': sess.run(W1_tf),
                        'Iniaff_b1': sess.run(b1_tf),

                        'beta': sess.run(beta),
                        'gamma': sess.run(gamma),
                        'alpha': sess.run(alpha),
                        'InnerW1': InnerW1_tf,
                        'Innerb1' : Innerb1_tf, 
                    },
                             format='5'
                             )

                print('Parameters saved')



        batch_H, batch_Y , batch_X, batch_Homega, batch_sigma  = generate_data_Onebit_MIMO( train_batch_size, K, N, snr_low, snr_high)
        _,traingamma, trainavg_loss,  trainBER = sess.run( [train_step,gamma, loss2,  BER], feed_dict={H: batch_H, Y: batch_Y, X_true: batch_X, Homega: batch_Homega, sigma: batch_sigma  })

        if i % epoch == 0:

            print('\n step %d, train avg loss %g,  train BER %g' % (i, trainavg_loss, trainBER))
            print('\n step %d, test avg loss %g, test BER %g' % ( i, testavg_loss, testBER))
 
            Strainavg_loss.append(trainavg_loss )
            StrainBER.append(trainBER)

            Stestavg_loss.append(testavg_loss)
            StestBER.append(testBER)


    xlab = np.arange(train_iter/epoch+1)
    fig = plt.figure()
    plt.subplot(1,2, 1)
    plt.plot(xlab, Strainavg_loss, label='Training Avg Loss',marker = 'o')
    plt.plot(xlab, Stestavg_loss, label='Testing Avg Loss',marker = 'o')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.title("Avg Loss")
    plt.xlabel('Iteration %d' % epoch)
    plt.ylabel("Avg Loss")

    plt.subplot(1,2, 2)
    plt.semilogy(xlab,StrainBER,label = 'Training BER',marker = 'o')
    plt.semilogy(xlab,StestBER,label = 'Testing BER',marker = 'o')
    plt.legend(loc='best',fancybox = True,shadow = True)
    plt.title("BER")
    plt.xlabel('Iteration %d'%epoch)
    plt.ylabel("BER")
    fig.savefig( directory + 'Loss_record.png')

    end = time.time()
    print('*' * 50)
    print('training finish.\ncost time:', int(end - st), 'seconds')



    snrdb_list = np.linspace(snrdb_low_test, snrdb_high_test, num_snr)
    snr_list = 10.0 ** (snrdb_list/10.0)
    bers = np.zeros((1, num_snr))
    times = np.zeros((1, num_snr))
    tmp_bers = np.zeros((1, test_iter))
    tmp_times = np.zeros((1, test_iter))
    for j in range(num_snr):
        print("Testing SNR:", snrdb_list[j])
        print("Overall test iteration:",test_iter)
        for jj in range(test_iter):
            batch_H, batch_Y, batch_X, batch_Homega, batch_sigma  = generate_data_Onebit_MIMO( test_batch_size, K, N, snr_list[j], snr_list[j])
            tic = tm.time()
            tmp_bers[:, jj] = np.array(sess.run([  BER], feed_dict={H: batch_H, Y: batch_Y, X_true: batch_X, Homega: batch_Homega, sigma: batch_sigma }))
            toc = tm.time()
            tmp_times[0][jj] = toc - tic
        bers[0][j] = np.mean(tmp_bers, 1)
        times[0][j] = np.mean(tmp_times[0]) / test_batch_size

    fig2 = plt.figure()
    plt.semilogy(snrdb_list,bers[0],marker = 'o')
    plt.title("Testing BER")
    plt.xlabel("SNR")
    plt.ylabel("Testing BER")
    fig2.savefig(directory + 'TestLoss_record.png')


    print('snrdb_list')
    print(snrdb_list)
    print('bers')
    print(bers)
    print('times')
    print(times)

    spio.savemat(directory + 'tests_pre.mat', {'snrdb_list': snrdb_list, 'bers': bers, 'times':times})

    plt.show()
