###   DeepHOTML for Classical MIMO Detection Problem ################
# This program performs training for the DeepHOTML in the following paper
# ``Binary MIMO Detection via Homotopy Optimization and Its Deep Adaptation'' by Mingjie Shao and Wing-Kin Ma
# The code is for classical MIMO detection with correlated Gaussian channels.
# If you have any questions, please contact mjshao@link.cuhk.edu.hk
# Nov. 27, 2020
#####################################################################

# if one uses tensorflow 2.0 or above, use the following command
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# otherwise use the following command
# import tensorflow as tf
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


# generate training settings
def generate_data_classical_MIMO( B,M,N,snr_low,snr_high):
    H_real = np.random.randn(B,N,M)/np.sqrt(2)
    H_imag = np.random.randn(B, N, M)/np.sqrt(2) 
    H1 = np.concatenate( [H_real,  -H_imag],axis=2)
    H2 = np.concatenate( [H_imag, H_real],axis=2)
    H =  np.concatenate( [H1,H2],axis=1 ) 
    HTH = np.zeros([B, 2 * M, 2 * M])
    X_real = np.sign(np.random.rand(B, M )-0.5)
    X_imag = np.sign(np.random.rand(B, M) - 0.5)
    X =  np.concatenate([ X_real, X_imag],axis=1)
 
    Y = np.zeros([B,2*N ])
    Ynoise = np.zeros([B,2*N ])
    Hynoise =np.zeros([B,2*M ])
    SNR = np.zeros([B])
    sigma = np.zeros([B])
    for i in range(B):
        Hslide = H[i,:, :]
        HTH[i, :, :] = Hslide.T.dot(Hslide) 
        x = X[ i  ,: ]
        SNR[i] = np.random.uniform(low=snr_low, high=snr_high)
        y =  Hslide.dot(x) 
        Y[ i  ,: ] = y
        sigma[i] = np.sqrt(2*M/ SNR[i]  )
        w = sigma[i] * np.random.randn(2*N)/np.sqrt(2)
        Ynoise[i  ,:] =  y+ w
        Hynoise[i, :] = Hslide.T.dot(Ynoise[i  ,:]) 
    return H, Ynoise ,X ,Hynoise ,HTH, sigma

def affine_layer(x,input_size,output_size,Layer_num):
    with tf.variable_scope(Layer_num):      
        weigth = tf.get_variable(name= "weigth", shape=[input_size, output_size], initializer=tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable(name= "bias", shape=[output_size], initializer=tf.random_normal_initializer(stddev=0.1) ) 
    y = tf.matmul(x,weigth)+bias
    return y


# result analysis, create directory for saving results
# you may change the directory name to save the parameters for differennt settings
if not os.path.exists('DeepHOTML_Classical_MIMO_iid_channel/'):
    os.makedirs('DeepHOTML_Classical_MIMO_iid_channel/')
directory = './DeepHOTML_Classical_MIMO_iid_channel/'

#parameters 
M = 40   # number of receive antennas
N = 40   # number of transmit antennas
# Note that these are complex-valued dimension.

snrdb_low = 12 # training SNR lower bound
snrdb_high = 18  # training SNR upper bound
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)


train_iter = 20000    # training iteration
train_batch_size = 512    # training batch size
epoch = 200     #  training  epoch 


# parameters for test 
snrdb_low_test = 0  # test SNR lower bound
snrdb_high_test = 20 # test SNR upper bound
snr_low_test = 10.0 ** (snrdb_low_test/10.0)
snr_high_test = 10.0 ** (snrdb_high_test/10.0)
num_snr = 5
test_batch_size =  train_batch_size
test_iter = 1000


# training learning rate and decay factor
startingLearningRate = 0.001
decay_factor = 0.85
decay_step_size = 500

# Architecture of DeepHOTML
LayerNo = 20  # number of layers in Deep HOTML
Y = tf.placeholder(tf.float32, shape=[ None,2*N], name='Y')
H = tf.placeholder(tf.float32, shape=[None,2*N,2*M ], name='H')
HTH = tf.placeholder(tf.float32, shape=[None,2*M,2*M ], name='HTH')
HY = tf.placeholder(tf.float32, shape=[None,2*M ], name='HY')
X_true = tf.placeholder(tf.float32, shape=[None,2*M ], name='X_true')
sigma = tf.placeholder(tf.float32, shape=[None], name='sigma')
batch_size = tf.shape(HY)[0]


#####  Deep HOTML ##############
# training parameters of each layer, except the input layer
beta = [] # beta in Eqn. 29
omega = [] # omega in Eqn. 29
alpha = [] # alpha in Eqn. 29
gamma= [] # gamma in Eqn. 29
for i in range(LayerNo):
    beta.append(tf.Variable(np.float32(0.01), name=('beta_' + np.str(i))))
    omega.append(tf.Variable(np.float32(0.01), name=('omega_' + np.str(i))))
    alpha.append(tf.Variable(np.float32(0.5), name=('alpha_' + np.str(i))))
    gamma.append(tf.Variable(np.float32(0.001), name=('gamma_' + np.str(i))))
LOSS = []

# input layer
Xup = tf.maximum(tf.minimum(affine_layer(Y, 2 * N, 2 * M, 'relu1'), 1), -1) 

V = np.zeros([train_batch_size, 2 * M], dtype=np.float32)
Z = Xup

for j in range(LayerNo):

    Xpre = Xup
    R = Z - beta[j] * tf.squeeze(tf.matmul(tf.expand_dims(Z, 1), HTH), 1) + omega[j] * HY +  gamma[j]*V
    Xup = tf.maximum(tf.minimum(R, 1), -1)
    Z = Xup + alpha[j]* tf.subtract(Xpre, Xup)
    V = Xup

    LOSS.append( np.log(j+1)* (tf.reduce_mean( tf.square(Xup - X_true) )))

# training objective
TOTAL_LOSS=tf.add_n(LOSS)
X_est = tf.identity( Xup  , name="X_est")
X_estsign = tf.identity( tf.sign(Xup) , name="X_estsign")
loss2 =  tf.reduce_mean( tf.square(X_est - X_true) )
BER = tf.reduce_mean(tf.cast(tf.not_equal(X_true,X_estsign), tf.float32))

# training optimizer
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss2)

saver = tf.train.Saver()
minimum_loss = np.float32(500)
StrainTOTAL_LOSS = []
Strainavg_loss = []
StrainBER = []
Stestavg_loss = []
StestBER = []
StestTOTAL_LOSS = []

# training DeepHOTML

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    st = time.time()
    print('training DeepHOTML')

    for i in range(train_iter+1):
        

        # save parameters. Here we use validation every epoch to decide whether the parameters should be saved or not. If the tested loss is less than the current best, then save the parameters of this iteration.
        # We also save the paramters of the last 10 epoches. Usually, we choose the last saved parameters except the last 10 epoches. But if that does not work well, we choose one of the saved parameters from the last 10 epoches.
        if i % epoch == 0:
            testbatch_H, testbatch_Y , testbatch_X, testbatch_HY, testbatch_HTH, testbatch_sigma  = generate_data_classical_MIMO( test_batch_size, M, N, snr_low, snr_high)
            testTOTAL_LOSS,testavg_loss,  testBER = sess.run([TOTAL_LOSS,loss2, BER], feed_dict={ H: testbatch_H,Y: testbatch_Y,  X_true: testbatch_X, HTH: testbatch_HTH, HY:testbatch_HY, sigma:testbatch_sigma })
            
            if i>= train_iter - 10 * epoch:
                saver.save(sess, directory + 'my-test-modellast'+ str(int(i/epoch)))

                with tf.variable_scope('',reuse=True):
                    W1_tf= tf.get_variable(name='relu1/weigth', shape=[ 2 * N, 2 * M])
                    b1_tf = tf.get_variable(name="relu1/bias", shape=[   2 * M])


                    spio.savemat(directory +'DeepHOTML_last_'+ str(N)+'by'+ str(M) +'SNR'+str(snrdb_low)+'_'+str(snrdb_high)+ 'Layer'+str(LayerNo)+'_'+ str(int(i/epoch))+'.mat', {
                        'W1': sess.run(W1_tf),
                        'b1': sess.run(b1_tf),


                        'beta': sess.run(beta), 
                        'omega': sess.run(omega), 
                        'alpha': sess.run(alpha), 
                        'gamma': sess.run(gamma)
                    },
                             format='5'
                             )

                print('Parameters saved')


            elif testavg_loss <= minimum_loss:
                minimum_loss = testavg_loss
                saver.save(sess, directory + 'my-test-model'+ str(int(i/epoch)))

                with tf.variable_scope('',reuse=True):
                    W1_tf= tf.get_variable(name='relu1/weigth', shape=[ 2 * N, 2 * M])
                    b1_tf = tf.get_variable(name="relu1/bias", shape=[   2 * M])


                    spio.savemat(directory +'DeepHOTML_'+ str(N)+'by'+ str(M) +'SNR'+str(snrdb_low)+'_'+str(snrdb_high)+ 'Layer'+str(LayerNo)+'_'+ str(int(i/epoch))+'.mat', {
                        'W1': sess.run(W1_tf),
                        'b1': sess.run(b1_tf),


                        'beta': sess.run(beta), 
                        'omega': sess.run(omega), 
                        'alpha': sess.run(alpha), 
                        'gamma': sess.run(gamma),

                    },
                             format='5'
                             )

                print('Parameters saved')



        batch_H, batch_Y , batch_X, batch_HY, batch_HTH , batch_sigma= generate_data_classical_MIMO( train_batch_size, M, N, snr_low, snr_high)
        _,traingamma,trainTOTAL_LOSS, trainavg_loss, trainBER = sess.run( [train_step,gamma,TOTAL_LOSS, loss2,  BER], feed_dict={H: batch_H, Y: batch_Y, X_true: batch_X, HTH: batch_HTH, HY: batch_HY , sigma:batch_sigma})
 
        if i % epoch == 0:

            print('\n step %d, train avg loss %g, train TOTAL_LOSS %g, train BER %g' % (i, trainavg_loss, trainTOTAL_LOSS,trainBER))
            print('\n step %d, test avg loss %g, test TOTAL_LOSS %g,  train BER %g' % ( i, testavg_loss, testTOTAL_LOSS,testBER))

            StrainTOTAL_LOSS.append(trainTOTAL_LOSS)
            StestTOTAL_LOSS.append(testTOTAL_LOSS)
            Strainavg_loss.append(trainavg_loss )
            StrainBER.append(trainBER)
            Stestavg_loss.append(testavg_loss)
            StestBER.append(testBER)

    # plot the training loss
    xlab = np.arange(train_iter/epoch+1)
    fig = plt.figure()
    plt.subplot(1,3, 1)
    plt.plot(xlab, StrainTOTAL_LOSS, label='Training TOTAL_LOSS', marker='o')
    plt.plot(xlab, StestTOTAL_LOSS, label='Testing TOTAL_LOSS', marker='o')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.title("TOTAL_LOSS")
    plt.xlabel('Iteration %d' % epoch)
    plt.ylabel("TOTAL_LOSS")
    plt.subplot(1,3, 2)
    plt.plot(xlab, Strainavg_loss, label='Training Avg Loss',marker = 'o')
    plt.plot(xlab, Stestavg_loss, label='Testing Avg Loss',marker = 'o')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.title("Avg Loss")
    plt.xlabel('Iteration %d' % epoch)
    plt.ylabel("Avg Loss")
    plt.subplot(1,3, 3)
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


    # testing the DeepHOTML
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
            batch_H, batch_Y, batch_X, batch_HY, batch_HTH ,batch_sigma= generate_data_classical_MIMO( test_batch_size, M, N, snr_list[j], snr_list[j])
            tic = tm.time()
            tmp_bers[:, jj] = np.array(sess.run([ BER], feed_dict={H: batch_H, Y: batch_Y, X_true: batch_X, HTH: batch_HTH, HY: batch_HY, sigma:batch_sigma }))
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


    spio.savemat(directory + 'tests_pre.mat', {'snrdb_list': snrdb_list, 'bers': bers})
    
    plt.show()
