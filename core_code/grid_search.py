import sys
import datetime

saving_name = "grid_search10_new_loss_10run"

# name of log file
Log_file = "log/"+saving_name+" .log"
saved_result_data = 'result_data/'+saving_name+'.pkl'
print('saving log: ', Log_file)
print('saving result: ', saved_result_data)
 

# logging with both screen output and saving to log
# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(Log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

# log header
print('\n\n\n New excution on: ',datetime.datetime.now(), '\n' )


################################### Your Program Starts here ###################################

# LeNet for MNIST using Keras and TensorFlow

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from keras.utils import np_utils
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.core import Permute, Reshape
from keras import backend as K
from keras import optimizers
from keras.models import load_model
np.random.seed(2)


# use GPU mem incrementally
import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)



# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


from utils import *
from neurallogic import *
from lenet import *
from data_prepare import *
from intergrated_model import *

# # Seed value
# # Apparently you may use different seed values at each stage
# seed_value= 0
# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
# import os
# os.environ['PYTHONHASHSEED']=str(seed_value)
# # 2. Set the `python` built-in pseudo-random generator at a fixed value
# import random
# random.seed(seed_value)
# # 3. Set the `numpy` pseudo-random generator at a fixed value
# import numpy as np
# np.random.seed(seed_value)
# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.random.set_seed(seed_value)

# # # 5. Configure a new global `tensorflow` session
# # from keras import backend as K
# # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# # K.set_session(sess)


global_iter = 10

saved_data = []


for iter_current in range(global_iter):
    print('#################### global iter: ', iter_current+1)


    simulation_name = 'simulation_1'

    ############## CE definition ##############

    NL_model_name = 'NL_' + simulation_name
    PL_model_name = 'PNL_' + simulation_name


    num_event_type = 10   # total num of unique events  3x3 + 1 unknown
    num_attribute = 1      # total num of attribute (int)

    ce_fsm_list = [ [1, 2, 3],  [4,5,6],  [7,8,9], [1, 10] ]
    ce_time_list = [ np.array([ [INF, INF], [0, 0]]),  
                    np.array([ [INF, INF ], [0, 0 ]]),  
                    np.array([ [INF, INF], [0, 0]]),
                    np.array([[INF], [0]])]

    event_num = 100000
    window_size = 10

    train_neurallogic_model(NL_model_name,
                                num_event_type, num_attribute,
                                ce_fsm_list, ce_time_list,
                                window_size,
                                event_num ,
                                verify_logic = False,
                                diagnose = True)

    # logic verification takes long time...  (uniq_e_num ^ window_size)

    mnist_event_num = 100000

    mnist_data_event_path = PL_model_name+'_data_' + str(mnist_event_num) +'.npz'
    mnist_data_event, data_feature, data_label = generate_mnist_event_data(num_event_type, num_attribute, 
                                                                              ce_fsm_list, ce_time_list,
                                                                              mnist_event_num, window_size,
                                                                              mnist_data_event_path)

    if np.ndim(data_label)==1:
        data_label = np.expand_dims(data_label, axis = 1)

    # valid_index = (data_label!= 999)# choose valid data samples
    valid_index = (data_label.sum(axis = 1)!= 0)
    v_mnist_data_event  =  mnist_data_event[valid_index, ]
    v_data_label = data_label[valid_index, ]
    print('Valid data (with valid CE) shape: ')
    print("Data:\t", v_mnist_data_event.shape)
    print("Label:\t", v_data_label.shape)

    # prepare the testing MNIST dataset for customized LeNet

    mnist_label = list(range(num_event_type)) # 
    mnist_x_test, mnist_y_test = generate_customized_mnist_data(mnist_label, num_event_type)

    ##### grid search ####
    def ce_model_acc(v_mnist_data_event, v_data_label, model):
        pred_score = model.predict(v_mnist_data_event)
        accuracy = sum(v_data_label == pred_score.round() ) / v_data_label.shape[0]
        avg_acc = accuracy.mean()
        return accuracy, avg_acc

    history_list = []
    score_list = []
    acc_list = []

    # omega_list = list(range(0,11, 1))
    # omega_list = [i/10 for i in omega_list]
    omega_list = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 0]
    print(omega_list)


    for omega in omega_list:

        print('============= Current Omega: ',omega, ' =============')


        # generate a new LeNet model from scratch
        lenetModel = LeNet_mnist(num_output = num_event_type)  # 2 events
        lenetModel.name="lenet"
        score = lenetModel.evaluate(mnist_x_test, mnist_y_test, verbose=0)
        print('Test loss: %3f,  \t \tTest Accuracy: %4f'%(score[0], score[1]))

        # Returns a compiled model identical to the previous one
        loading_path = 'saved_model/'+NL_model_name+'.hdf5'
        neuralLogic_model = load_model(loading_path)
        neuralLogic_model.name="neurallogic"
        print('Loading model successfully from ', loading_path)

        final_model = intergrated_model(lenetModel, neuralLogic_model, 
                                        window_size, num_event_type,
                                        omega_value = omega,
                                        load_nl_weights = True,
                                        nl_trainable = False,
                                        loss = 'combined_loss',
                                        diagnose = False)
        # loss = combined_loss / mse_loss



        epochs = 10000
        diagnose = False
        save_path = 'temp/'+ PL_model_name +'.hdf5'
        es = EarlyStopping(monitor='val_MAE', mode='min', verbose=1, patience=30)
        mc = ModelCheckpoint(save_path, monitor='val_MAE', mode='min', verbose=diagnose, save_best_only=True)
        cb_list = [es, mc]

        print('The maximum training epochs is: ', epochs)
        H = final_model.fit( v_mnist_data_event ,v_data_label, 
                                batch_size = 256, 
                                epochs = epochs,
                                verbose=diagnose,
                                shuffle=True,
                                callbacks=cb_list,
                                validation_split = 0.2)
        #                         validation_data=(v_mnist_data_event, v_data_label))

        final_score = final_model.evaluate(v_mnist_data_event, v_data_label, verbose = 0)
        _, final_acc = ce_model_acc(v_mnist_data_event, v_data_label, final_model)

        lenet_score = lenetModel.evaluate(mnist_x_test, mnist_y_test, verbose=0)
        print('Test loss: %3f,  \t \tTest Accuracy: %4f'%(lenet_score[0], lenet_score[1]))


        print('Trained result: ', final_score, 'Acc: ',  final_acc, '\n\n')
        score_list.append(final_score)
        acc_list.append(final_acc)

        hist_NL = H.history
        history_list.append(hist_NL)
        
        
    
    # append result for a single iter
    result_data = [acc_list, score_list, history_list]
    saved_data.append(result_data)
        
        
import pickle
with open(saved_result_data, 'wb') as f:
    pickle.dump(saved_data, f)
