from utils import *
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.core import Permute, Reshape
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
import os
import itertools


def train_neurallogic_model(model_name,
                            num_event_type, num_attribute,
                            ce_fsm_list, ce_time_list,
                            window_size,
                            event_num ,
                            verify_logic,
                            diagnose = False):
    """
    better to create a class later...
    
    This module generates the event data(windowed), and event label for the CE specs in the definition.
    
    The inputs:
    model_name: name of the NeuralLogic model to be trained.
    num_event_type: num of unique events
    num_attribute: number of event attributes, usually 1. SID
    ce_fsm_list:  The pattern of events in CE. This is a list of lists. Each sub-list store the pattern of a CE
    ce_time_list: The temporal constraints of CE. Only gives the min/max time interval between two consecutive events.
    window_size: the size of detection window in Neural Logic
    event_num: total num of events data generated randomly
    stop_condition: when training stops (both MSE and MAE < then this threshold)
    
    First check if the NeuralLogic model exists:
    if not, start training. else, pass.
    
    
    """
    ############## Check if the required NL model exist in saved_model dir ##############
    if (model_name+'.hdf5') in os.listdir('saved_model'):
        print('NeuralLogic model exists in: ',  'saved_model/'+ model_name+'.hdf5')
        return
    else:
        print('No existing model found, start training....')
    

    ############## generate data samples (contain CE) and label them ##############
    data_feature, data_label = data_generator(num_event_type, num_attribute, 
                                               ce_fsm_list, ce_time_list,
                                               event_num, window_size )

    data_feature = data_feature[:, :, 0:num_event_type] ## only look at the pattern, (ignore timestamp and attributes)
    print(data_feature.shape)

    # visualization of data class distribution
    data_class_distribution(data_label, y_lim = event_num)


    # split for training and testing    ---# no validation here 
    # TRAIN - TEST
    p_train = 0.8
    rnd_indices = np.random.rand(data_label.shape[0]) < p_train

    train_data = data_feature[rnd_indices]
    train_label = data_label[rnd_indices]

    test_data = data_feature[~rnd_indices]
    test_label = data_label[~rnd_indices]

    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)




    ############## define Neural Logic Model ##############
    num_ce = len(ce_fsm_list)
    num_hidden_lstm = 64
    _, win_len, dim = train_data.shape
    print('Length of time window: ', win_len, 
          "\n Dimension of data: ", dim)

    num_classes = num_ce

    print('building the model ... ')
    model = Sequential()
    model.add(LSTM(num_hidden_lstm, 
               input_shape=(win_len,dim), 
               return_sequences=False))
    # model.add(Dropout(0.5))
    # model.add(LSTM(32, return_sequences=False))
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='linear') ) 
    model.summary()


    ############## training Neural Logic Model ##############
    epochs = 1000
    batch_size = 128

    sgd = optimizers.SGD(lr=0.01)
    adam = optimizers.Adam(lr=0.001)
#     sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
#     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)

    # use mean_squared_error here for multilabel classify  (mae, mse, rmse as metric)
    model.compile(loss= "mean_squared_error", optimizer= adam,  
                  metrics=['MAE'])


    ############## training ##############   
    save_path = 'saved_model/'+ model_name+'.hdf5'
    es = EarlyStopping(monitor='val_MAE', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(save_path, monitor='val_MAE', mode='min', verbose=diagnose, save_best_only=True)
    cb_list = [es, mc]
    
    print('============ Start NL Model Training ===========\n')
    print('The maximum training epochs is: ', epochs)
    H = model.fit(train_data, train_label,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                shuffle=True,
                callbacks = cb_list,
                validation_data=(test_data, test_label))
    
#     while True:
#         H = model.fit(train_data, train_label,
#                 batch_size=batch_size,
#                 epochs=epochs,
#                 verbose=0,
#                 shuffle=True,
#                 validation_data=(test_data, test_label))

#         score = model.evaluate(test_data, test_label, verbose=0)
#         if (score[1]<stop_condition and score[0]<stop_condition):
#             break
#         else:
#             print("Training progress: ", score)
#     print('Training finished sucessfully.')      
#     print('Test loss:', score[0])
#     print('Test MAE:', score[1])



    ############## visualize prediction on testing data #############
    # p_result = model.predict(test_data)

    # import matplotlib.pyplot as plt

    # plot_num = 30

    # for i in range(p_result.shape[1]):

    #     fig = plt.figure(figsize=(12,8))
    #     plt.plot(p_result[0:plot_num, i], 'r-')
    #     plt.plot(test_label[0:plot_num], 'b-')
    #     plt.legend(['predict', 'true'])
    #     plt.show()


    ############## visualize learning curves #################
    # print(H.history.keys() )

    # fig = plt.figure(figsize=(12,8))

    # plt.plot(H.history['val_loss'],  'r-')
    # plt.plot(H.history['loss'],  'b-')

    # plt.plot(H.history['val_mean_absolute_error'],  'r--')
    # plt.plot(H.history['mean_absolute_error'],  'b--')

    # plt.ylim([0, 30])

    # plt.legend(['Validation loss', 'Loss', 'Validation_MAE', 'MAE'])
    # plt.show()


    ############### Saving training models ################
    from keras.models import load_model
  
    trained_model = load_model(save_path)
    score = trained_model.evaluate(test_data, test_label, verbose=0)
    print('Valid loss:', score[0], '\t Valid MAE:', score[1])
    print('Neural Logic model saved at: ', save_path)
    if verify_logic:
        logic_model_verify(trained_model, 
                           ce_fsm_list, ce_time_list, 
                           window_size, num_event_type,
                           constraint_selection = False)

    
    

def logic_model_verify(neuralLogic_model, 
                       ce_fsm_list, ce_time_list, 
                       window_size, num_event_type,
                       constraint_selection = False):
    """
    Verify the logic of NeuralLogic model: generate the truth table, check if the NN outputs match
    the logic output. The length of the Truth table is num(uniq_event)^window_size
    
    # the num(uniq_event) here == event_num used in some other functions
    
    Input:
    neuralLogic_model, 
    ce_fsm_list, ce_time_list, 
    window_size
    constraint_selection: if the constraints are also checked in the Logical method.
    
    Output: 
    verify_result: truth iff all the test cases pass
    
    """
    
    verify_result = True
    
    uniq_event = list(range(num_event_type))   # include all event : either informed or unknown
#     print(uniq_event)
    
    from tqdm import tqdm_notebook
    # for i in tqdm( itertools.product(uniq_event, repeat= window_size)):
    # in ipynb notebook, tqdm would create new line, so use tqdm_notebook instead
    total_len = len(uniq_event)**window_size
#     total_len = len(list(itertools.product(uniq_event, repeat= window_size)))
    for i in tqdm_notebook( itertools.product(uniq_event, repeat= window_size), total = total_len):
#         print(i)
        event_stream = np.array(list(i)) -1  # minus one to adjust the event_id / ind
        event_feature = np.zeros([window_size, len(uniq_event)])
        event_feature[np.arange(window_size), event_stream] = 1

        # logic result
        logic_result = detection_machine(event_feature, ce_fsm_list, ce_time_list, window_size, constraint_selection = constraint_selection)
        # NN model result
        nn_result = np.around( neuralLogic_model.predict(np.expand_dims(event_feature, axis = 0 )) )
        
        if (logic_result != nn_result).all():
            print(event_stream)
            print('Logic: ',logic_result , '\tNN: ', nn_result)
            verify_result = False
#         break
    if verify_result == True:
        print('NN model logic verified!')
    else:
        print('NN model logic has flaws...')

    return verify_result