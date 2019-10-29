import os
from keras.datasets import mnist
from keras.utils import np_utils
from utils import *


def generate_customized_mnist_data(mnist_label, 
                          num_event_type):
    """
    Prepare the testing MNIST dataset for customized LeNet
    Input: 
    mnist_label: the list of mnist label used.
     # the mnist label used in this task (typically the [0,0,1] event is digit 2. --argmax)
    num_event_type: number of unique events
    
    Output: 
    mnist_x_test: np array with shape [num, 28, 28, 1]
    mnist_y_test: np array with shape [num, num_event_type]
    """

    # loading MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    # the mnist label used in this task (typically the [0,0,1] event is digit 2. --argmax)

    if len(mnist_label) == num_event_type:
        pass
    else:
        raise ValueError('The assigned MNIST labels: ', mnist_label, ' donot match unique event number: ', num_event_type)

    testing_ind = (y_test == -1) # index with all Falses
    for label_i in mnist_label:
        testing_ind = (testing_ind)|(y_test == label_i)

    mnist_x_test = x_test[testing_ind,]
    mnist_y_test =  y_test[testing_ind,]

    mnist_x_test = np.expand_dims(mnist_x_test, axis=3)
    mnist_y_test = np_utils.to_categorical(mnist_y_test, num_event_type)

    print('Testing mnist feature shape: ',mnist_x_test.shape)
    print('Testing mnist label shape: ', mnist_y_test.shape)  
    return mnist_x_test, mnist_y_test



def generate_mnist_event_data(num_event_type, num_attribute, 
                              ce_fsm_list, ce_time_list,
                              event_num, window_size,
                              mnist_data_event_path):
    """
    generate the windowed event data with CE in it. Each event is a mnist image.
    input:
    num_event_type, num_attribute
    ce_fsm_list, ce_time_list
    mnist_event_num (how many mnist event randomly generated), window_size
    
    output:
    mnist_data_event:
    data_feature:
    data_label:
    
    """
    
    ############## Check if data file exists: in data dir ##############
    if mnist_data_event_path in os.listdir('data/'):
        print('Data file exists in: ',  'data/'+ mnist_data_event_path)
        npzfile = np.load('data/'+ mnist_data_event_path)
        return npzfile['mnist_data_event'], npzfile['data_feature'], npzfile['data_label']
    else:
        print('No existing data found, start generating....')

    
    # loading MNIST dataset
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    mnist_data_event, data_feature, data_label = mnist_data_generator(num_event_type, num_attribute, 
                                                                       ce_fsm_list, ce_time_list,
                                                                       event_num, window_size, 
                                                                       x_train, y_train)
    # abandon the last several column features (attributes, timestamps)
    data_feature = data_feature[:, :, 0:num_event_type]
    
    # visualization
    data_class_distribution(data_label, y_lim = event_num)
    
    # changing the shape of mnist_data_event
    mnist_data_event = np.expand_dims(mnist_data_event, axis=4)  # expand dim of mnist with a 3rd channel
    print(mnist_data_event.shape)
    print(data_label.shape)
    
    print('Dim of MNIST event data: ',mnist_data_event.shape)
    print('Dim of event label data: ',data_feature.shape)
    print('Dim of MNIST event label: ',data_label.shape)
    
    save_path = 'data/'+ mnist_data_event_path
    np.savez(save_path ,mnist_data_event = mnist_data_event,data_feature = data_feature, data_label = data_label )
    print('Saved data file to: '+ save_path)
    
    return mnist_data_event, data_feature, data_label
    
    
 