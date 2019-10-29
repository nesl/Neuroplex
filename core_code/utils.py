import numpy as np
import time
import copy 
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool

INF = float('inf')



def event_generator(num_event_type, num_attribute, last_timestamp):
    """
    Any event should have: 
    event_type, and attributes of discrete value (user_id) and continuous value (timestamp) to express different logic
    Generate event vector of the format:  event_type[1xnum_e_type], event_time[1], event_user_id[1]
    
    Inputs:
    - num_event_type
    - num_attribute
    - last_timestamp
    
    Outputs:
    - new_event: 
    - event_time: 
    
    e.g.
    ABCD event types, for event_type == 4
    Event users 123, for event_attribute == 3
    sample event: [ 0 0 0 1, 1.234, 2 ]
    """
    event_type = np.zeros(num_event_type)
    e_type_id = np.random.randint(num_event_type)
    event_type[e_type_id] = 1
    
    event_time = last_timestamp + np.random.rand()  # uniform distribution over 0-1
    event_user_id = np.random.randint(num_attribute) # user id is not one-hot encoded here
    
    new_event = np.concatenate((event_type, np.array([event_time]),  np.array([event_user_id]) ), axis = 0)
    
    return new_event, event_time



def check_constraints(event_data, path_i, timing_list):
    """
    check if path i in event_data satisfies the timing constraints
    
    Inputs:
    - event_data:  all event data array
    - path_i:      one of the candicate path returned from FSM detection of current CE
    - timing_list: the timing constraints list of current CE
    
    Outputs:
    True if satisfies, False otherwise
    
    Three flags in total: feature_flag, smaller_flag, larger_flag
    """
        
    # Initialize all flags:
    feature_flag = True
    smaller_flag = True
    larger_flag = True
    
    # check features:
    feature_loc = event_data.shape[1]-1
    ce_features = event_data[path_i[0], feature_loc]
    for i in path_i:
        if event_data[i, feature_loc] != ce_features:
            feature_flag = False
            
    # check smaller:
    time_loc = event_data.shape[1] - 2
    for i in range(len(path_i)-1):
        if event_data[path_i[i+1], time_loc] - event_data[path_i[i], time_loc] > timing_list[0, i]:
            smaller_flag = False
    
    # check larger:
    for i in range(len(path_i)-1):
        if event_data[path_i[i+1], time_loc] - event_data[path_i[i], time_loc] < timing_list[1, i]:
            larger_flag = False    
            
    final_flag = feature_flag and smaller_flag and larger_flag
#     print(feature_flag , smaller_flag , larger_flag)
    return final_flag



def find_ce_path(event_type, event_list):
    """
    Recursive function, use idea of dynamic programming
    Inputs:
    - event_type: data event type, n x 1 list
    - event_list: CE event pattern
    Outputs:
    - path list:  a list of lists, where each list store the index of relevant events of a single path
    
    Notes: the path does not have to end and the ending events. (e.g. in abcdc find ac, it can be [0,2] and [0, 4])
    
    """
    # if only one event 
    if len(event_list) == 1:
        path = [ [i] for i, x in enumerate(event_type) if x == event_list[0]]
        return path
    else:
        path = []
        for i in range(len(event_type)):
            if event_type[i] == event_list[0]:
                # recursion part, need to fix the index issue
                path_i = find_ce_path(event_type[i+1:], event_list[1:])
                path_i = [ [k+1+i for k in path_i_j] for path_i_j in path_i]
                
                new_path_i = [ [i, *path_i_j] for path_i_j in path_i]  # append current index to all paths

                path = path + new_path_i
                
    return path



def detection_machine(event_data, ce_fsm_list, ce_time_list, window_size, constraint_selection = False):
    """    
    Define detection function that can provide ground truth
    output can be (1) number of CE for each CE, or (2) binary detection for each CE
    
    Inputs:
    - event_data is the already windowed event_data
    - ce_fsm_list is the known Event sequence of Complex Events
    - ce_time_list is the timing constraints
    - window_size is the size of event_data, just to check if the event_data is windowed correctly.
    
    Return:
    A 1 x num_ce vector, each entry stores the detection result for every complex event.
        num_ce == len(ce_fsm_list)
    
    d_flag_1 and d_flag_2 are flags for detection and selection (FSM and other constraints)
    """    
    
    # check consistency of inputs
    if event_data.shape[0] == window_size:
        pass
    else:
        raise ValueError('Windowed Event data size: ', event_data.shape[0], ' is not equal to  window_size', window_size)
    
    num_ce = len(ce_fsm_list)
    # udpating num_event calculation: based on unique events
    flat_ce_list = [item for sublist in ce_fsm_list for item in sublist]
    num_event = len(list(set(flat_ce_list) ) )
    # num_event = event_data.shape[1] - 2
    
    # converting event type for FSM detection
    event_type_data = event_data[:, 0:num_event]
    event_type = np.argmax(event_data[:, 0:num_event], axis=1)+1  # change event type to 1, 2, 3, 4 ...
    
    # initialize return vector
    ce_result = np.zeros([1,num_ce])
    
    d_flag_1 = d_flag_2 =0
    
    for ce in range(num_ce):
        
        event_list = ce_fsm_list[ce]
        timing_list = ce_time_list[ce]
        
        # FSM detection
        if event_type[-1]!=event_list[-1]:
            d_flag_1 = 0
        else:
            path = find_ce_path(event_type, event_list)
                
            # delete the path that doesnt end with last event
            invalid_path = []
            for path_i in path:
                if path_i[-1] != (window_size-1):
                    invalid_path.append(path_i) 
            for in_path_i in invalid_path:
                path.remove(in_path_i) 
            # assign value to flag_1 for valid path(satisfies fsm)
            if path == []:
                d_flag_1 = 0
            else:
                d_flag_1 = len(path)  # number of CE candidates
                
        # return directly if not detected.
        if d_flag_1 == 0:
            continue
        
        
        # Constraints selection
        if constraint_selection == False:
            ce_result[0,ce] = d_flag_1
        else:
            # if anyone of the candidate satisfied, then the detection result +1.
            for path_i in path:
                if check_constraints(event_data, path_i, timing_list):
                    d_flag_2 = d_flag_2 + 1
            ce_result[0,ce] = d_flag_2
        
    return ce_result



def data_generator(num_event_type, num_attribute, 
                   ce_fsm_list, ce_time_list,
                   event_num, window_size ):
    """
    High-level function for generating event data used for training NeuralFSM.
    Consists of 4 steps: 
    checking consistency, generate events, split data and generate label, output array
    
    Inputs:
    - num_event_type: total num of unique events
    - num_attribute:  total num of unique attributs
    - ce_fsm_list:    fsm patterns of CEs
    - ce_time_list:   timing constraints of CEs
    - event_num:      number of total generated data
    - window_size:    window size for splitting data
    
    Outputs:
    - data_feature: numpy array of training X
    - data_label:   numpy array of training Y
    
    """
    
    # ################### Checking consistency of inputs ###################
    if set( [item for sublist in ce_fsm_list for item in sublist] ).issubset(set(range(1, num_event_type+1)) ):
        pass
    else:
        raise ValueError('Events in CE def: ', set( [item for sublist in ce_fsm_list for item in sublist] ), 
                         ' is beyond the space of all events: ', set(range(1, num_event_type+1)) )
    print('Inputs consistency checked. ')
    
    # ################### Generating event data ###################
    event_list = []
    init_time = 0
    last_time = init_time
    # creating event data
    for i in range(event_num):
        new_event, event_time = event_generator(num_event_type, num_attribute, last_time)
        event_list.append(new_event)
        last_time = event_time
    event_data = np.array(event_list)
    print('Event data generated with size: ', event_data.shape)

    # ################### Generating event data ###################
    data_feature = []
    data_class = []
    print('Total num of event_data: %d' %(event_data.shape[0]- window_size +1) )
    start_time = time.time()
    
    for i in range(event_data.shape[0]-window_size+1):
        # Need to: copy a seperate copy of the array instead of touching original
        event_data_i = copy.deepcopy( event_data[i:i+window_size,:]  )
        # use time difference (with initial events) for timestamp (feature 5)
        event_data_i[:, num_event_type] = event_data_i[:, num_event_type]-event_data_i[0, num_event_type]
        data_feature.append(event_data_i)

        data_class.append(detection_machine(event_data_i, ce_fsm_list, ce_time_list, window_size))
    #     del event_data_i
        remaining_time = (time.time()-start_time)*( (event_data.shape[0] -(i+1))/(i+1) )
        # visualize processing process
        # IOPub message rate exceeded.?
    #     print_str = '\rProcessing data: %d - %d%% \t Remaining time: %d sec) '\
    #           % (i+1, (100*(i+1))//event_data.shape[0], remaining_time)
    #     sys.stdout.write(print_str)
    #     sys.stdout.flush()
        print('Processing data: %d - %d%% \t Remaining time: %d sec) '\
              % (i+1, (100*(i+1))//event_data.shape[0], remaining_time), 
              end='\r' if i<event_data.shape[0]-1 else '')
    print('Data labeling complete!') 
    
    # ################### Generating event data ###################
    data_feature = np.array(data_feature)
    data_label = np.squeeze(np.array(data_class)  )
    print('The generated data with size:  ')
    print('Data_feature: ', data_feature.shape)
    print('Data_label: ', data_label.shape)
    
    return data_feature, data_label


def mnist_data_generator(num_event_type, num_attribute, 
                           ce_fsm_list, ce_time_list,
                           event_num, window_size, 
                           mnist_image_data, mnist_image_label):
    """
    High-level function for generating event data used for training NeuralFSM.
    Consists of 4 steps: 
    checking consistency, generate events, split data and generate label, output array
    
    Inputs:
    - num_event_type: total num of unique events
    - num_attribute:  total num of unique attributs
    - ce_fsm_list:    fsm patterns of CEs
    - ce_time_list:   timing constraints of CEs
    - event_num:      number of total generated data
    - window_size:    window size for splitting data
    
    Outputs:
    - data_feature: numpy array of training X
    - data_label:   numpy array of training Y
    
    """
    
    data_feature, data_label = data_generator(num_event_type, num_attribute, 
                                               ce_fsm_list, ce_time_list,
                                               event_num, window_size )
    data_event_label= np.argmax(data_feature[:,:, 0:num_event_type], axis=2)
    mnist_data_event = np.zeros([data_event_label.shape[0],data_event_label.shape[1], 28, 28], dtype='f')
    
    
    start_time = time.time()
    for i in range(data_event_label.shape[0]):
        for j in range(data_event_label.shape[1]):
            # select mnist image
            img_label = data_event_label[i,j]
            rand_img = random_select_row( mnist_image_data[mnist_image_label==img_label,] )[0]
            mnist_data_event[i,j,:,:] = rand_img
        
        remaining_time = (time.time()-start_time)*( (data_event_label.shape[0] -(i+1))/(i+1) )
        print('Generating data: %d - %d%% \t Remaining time: %d sec) '\
              % (i+1, (100*(i+1))//data_event_label.shape[0], remaining_time), 
              end='\r' if i<data_event_label.shape[0]-1 else '')
    print('\n MNIST events data generated!') 
    
    
#     def rand_img(label):
#         rand_img = random_select_row( mnist_image_data[mnist_image_label==label,] )[0]
#         return rand_img

#     my_result = []
#     pool = Pool()
#     for i, r_i in enumerate(pool.imap(rand_img, data_event_label.reshape(-1,1)), 1): # our case order sensitive!
#         # imap and map both provide result in order, but map is slower. (break the list into chunks)
#         # for i, r_i in enumerate(pool.imap_unordered(test1_image, data_event_label.reshape(-1,1)), 1):   # order insensitive
#         sys.stderr.write('\rdone {0:%}'.format(i/np.prod(data_event_label.shape)))
#         my_result.append(r_i)
#     pool.close()

#     mnist_data_event = np.array(my_result).reshape(data_event_label.shape[0], data_event_label.shape[1],28,28)
#     print('\n MNIST events data generated!') 
     
    return mnist_data_event, data_feature, data_label

    
    
    
def random_select_row(A):
    idx = np.arange(A.shape[0])
    mask = np.zeros_like(idx, dtype=bool)

    selected = np.random.choice(idx, 1, replace=False)
    mask[selected] = True

    B = A[mask]
    # C = A[~mask]
    return B



def data_class_distribution(data_label, y_lim = 5000):
    """
    Plot the class distribution of data for each CE
    Input:
    - data_label: n x num_ce array
    - y_lim: range of y axis
    """
    
    if data_label.ndim == 1:
        data_label_i = data_label[:,]
        bin_count = np.bincount(data_label_i.astype(int))
        all_value = np.arange(len(bin_count) )
        
        fig = plt.figure(figsize=(8,2))
        plt.title("CE # 1.")
        plt.xlabel('Num of CE')
        plt.ylabel('Count')
        plt.ylim(0, y_lim)
        plt.bar(all_value, bin_count,  align='center', alpha=0.5)
        plt.show()
        valid_sample_r = sum(data_label_i != 0)/data_label_i.shape
        print('%.3f valid samples in generated data.'%valid_sample_r)
        
    else:
        for i in range(data_label.shape[1]):
            data_label_i = data_label[:,i]
            bin_count = np.bincount(data_label_i.astype(int))
            all_value = np.arange(len(bin_count) )

            fig = plt.figure(figsize=(8,2))
            plt.title("CE # %d"%i)
            plt.xlabel('Num of CE')
            plt.ylabel('Count')
            plt.ylim(0, y_lim)
            plt.bar(all_value, bin_count,  align='center', alpha=0.5)
            plt.show()
            valid_sample_r = sum(data_label_i != 0)/data_label_i.shape
            print('%.3f valid samples in generated data.'%valid_sample_r)
        
        
        
def label_binarize(data_label):
    """
    binarize label to perform multiple-label classification task
    """
    return (data_label>0).astype(int)


def show_img(img):
    """
    displaying random image
    input: img should be a 2-dim numpy array
    """
    from matplotlib import pyplot as plt
    plt.imshow(img, interpolation='nearest')
    plt.show()

    
    
    
def evaluate_model(model_load, Test_data, Test_label):
    """
    evaluate the trained model: accuracy, confusion matrix
    input:
    model_load: trained model
    Test_data: testing data
    Test_label: one-hot encoded label data
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score


    pred_data = Test_data
    pred_label = Test_label

    y_pred = np.argmax(model_load.predict(pred_data), axis=1)
    y_true = np.argmax(pred_label, axis=1)
    print('Accuracy on testing data:',sum(y_pred==y_true)/y_true.shape[0])
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None)*100)*0.01
    print('the mean-f1 score: {:.2f}'.format(np.mean(class_wise_f1)))
    
    
    

def generate_ce_rule():
    """
    defining rules for CEs automatically
    """
    ##############################################################################
    # num_ce = 3

    # ce_fsm_list = []
    # ce_time_list = []

    # for i in range(num_ce):
    #     # FSM constraints
    #     event_list = []
    #     ce_fsm_list.append(event_list)

    #     # Timing constraints

    #     # timing matrix: 2 x (num_events-1). 
    #     # First row stores the largest interval(by default INF)
    #     # Second row stores the smallest interval (by default 0)
    #     time_list = []
    #     ce_time_list.append(time_list)

    #     # features constraints: omitted ---- by default, CE has events with same "feature"
    ##############################################################################
    # ce_fsm_list = [ [1, 2, 3],  [4, 3],  [3, 2, 4] ]
    # ce_time_list = [ np.array([ [INF, INF], [0, 0]]),  
    #                 np.array([ [INF ], [0 ]]),  
    #                 np.array([ [INF, INF], [0, 0]]) ]


    # ce_fsm_list = [ [1, 2, 3],  [4, 3],  [3, 2, 4] ]
    # ce_time_list = [ np.array([ [3, INF], [0, 1]]),  
    #                 np.array([ [10 ], [0 ]]),  
    #                 np.array([ [5, 5], [1, 2]]) ]
    # for CE: ABC, DC, CBD