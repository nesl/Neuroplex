import keras
from keras import backend as K
from keras.layers import Lambda
from keras.engine.input_layer import Input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import LeakyReLU
from keras.models import Model
from keras.layers import concatenate
from keras.layers import Reshape
import tensorflow as tf

def intergrated_model(lenetModel, neuralLogic_model, 
                      window_size, num_event_type,
                      omega_value = 0,
                      load_nl_weights = True,
                      nl_trainable = False,
                      loss = 'mse_loss',
                      diagnose = False):
    """
    generate an intergrated model using lenetmodel + neurallogic model(trained)
    input:
    Lenet
    NeuralLogic
    window_size: wind size in logic
    num_event_type: unique num of events
    
    Output: 
    intergrated_model
    """
    
    
    # Define final model
    input_layer = Input( shape=(window_size, 28, 28, 1) )
    split = Lambda( lambda x: tf.split(x, num_or_size_splits= window_size,axis=1))(input_layer)
    input_split = [Reshape((28,28,1))(i) for i in split]
    lenet_out = []
    for i in range(window_size):
        lenet_out.append( lenetModel( input_split[i] ) )
    cont_result = concatenate(lenet_out, axis=-1)
    reshape_result = Reshape(( -1, num_event_type), name = 'lenet_output' )(cont_result)
    
    ################# adding neural logic model here #############
    num_classes = neuralLogic_model.output.shape[1]
    num_hidden_lstm = 64
    _, win_len, dim = neuralLogic_model.input.shape
    
    lstm1 = LSTM(num_hidden_lstm, 
               input_shape=(win_len, dim), 
               return_sequences=False)(reshape_result)
    dense1 = Dense(num_classes, activation='linear')(lstm1)
    
#     neuralloigc_out = neuralLogic_model(reshape_result)
#     model = Model(inputs=input_layer, outputs=neuralloigc_out)
    ############ NL model finish #############
    
    new_model = Model(inputs=input_layer, outputs=dense1)
    new_model.name="Final_Model"

    # model.summary()
    print('Model input: ', new_model.input)
    print('Model output: ', new_model.output)
    
    
    ############ loading the NL weights from pre-trained model ###########
    if load_nl_weights:
        print('\n===== NL model weights loaded =====')
        for new_layer, nl_layer in zip(new_model.layers[-2:], neuralLogic_model.layers):
            new_layer.set_weights(nl_layer.get_weights())
    else:
        print('\n===== NL model weights NOT loaded! =====')


    ############ freeze the last layer ############
    if not nl_trainable:
        print('\n===== Neural Logic module freezed.=====')
        new_model.layers[-2].trainable = nl_trainable   # False
        new_model.layers[-1].trainable = nl_trainable   # False
    else:
        print('\n===== Neural Logic module NOT Freezed.=====')
    new_model.layers[-2].trainable = nl_trainable   # False
    new_model.layers[-1].trainable = nl_trainable   # False
    if diagnose:
        for i in new_model.layers:
            print(i, '\t\t\t', i.trainable)

    
    ############ define loss function and compile model ############
#     def combined_loss(input_tensor, omega_value):
#         def custom_loss(y_true, y_pred):
#             omega = omega_value
#             mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
#             input_max = K.expand_dims(K.max(input_tensor, axis = 2), axis = 2)
#             logic_loss = K.sum( K.prod(K.exp(input_tensor - input_max), axis = 2), axis = 1 )
#             new_loss = omega*logic_loss + (1-omega)*mse_loss
#             return new_loss
#         return custom_loss
    
    # loss from ICML paper
    def combined_loss(input_tensor, omega_value):
        def custom_loss(y_true, y_pred):
            omega = omega_value
            mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
            
            t_vec = input_tensor/(1-input_tensor + K.epsilon()) 
            t_prod = K.expand_dims( K.prod(1-input_tensor + K.epsilon() , axis = 2), axis = 2 )
            logic_loss = K.sum( K.sum( t_vec * t_prod, axis =2), axis = 1)

            new_loss = omega*logic_loss + (1-omega)*mse_loss
            return new_loss
        return custom_loss

    
    
    if loss == 'mse_loss':
        model_loss = "mean_squared_error"
        print('\n===== Loss Func: MSE =====')
    elif loss == 'combined_loss':
        model_loss = combined_loss(new_model.get_layer('lenet_output').output, omega_value)
        print('\n===== Loss Func: Combined Loss =====')
        print(' Omega = ', omega_value)
    else:
        print('Loss function not supported!')
    
    # Compile the model
    new_model.compile(optimizer=Adam(lr = 0.001),  # originally use SGD(lr = 0.0001)
                  loss = model_loss,
                  metrics=['MAE'])
    
    if diagnose:
        new_model.summary()

    return new_model