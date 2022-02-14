import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


"""Load architecture parameters of the neural network."""
nnParams = {'label': 'MultiNet',
                'rfdims': 12,
                'nVST': 12,
                'nMST': 128,
                'tfConvPadding': 'VALID'}
nnParams['saveDir'] = 'results/' + nnParams['label']


"""Load training parameters for the neural network."""
tParams = {'epochs': 50,
                'batch_size': 128,
                'drop_rate': 0.5,
                'resultsDir': 'results/'}

"""Build the neural net graph.

Args:
drop_rate: an input scalar between 0-1

Returns:
the network

[DEPENDENCIES]
+ tensorflow==1.12.0

During training drop_rate > 0.0. During testing drop_rate == 0. If
drop_rate == 0 the activations are separated from the layers to allow
interogation of activity before and after non-linear activation is applied."""

def custom_loss_function(y_true,y_pred):
    loss = y_true-y_pred
    # Visual (1), vestibular (0.5)
    loss = loss * [0.5,0.5,0.5,0.5]
    loss = tf.keras.backend.square(loss)
    loss = tf.keras.backend.mean(loss,axis=1)
    return loss

def nn(drop_rate):
    if drop_rate>0.0:
        activation = 'relu'
    else:
        activation = 'linear'
############################# FIRST VESTIBULAR NETWORK ########################
    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)

    input_vest = tf.keras.layers.Input(shape=(32,4,))

    vest = tf.keras.layers.Flatten()(input_vest)
    vest = tf.keras.layers.Dense(nnParams['nVST'],
                                      kernel_initializer=kernel_initializer,
                                      activation=activation,
                                      name='VST')(vest)
    if drop_rate>0.0:
        vest = tf.keras.layers.Dropout(drop_rate)(vest)
    else:
        vest = tf.keras.layers.Activation('relu')(vest)

    vest = tf.keras.layers.Dense(4,activation='sigmoid',
                                   name='binary_output')(vest)
    vest = tf.keras.models.Model(inputs=input_vest,outputs=vest)
    print(vest.summary())
############################# SECOND VESTIBULAR NETWORK ########################
    kernel_initializer_2=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)

    input_vest_2 = tf.keras.layers.Input(shape=(32,4,))

    vest_2 = tf.keras.layers.Flatten()(input_vest_2)
    vest_2 = tf.keras.layers.Dense(nnParams['nVST'],
                                      kernel_initializer=kernel_initializer_2,
                                      activation=activation,
                                      name='VST_2')(vest_2)
    if drop_rate>0.0:
        vest_2 = tf.keras.layers.Dropout(drop_rate)(vest_2)
    else:
        vest_2 = tf.keras.layers.Activation('relu')(vest_2)

    vest_2 = tf.keras.layers.Dense(4,activation='sigmoid',
                                   name='binary_2_output')(vest_2)
    vest_2 = tf.keras.models.Model(inputs=input_vest_2,outputs=vest_2)
    print(vest_2.summary())

    combined = tf.keras.layers.concatenate([vest.output,vest_2.output])
    combined = tf.keras.layers.Dense(nnParams['nMST'],
                                  kernel_initializer=kernel_initializer,
                                  activation=activation,
                                  name='MST')(combined)
    if drop_rate>0.0:
        combined = tf.keras.layers.Dropout(drop_rate)(combined)
    else:
        combined = tf.keras.layers.Activation('relu')(combined)
    combined_reg = tf.keras.layers.Dense(4,
                                         kernel_initializer=kernel_initializer,
                                         name='regression_output')(combined)
    combined_bin = tf.keras.layers.Dense(4,
                                         activation='sigmoid',
                                         name='binary_c_output')(combined)
    losses = {'regression_output': custom_loss_function,
        	  'binary_c_output': 'binary_crossentropy'}
    loss_weights = {'regression_output': 1.0, 'binary_c_output': 0.2}
    network = tf.keras.models.Model(inputs=[vest.input,vest_2.input],outputs=[combined_reg,combined_bin])

    network.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                    loss=losses,
                    loss_weights=loss_weights,
                    metrics=['accuracy'])

    return network
  
  ''' Script to train a multisensory convolutional neural network
(i.e., MultiNet) as implemented in Rideaux, Storrs, Maiello and Welchman,
Proceedings of the National Academy of Sciences, 2021

** The training image files (training_image_sequences.1-8)
must be placed into the 'dataset' folder prior to running this script. **

[DEPENDENCIES]
+ tensorflow==1.12.0
+ numpy
+ pickle
+ gzip
+ sklearn

[EXAMPLE]
To run the script, please launch your terminal, move to the <MultiNet> folder
and run the following command:
>> python train_network.py

'''

# Define 1D gaussian function
def gauss(x, x0, xsigma):
   return np.exp(-.5*((x-x0)/xsigma)**2)

# Vestibular x dim
x = np.linspace(-8, 8, 32)

# 75% train , 25% test
n_test = np.array((1600, 4))
n_train = np.array((4800, 4))
# Generate vestibular signal sigma
xsigma_train = np.random.uniform(1, 8, np.prod(n_train))
xsigma_test = np.random.uniform(1, 8, np.prod(n_test))

# Generate vestibular signal mean
vest_label_train = np.random.uniform(-4, 4, np.prod(n_train))
vest_label_test = np.random.uniform(-4, 4, np.prod(n_test))
vest_label_train = vest_label_train.flatten()
vest_label_test = vest_label_test.flatten()

# Generate vestibular signals
vest_data_train = gauss(x.reshape(-1, 1), vest_label_train, xsigma_train)
vest_data_test = gauss(x.reshape(-1, 1), vest_label_test, xsigma_test)

# Reshape vestibular signals
vest_data_train = vest_data_train.reshape(len(x), n_train[0], n_train[1])
vest_data_test = vest_data_test.reshape(len(x), n_test[0], n_test[1])
vest_data_train = np.swapaxes(vest_data_train, 0, 1)
vest_data_test = np.swapaxes(vest_data_test, 0, 1)
vest_label_train = vest_label_train.reshape(n_train[0], n_train[1])
vest_label_test = vest_label_test.reshape(n_test[0], n_test[1])

## Add noise to vestibular signal
vest_data_train += np.random.normal(0, .3, vest_data_train.shape)
vest_data_test += np.random.normal(0, .3, vest_data_test.shape)

################################### Vest 2 dataset #############################
x = np.linspace(-8, 8, 32)

# Generate vestibular signal sigma
xsigma_train = np.random.uniform(1, 8, np.prod(n_train))
xsigma_test = np.random.uniform(1, 8, np.prod(n_test))

# Generate vestibular signal mean
vest_2_label_train = np.random.uniform(-4, 4, np.prod(n_train))
vest_2_label_test = np.random.uniform(-4, 4, np.prod(n_test))
vest_2_label_train = vest_2_label_train.flatten()
vest_2_label_test = vest_2_label_test.flatten()

# Generate vestibular signals
vest_2_data_train = gauss(x.reshape(-1, 1), vest_2_label_train, xsigma_train)
vest_2_data_test = gauss(x.reshape(-1, 1), vest_2_label_test, xsigma_test)

# Reshape vestibular signals
vest_2_data_train = vest_2_data_train.reshape(len(x), n_train[0], n_train[1])
vest_2_data_test = vest_2_data_test.reshape(len(x), n_test[0], n_test[1])
vest_2_data_train = np.swapaxes(vest_2_data_train, 0, 1)
vest_2_data_test = np.swapaxes(vest_2_data_test, 0, 1)
vest_2_label_train = vest_2_label_train.reshape(n_train[0], n_train[1])
vest_2_label_test = vest_2_label_test.reshape(n_test[0], n_test[1])

## Add noise to vestibular signal
vest_2_data_train += np.random.normal(0, .3, vest_2_data_train.shape)
vest_2_data_test += np.random.normal(0, .3, vest_2_data_test.shape)
################################### Generate Labels #############################
# Define label placeholders []
label_train = np.empty([n_train[0], 8])
label_test = np.empty([n_test[0], 8])

# Average (Fusion)
label_train[:, :4] = (vest_label_train+vest_2_label_train)/2
label_test[:, :4] = (vest_label_test+vest_2_label_test)/2

# Binary causal inference 
label_train[:, 4:] = np.abs(
    vest_label_train-vest_2_label_train) < np.median(np.abs(vest_label_train-vest_2_label_train))
label_test[:, 4:] = np.abs(
    vest_label_test-vest_2_label_test) < np.median(np.abs(vest_label_test-vest_2_label_test))

print(label_test[:, 4:].shape)
# Define network
network = nn(drop_rate=0.5)

#visualize_nn(network, description=True, figsize=(10,8))

# Train network
network.fit([vest_data_train, vest_2_data_train],
            [label_train[:, :4], label_train[:, 4:]],
            epochs=50,
            batch_size=64,
            shuffle=True,
            verbose=1,
            validation_data=([vest_data_test, vest_2_data_test],
                             [label_test[:, :4], label_test[:, 4:]]))

# Save network
network.save(nnParams['saveDir'] + '.h5')

print('done.')
