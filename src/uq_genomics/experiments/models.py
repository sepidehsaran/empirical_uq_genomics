import numpy as np
import os
import random
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp

dim = 4
num_units = 50
ls_conv_filters = [90, 100]
ls_conv_kernel_sizes = [7, 7]
ls_pool_sizes = [4, 10]
ls_pool_strides = [2, 5]

def build_deterministic_gex_mlp(config):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function build_deterministic_gex_mlp')
    n_classes = 4  # Number of output classes
    input_dim = 20  # Dimension of input features
    nodes_dense_1 = 10 #int(config.get('model_selection', 'nodes_dense_1'))
    nodes_dense_2 = 10 # int(config.get('model_selection', 'nodes_dense_2'))
    dropout_rate_1 = float(0.3)
    dropout_rate_2 = float(0.3)
    multi_label = False
    final_activation = 'softmax'
    
    input = tf.keras.layers.Input(shape=(input_dim,), name="input")
    dense1 = tf.keras.layers.Dense(nodes_dense_1, activation='relu', name="dense1")(input)
    drop1 = tf.keras.layers.Dropout(dropout_rate_1, name="drop1")(dense1)
    dense2 = tf.keras.layers.Dense(nodes_dense_2, activation='relu', name="dense2")(drop1)
    drop2 = tf.keras.layers.Dropout(dropout_rate_2, name="drop2")(dense2)
    output = tf.keras.layers.Dense(n_classes, activation='softmax', name="output")(drop2)
    model = tf.keras.Model(inputs=input, outputs=output)
    if verbose: print(model.summary())
    return model

def build_deterministic_deepripe(config):
    """
    Builds a deterministic deepripe model
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print('in function build_deterministic_deepripe')
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    seq_length = int(config.get('input_data_selection', 'seq_length'))
    dropout_rate_1 = float(config.get('model_selection', 'dropout_rate_1'))
    dropout_rate_2 = float(config.get('model_selection', 'dropout_rate_2'))
    dropout_rate_3 = float(config.get('model_selection', 'dropout_rate_3'))
    nodes_dense_1 = int(config.get('model_selection', 'nodes_dense_1'))
    multi_label = config.getboolean('input_data_selection', 'multi_label')
    if multi_label:
        final_activation = 'sigmoid'
    else:
        final_activation = 'softmax'

    input = tf.keras.layers.Input(shape=(seq_length, dim), name="input")
    conv1 = tf.keras.layers.Conv1D(filters=ls_conv_filters[0],
                                   kernel_size=ls_conv_kernel_sizes[0],
                                   padding='valid',
                                   activation="relu",
                                   name="conv1")(input)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=ls_pool_sizes[0],
                                         strides=ls_pool_strides[0],
                                         name="pool1")(conv1)
    drop1 = tf.keras.layers.Dropout(dropout_rate_1, name="drop1")(pool1)
    conv2 = tf.keras.layers.Conv1D(filters=ls_conv_filters[1],
                                   kernel_size=ls_conv_kernel_sizes[1],
                                   padding='valid',
                                   activation="relu",
                                   name="conv2")(drop1)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=ls_pool_sizes[1],
                                         strides=ls_pool_strides[1],
                                         name="pool2")(conv2)
    drop2 = tf.keras.layers.Dropout(dropout_rate_2, name="drop2")(pool2)
    flat = tf.keras.layers.Flatten()(drop2)
    dense1 = tf.keras.layers.Dense(nodes_dense_1, activation='relu', name="dense1")(flat)
    drop3 = tf.keras.layers.Dropout(dropout_rate_3, name="drop3")(dense1)
    output = tf.keras.layers.Dense(n_classes, activation=final_activation, name="output")(drop3)
    model = tf.keras.Model(inputs=input, outputs=output)
    if verbose:
        print(model.summary())
    return model

def build_ensemble_gex_mlp(config):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function build_ensemble_gex_mlp')
    n_classes = 4  # Number of output classes
    input_dim = 20  # Dimension of input features
    nodes_dense_1 = 10 #int(config.get('model_selection', 'nodes_dense_1'))
    nodes_dense_2 = 10 # int(config.get('model_selection', 'nodes_dense_2'))
    dropout_rate_1 = float(0.3)
    dropout_rate_2 = float(0.3)
    multi_label = False
    final_activation = 'softmax'

    
    ls_random_init_seeds = [int(s.strip()) for s in config.get('model_selection', 'ls_random_init_seeds').split(',')]


    if verbose: print('generating {0} deterministic models to create a deep ensemble'.format(len(ls_random_init_seeds)))
    dict_ens_models = dict() 
    
    for rand_init_seed in ls_random_init_seeds:
        os.environ['PYTHONHASHSEED']=str(rand_init_seed) 
        random.seed(rand_init_seed)
        np.random.seed(rand_init_seed)
        tf.random.set_seed(rand_init_seed)
        tf.keras.utils.set_random_seed(rand_init_seed)
        tf_keras.utils.set_random_seed(rand_init_seed)
    
        input = tf.keras.layers.Input(shape=(input_dim,), name="input")
        dense1 = tf.keras.layers.Dense(nodes_dense_1, activation='relu', name="dense1")(input)
        drop1 = tf.keras.layers.Dropout(dropout_rate_1, name="drop1")(dense1)
        dense2 = tf.keras.layers.Dense(nodes_dense_2, activation='relu', name="dense2")(drop1)
        drop2 = tf.keras.layers.Dropout(dropout_rate_2, name="drop2")(dense2)
        output = tf.keras.layers.Dense(n_classes, activation='softmax', name="output")(drop2)
        model = tf.keras.Model(inputs=input, outputs=output)
        dict_ens_models[rand_init_seed] = model
        
    return dict_ens_models

def build_ensemble_deepripe(config):
    """
    Builds a deep ensemble of deterministic deepripe models
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print('in function build_ensemble_deepripe')
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    seq_length = int(config.get('input_data_selection', 'seq_length'))
    dropout_rate_1 = float(config.get('model_selection', 'dropout_rate_1'))
    dropout_rate_2 = float(config.get('model_selection', 'dropout_rate_2'))
    dropout_rate_3 = float(config.get('model_selection', 'dropout_rate_3'))
    nodes_dense_1 = int(config.get('model_selection', 'nodes_dense_1'))
    multi_label = config.getboolean('input_data_selection', 'multi_label')
    if multi_label:
        final_activation = 'sigmoid'
    else:
        final_activation = 'softmax'

    ls_random_init_seeds = [int(s.strip()) for s in config.get('model_selection', 'ls_random_init_seeds').split(',')]

    if verbose:
        print('generating {0} deterministic models to create a deep ensemble'.format(len(ls_random_init_seeds)))
    dict_ens_models = dict()

    for rand_init_seed in ls_random_init_seeds:
        # setting all seeds - not all might be necessary - but just to be sure
        os.environ['PYTHONHASHSEED'] = str(rand_init_seed)
        random.seed(rand_init_seed)
        np.random.seed(rand_init_seed)
        tf.random.set_seed(rand_init_seed)
        tf.keras.utils.set_random_seed(rand_init_seed)

        input = tf.keras.layers.Input(shape=(seq_length, dim), name="input")
        conv1 = tf.keras.layers.Conv1D(filters=ls_conv_filters[0],
                                       kernel_size=ls_conv_kernel_sizes[0],
                                       padding='valid',
                                       activation="relu",
                                       name="conv1")(input)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=ls_pool_sizes[0],
                                             strides=ls_pool_strides[0],
                                             name="pool1")(conv1)
        drop1 = tf.keras.layers.Dropout(dropout_rate_1, name="drop1")(pool1)
        conv2 = tf.keras.layers.Conv1D(filters=ls_conv_filters[1],
                                       kernel_size=ls_conv_kernel_sizes[1],
                                       padding='valid',
                                       activation="relu",
                                       name="conv2")(drop1)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=ls_pool_sizes[1],
                                             strides=ls_pool_strides[1],
                                             name="pool2")(conv2)
        drop2 = tf.keras.layers.Dropout(dropout_rate_2, name="drop2")(pool2)
        flat = tf.keras.layers.Flatten()(drop2)
        dense1 = tf.keras.layers.Dense(nodes_dense_1, activation='relu', name="dense1")(flat)
        drop3 = tf.keras.layers.Dropout(dropout_rate_3, name="drop3")(dense1)
        output = tf.keras.layers.Dense(n_classes, activation=final_activation, name="output")(drop3)

        model = tf.keras.Model(inputs=input, outputs=output)
        dict_ens_models[rand_init_seed] = model

    return dict_ens_models

def build_bayesian_gex_mlp(config):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function build_bayesian_gex_mlp')
    n_classes = 4  # Number of output classes
    input_dim = 20  # Dimension of input features
    nodes_dense_1 = 10 #int(config.get('model_selection', 'nodes_dense_1'))
    nodes_dense_2 = 10 # int(config.get('model_selection', 'nodes_dense_2'))
    dropout_rate_1 = float(0.3)
    dropout_rate_2 = float(0.3)
    multi_label = False
    final_activation = 'softmax'

    model = tf_keras.Sequential([
    tf_keras.Input(shape=(input_dim,), name="input"),
    tfp.layers.DenseFlipout(nodes_dense_1, activation='relu', name="dense1"),
    tfp.layers.DenseFlipout(nodes_dense_2, activation='relu', name="dense2"),
    tfp.layers.DenseFlipout(n_classes, activation=None, name="output")
    ])
    if verbose: print(model.summary())
    return model


def build_bayesian_deepripe(config):
    """
    Builds a Bayesian deepripe model
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print('in function build_bayesian_deepripe')
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    seq_length = int(config.get('input_data_selection', 'seq_length'))

    # NOTE: In TensorFlow 2.16+, tf.keras (and tf.initializers, tf.losses, and tf.optimizers) refers to Keras 3.
    # TensorFlow Probability is not compatible with Keras 3 -- instead TFP is continuing to use Keras 2,
    # which is now packaged as tf-keras and tf-keras-nightly and is imported as tf_keras.

    model = tf_keras.Sequential([
        tf_keras.Input(shape=(seq_length, dim), name="input"),
        tfp.layers.Convolution1DFlipout(filters=ls_conv_filters[0],
                                        kernel_size=ls_conv_kernel_sizes[0],
                                        padding='valid',
                                        activation="relu",
                                        name="conv1"),
        tf_keras.layers.MaxPooling1D(pool_size=ls_pool_sizes[0],
                                     strides=ls_pool_strides[0],
                                     name="pool1"),
        tfp.layers.Convolution1DFlipout(filters=ls_conv_filters[1],
                                        kernel_size=ls_conv_kernel_sizes[1],
                                        padding='valid',
                                        activation="relu",
                                        name="conv2"),
        tf_keras.layers.MaxPooling1D(pool_size=ls_pool_sizes[1],
                                     strides=ls_pool_strides[1],
                                     name="pool2"),
        tf_keras.layers.Flatten(),
        #         tfp.python.layers.DenseFlipout(250,
        #                                        activation=None),
        tfp.layers.DenseFlipout(n_classes,
                                activation=None)

    ])

    if verbose:
        print(model.summary())
    return model

def build_mcd_gex_mlp(config):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function build_mcd_gex_mlp')
    n_classes = 4  # Number of output classes
    input_dim = 20  # Dimension of input features
    nodes_dense_1 = 10 #int(config.get('model_selection', 'nodes_dense_1'))
    nodes_dense_2 = 10 # int(config.get('model_selection', 'nodes_dense_2'))
    dropout_rate_1 = float(0.3)
    dropout_rate_2 = float(0.3)
    multi_label = False
    final_activation = 'softmax'
    mcd_dropout_at_ineference_1 = True
    mcd_dropout_at_ineference_2 = True
    input = tf.keras.layers.Input(shape=(input_dim,), name="input")
    dense1 = tf.keras.layers.Dense(nodes_dense_1, activation='relu', name="dense1")(input)
    drop1 = tf.keras.layers.Dropout(dropout_rate_1, name="drop1")(dense1, training=mcd_dropout_at_ineference_1)
    dense2 = tf.keras.layers.Dense(nodes_dense_2, activation='relu', name="dense2")(drop1)
    drop2 = tf.keras.layers.Dropout(dropout_rate_2, name="drop2")(dense2, training=mcd_dropout_at_ineference_2)
    output = tf.keras.layers.Dense(n_classes, activation='softmax', name="output")(drop2)
    model = tf.keras.Model(inputs=input, outputs=output)
    if verbose: print(model.summary())
    return model


def build_mcd_deepripe(config):
    """
    Builds a Monte Carlo Dropout deepripe model
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print('in function build_mcd_deepripe')
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    seq_length = int(config.get('input_data_selection', 'seq_length'))
    dropout_rate_1 = float(config.get('model_selection', 'dropout_rate_1'))
    dropout_rate_2 = float(config.get('model_selection', 'dropout_rate_2'))
    dropout_rate_3 = float(config.get('model_selection', 'dropout_rate_3'))
    nodes_dense_1 = int(config.get('model_selection', 'nodes_dense_1'))
    mcd_dropout_at_ineference_1 = config.getboolean('model_selection', 'mcd_dropout_at_ineference_1')
    mcd_dropout_at_ineference_2 = config.getboolean('model_selection', 'mcd_dropout_at_ineference_2')
    mcd_dropout_at_ineference_3 = config.getboolean('model_selection', 'mcd_dropout_at_ineference_3')
    multi_label = config.getboolean('input_data_selection', 'multi_label')

    if verbose and ~ mcd_dropout_at_ineference_1 and ~ mcd_dropout_at_ineference_2 and ~ mcd_dropout_at_ineference_3:
        print('MC-Dropout is selected as the model but all dropout layers are deactivated an inference,')
        print('switching to default setting with only last dropout layer activated at inference time.')
        mcd_dropout_at_ineference_3 = True

    if multi_label:
        final_activation = 'sigmoid'
    else:
        final_activation = 'softmax'

    input = tf.keras.layers.Input(shape=(seq_length, dim), name="input")
    conv1 = tf.keras.layers.Conv1D(filters=ls_conv_filters[0],
                                   kernel_size=ls_conv_kernel_sizes[0],
                                   padding='valid',
                                   activation="relu",
                                   name="conv1")(input)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=ls_pool_sizes[0],
                                         strides=ls_pool_strides[0],
                                         name="pool1")(conv1)
    drop1 = tf.keras.layers.Dropout(dropout_rate_1, name="drop1")(pool1, training=mcd_dropout_at_ineference_1)
    conv2 = tf.keras.layers.Conv1D(filters=ls_conv_filters[1],
                                   kernel_size=ls_conv_kernel_sizes[1],
                                   padding='valid',
                                   activation="relu",
                                   name="conv2")(drop1)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=ls_pool_sizes[1],
                                         strides=ls_pool_strides[1],
                                         name="pool2")(conv2)
    drop2 = tf.keras.layers.Dropout(dropout_rate_2, name="drop2")(pool2, training=mcd_dropout_at_ineference_2)
    flat = tf.keras.layers.Flatten()(drop2)
    dense1 = tf.keras.layers.Dense(nodes_dense_1, activation='relu', name="dense1")(flat)
    drop3 = tf.keras.layers.Dropout(dropout_rate_3, name="drop3")(dense1, training=mcd_dropout_at_ineference_3)
    output = tf.keras.layers.Dense(n_classes, activation=final_activation, name="output")(drop3)
    model = tf.keras.Model(inputs=input, outputs=output)
    if verbose:
        print(model.summary())
    return model

