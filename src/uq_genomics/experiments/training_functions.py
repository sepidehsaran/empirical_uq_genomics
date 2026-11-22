import configparser
from pathlib import Path
import pickle
import tensorflow as tf
import tf_keras
import pandas as pd
import time
from datetime import datetime
from models import *
from utils import *
from eval_functions import *


class TrainingFunctions:

    def __init__(self, config):
        self.ls_temp_1 = []
        self.ls_temp_2 = []
        self.learning_rate = float(config.get('model_selection', 'learning_rate'))
        self.optimizer = tf_keras.optimizers.Adam(learning_rate=self.learning_rate)
        model_type = config.get('model_selection', 'model_type') 
        classification_task = config.get('input_data_selection', 'classification_task')
        if model_type == 'bnn': 
            if classification_task == 'gex':
                self.bnn_model = build_bayesian_gex_mlp(config)
            else:
                self.bnn_model = build_bayesian_deepripe(config)
        else:
            self.bnn_model = None

    def run_train_det(self, config, run):
        """
        Function to train deterministic model
        :param config:
        :param run:
        :return:
        """
        verbose = config.getboolean('general', 'verbose')
        if verbose:
            print('in function run_train_det')
        # read input dataset and params
        x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config)
        rand_seed = int(config.get('general', 'random_seed'))
        np.random.seed(rand_seed)
        tf.random.set_seed(rand_seed)
        version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
        multi_label = config.getboolean('input_data_selection', 'multi_label')
        batch_size = int(config.get('model_selection', 'batch_size'))
        epochs = int(config.get('model_selection', 'epochs'))
        patien = int(config.get('model_selection', 'patience'))
        early_stopping = config.getboolean('model_selection', 'early_stopping')
        path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
        path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
        classification_task = config.get('input_data_selection', 'classification_task')
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patien, verbose=0)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path_models + 'det_model_' + version_rand + '.keras',
                                                        monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        Path(path_models).mkdir(parents=True, exist_ok=True)
        Path(path_results).mkdir(parents=True, exist_ok=True)

        ls_callbacks = [checkpoint]
        if early_stopping:
            ls_callbacks.append(earlystopper)
        # Note that for MNIST we used categorical_crossentropy and for RBP binary_crossentropy
        if classification_task in ['rbp', 'tf', 'virus']:
            task_loss = tf.keras.losses.binary_crossentropy
        elif classification_task in ['mnist', 'gex'] :
            task_loss = tf.keras.losses.categorical_crossentropy
        else:
            print('invalid value for setting classification_task')

        print("deterministic model training: ")
        # create model
        if classification_task in ['rbp', 'tf', 'virus', 'mnist']:
            det_model = build_deterministic_deepripe(config)
        elif classification_task == 'gex':
            det_model = build_deterministic_gex_mlp(config)
        

        det_model.compile(loss=task_loss, optimizer="adam", metrics=["accuracy"])

        hist = det_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                  validation_data=(x_valid, y_valid),
                                  callbacks=ls_callbacks)

        plot_training_acc_loss(hist, run)
        pred_test = det_model(x_test).numpy()
        if multi_label:
            acc_test = accuracy_multi_label(pred_test, y_test, apply_sigmoid=False)
        else:
            acc_test = accuracy_single_label(pred_test, y_test)
        run['acc_test'] = acc_test
        df_test_pred = pd.DataFrame(pred_test)
        df_test_pred.to_csv(path_results+'df_test_pred_det_'+version_rand+'.csv', sep='|')
        run["artifacts/df_test_pred"].track_files(path_results+'df_test_pred_det_'+version_rand+'.csv')

        return

    def run_train_ens(self, config, run):
        """
        Function to train ensemble model
        :param config:
        :param run:
        :return:
        """
        verbose = config.getboolean('general', 'verbose')
        if verbose:
            print('in function run_train_ens')
        # read input dataset and params
        x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config)
        rand_seed = int(config.get('general', 'random_seed'))
        np.random.seed(rand_seed)
        tf.random.set_seed(rand_seed)
        version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
        multi_label = config.getboolean('input_data_selection', 'multi_label')
        batch_size = int(config.get('model_selection', 'batch_size'))
        epochs = int(config.get('model_selection', 'epochs'))
        patien = int(config.get('model_selection', 'patience'))
        early_stopping = config.getboolean('model_selection', 'early_stopping')
        path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
        path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
        classification_task = config.get('input_data_selection', 'classification_task')
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patien, verbose=0)

        Path(path_models).mkdir(parents=True, exist_ok=True)
        Path(path_results).mkdir(parents=True, exist_ok=True)

        if classification_task in ['rbp', 'tf', 'virus']:
            task_loss = tf.keras.losses.binary_crossentropy
        elif classification_task in ['mnist', 'gex'] :
            task_loss = tf.keras.losses.categorical_crossentropy
        else:
            print('invalid value for setting classification_task')

        print("ensemble model training: ")
        # create model
        if classification_task in ['rbp', 'tf', 'virus', 'mnist']:
            dict_ens_models = build_ensemble_deepripe(config)
        elif classification_task == 'gex':
            dict_ens_models = build_ensemble_gex_mlp(config)
        ls_acc_test = []
        ls_pred_test = []

        
        ls_acc_test = []
        ls_pred_test = []

        for rand_init_seed in dict_ens_models.keys():
            # setting all seeds - not all might be necessary - but just to be sure 
            os.environ['PYTHONHASHSEED'] = str(rand_init_seed)
            random.seed(rand_init_seed)
            np.random.seed(rand_init_seed)
            tf.random.set_seed(rand_init_seed)
            tf.keras.utils.set_random_seed(rand_init_seed)

            # shuffle training set internally 
            train_idx = np.arange(x_train.shape[0])
            np.random.shuffle(train_idx)
            x_train = x_train[train_idx]
            y_train = y_train[train_idx]

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=path_models + 'ens_member_model_' + str(rand_init_seed) + '_' + version_rand + '.keras',
                monitor='val_loss', verbose=0, save_best_only=True, mode='min')

            ls_callbacks = [checkpoint]
            if early_stopping:
                ls_callbacks.append(earlystopper)
            ens_member_model = dict_ens_models[rand_init_seed]
            ens_member_model.compile(loss=task_loss, optimizer="adam", metrics=["accuracy"])
            hist = ens_member_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                        validation_data=(x_valid, y_valid),
                                        callbacks=ls_callbacks)
            # plot_training_acc_loss(hist, run)
            pred_test = ens_member_model(x_test).numpy()
            if multi_label:
                acc_test = accuracy_multi_label(pred_test, y_test, apply_sigmoid=False)
            else:
                acc_test = accuracy_single_label(pred_test, y_test)
            ls_acc_test.append(acc_test)
            ls_pred_test.append(pred_test)

        np_acc_test_ens = np.array(ls_acc_test).mean()
        np_pred_test_ens = np.array(ls_pred_test).mean(axis=0)
        run['acc_test'] = np_acc_test_ens
        df_test_pred = pd.DataFrame(np_pred_test_ens)
        df_test_pred.to_csv(path_results+'df_test_pred_ens_'+version_rand+'.csv', sep='|')
        run["artifacts/df_test_pred"].track_files(path_results+'df_test_pred_ens_'+version_rand+'.csv')

        return
        
    
    def run_pytorch_train_det(self, config, run):
        verbose = config.getboolean('general', 'verbose')
        if verbose: print('in function run_pytorch_train_det')
        # read input dataset and params
        x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config, run)
        rand_seed = int(config.get('general', 'random_seed'))
        np.random.seed(rand_seed)
        tf.random.set_seed(rand_seed)
        version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
        multi_label = config.getboolean('input_data_selection', 'multi_label')
        batch_size = int(config.get('model_selection', 'batch_size'))
        epochs =  int(config.get('model_selection', 'epochs'))
        patien =  int(config.get('model_selection', 'patience'))
        early_stopping = config.getboolean('model_selection', 'early_stopping')
        path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
        path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
        classification_task = config.get('input_data_selection', 'classification_task')
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patien, verbose=0)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path_models + 'det_model_'+version_rand,
                                                        monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        Path(path_models).mkdir(parents=True, exist_ok=True)
        Path(path_results).mkdir(parents=True, exist_ok=True)

        ls_callbacks = [checkpoint]
        if early_stopping:
            ls_callbacks.append(earlystopper)

        # Note that for MNIST we used categorical_crossentropy and for RBP binary_crossentropy
        if classification_task in ['rbp', 'tf', 'virus']:
            task_loss = tf.keras.losses.binary_crossentropy
        elif classification_task == 'mnist':
            task_loss = tf.keras.losses.categorical_crossentropy
        else:
            print('invalid value for setting classification_task')

        print("deterministic model training: ")
        # create model
        det_model = build_deterministic_deepripe(config)

        det_model.compile(loss=task_loss, optimizer="adam", metrics=["accuracy"])

        hist = det_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                  validation_data=(x_valid, y_valid),
                                  callbacks=ls_callbacks)

        plot_training_acc_loss(hist, run)
        pred_test = det_model(x_test).numpy()
        if multi_label:
            acc_test = accuracy_multi_label(pred_test, y_test, apply_sigmoid=False)
        else:
            acc_test = accuracy_single_label(pred_test, y_test)
        run['acc_test'] = acc_test
        if verbose: print('test accuracy is %.2f' % acc_test)

        df_test_pred = pd.DataFrame(pred_test)
        df_test_pred.to_csv(path_results+'df_test_pred_det_'+version_rand+'.csv', sep='|')
        run["artifacts/df_test_pred"].track_files(path_results+'df_test_pred_det_'+version_rand+'.csv')
        # run["artifacts/training_history"].upload(hist)

        return

    def run_train_mcd(self, config, run):
        verbose = config.getboolean('general', 'verbose')
        if verbose:
            print('in function run_train_mcd')
        # read input dataset and params
        x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config)
        rand_seed = int(config.get('general', 'random_seed'))
        np.random.seed(rand_seed)
        tf.random.set_seed(rand_seed)
        version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
        multi_label = config.getboolean('input_data_selection', 'multi_label')
        batch_size = int(config.get('model_selection', 'batch_size'))
        epochs = int(config.get('model_selection', 'epochs'))
        patien = int(config.get('model_selection', 'patience'))
        early_stopping = config.getboolean('model_selection', 'early_stopping')
        path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
        path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
        classification_task = config.get('input_data_selection', 'classification_task')
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patien, verbose=0)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path_models + 'mcd_model_' + version_rand + '.keras',
                                                        monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        Path(path_models).mkdir(parents=True, exist_ok=True)
        Path(path_results).mkdir(parents=True, exist_ok=True)

        ls_callbacks = [checkpoint]
        if early_stopping:
            ls_callbacks.append(earlystopper)

        # Note that for MNIST we used categorical_crossentropy and for RBP binary_crossentropy
        if classification_task in ['rbp', 'tf', 'virus']:
            task_loss = tf.keras.losses.binary_crossentropy
        elif classification_task in ['mnist', 'gex'] :
            task_loss = tf.keras.losses.categorical_crossentropy
        else:
            print('invalid value for setting classification_task')

        print("MCD model training: ")
        # create model
        if classification_task in ['rbp', 'tf', 'virus', 'mnist']:
            mcd_model = build_mcd_deepripe(config)
        elif classification_task == 'gex':
            mcd_model = build_mcd_gex_mlp(config)

        mcd_model.compile(loss=task_loss, optimizer="adam", metrics=["accuracy"])

        hist = mcd_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                validation_data=(x_valid, y_valid),
                                callbacks=ls_callbacks)

        plot_training_acc_loss(hist, run)
        pred_test = mcd_model(x_test).numpy()
        if multi_label:
            acc_test = accuracy_multi_label(pred_test, y_test, apply_sigmoid=False)
        else:
            acc_test = accuracy_single_label(pred_test, y_test)
        run['acc_test'] = acc_test
        df_test_pred = pd.DataFrame(pred_test)
        if verbose:
            print('test accuracy is %.2f' % acc_test)
            print('test results stored under:')
            print(path_results + 'df_test_pred_mcd_' + version_rand + '.csv')
        df_test_pred.to_csv(path_results + 'df_test_pred_mcd_' + version_rand + '.csv', sep='|')
        run["artifacts/df_test_pred"].track_files(path_results + 'df_test_pred_mcd_' + version_rand + '.csv')

        return

    # NOTE: In TensorFlow 2.16+, tf.keras (and tf.initializers, tf.losses, and tf.optimizers) refers to Keras 3.
    # TensorFlow Probability is not compatible with Keras 3 -- instead TFP is continuing to use Keras 2,
    # which is now packaged as tf-keras and tf-keras-nightly and is imported as tf_keras.

    @tf.function
    def elbo_loss_single_label(self, labels, logits):
        self.ls_temp_1.append(labels)
        self.ls_temp_2.append(logits)
        # Note: output of the softmax_cross_entropy has one dimension less than sigmoid_cross_entropy
        loss_en = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)
        loss_kl = tf_keras.losses.KLD(labels, logits)
        loss = tf.reduce_mean(tf.add(loss_en, loss_kl))
        return loss

    @tf.function
    def train_step_single_label(self, x_train, labels):
        with tf.GradientTape() as tape:
            logits = self.bnn_model(x_train)
            loss = self.elbo_loss_single_label(labels, logits)
        gradients = tape.gradient(loss, self.bnn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.bnn_model.trainable_variables))
        return loss

    @tf.function
    def elbo_loss_multi_label(self, labels, logits):
        self.ls_temp_1.append(labels)
        self.ls_temp_2.append(logits)
        # Note: output of the softmax_cross_entropy has one dimension less than sigmoid_cross_entropy
        loss_en = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)
        # required when using sigmoid
        loss_en = tf.reduce_sum(loss_en, axis=1)
        loss_kl = tf_keras.losses.KLD(labels, logits)
        loss = tf.reduce_mean(tf.add(loss_en, loss_kl))
        return loss

    @tf.function
    def train_step_multi_label(self, x_train, labels):
        with tf.GradientTape() as tape:
            logits = self.bnn_model(x_train)
            loss = self.elbo_loss_multi_label(labels, logits)
        gradients = tape.gradient(loss, self.bnn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.bnn_model.trainable_variables))
        return loss

    def run_train_bnn(self, config, run):
        """
        Function to train Bayesian Neural Network model
        :param config:
        :param run:
        :return:
        """
        verbose = config.getboolean('general', 'verbose')
        if verbose:
            print('in function run_train_bnn')

        # read input dataset and params
        x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config)
        rand_seed = int(config.get('general', 'random_seed'))
        np.random.seed(rand_seed)
        tf.random.set_seed(rand_seed)
        version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
        multi_label = config.getboolean('input_data_selection', 'multi_label')
        batch_size = int(config.get('model_selection', 'batch_size'))
        epochs = int(config.get('model_selection', 'epochs'))
        patien = int(config.get('model_selection', 'patience'))
        early_stopping = config.getboolean('model_selection', 'early_stopping')
        path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
        path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
        Path(path_models).mkdir(parents=True, exist_ok=True)
        Path(path_results).mkdir(parents=True, exist_ok=True)
        print(path_models)
        print(20*'----**')
        times = []
        accs = []
        val_accs = []
        losses = []
        val_losses = []
        prev_val_loss = 10000
        best_val_loss = 10000
        count_for_patience = 0
        i = 0
        continue_training = True
        best_model = self.bnn_model
        while continue_training and i < epochs:
            if verbose:
                print("epoch {0} at {1}".format(i, datetime.now()))
            tic = time.time()
            if multi_label:
                loss = self.train_step_multi_label(x_train, y_train)
            else:
                loss = self.train_step_single_label(x_train, y_train)
            preds = self.bnn_model(x_train)
            if multi_label:
                acc = accuracy_multi_label(preds, y_train, apply_sigmoid=True)
            else:
                acc = accuracy_single_label(preds, y_train)
            accs.append(acc)
            losses.append(loss)

            val_preds = self.bnn_model(x_valid)
            if multi_label:
                val_loss = self.elbo_loss_multi_label(y_valid, val_preds)
            else:
                val_loss = self.elbo_loss_single_label(y_valid, val_preds)
            if multi_label:
                val_acc = accuracy_multi_label(preds, y_train, apply_sigmoid=True)
            else:
                val_acc = accuracy_single_label(preds, y_train)

            val_accs.append(val_acc)
            val_losses.append(val_loss)
            tac = time.time()
            train_time = tac - tic
            times.append(train_time)
            if verbose:
                print(
                    "Epoch: {}: loss = {:7.3f} , accuracy = {:7.3f}, val_loss = {:7.3f}, val_acc={:7.3f} time: {:7.3f}".format(
                        i, loss, acc, val_loss, val_acc, train_time))

            # save the best model based on lowest validation loss
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_model = self.bnn_model
            # if current validation loss is not improved compared to the one from last epoch, count for paitence
            # only reset patience count if the val loss is better than our previous bests, it stopped too early
            if val_loss >= prev_val_loss:
                count_for_patience += 1
            elif val_loss <= best_val_loss:
                count_for_patience = 0
            # if reached maximum patience, stop the training
            if count_for_patience == patien:
                continue_training = False
                if verbose:
                    print("stopped training by early stopping")

            prev_val_loss = val_loss
            i += 1

        best_model.save(path_models + 'bnn_model_' + version_rand)
        best_model.save_weights(path_models + 'bnn_model_' + version_rand + '_weights.h5')

        hist = {"ls_acc": accs, "ls_val_acc": val_accs,
                "ls_losses": losses, "ls_val_losses": val_losses}

        plot_training_acc_loss(hist, run, bnn_hist=True)
        pred_test = best_model(x_test).numpy()
        if multi_label:
            acc_test = accuracy_multi_label(pred_test, y_test, apply_sigmoid=True)
        else:
            acc_test = accuracy_single_label(pred_test, y_test)
        run['acc_test'] = acc_test
        df_test_pred = pd.DataFrame(pred_test)
        if verbose:
            print('test accuracy is %.2f' % acc_test)
            print('test results stored under:')
            print(path_results + 'df_test_pred_bnn_' + version_rand + '.csv')
        df_test_pred.to_csv(path_results + 'df_test_pred_bnn_' + version_rand + '.csv', sep='|')
        run["artifacts/df_test_pred"].track_files(path_results + 'df_test_pred_bnn_' + version_rand + '.csv')
        return
