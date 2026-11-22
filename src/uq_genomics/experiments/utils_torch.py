import os
import sys
import numpy as np
from datetime import date
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import scipy
from scipy import stats
import argparse
import math
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import wilcoxon, pearsonr, spearmanr
import h5py
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_curve, accuracy_score, average_precision_score, roc_curve, auc, roc_auc_score
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import matplotlib
from scipy.stats import wilcoxon, pearsonr, spearmanr
import statsmodels.stats.multitest as smt
from datetime import datetime
import pickle
import warnings
import neptune as neptune
# import neptune.integrations.sklearn as npt_utils
from math import sqrt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc, classification_report

import matplotlib.pyplot as plt


####### config and run setting
def assign_run_variables(config, run):
    # read relevant config and set run params
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('assign_run_variables'))
    run['version'] = int(config.get('general', 'version'))
    run['run_date'] = str(date.today()) + "_" + config.get('general', 'version')
    run['run_mode'] = config.get('general', 'run_mode')
    run['random_seed']  = int(config.get('general', 'random_seed'))
    run['ls_random_seeds'] = [int(s.strip()) for s in config.get('general',
                                                                 'ls_random_seeds').split(',')]
    run['model_type'] = config.get('model_selection', 'model_type')
    run['n_classes'] = int(config.get('input_data_selection', 'n_classes'))
    run['seq_length'] = int(config.get('input_data_selection', 'seq_length'))

    run['balanced_dataset'] = config.getboolean('input_data_selection', 'balanced_dataset')
    run['multi_label'] = config.getboolean('input_data_selection', 'multi_label')
    run['noise'] = config.get('input_data_selection', 'noise')
#     run['rbp_dataset'] = config.get('input_data_selection', 'rbp_dataset')
    run['data_type_species'] = config.get('input_data_selection', 'data_type_species')

    run['dropout_rate_1'] = float(config.get('model_selection', 'dropout_rate_1'))
    run['dropout_rate_2'] = float(config.get('model_selection', 'dropout_rate_2'))
    run['dropout_rate_3'] = float(config.get('model_selection', 'dropout_rate_3'))
    run['nodes_dense_1'] = int(config.get('model_selection', 'nodes_dense_1'))
    run['mcd_dropout_at_ineference_1'] = config.getboolean('model_selection', 'mcd_dropout_at_ineference_1')
    run['mcd_dropout_at_ineference_2'] = config.getboolean('model_selection', 'mcd_dropout_at_ineference_2')
    run['mcd_dropout_at_ineference_3'] = config.getboolean('model_selection', 'mcd_dropout_at_ineference_3')
    run['learning_rate'] = float(config.get('model_selection', 'learning_rate'))
    run['epochs'] = int(config.get('model_selection', 'epochs'))
    run['patience'] = int(config.get('model_selection', 'patience'))
    run['batch_size'] = int(config.get('model_selection', 'batch_size'))
    run['early_stopping'] = config.getboolean('model_selection', 'early_stopping')
    run['xai_method'] = config.get('xai', 'xai_method')
    run['xai_baseline'] = config.get('xai', 'baseline')
    run['application_mode'] = config.getboolean('input_data_selection', 'application_mode')
    return



def read_splitted_dataset(config, run, print_plot_imbalance=True):
    verbose = config.getboolean('general', 'verbose')
    path_datasets = config.get('general', 'path_datasets')
    balanced_dataset = config.getboolean('input_data_selection', 'balanced_dataset')
    multi_label = config.getboolean('input_data_selection', 'multi_label')
    noise = config.get('input_data_selection', 'noise')
#     rbp_dataset = config.get('input_data_selection', 'rbp_dataset')
    data_type_species  = config.get('input_data_selection', 'data_type_species')
    classification_task = config.get('input_data_selection', 'classification_task')
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    application_mode = config.getboolean('input_data_selection', 'application_mode')
    rand_seed = int(config.get('general', 'random_seed'))
    path_to_dataset = path_datasets
    
    if classification_task == 'rbp':
        # note that I have changed rbp_dataset from real/simulated to real_human, sim_human
        dataset_setting_name = classification_task +'/'+ data_type_species + '/'
        # (im) / balanced
        if balanced_dataset:
            dataset_setting_name = dataset_setting_name + 'balanced/'
        else: 
            dataset_setting_name = dataset_setting_name + 'imbalanced/'
        # multi/single label
        if multi_label:
            dataset_setting_name = dataset_setting_name + 'multi_label/'
        else:
            dataset_setting_name = dataset_setting_name + 'single_label/'
            
        dataset_setting_name = dataset_setting_name + str(n_classes)+'class_split'+str(rand_seed)
        path_to_dataset = path_datasets + dataset_setting_name + '.pickle'
        
        print('dataset setting name is:')
        print(dataset_setting_name)
        print('path to data is: ')
        print(path_to_dataset)
            
#         if application_mode:
#             dataset_setting_name = config.get('input_data_selection', 'application_setting_name')
#         # simulated 6-class
#         elif rbp_dataset == 'simulated' and n_classes == 6 and balanced_dataset and not multi_label and noise == 'none':
#             dataset_setting_name = 'simulated_balanced_singlelabel_6class_nonoise'    

#         elif rbp_dataset == 'simulated' and n_classes == 6 and ~balanced_dataset and not multi_label and noise == 'none':
#             dataset_setting_name = 'simulated_imbalanced_singlelabel_6class_nonoise'

#         elif rbp_dataset == 'simulated' and n_classes == 6 and balanced_dataset and not multi_label and noise == 'type1':
#             dataset_setting_name = 'simulated_balanced_singlelabel_6class_noisetype1'    

#         elif rbp_dataset == 'simulated' and n_classes == 6 and balanced_dataset and not multi_label and noise == 'type2':
#             dataset_setting_name = 'simulated_balanced_singlelabel_6class_noisetype2'  

#         elif rbp_dataset == 'simulated' and n_classes == 6 and balanced_dataset and not multi_label and noise == 'nonoise_2rep':
#             dataset_setting_name = 'simulated_balanced_singlelabel_6class_nonoise_2rep'    

#         elif rbp_dataset == 'simulated' and n_classes == 6 and ~balanced_dataset and not multi_label and noise == 'nonoise_2rep':
#             dataset_setting_name = 'simulated_imbalanced_singlelabel_6class_nonoise_2rep'   

#         # simulated 3-class
#         elif rbp_dataset == 'simulated' and n_classes == 3 and balanced_dataset and not multi_label and noise == 'none':
#             dataset_setting_name = 'simulated_balanced_singlelabel_3class_nonoise'    

#         elif rbp_dataset == 'simulated' and n_classes == 3 and ~balanced_dataset and not multi_label and noise == 'none':
#             dataset_setting_name = 'simulated_imbalanced_singlelabel_3class_nonoise'

#         elif rbp_dataset == 'simulated' and n_classes == 3 and balanced_dataset and multi_label and noise == 'none':
#             dataset_setting_name = 'simulated_balanced_multilabel_3class_nonoise'

#         elif rbp_dataset == 'simulated' and n_classes == 3 and balanced_dataset and not multi_label and noise == 'type1':
#             dataset_setting_name = 'simulated_balanced_singlelabel_3class_noisetype1'    

#         elif rbp_dataset == 'simulated' and n_classes == 3 and balanced_dataset and multi_label and noise == 'nonoise_specialcase':
#             dataset_setting_name = 'simulated_balanced_multilabel_3class_nonoise_specialcase'

#         # real 3-class       
#         elif rbp_dataset == 'real' and n_classes == 3 and balanced_dataset and not multi_label and noise == 'none':
#             dataset_setting_name = 'real_balanced_singlelabel_3class_nonoise'

#     #     elif rbp_dataset == 'real' and n_classes == 3 and not balanced_dataset and not multi_label and noise == 'none':
#     #         dataset_setting_name = 'real_imbalanced_singlelabel_3class_nonoise'

#         elif rbp_dataset == 'real' and n_classes == 3 and balanced_dataset and multi_label and noise == 'none':
#             dataset_setting_name = 'real_balanced_multilabel_3class_nonoise'

#         elif rbp_dataset == 'real' and n_classes == 3 and not balanced_dataset and multi_label and noise == 'none':
#             dataset_setting_name = 'real_imbalanced_multilabel_3class_nonoise'

#         # other 
#         elif rbp_dataset == 'simulated' and noise == 'nomotif_testset':
#             dataset_setting_name = 'simulated_testset_nomotif'

#         else:
#             print('dataset input selection is invalid \n.'
#                   'for now switching to default input dataset setting: \n'
#                   'simulated, balanced, single label data with 6 classes and no noise')
#
#             #str_path = path_datasets + config.get('data_files', 'simulated_balanced_singlelabel_6class_nonoise')
#             dataset_setting_name = 'simulated_balanced_singlelabel_6class_nonoise'
            
#     elif classification_task == 'tf':
#         # simulated 6-class
#         if  balanced_dataset and not multi_label and noise == 'none':
#             dataset_setting_name = 'tf_simulated_balanced_singlelabel_6class_nonoise'    

#         elif ~balanced_dataset and not multi_label and noise == 'none':
#             dataset_setting_name = 'tf_simulated_imbalanced_singlelabel_6class_nonoise'
            
#
    
#     elif classification_task == 'virus':
#         with open(str_path, 'rb') as handle:
        
#         str_path = path_datasets + config.get('data_files', 'real_balanced_singlelabel_3class_nonoise')
#         with open(str_path, 'rb') as handle:
#             dict_ds = pickle.load(handle)

#         # handle dataset inconsistencies wrt caps 
#         ls_keys = [i for i in dict_ds.keys()]
#         for k in ls_keys:
#             dict_ds[k.lower()] = dict_ds[k]

#         y_test[:,0] = 1
        
#     else:
#         print('dataset input selection is invalid \n.'
#               'for now switching to default input rbp dataset setting: \n'
#               'rbp, simulated, balanced, single label data with 6 classes and no noise')
#
#         #str_path = path_datasets + config.get('data_files', 'simulated_balanced_singlelabel_6class_nonoise')
#         dataset_setting_name = 'simulated_balanced_singlelabel_6class_nonoise'

#     if classification_task != 'virus': 
#         str_path = path_datasets + config.get('data_files', dataset_setting_name)

#         with open(str_path, 'rb') as handle:
#             dict_ds = pickle.load(handle)

#         # handle dataset inconsistencies wrt caps 
#         ls_keys = [i for i in dict_ds.keys()]
#         for k in ls_keys:
#             dict_ds[k.lower()] = dict_ds[k]

    with open(path_to_dataset, 'rb') as handle:
            dict_ds = pickle.load(handle)
            x_train = dict_ds['x_train']
            y_train = dict_ds['y_train']
            x_valid = dict_ds['x_valid']
            y_valid = dict_ds['y_valid']
            x_test = dict_ds['x_test']
            y_test = dict_ds['y_test']

    if print_plot_imbalance:
        print('dataset: ', dataset_setting_name)
        print(x_train.shape, y_train.shape)
        print(x_valid.shape, y_valid.shape)
        print(x_test.shape, y_test.shape)
        print('train (im)balance:', y_train.sum(axis=0))
        print('valid (im)balance:', y_valid.sum(axis=0))
        print('test (im)balance:', y_test.sum(axis=0))
        print('training - single labels ', len([i for i in y_train if i.sum() == 1]), 'multi labels ',
              len([i for i in y_train if i.sum() > 1]))
        print('validation - single labels ', len([i for i in y_valid if i.sum() == 1]), 'multi labels ',
              len([i for i in y_valid if i.sum() > 1]))
        print('test - single labels ', len([i for i in y_test if i.sum() == 1]), 'multi labels ',
              len([i for i in y_test if i.sum() > 1]))
        
        y_all = np.concatenate((y_train, y_valid,y_test))
#         plot_class_imbalance(config, run, y_all, 'entire_dataset')
#         plot_class_imbalance(config, run, y_test, 'only_testset')
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name




# custom sigmoid function
def sigmoid(x):
      return 1 / (1 + math.exp(-x))
# define vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def plot_training_acc_loss(hist, run, bnn_hist=False):
    # plotting accuracy
    fig1 = plt.figure()
    if bnn_hist:
        # for bnn - custom hist dictionary
        plt.plot(hist['ls_acc'])
        plt.plot(hist['ls_val_acc'])
    else:
        # for det or mcd - tf's history object
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()
    run["figures/training_accuracy_plot"].upload(neptune.types.File.as_image(fig1))

    # Plot training & validation loss values
    fig2 = plt.figure()
    if bnn_hist:
        # for bnn - custom hist dictionary
        plt.plot(hist['ls_losses'])
        plt.plot(hist['ls_val_losses'])
    else:
        # for det or mcd - tf's history object
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()
    run["figures/training_loss_plot"].upload(neptune.types.File.as_image(fig2))

    return




def plot_pr_roc(config, run, y_true, y_pred, path_results, ls_selected_classes = []):
    
    
    class_names = get_class_names(config)
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    model_type = config.get('model_selection', 'model_type')
    rand_seed = int(config.get('general', 'random_seed'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    class_names = class_names[:n_classes]
    precision = [None] * n_classes
    recall = [None] * n_classes
    average_precision = [None] * n_classes 
    fpr = [None] * n_classes
    tpr = [None] * n_classes
    roc_auc = [None] * n_classes
    pr_auc = [None] * n_classes

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])
        pr_auc [i]= auc(recall[i],precision[i])
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    #### define colors 
    colors = [(39,64,139),  (214, 39, 40), (44, 160, 44),
              (152, 223, 138), (139,34,82) , (148, 103, 189),
              (255, 127, 14), (31, 119, 180), (122,55,139),  
              (255, 152, 150),(247, 182, 210), (140, 86, 75),
              (197, 176, 213), (0,128,128),(255, 187, 120),  
              (174, 199, 232) ]

    for i in range(len(colors)):  
        r, g, b = colors[i]  
        colors[i] = (r / 255., g / 255., b / 255.)


    # modifying parameters for plot
    from math import sqrt
    golden_mean = (sqrt(5)-1.0)/2.0 #used for size=
    fig_width = 4.5 # fig width in inches
    fig_height = fig_width*golden_mean # fig height in inches

    #plot
    fig1 = plt.figure()
    if len(ls_selected_classes) == 0 :
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2, 
                     label='{0} (AUPR = {1:0.2f} , AP= {2:0.2f})' ''.format(class_names[i],
                                                                            pr_auc[i], 
                                                                            average_precision[i]))
    else: 
        print('selected classes are:',  [class_names[i] for i in ls_selected_classes])
        for i in range (len(ls_selected_classes)):
            plt.plot(recall[ls_selected_classes[i]], precision[ls_selected_classes[i]], color=colors[i], lw=2, 
                     label='{0} (AUPR = {1:0.2f} , AP= {2:0.2f})' ''.format(class_names[ls_selected_classes[i]],
                                                                            pr_auc[ls_selected_classes[i]], 
                                                                            average_precision[ls_selected_classes[i]]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tick_params(axis='x',which='both',top='off')
    plt.tick_params(axis='y',which='both',right='off')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.gca().set_aspect('equal', adjustable='box')
#     plt.legend(loc=2,ncol=1,bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.legend()
    plt.savefig(path_results+model_type+'_'+version_rand+"_pr_curve.png") 
    plt.tight_layout()
    plt.show()
    run["percision_recall_curve"].upload(neptune.types.File.as_image(fig1))



    fig2 = plt.figure()
    if len(ls_selected_classes) == 0 :
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, 
                     label='{0} (AUROC = {1:0.2f} , AP= {2:0.2f})' ''.format(class_names[i],
                                                                             roc_auc[i], 
                                                                             average_precision[i]))
    else:
        print('selected classes are:',  [class_names[i] for i in ls_selected_classes])
        for i in range (len(ls_selected_classes)):
            plt.plot(fpr[ls_selected_classes[i]], tpr[ls_selected_classes[i]], color=colors[i], lw=2, 
                     label='{0} (AUROC = {1:0.2f} , AP= {2:0.2f})' ''.format(class_names[ls_selected_classes[i]],
                                                                             roc_auc[ls_selected_classes[i]], 
                                                                             average_precision[ls_selected_classes[i]]))
            
        


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tick_params(axis='x',which='both',top='off')
    plt.tick_params(axis='y',which='both',right='off')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.gca().set_aspect('equal', adjustable='box')
#     plt.legend(loc=2,ncol=1,bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.legend()
    plt.savefig(path_results+model_type+'_'+version_rand+"_roc_curve.png") 
    plt.tight_layout() 
    plt.show()
    run["roc_recall_curve"].upload(neptune.types.File.as_image(fig2))
    return



def plot_class_imbalance(config, run, y_labels, save_fig_name = 'entire_dataset', ls_selected_classes = []):
    
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    classification_task = config.get('input_data_selection', 'classification_task')
    class_names = get_class_names(config)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    
    if len(ls_selected_classes) == 0 :
        ax.bar(class_names,y_labels.sum(axis=0))
    else: 
        y_all = y_labels.sum(axis=0)
        y_selected = [y_all[i] for i in ls_selected_classes]
        y_names = [class_names[i] for i in ls_selected_classes]
        ax.bar(y_names,y_selected)
    
    if classification_task in ['rbp' , 'virus']:
        plt.xlabel('RBP classes')
        plt.ylabel('Class size')
        ax.set_xlabel("RBP classes")
        ax.set_ylabel('Class size')
    else:
        plt.xlabel('MNISTclasses')
        plt.ylabel('Class size')
        ax.set_xlabel("MNIST classes")
        ax.set_ylabel('Class size')
        
    #fig.savefig(save_fig_name) 
    plt.show()
    run[save_fig_name+'_imbalance'].upload(neptune.types.File.as_image(fig))
    
    return


def load_model(config, run, explicit_model_type=''):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function load_model')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config, run, print_plot_imbalance=False)
    path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    model_type = config.get('model_selection', 'model_type')
    # overwrite if model type is specified - used for comparison functions where several model types are loaded and compared
    if explicit_model_type in ['det','mcd', 'bnn']:
        model_type = explicit_model_type
    model_path = path_models + model_type + '_model_'+version_rand
    
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model


def get_class_names(config):
    
    classification_task = config.get('input_data_selection', 'classification_task')
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    application_mode = config.getboolean('input_data_selection', 'application_mode')
    application_setting_name = config.get('input_data_selection', 'application_setting_name')
    if application_mode:
#         if application_setting_name == 'app_rbp_real_imbalanced_multilabel_27classlow_nonoise': 
        class_names = ['MBNL1', 'P53_NONO', 'PUM2', 'QKI', 'AGO3', 'FUS', 'TAF15', 'ZFP36', 'DICER1', 'EIF3A', 'EIF3D',
            'EIF3G', 'SSB', 'PAPD5', 'CPSF4', 'CPSF3', 'RTCB', 'FXR1', 'NOP58', 'NOP56', 'FBL', 'LIN28A', 'LIN28B', 'UPF1',
            'G35', 'G45', 'XPO5']
    elif classification_task in ['rbp' , 'virus']:
        class_names = ['MBNL1','PUM2', 'QKI', 'ELAVL1', 'RBFOX2', 'TARDBP']
        class_names = class_names[:n_classes]
    elif classification_task == 'tf':
        class_names = ['ctcf', 'hnf4a', 'jun', 'mef2a', 'myc', 'tal1']
        class_names = class_names[:n_classes]
    else:
        # MNIST
        class_names = [str(i) for i in [0,1,2,3,4,5,6,7,8,9]]
    
    return class_names 


def get_dict_class_names(config):
    
    class_names = get_class_names(config)
    dict_class_names = dict()
    for i in range(len(class_names)):
        dict_class_names[i] = class_names[1]
    return dict_class_names 