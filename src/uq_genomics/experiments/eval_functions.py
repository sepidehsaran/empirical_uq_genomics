import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn import metrics
from matplotlib import pyplot
from scipy import stats
from utils import *

FIGURE_WIDTH = 8.27


def accuracy_single_label(preds, labels):
    """
    calculate accuracy for single label setting
    :param preds:
    :param labels:
    :return:
    """
    return np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1))


def accuracy_multi_label(preds, labels, apply_sigmoid, th=0.5):
    """
    calculate accuracy for multi label setting

    # apply sigmoid when evaluating the bayesian NN , as the final layer has no activation
    # MCD has sigmoid in last layer - no need to apply sigmoid
    :param preds:
    :param labels:
    :param apply_sigmoid:
    :param th:
    :return:
    """
    if apply_sigmoid:
        arr_preds = np.array([sigmoid_v(np.array(x)) for x in preds])
    else:
        arr_preds = np.array([np.array(x) for x in preds])
    th_preds = np.where(arr_preds < th, 0.0, 1.0)
    acc = np.mean(th_preds == labels)
    return acc


def print_correct_per_class(n_classes, y_test, y_pred):
    """
    print the number of correct predictions per class
    :param n_classes:
    :param y_test:
    :param y_pred:
    :return:
    """
    for class_i in range(n_classes):
        ls_class_correct = [i for i in range(len(y_test)) if
                            y_test[i, class_i] == 1 and y_pred[i, :].argmax() == class_i]
        ls_class_incorrect = [i for i in range(len(y_test)) if
                              y_test[i, class_i] == 1 and y_pred[i, :].argmax() != class_i]
    print('class: ', class_i, "correct: ", len(ls_class_correct), "incorrect: ", len(ls_class_incorrect))
    return


def infer_probabilistic_model(prob_model, x_test, n_iter, mc_dropout=False, batch_size=1024):
    """
    obtain predictions from the trained probabilistic model - MC sampling
    :param prob_model: trained probabilistic model
    :param x_test: features of the test set
    :param n_iter: number of iterations for each test sample to build the predictive distribution
    :param mc_dropout: call the model in training mode so that its dropout layers are active.
        Required for MC-dropout under Keras 3, whose model.predict() ignores the training
        argument recorded on the dropout layers at build time (every pass would be identical).
        Leave False for BNN (Flipout samples in every mode) and for deterministic ensemble members.
    :return: iter_predictions: array of predictive distributions
    """
    prob_predictions = []
    for i in range(n_iter):
        if mc_dropout:
            parts = [np.asarray(prob_model(x_test[j:j + batch_size], training=True))
                     for j in range(0, len(x_test), batch_size)]
            temp_pred = np.concatenate(parts)
        else:
            temp_pred = prob_model.predict(x_test)
        prob_predictions.append(temp_pred)
    return np.array(prob_predictions)


def stat_probabilistic_model(prob_predictions):
    """
    obtain mean and standard deviation of the predictive distribution
    :param prob_predictions:
    :return: dataframe of predictions with their stats
    """
    prediction_df = pd.DataFrame()
    pred_array = np.array(prob_predictions)
    prediction_df['mean'] = pred_array.mean(axis=0).reshape(-1, )
    prediction_df['std'] = pred_array.std(axis=0).reshape(-1, )
    return prediction_df


def stat_probabilistic_model_not_flattened(prob_predictions):
    """
    obtain mean and standard deviation of the predictive distribution
    :param prob_predictions:
    :return: dictionary - not flattened
    """
    prediction_dict = dict()
    pred_array = np.array(prob_predictions)
    prediction_dict['mean'] = pred_array.mean(axis=0)
    prediction_dict['std'] = pred_array.std(axis=0)
    return prediction_dict


def get_list_std_and_mean_based_on_label(dict_stat_unflattened, y_test):
    """
    get the standard deviation and mean of the predicted class for each test sample based on ground truth class
    :param dict_stat_unflattened: - output of stat_probabilistic_model_not_flattened
    :param y_test: labels of the test set
    :return: lists of standard deviation and mean of the predictive distribution
    """
    ls_stds = []
    ls_means = []
    len_test_samples = len(y_test)
    for i in range(len_test_samples):
        std_vector10 = dict_stat_unflattened["std"][i]
        mean_vector10 = dict_stat_unflattened["mean"][i]
        ls_stds.append(std_vector10[y_test[i].argmax()])
        ls_means.append(mean_vector10[y_test[i].argmax()])
    return ls_stds, ls_means


def get_list_std_and_mean_based_on_prediction(dict_stat_unflattened, y_test):
    """
    get the standard deviation and mean of the predicted class for each test sample based on predicted class
    :param dict_stat_unflattened: - output of stat_probabilistic_model_not_flattened
    :param y_test: labels of the test set
    :return: lists of standard deviation and mean of the predictive distribution
    """
    ls_stds = []
    ls_means = []
    len_test_samples = len(y_test)
    # print("total test samples ", len_test_samples)
    for i in range(len_test_samples):
        std_vector10 = dict_stat_unflattened["std"][i]
        mean_vector10 = dict_stat_unflattened["mean"][i]
        ls_stds.append(std_vector10[mean_vector10.argmax()])
        ls_means.append(mean_vector10[mean_vector10.argmax()])
    return ls_stds, ls_means


def find_uncertain_samples(iter_predictions, topn):
    """
    Find top n samples with the highest uncertainty.
    :param iter_predictions: array of predictive distributions
    :param topn: int, the desired number of top samples with the highest uncertainty
    :return: vector of stds for all samples and index of top n samples with the highest uncertainty.
    """
    max_std = []
    for idx in range(len(iter_predictions[0])):
        px = np.array([p[idx] for p in iter_predictions])
        max_std.append(px.std(axis=0)[px.mean(axis=0).argmax()])
    candid_idx = (-np.array(max_std)).argsort()[:topn]
    return max_std, candid_idx


def find_uncertain_samples_per_class(y_test, iter_predictions, class_i, topn):
    """
    Find top n samples with the highest uncertainty for a given class.
    :param y_test:
    :param iter_predictions: array of predictive distributions
    :param class_i:
    :param topn:
    :return:
    """
    max_std = []
    # take index of test samples in a given class
    ls_class_index = [i for i in range(len(y_test)) if y_test[i].argmax() == class_i]
    iter_predictions_class = np.take(iter_predictions, ls_class_index, axis=1)
    # for number of test samples in a given class
    # iter_predictions_class.shape == #iter,#test_samples_of_the_class,#classes
    for idx in range(len(iter_predictions_class[0])):
        # get all iter predictions for that test sample
        # px.shape == #test_sample,#classes
        px = np.array([p[idx] for p in iter_predictions_class])
        # get the mean overl all iterations
        # px.mean(axis=0).shape == (10,) -> argmax to get the predicted class
        # get the standard deviation of that predicted class over the iterations for this test sample
        max_std.append(px.std(axis=0)[px.mean(axis=0).argmax()])
        # sort descending and take top n
    candid_idx_class = (-np.array(max_std)).argsort()[:topn]
    candid_idx_orig = [ls_class_index[j] for j in candid_idx_class]
    return max_std, candid_idx_orig


# Finding the most certain samples by SD
def find_certain_samples(iter_predictions, topn):
    """
    Find top n samples with the highest certainty.
    :param iter_predictions: array of predictive distributions
    :param topn: int, the desired number of top samples with the highest certainty
    :return: vector of stds for all samples and index of top n samples with the highest certainty.
    """
    max_std = []
    for idx in range(len(iter_predictions[0])):
        px = np.array([p[idx] for p in iter_predictions])
        max_std.append(px.std(axis=0)[px.mean(axis=0).argmax()])
        candid_idx = (np.array(max_std)).argsort()[:topn]
    return max_std, candid_idx


def prob2labels(probs, tr):
    """
    Convert probabilities to labels based on a threshold
    :param probs:
    :param tr:
    :return:
    """
    return (probs >= tr).astype(int)


def get_list_index_correct_classified(dict_stat_unflattened, y_test):
    """
    Get the index of correctly classified samples
    :param dict_stat_unflattened:
    :param y_test:
    :return:
    """
    ls_index_correct = []
    for i in range(len(y_test)):
        if (dict_stat_unflattened["mean"][i].argmax() == y_test[i].argmax()):
            ls_index_correct.append(i)
    return ls_index_correct


def compare_stats(y_test, iter_bay_pred, iter_mcd_pred, iterations_version, path_to_save_results):
    """
    get stats and calculate kendall values for STD correlation among the most uncertain samples of each class
    :param y_test:
    :param iter_bay_pred:
    :param iter_mcd_pred:
    :param iterations_version:
    :param path_to_save_results:
    :return:
    """
    # results in the form of test_description ,tau, p values
    ls_kendall_results = []
    # other results in form of description, value
    ls_other_results = []
    dict_per_class_uncertainties = dict()
    # get standard devations and means of iterations per test sample  (all classes)
    dict_stats_bay_unflattened = stat_probabilistic_model_not_flattened(iter_bay_pred)
    dict_stats_mcd_unflattened = stat_probabilistic_model_not_flattened(iter_mcd_pred)
    # get list of standard deviations and means for the class of interest only
    # meaning: only the label class or predicted class
    ls_label_stds_bay, ls_label_means_bay = get_list_std_and_mean_based_on_label(
        dict_stats_bay_unflattened, y_test)
    ls_pred_stds_bay, ls_pred_means_bay = get_list_std_and_mean_based_on_prediction(
        dict_stats_bay_unflattened, y_test)
    ls_label_stds_mcd, ls_label_means_mcd = get_list_std_and_mean_based_on_label(
        dict_stats_mcd_unflattened, y_test)
    ls_pred_stds_mcd, ls_pred_means_mcd = get_list_std_and_mean_based_on_prediction(
        dict_stats_mcd_unflattened, y_test)
    # look into most uncertain samples
    # separate cases that are correctly classified in both models, from those that do not agree in their predictions
    # hopefully the uncertainty is high in correctly classiefied ones compared to incorrect ones

    # first those that are both correctly classified - based on the mean predicted value per test sample
    ls_correct_bay = get_list_index_correct_classified(
        dict_stats_bay_unflattened, y_test)
    ls_correct_mcd = get_list_index_correct_classified(
        dict_stats_mcd_unflattened, y_test)
    ls_correct_both = [i for i in ls_correct_bay if i in ls_correct_mcd]
    ls_other_results.append(["count total test samples", len(y_test)])
    ls_other_results.append(["count correctly classified by bayesian", len(ls_correct_bay)])
    ls_other_results.append(["count correctly classified by mc dropout", len(ls_correct_mcd)])
    ls_other_results.append(["count correctly classified by both", len(ls_correct_both)])

    # kendall test for correlation of STDs for all test samples
    tau, p_value = stats.kendalltau(ls_label_stds_bay, ls_label_stds_mcd)
    ls_kendall_results.append(["all test samples", tau, p_value])
    # kendall test for correlation of STDs for all correctly classified samples across the two models
    tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in ls_correct_both],
                                    [ls_label_stds_mcd[j] for j in ls_correct_both])
    ls_kendall_results.append(["correctly classified samples by both", tau, p_value])

    # kendall test for correlation of STDs for most uncertain samples across all classes

    # 3000 top most uncertain
    max_var_bay_all_3k, candid_idx_bay_all_3k = find_uncertain_samples(iter_bay_pred, 3000)
    max_var_mcd_all_3k, candid_idx_mcd_all_3k = find_uncertain_samples(iter_mcd_pred, 3000)
    ls_idx_common_uncertains_3k = [i for i in candid_idx_bay_all_3k if i in candid_idx_mcd_all_3k]
    tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in candid_idx_bay_all_3k],
                                    [ls_label_stds_mcd[j] for j in candid_idx_bay_all_3k])
    ls_kendall_results.append(["top 3K across all classes, based on bayesian most uncertain index", tau, p_value])
    tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in candid_idx_mcd_all_3k],
                                    [ls_label_stds_mcd[j] for j in candid_idx_mcd_all_3k])
    ls_kendall_results.append(["top 3K across all classes, based on mc dropout most uncertain index", tau, p_value])
    tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in ls_idx_common_uncertains_3k],
                                    [ls_label_stds_mcd[j] for j in ls_idx_common_uncertains_3k])
    ls_kendall_results.append(["top 3K across all classes, based on common most uncertain index", tau, p_value])

    # 1000 top most uncertain
    max_var_bay_all_1k, candid_idx_bay_all_1k = find_uncertain_samples(iter_bay_pred, 1000)
    max_var_mcd_all_1k, candid_idx_mcd_all_1k = find_uncertain_samples(iter_mcd_pred, 1000)
    ls_idx_common_uncertains_1k = [i for i in candid_idx_bay_all_1k if i in candid_idx_mcd_all_1k]
    tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in candid_idx_bay_all_1k],
                                    [ls_label_stds_mcd[j] for j in candid_idx_bay_all_1k])
    ls_kendall_results.append(["top 1K across all classes, based on bayesian most uncertain index", tau, p_value])
    tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in candid_idx_mcd_all_1k],
                                    [ls_label_stds_mcd[j] for j in candid_idx_mcd_all_1k])
    ls_kendall_results.append(["top 1K across all classes, based on mc dropout most uncertain index", tau, p_value])
    tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in ls_idx_common_uncertains_1k],
                                    [ls_label_stds_mcd[j] for j in ls_idx_common_uncertains_1k])
    ls_kendall_results.append(["top 1K across all classes, based on common most uncertain index", tau, p_value])

    # look at the most uncertain samples per class and get kendall tau between the STDs of the most uncertain samples
    num_classes = iter_bay_pred.shape[-1]
    for i in range(num_classes):
        max_var_bay_class_100, candid_idx_bay_class_100 = find_uncertain_samples_per_class(iter_bay_pred, i, 100)
        max_var_mcd_class_100, candid_idx_mcd_class_100 = find_uncertain_samples_per_class(iter_mcd_pred, i, 100)
        ls_idx_common_uncertains_100 = [i for i in candid_idx_bay_class_100 if i in candid_idx_mcd_class_100]
        dict_per_class_uncertainties["class_" + str(i)] = {"uncertain_bay_100": candid_idx_bay_class_100,
                                                           "uncertain_mcd_100": candid_idx_mcd_class_100,
                                                           "uncertain_common_100": ls_idx_common_uncertains_100}
        ls_other_results.append(
            ["class {0}: number of common uncertain samples out of 100".format(i), len(ls_idx_common_uncertains_100)])
        tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in candid_idx_bay_class_100],
                                        [ls_label_stds_mcd[j] for j in candid_idx_bay_class_100])
        ls_kendall_results.append(
            ["top 100 uncertain for class {0} based on bayesian most uncertain index".format(i), tau, p_value])
        tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in candid_idx_mcd_class_100],
                                        [ls_label_stds_mcd[j] for j in candid_idx_mcd_class_100])
        ls_kendall_results.append(
            ["top 100 uncertain for class {0} based on mc dropout most uncertain index".format(i), tau, p_value])
        tau, p_value = stats.kendalltau([ls_label_stds_bay[j] for j in ls_idx_common_uncertains_100],
                                        [ls_label_stds_mcd[j] for j in ls_idx_common_uncertains_100])
        ls_kendall_results.append(
            ["top 100 uncertain for class {0} based on common most uncertain index".format(i), tau, p_value])

    df_other_results = pd.DataFrame(ls_other_results, columns=["description", "value"])
    df_kendall_results = pd.DataFrame(ls_kendall_results, columns=["description", "tau", "p_value"])
    df_other_results.to_csv(path_to_save_results + "other_stat_results_{0}.csv".format(str(iterations_version)),
                            sep='|')
    df_kendall_results.to_csv(path_to_save_results + "kendall_results_{0}.csv".format(str(iterations_version)), sep='|')
    with open(path_to_save_results + "dict_per_class_{0}.pickle".format(str(iterations_version)), 'wb') as handle:
        pickle.dump(dict_per_class_uncertainties, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df_kendall_results, df_other_results, dict_per_class_uncertainties


def get_kendall_stats(y_test, pred_mean_model1, pred_sd_model1, pred_mean_model2, pred_sd_model2,
                      str_version_rand, path_to_save_results, str_model1_model2):
    """
    get stats and calculate kendall values for STD correlation among the most uncertain samples of each class
    :param y_test:
    :param pred_mean_model1:
    :param pred_sd_model1:
    :param pred_mean_model2:
    :param pred_sd_model2:
    :param str_version_rand:
    :param path_to_save_results:
    :param str_model1_model2:
    :return:
    """
    # results in the form of test_description ,tau, p values
    ls_kendall_results = []
    ls_mean_model1 = []
    ls_mean_model2 = []
    ls_sd_model1 = []
    ls_sd_model2 = []
    ls_idx_correct_both = []
    # note that doesn't handle multilabel properly - just takes one of the labels  redo later - now too slow
    ls_label_idx = y_test.argmax(axis=1)
    for i in range(y_test.shape[0]):
        ls_mean_model1.append(pred_mean_model1[i, ls_label_idx[i]])
        ls_sd_model1.append(pred_sd_model1[i, ls_label_idx[i]])
        ls_mean_model2.append(pred_mean_model2[i, ls_label_idx[i]])
        ls_sd_model2.append(pred_sd_model2[i, ls_label_idx[i]])
        if pred_mean_model1[i].argmax() == ls_label_idx[i]:
            if pred_mean_model2[i].argmax() == ls_label_idx[i]:
                ls_idx_correct_both.append(i)
    print('number of correctly classified by both is ', str(len(ls_idx_correct_both)),
          ' out of ', str(y_test.shape[0]))
    # kendall test for correlation of STDs for all test samples
    tau, p_value = stats.kendalltau(ls_sd_model1, ls_sd_model2)
    ls_kendall_results.append(["all test samples", tau, p_value])

    # kendall test for correlation of STDs for all correctly classified samples across the two models
    tau, p_value = stats.kendalltau([ls_sd_model1[j] for j in ls_idx_correct_both],
                                    [ls_sd_model2[j] for j in ls_idx_correct_both])

    ls_kendall_results.append(["correctly classified samples by both", tau, p_value])
    print('tau for corrects by both', str_model1_model2, str(tau))
    df_kendall_results = pd.DataFrame(ls_kendall_results, columns=["description", "tau", "p_value"])

    df_kendall_results.to_csv(path_to_save_results + "kendall_results_{0}_{1}.csv".format(
        str_version_rand, str_model1_model2), sep='|')

    return df_kendall_results

def show_diff_uncertain_perclass(x_test, selected_dict_per_class):
    """
    show the top 100 most uncertain samples per class that are not common between the two models
    :param x_test:
    :param selected_dict_per_class:
    :return:
    """
    # for each class
    for i in range(3):
        print(30 * "-")
        print("top 100 most uncertain samples in class ", i)
        print("len common ones ", len(selected_dict_per_class["class_" + str(i)]["uncertain_common_100"]))
        print("showing 10 uncertain samples that are not common between the two models")
        print("left: bayesian, right: mcd")
        ls_uncommon_bay = [k for k in selected_dict_per_class["class_" + str(i)]["uncertain_bay_100"] if
                           k not in selected_dict_per_class["class_" + str(i)]["uncertain_common_100"]]
        ls_uncommon_mcd = [k for k in selected_dict_per_class["class_" + str(i)]["uncertain_mcd_100"] if
                           k not in selected_dict_per_class["class_" + str(i)]["uncertain_common_100"]]
        f, axarr = plt.subplots(10, 2)
        for j in range(10):
            axarr[j, 0].imshow(x_test[ls_uncommon_bay[j]])
            axarr[j, 1].imshow(x_test[ls_uncommon_mcd[j]])
        plt.show()

    return


def std_diff_uncertain_perclass(path_results, selected_dict_per_class, iter_pred_bay, iter_pred_mcd, comparison_name):
    ls_std_diff = []
    # for each class
    for i in range(3):
        ls_uncommon_bay = [k for k in selected_dict_per_class["class_" + str(i)]["uncertain_bay_100"] if
                           k not in selected_dict_per_class["class_" + str(i)]["uncertain_common_100"]]
        ls_uncommon_mcd = [k for k in selected_dict_per_class["class_" + str(i)]["uncertain_mcd_100"] if
                           k not in selected_dict_per_class["class_" + str(i)]["uncertain_common_100"]]
        # compare STDs of the test samples for the uncommon samples
        for ix in ls_uncommon_bay:
            ls_std_diff.append([i, "based_on_bay", iter_pred_bay[:, ix, i].std(), iter_pred_mcd[:, ix, i].std()])
        for ix in ls_uncommon_mcd:
            ls_std_diff.append([i, "based_on_mcd", iter_pred_bay[:, ix, i].std(), iter_pred_mcd[:, ix, i].std()])
    df_diff_stds = pd.DataFrame(ls_std_diff, columns=["class", "based_on", "bay_std", "mcd_std"])
    df_diff_stds.to_csv(path_results + "diff_uncommon_uncertain_per_class_" + comparison_name + ".csv")
    return df_diff_stds


def generate_iterations(config, run):
    """
    read data and the dataset setting name, load the pretrained probabilistic model, generate iterations and
    store the results - function used for deterministic, bayesian and MCD models
    :param config:
    :param run:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print('in function generate_iterations')
    # read input dataset and params
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(
        config, print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    n_iter = int(config.get('evaluation', 'n_iterations'))
    model_type = config.get('model_selection', 'model_type')
    if verbose: print("start loading model at {1}".format(dataset_setting_name, datetime.now()))
    # curr_model = tf.keras.models.load_model(path_models + model_type +'_model_' + version_rand)
    curr_model = load_model(config, run, explicit_model_type=model_type)
    if verbose: print("start iterations of inference for {0} model at {1}".format(dataset_setting_name, datetime.now()))
    # note that we do not apply the final activation layer for bnn in inference in generate iterations
    iter_pred_curr = infer_probabilistic_model(curr_model, x_test, n_iter, mc_dropout=(model_type == 'mcd'))
    if verbose: print("start storing the predictions at ", datetime.now())
    with open(path_results + "iter_preds_{0}_{1}_{2}.pickle".format(model_type, version_rand, iter_pred_curr.shape[0]),
              'wb') as handle:
        pickle.dump(iter_pred_curr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose: print("done calculating iterations at ", datetime.now())
    return


def generate_iterations_for_ens(config, run):
    """
        read data and the dataset setting name, load the pretrained probabilistic model, generate iterations and
        store the results - function used for only for ensemble models
        :param config:
        :param run:
        :return:
        """
    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print('in function generate_iterations_for_ens')
    # read input dataset and params
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config,
                                                                                                     print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)

    model_type = 'ens'

    if verbose:
        print("start loading member-models at {1}".format(dataset_setting_name, datetime.now()))
    dict_curr_model = load_ensemble_model(config, run)
    n_iter_each_member = int(
        int(config.get('evaluation', 'n_iterations')) / len(dict_curr_model.keys()))  # should be 100/10 => 10
    ls_iter_pred_ens = []
    if verbose:
        print("start iterations of inference for {0} model at {1}".format(dataset_setting_name, datetime.now()))
    for rand_init_seed in dict_curr_model.keys():
        i_iter_pred_curr = infer_probabilistic_model(dict_curr_model[rand_init_seed], x_test, n_iter_each_member)
        ls_iter_pred_ens.append(i_iter_pred_curr)
    iter_pred_curr = np.concatenate(ls_iter_pred_ens)
    if verbose:
        print('iter_pred_curr.shape for all ensemble member models is ', iter_pred_curr.shape)
        print("start storing the predictions at ", datetime.now())
    with open(path_results + "iter_preds_{0}_{1}_{2}.pickle".format(model_type, version_rand, iter_pred_curr.shape[0]),
              'wb') as handle:
        pickle.dump(iter_pred_curr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print("done calculating iterations at ", datetime.now())
    return


def plot_pr_roc_for_mean_iterations(config, run, ls_selected_class_idx=[],
                                    filter_for_uncertainty_score=False,
                                    uncertainty_score_threshold=0.0):
    """
    plot the PR and ROC curves for the predictions (i.e. mean of the predictive distribution)
    :param config:
    :param run:
    :param ls_selected_class_idx:
    :param filter_for_uncertainty_score:
    :param uncertainty_score_threshold:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print('in function run_plot_pr_roc')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(
        config,  print_plot_imbalance=False)

    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    path_models = config.get('general', 'path_models') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    model_type = config.get('model_selection', 'model_type')
    print(path_results + "iter_preds_{0}_{1}.pickle".format(model_type, version_rand))
    arr_iter_pred = pd.read_pickle(path_results + "iter_preds_{0}_{1}.pickle".format(model_type, version_rand))
    # if apply_softmax:
    mean_iter_pred = arr_iter_pred.mean(axis=0)
    sd_iter_pred = arr_iter_pred.std(axis=0)
    print(mean_iter_pred.shape)
    print(sd_iter_pred.shape)
    y_predicted_idx = mean_iter_pred.argmax(axis=1)
    print('y_predicted_idx[0]', y_predicted_idx[0])
    ls_sd_only_for_predicted_class = []
    # for each sample
    for i in range(mean_iter_pred.shape[0]):
        ls_sd_only_for_predicted_class.append(sd_iter_pred[i, y_predicted_idx[i]])
    ls_sd_only_for_predicted_class = np.array(ls_sd_only_for_predicted_class)

    if filter_for_uncertainty_score:
        print('uncertainty score is : ', uncertainty_score_threshold)
        high_quality_pred = mean_iter_pred[(ls_sd_only_for_predicted_class < uncertainty_score_threshold), :]
        respective_y_test = y_test[(ls_sd_only_for_predicted_class < uncertainty_score_threshold), :]
        print(high_quality_pred.shape, respective_y_test.shape)
        plot_pr_roc(config, run, respective_y_test, high_quality_pred, path_results,
                    ls_selected_classes=ls_selected_class_idx)
    else:
        plot_pr_roc(config, run, y_test, mean_iter_pred, path_results,
                    ls_selected_classes=ls_selected_class_idx)
    return


def entropy(imp_scores, pseudocount=0.001):
    """
    function code credit:
    https://github.com/amtseng/fourier_attribution_priors/blob/481d207499f7164b109db481f3ba989b098fc0e9/notebooks/view_profile_predictions.ipynb
    For an N x I x 4 array of actual importance scores, computes the entropy
    of each track. Returns an N-array of entropy values.

    :param imp_scores:
    :param pseudocount:
    :return:
    """
    scores = []
    # Normalize
    imp_scores_sum = np.sum(np.abs(imp_scores), axis=2)  # Make into N x I
    imp_scores_sum = imp_scores_sum + pseudocount
    imp_scores_norm = imp_scores_sum / np.sum(imp_scores_sum, axis=1, keepdims=True)

    return -np.sum(imp_scores_norm * np.log2(imp_scores_norm), axis=1)
