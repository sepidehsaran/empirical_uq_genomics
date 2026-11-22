from eval_functions import *
from xai_functions import *
import neptune as neptune
import logomaker as lm

def compare_commonality_most_uncertain_samples(config, run, arr_iter_pred_1, arr_iter_pred_2):



    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function compare_commonality_most_uncertain_samples')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config,
                                                                                                     print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    model_type = config.get('model_selection', 'model_type')
    ls_classes = []
    ls_count_common = []
    top_n_uncertain = 50
    x_test_float = x_test.astype('float32')
    arr_mean_pred_1 = arr_iter_pred_1.mean(axis=0)
    arr_sd_pred_1 = arr_iter_pred_1.std(axis=0)
    arr_mean_pred_2 = arr_iter_pred_2.mean(axis=0)
    arr_sd_pred_2 = arr_iter_pred_2.std(axis=0)
    # for each class
    for class_i in range(n_classes):
        ls_max_std_1, ls_cand_id_1 = find_uncertain_samples_per_class(y_test, arr_iter_pred_1, class_i,
                                                                        top_n_uncertain)
        ls_max_std_2, ls_cand_id_2 = find_uncertain_samples_per_class(y_test, arr_iter_pred_2, class_i,
                                                                        top_n_uncertain)
        ls_common_class = [ix for ix in ls_cand_id_1 if ix in ls_cand_id_2]
        ls_uncommon_class_1 = [ix for ix in ls_cand_id_1 if ix not in ls_cand_id_2]
        ls_uncommon_class_2 = [ix for ix in ls_cand_id_2 if ix not in ls_cand_id_1]
        ls_ix_incorrect_1 = [i for i in range(len(y_test)) if
                               y_test[i, class_i] == 1 and arr_mean_pred_1[i, :].argmax() != class_i]
        ls_ix_incorrect_2 = [i for i in range(len(y_test)) if
                               y_test[i, class_i] == 1 and arr_mean_pred_2[i, :].argmax() != class_i]
        # joins
        ls_cand_id_bay_ls_ix_incorrect_1 = [i for i in ls_cand_id_1 if i in ls_ix_incorrect_1]
        ls_cand_id_mcd_ls_ix_incorrect_2 = [i for i in ls_cand_id_2 if i in ls_ix_incorrect_2]
        ls_common_class_ls_ix_incorrect_1 = [i for i in ls_common_class if i in ls_ix_incorrect_1]
        ls_common_class_ls_ix_incorrect_2 = [i for i in ls_common_class if i in ls_ix_incorrect_2]
        print(
            "first has {} incorrect samples, {} of them among the top {} most uncertains".format(
                len(ls_ix_incorrect_1), len(ls_cand_id_bay_ls_ix_incorrect_1), top_n_uncertain))
        print(
            "second has {} incorrect samples, {} of them among the top {} most uncertains".format(
                len(ls_ix_incorrect_2), len(ls_cand_id_mcd_ls_ix_incorrect_2), top_n_uncertain))

        print("len common ones ", len(ls_common_class), "len uncommon ones", len(ls_uncommon_class_1), "for class",
              class_i)
        print("len common that are incorrect by first is {}, and by second is {}".format(
            len(ls_common_class_ls_ix_incorrect_1),
            len(ls_common_class_ls_ix_incorrect_2)))

        print(40 * "=")

    return


def compare_rbp_ig_att_map_common_uncertain_sample(config, run, arr_iter_pred_1, arr_iter_pred_2):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function compare_commonality_most_uncertain_samples')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config,
                                                                                                     print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    model_type = config.get('model_selection', 'model_type')

    dict_class_names = {0: 'mbnl1', 1: 'pum2', 2: 'qki', 3: 'elavl1', 4: 'rbfox2', 5: 'tardbp'}

    ls_classes = []
    ls_count_common = []
    top_n_uncertain = 50
    x_test_float = x_test.astype('float32')
    arr_mean_pred_1 = arr_iter_pred_1.mean(axis=0)
    arr_sd_pred_1 = arr_iter_pred_1.std(axis=0)
    arr_mean_pred_2 = arr_iter_pred_2.mean(axis=0)
    arr_sd_pred_2 = arr_iter_pred_2.std(axis=0)

    # for each class
    for class_i in range(n_classes):
        ls_max_std_1, ls_cand_id_1 = find_uncertain_samples_per_class(y_test, arr_iter_pred_1, class_i,
                                                                        top_n_uncertain)
        ls_max_std_2, ls_cand_id_2 = find_uncertain_samples_per_class(y_test, arr_iter_pred_2, class_i,
                                                                        top_n_uncertain)
        ls_common_class = [ix for ix in ls_cand_id_1 if ix in ls_cand_id_2]

        for i in ls_common_class[:5]:
            print("common uncertain sample, based on interpretability in first")
            print("true class is ", class_i, " - predicted class by first is ", arr_mean_pred_1[i].argmax())
            sample_rbp = x_test_float[i]
            target_class_idx = y_test[i].argmax()
            baseline = tf.zeros(sample_rbp.shape)
            attributions = integrated_gradients(baseline=baseline,
                                                test_sample=sample_rbp,
                                                target_class_idx=target_class_idx,
                                                m_steps=50, model_type='bay')
            all_attributions = np.array([attributions])

            #         print("for class ", dict_class_names[target_class_idx])
            saliency_mat_df = pd.DataFrame(np.array(attributions), columns=['A', 'C', 'G', 'U'])
            # print(saliency_mat_df.head())
            lm.Logo(saliency_mat_df, figsize=(17, 2))
            plt.show()
            plt.close("all")
            print("common uncertain sample, based on interpretability in second")
            print("true class is ", class_i, " - predicted class by second is ", arr_mean_pred_2[i].argmax())
            sample_rbp = x_test_float[i]
            target_class_idx = y_test[i].argmax()
            baseline = tf.zeros(sample_rbp.shape)
            attributions = integrated_gradients(baseline=baseline,
                                                test_sample=sample_rbp,
                                                target_class_idx=target_class_idx,
                                                m_steps=50, model_type='mcd')
            all_attributions = np.array([attributions])

            #         print("for class ", dict_class_names[target_class_idx])
            saliency_mat_df = pd.DataFrame(np.array(attributions), columns=['A', 'C', 'G', 'U'])
            # print(saliency_mat_df.head())
            fig_i = plt.figure()
            lm.Logo(saliency_mat_df, figsize=(17, 2))
            plt.show()
            run['att_map_common_second_class_{0}_sample_{1}'.format(dict_class_names[class_i], str(i))].upload(
                neptune.types.File.as_image(fig_i))
            plt.close("all")

        print(40 * "=")
        print(40 * "=")

    return


def compare_rbp_ig_att_map_uncommon_uncertain_sample(config, run, arr_iter_pred_1, arr_iter_pred_2):


    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function compare_commonality_most_uncertain_samples')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config,
                                                                                                     print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    model_type = config.get('model_selection', 'model_type')

    dict_class_names = {0: 'mbnl1', 1: 'pum2', 2: 'qki', 3: 'elavl1', 4: 'rbfox2', 5: 'tardbp'}

    ls_classes = []
    ls_count_common = []
    top_n_uncertain = 50
    x_test_float = x_test.astype('float32')
    arr_mean_pred_1 = arr_iter_pred_1.mean(axis=0)
    arr_sd_pred_1 = arr_iter_pred_1.std(axis=0)
    arr_mean_pred_2 = arr_iter_pred_2.mean(axis=0)
    arr_sd_pred_2 = arr_iter_pred_2.std(axis=0)


    # for each class
    for class_i in range(n_classes):
        ls_max_std_1, ls_cand_id_1 = find_uncertain_samples_per_class(y_test, arr_iter_pred_1, class_i,
                                                                            top_n_uncertain)
        ls_max_std_2, ls_cand_id_2 = find_uncertain_samples_per_class(y_test, arr_iter_pred_2, class_i,
                                                                            top_n_uncertain)
        ls_common_class = [ix for ix in ls_cand_id_1 if ix in ls_cand_id_2]
        ls_uncommon_class_1 = [ix for ix in ls_cand_id_1 if ix not in ls_cand_id_2]
        ls_uncommon_class_2 = [ix for ix in ls_cand_id_2 if ix not in ls_cand_id_1]
        ls_ix_incorrect_1 = [i for i in range(len(y_test)) if
                               y_test[i, class_i] == 1 and arr_mean_pred_1[i, :].argmax() != class_i]
        ls_ix_incorrect_2 = [i for i in range(len(y_test)) if
                               y_test[i, class_i] == 1 and arr_mean_pred_2[i, :].argmax() != class_i]
        # joins
        ls_cand_id_bay_ls_ix_incorrect_1 = [i for i in ls_cand_id_1 if i in ls_ix_incorrect_1]
        ls_cand_id_mcd_ls_ix_incorrect_2 = [i for i in ls_cand_id_2 if i in ls_ix_incorrect_2]
        ls_common_class_ls_ix_incorrect_1 = [i for i in ls_common_class if i in ls_ix_incorrect_1]
        ls_common_class_ls_ix_incorrect_2 = [i for i in ls_common_class if i in ls_ix_incorrect_2]

        print("only in first")
        for i in ls_uncommon_class_1[:5]:
            print("true class is ", class_i, " - predicted class by first is ", arr_mean_pred_1[i].argmax())
            sample_rbp = x_test_float[i]
            target_class_idx = y_test[i].argmax()
            baseline = tf.zeros(sample_rbp.shape)
            attributions = integrated_gradients(baseline=baseline,
                                                test_sample=sample_rbp,
                                                target_class_idx=target_class_idx,
                                                m_steps=50, model_type='bay')
            all_attributions = np.array([attributions])

            #         print("for class ", dict_class_names[target_class_idx])
            saliency_mat_df = pd.DataFrame(np.array(attributions), columns=['A', 'C', 'G', 'U'])
            # print(saliency_mat_df.head())
            fig_1 = plt.figure()
            lm.Logo(saliency_mat_df, figsize=(17, 2))
            plt.show()
            run['att_map_uncommon_first_class_{0}_sample_{1}'.format(dict_class_names[class_i], str(i))].upload(
                neptune.types.File.as_image(fig_i))
            plt.close("all")

        print("only in second")
        for i in ls_uncommon_class_2[:5]:
            print("true class is ", class_i, " - predicted class by second is ", arr_mean_pred_2[i].argmax())
            sample_rbp = x_test_float[i]
            target_class_idx = y_test[i].argmax()
            baseline = tf.zeros(sample_rbp.shape)
            attributions = integrated_gradients(baseline=baseline,
                                                test_sample=sample_rbp,
                                                target_class_idx=target_class_idx,
                                                m_steps=50, model_type='mcd')
            all_attributions = np.array([attributions])

            #         print("for class ", dict_class_names[target_class_idx])
            saliency_mat_df = pd.DataFrame(np.array(attributions), columns=['A', 'C', 'G', 'U'])
            # print(saliency_mat_df.head())
            fig_i = plt.figure()
            lm.Logo(saliency_mat_df, figsize=(17, 2))
            plt.show()
            run['att_map_uncommon_second_class_{0}_sample_{1}'.format(dict_class_names[class_i],str(i))].upload(
                neptune.types.File.as_image(fig_i))
            plt.close("all")

        print(40 * "=")
        print(40 * "=")

    return


def show_sd_distribution_side_by_side_most_uncertain_per_class_imbalance(y_test_b, y_test_im, arr_iter_preds_balanced,
                                                                         arr_iter_preds_imbalanced, n_classes,
                                                                         n_samples, plot_name_path):
    # for each class
    plt.clf()
    print("distributation of SD for most uncertain samples per class")
    print("left: balanced, right: imbalanced")
    f, axarr = plt.subplots(n_classes, 2, figsize=(15, 15))
    for i in range(n_classes):
        print(30 * "-")
        class_size = sum(y_test[:, i] == 1)
        print("top {0} most uncertain samples in class {1}, with class size {2}".format(n_samples, i, class_size))
        ls_max_std_b, ls_cand_id_b = find_uncertain_samples_per_class(y_test_b, arr_iter_preds_balanced, i, n_samples)
        ls_max_std_im, ls_cand_id_im = find_uncertain_samples_per_class(y_test_im, arr_iter_preds_imbalanced, i,
                                                                          n_samples)
        sns.distplot(ls_max_std_b, ax=axarr[i, 0])
        sns.distplot(ls_max_std_im, ax=axarr[i, 1])

    plt.savefig(plot_name_path, bbox_inches='tight')
    plt.show()


def show_sd_distribution_side_by_side_most_uncertain_per_class_model(y_test, arr_iter_preds_bay, arr_iter_preds_mcd,
                                                                     n_classes, n_samples, plot_name_path):
    # for each class
    plt.clf()
    print("distributation of SD for most uncertain samples per class")
    print("left: bay, right: mcd")
    f, axarr = plt.subplots(n_classes, 2, figsize=(15, 15))
    for i in range(n_classes):
        print(30 * "-")
        class_size = sum(y_test[:, i] == 1)
        print("top {0} most uncertain samples in class {1}, with class size {2}".format(n_samples, i, class_size))
        ls_max_std_bay, ls_cand_id_bay = find_uncertain_samples_per_class(y_test, arr_iter_preds_bay, i, n_samples)
        ls_max_std_mcd, ls_cand_id_mcd = find_uncertain_samples_per_class(y_test, arr_iter_preds_mcd, i, n_samples)
        sns.distplot(ls_max_std_bay, ax=axarr[i, 0])
        sns.distplot(ls_max_std_mcd, ax=axarr[i, 1])

    plt.savefig(plot_name_path, bbox_inches='tight')
    plt.show()


def show_sd_distribution_side_by_side_correct_incorrect_per_class(y_test, arr_iter_preds, n_classes, plot_name_path,
                                                                  show_by_label=True):
    # for each class
    plt.clf()
    print("distributation of SD for test set per class")
    print(
        "left: correctly classified,middle: incorrectly classified - SD of true class, right: incorrectly classified - SD of predicted class")
    if show_by_label:
        f, axarr = plt.subplots(n_classes, 3, figsize=(15, 15))
    else:
        f, axarr = plt.subplots(n_classes, 2, figsize=(15, 15))
    arr_mean_pred = arr_iter_preds.mean(axis=0)
    arr_sd_pred = arr_iter_preds.std(axis=0)

    for class_i in range(n_classes):
        print(30 * "-")
        class_size = sum(y_test[:, class_i] == 1)
        ls_ix_correct = [i for i in range(len(y_test)) if
                         y_test[i, class_i] == 1 and arr_mean_pred[i, :].argmax() == class_i]
        ls_ix_incorrect = [i for i in range(len(y_test)) if
                           y_test[i, class_i] == 1 and arr_mean_pred[i, :].argmax() != class_i]
        ls_sd_correct = [arr_sd_pred[i, class_i] for i in ls_ix_correct]
        ls_sd_incorrect_bylabel = [arr_sd_pred[i, class_i] for i in ls_ix_incorrect]
        ls_sd_incorrect_bypred = [arr_sd_pred[i, arr_mean_pred[i, :].argmax()] for i in ls_ix_incorrect]
        print("in class{} with size {}, correct {} and incorrect {}".format(class_i, class_size,
                                                                            len(ls_ix_correct), len(ls_ix_incorrect)))

        sns.distplot(ls_sd_correct, ax=axarr[class_i, 0])
        if show_by_label:
            sns.distplot(ls_sd_incorrect_bylabel, ax=axarr[class_i, 1])
            sns.distplot(ls_sd_incorrect_bypred, ax=axarr[class_i, 2])
        else:
            sns.distplot(ls_sd_incorrect_bypred, ax=axarr[class_i, 1])

    plt.savefig(plot_name_path, bbox_inches='tight')
    plt.show()


#
# ## to be used as follows
# # compare BNN vs MCDs SD ranges
# plot_name_path ="figures/figure_mcd_compare_all_sd_dist_bnn_mcd.png"
# # show_sd_distribution_side_by_side_most_uncertain_per_class(y_test, y_test, arr_iter_pred_bay, arr_iter_pred_mcd, 6, 100, plot_name_path)
# show_sd_distribution_side_by_side_most_uncertain_per_class_model(y_test, arr_iter_pred_bay, arr_iter_pred_mcd, 3, 200, plot_name_path)
#
# plot_name_path ="figures/figure_mcd_compare_50uncertain_sd_dist_bnn_mcd.png"
# # show_sd_distribution_side_by_side_most_uncertain_per_class(y_test, y_test, arr_iter_pred_bay, arr_iter_pred_mcd, 6, 100, plot_name_path)
# show_sd_distribution_side_by_side_most_uncertain_per_class_model(y_test, arr_iter_pred_bay, arr_iter_pred_mcd, 3, 50, plot_name_path)
#
# plot_name_path ="figures/correct_incorrect_sd_dist_bnn.png"
# show_sd_distribution_side_by_side_correct_incorrect_per_class(y_test, arr_iter_pred_bay, 3, plot_name_path, show_by_label=False)
#
# plot_name_path ="figures/correct_incorrect_sd_dist_mcd.png"
# show_sd_distribution_side_by_side_correct_incorrect_per_class(y_test, arr_iter_pred_mcd, 3, plot_name_path, show_by_label=False)
#
