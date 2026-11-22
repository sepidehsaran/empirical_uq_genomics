import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neptune as neptune
import logomaker as lm
from utils import *
import random
# from deeplift.dinuc_shuffle import dinuc_shuffle, traverse_edges, shuffle_edges, prepare_edges
from collections import Counter
# from deeplift.visualization import viz_sequence
import shap
# %matplotlib inline
plt.ion()


def interpolate_sample(baseline, sample, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(sample, axis=0)
    delta = input_x - baseline_x
    samples = baseline_x + alphas_x * delta
    return samples[0]


def compute_gradients_model(model, sample, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(sample)
        prob = model(sample)[:, target_class_idx]
    return tape.gradient(prob, sample)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    int_gradients = tf.math.reduce_mean(grads, axis=0)
    return int_gradients


@tf.function
def integrated_gradients(model, baseline, test_sample, target_class_idx, batch_size, m_steps=50):
    """
    function code credit: ?

    :param model:
    :param baseline:
    :param test_sample:
    :param target_class_idx:
    :param m_steps:
    :param batch_size:
    :return:
    """
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Initialize TensorArray outside loop to collect gradients.
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        #         print("shape is ", test_sample.shape)

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_sample(baseline=baseline, sample=test_sample, alphas=alpha_batch)
        #         print("shape 2 is ", interpolated_path_input_batch.shape)
        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients_model(model, sample=interpolated_path_input_batch,
                                                 target_class_idx=target_class_idx)


        #         print(tf.range(from_, to).dtype)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)
        # print("here")
    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    int_gradients = (test_sample - baseline) * avg_gradients
    
    return int_gradients


def get_shap(model, x_train, x_test, n_backgrounds=100):
    # reload(shap.explainers.deep.deep_tf)
    # reload(shap.explainers.deep)
    # reload(shap.explainers)
    # reload(shap)
    # select a set of background examples to take an expectation over
    background = x_train[np.random.choice(x_train.shape[0], n_backgrounds, replace=False)]
    # create explainer based on subset of training set
    explainer = shap.DeepExplainer(model, background)
    # get shap values on test set
    test_shap_values = explainer.shap_values(x_test)
    return test_shap_values


def rbp_ig_attribution_maps_for_correctly_classified(config, run):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function rbp_ig_attribution_maps_for_correctly_classified')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config,
                                                                                                     print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    model_type = config.get('model_selection', 'model_type')
    loaded_model = load_model(config, run)
    arr_iter_pred = pd.read_pickle(path_results + "iter_preds_{0}_{1}.pickle".format(model_type, version_rand))
    x_test_float = x_test.astype('float32')
    arr_mean_pred = arr_iter_pred.mean(axis=0)
    arr_sd_pred = arr_iter_pred.std(axis=0)
    class_names = get_class_names(config)
    dict_class_names = get_dict_class_names(config)#{0: 'mbnl1', 1: 'pum2', 2: 'qki', 3: 'elavl1', 4: 'rbfox2', 5: 'tardbp'}
    n_plot_samples_per_class = int(config.get('xai', 'n_plot_samples_per_class'))
    batch_size = int(config.get('model_selection', 'batch_size'))
    base = config.get('xai', 'baseline')
    xai_method = config.get('xai', 'xai_method')
    if base == 'zeros':
        baseline = tf.zeros(x_test[0].shape)
    else:
        print('invalid config setting for baseline of the xai method, switching to default zeros.')
        baseline = tf.zeros(x_test[0].shape)

    for class_i in range(n_classes):
        fig_per_class = plt.figure(figsize=[20, 25])
        if verbose: print("in class ", dict_class_names[class_i])
        ls_ix_correct = [i for i in range(len(y_test)) if
                         y_test[i, class_i] == 1 and arr_mean_pred[i, :].argmax() == class_i]

        random.shuffle(ls_ix_correct)

        counter = 0
        for i in ls_ix_correct[:n_plot_samples_per_class]:
            sample_rbp = x_test_float[i]
            target_class_idx = y_test[i].argmax()
            # baseline = tf.zeros(sample_rbp.shape)
            attributions = integrated_gradients(loaded_model,
                                                baseline=baseline,
                                                test_sample=sample_rbp,
                                                target_class_idx=target_class_idx,
                                                batch_size=batch_size,
                                                m_steps=50)
            #             all_attributions = np.array([attributions])
            saliency_mat_df = pd.DataFrame(np.array(attributions), columns=['A', 'C', 'G', 'U'])
            ax = plt.subplot2grid((n_plot_samples_per_class, 1), (counter, 0))
            lm.Logo(saliency_mat_df, ax=ax, figsize=(17, 2))
            ax.set_xticks([])
            ax.set_yticks([])
            sample_entropy = entropy(np.array(attributions))
            ax.set_title('correct class {0}, predicted class {1}, xai entropy {2}'.format(
                dict_class_names[class_i], 
                dict_class_names[arr_mean_pred[i,:].argmax()],
                sample_entropy))
            counter += 1

        plt.show()
        run['att_map_correct_classified_class_{0}'.format(dict_class_names[class_i])].upload(
            neptune.types.File.as_image(fig_per_class))
    plt.close("all")
    
    return attributions


def rbp_ig_attribution_maps_for_incorrectly_classified(config, run):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function rbp_ig_attribution_maps_for_incorrectly_classified')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config,
                                                                                                     print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    model_type = config.get('model_selection', 'model_type')
    loaded_model = load_model(config, run)
    arr_iter_pred = pd.read_pickle(path_results + "iter_preds_{0}_{1}.pickle".format(model_type, version_rand))
    x_test_float = x_test.astype('float32')
    arr_mean_pred = arr_iter_pred.mean(axis=0)
    arr_sd_pred = arr_iter_pred.std(axis=0)
    dict_class_names = get_dict_class_names(config) #{0: 'mbnl1', 1: 'pum2', 2: 'qki', 3: 'elavl1', 4: 'rbfox2', 5: 'tardbp'}
    n_plot_samples_per_class = int(config.get('xai', 'n_plot_samples_per_class'))
    batch_size = int(config.get('model_selection', 'batch_size'))
    base = config.get('xai', 'baseline')
    xai_method = config.get('xai', 'xai_method')
    if base == 'zeros':
        baseline = tf.zeros(x_test[0].shape)
    else:
        print('invalid config setting for baseline of the xai method, switching to default zeros.')
        baseline = tf.zeros(x_test[0].shape)

    for class_i in range(n_classes):
        fig_per_class = plt.figure(figsize=[20, 25])
        if verbose: print("in class ", dict_class_names[class_i])
        ls_ix_incorrect = [i for i in range(len(y_test)) if
                           y_test[i, class_i] == 1 and arr_mean_pred[i, :].argmax() != class_i]

        random.shuffle(ls_ix_incorrect)

        counter = 0
        for i in ls_ix_incorrect[:n_plot_samples_per_class]:
            sample_rbp = x_test_float[i]
            target_class_idx = y_test[i].argmax()
            # baseline = tf.zeros(sample_rbp.shape)
            attributions = integrated_gradients(loaded_model,
                                                baseline=baseline,
                                                test_sample=sample_rbp,
                                                target_class_idx=target_class_idx,
                                                batch_size=batch_size,
                                                m_steps=50)
            #             all_attributions = np.array([attributions])
            saliency_mat_df = pd.DataFrame(np.array(attributions), columns=['A', 'C', 'G', 'U'])
            sample_entropy = entropy(np.array(attributions))
            ax = plt.subplot2grid((n_plot_samples_per_class, 1), (counter, 0))
            lm.Logo(saliency_mat_df, ax=ax, figsize=(17, 2))
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title('correct class {0}, predicted class {1}, xai entropy {2}'.format(
                dict_class_names[class_i],
                dict_class_names[arr_mean_pred[i,:].argmax()],
                sample_entropy))
            counter += 1

        plt.show()
        run['att_map_incorrect_classified_class_{0}'.format(dict_class_names[class_i])].upload(
            neptune.types.File.as_image(fig_per_class))
    plt.close("all")

    return


def entropy(imp_scores, pseudocount=0.001):
    """
    modified function from:
    https://github.com/amtseng/fourier_attribution_priors/blob/481d207499f7164b109db481f3ba989b098fc0e9/notebooks/view_profile_predictions.ipynb

    For an I x 4 or N x I x 4 array of actual importance scores, computes the entropy
    of each track. Returns a single entropy value or an N-array of entropy values.
    """
    scores = []
    # Normalize
    imp_scores_sum = np.sum(np.abs(imp_scores), axis=len(imp_scores.shape) - 1)
    imp_scores_sum = imp_scores_sum + pseudocount
    imp_scores_norm = imp_scores_sum / np.sum(imp_scores_sum, axis=len(imp_scores.shape) - 2, keepdims=True)

    return -np.sum(imp_scores_norm * np.log2(imp_scores_norm), axis=len(imp_scores.shape) - 2)


def comparison_plot_sample_entropy_testset_bnn_vs_mcd(config, run):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function comparison_plot_sample_entropy_testset_bnn_vs_mcd')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config,
                                                                                                     print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
#     batch_size = int(config.get('model_selection', 'batch_size'))
    n_steps = int(config.get('xai', 'n_steps'))
    base = config.get('xai', 'baseline')
    xai_method = config.get('xai', 'xai_method')
    x_test_float = x_test.astype('float32')
    if base == 'zeros':
        baseline = tf.zeros(x_test[0].shape)
    else:
        print('invalid config setting for baseline of the xai method, switching to default zeros.')
        baseline = tf.zeros(x_test[0].shape)
    
    # det_loaded_model = load_model(config, run, explicit_model_type='det')
    bnn_loaded_model = load_model(config, run, explicit_model_type='bnn')
    bnn_arr_iter_pred = pd.read_pickle(path_results + "iter_preds_{0}_{1}.pickle".format('bnn', version_rand))
    bnn_arr_mean_pred = bnn_arr_iter_pred.mean(axis=0)
    bnn_arr_sd_pred = bnn_arr_iter_pred.std(axis=0)

    mcd_loaded_model = load_model(config, run, explicit_model_type='mcd')
    mcd_arr_iter_pred = pd.read_pickle(path_results + "iter_preds_{0}_{1}.pickle".format('mcd', version_rand))
    mcd_arr_mean_pred = mcd_arr_iter_pred.mean(axis=0)
    mcd_arr_sd_pred = mcd_arr_iter_pred.std(axis=0)

    dict_class_names = get_dict_class_names(config)
    ls_bnn_entropies = []
    ls_mcd_entropies = []
    if verbose: print('calculating Shanon entropy for attribution maps generated by '+ xai_method)
    if xai_method == 'integrated_gradients':
        for i in range(x_test_float.shape[0]):
            sample_rbp = x_test_float[i]
            target_class_idx = y_test[i].argmax()

            bnn_attributions = integrated_gradients(bnn_arr_iter_pred,
                                                    baseline=baseline,
                                                    test_sample=sample_rbp,
                                                    target_class_idx=target_class_idx,
                                                    batch_size=n_steps+1,
                                                    m_steps=n_steps)

            mcd_attributions = integrated_gradients(mcd_loaded_model,
                                                    baseline=baseline,
                                                    test_sample=sample_rbp,
                                                    target_class_idx=target_class_idx,
                                                    batch_size=n_steps+1,
                                                    m_steps=n_steps)

            ls_bnn_entropies.append(entropy(np.array(bnn_attributions)))
            ls_mcd_entropies.append(entropy(np.array(mcd_attributions)))

        df_compare_entropies = pd.DataFrame()
        df_compare_entropies['bnn_entropy'] = pd.Series(ls_bnn_entropies)
        df_compare_entropies['mcd_entropy'] = pd.Series(ls_mcd_entropies)
        
    elif xai_method == 'deep_shap':
        print('implement deep shap')
#
#
#         #project the importance at each position onto the base that's actually present
#         dinuc_shuff_explanations = np.sum(raw_shap_explanations,axis=-1)[:,:,None]*seqs_to_explain
#         for dinuc_shuff_explanation in dinuc_shuff_explanations:
#             viz_sequence.plot_weights(dinuc_shuff_explanation, subticks_frequency=20)
#
#         for i in range(len(mcd_shap_values)):
#
#             ls_bnn_entropies.append(entropy(np.array(mcd_shap_values)))
#             ls_mcd_entropies.append(entropy(np.array(mcd_shap_values)))
#
#         df_compare_entropies = pd.DataFrame()
#         df_compare_entropies['bnn_entropy'] = pd.Series(ls_bnn_entropies)
#         df_compare_entropies['mcd_entropy'] = pd.Series(ls_mcd_entropies)
#
        # plot the feature attributions
        # shap.image_plot(shap_values, -x_test[1:5])

    else:
        df_compare_entropies = pd.DataFrame()
        print('invalid value for xai method')
        
    print(df_compare_entropies.head())
    fig1 = plt.figure()
    sns.scatterplot(data=df_compare_entropies, x='bnn_entropy', y='mcd_entropy', alpha=0.2)
#     plt.plot([6.5, 6.5], [7.3, 7.3], linewidth=2)
    run["entropy_bnn_vs_mcd_test_scatter"].upload(neptune.types.File.as_image(fig1))
    plt.show()
    plt.close("all")

    return




def plot_single_sample_entropy_testset_bnn_vs_mcd(config, run):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function plot_single_sample_entropy_testset_bnn_vs_mcd')
    x_train, y_train, x_valid, y_valid, x_test, y_test, dataset_setting_name = read_splitted_dataset(config,
                                                                                                     print_plot_imbalance=False)
    path_results = config.get('general', 'path_results') + dataset_setting_name + '/'
    rand_seed = int(config.get('general', 'random_seed'))
    n_classes = int(config.get('input_data_selection', 'n_classes'))
    version_rand = 'v_' + str(config.get('general', 'version')) + '_s_' + str(rand_seed)
    batch_size = int(config.get('model_selection', 'batch_size'))
    base = config.get('xai', 'baseline')
    xai_method = config.get('xai', 'xai_method')
    x_test_float = x_test.astype('float32')
    if base == 'zeros':
        baseline = tf.zeros(x_test[0].shape)
    else:
        print('invalid config setting for baseline of the xai method, switching to default zeros.')
        baseline = tf.zeros(x_test[0].shape)

    bnn_loaded_model = load_model(config, run, explicit_model_type='bnn')

    mcd_loaded_model = load_model(config, run, explicit_model_type='mcd')

    dict_class_names = get_dict_class_names(config) #{0: 'mbnl1', 1: 'pum2', 2: 'qki', 3: 'elavl1', 4: 'rbfox2', 5: 'tardbp'}
    ls_bnn_entropies = []
    ls_mcd_entropies = []
    if verbose: print('calculating Shanon entropy for attribution maps generated by '+ xai_method)
    i = 5
    sample_rbp = x_test_float[i]
    target_class_idx = y_test[i].argmax()
#     ls_batch_size = [10,20,30,40,50]
    ls_batch_size = [51,60,70,80,90, 200, 300]
    for b in ls_batch_size:
        bnn_attributions = integrated_gradients(bnn_loaded_model,
                                                baseline=baseline,
                                                test_sample=sample_rbp,
                                                target_class_idx=target_class_idx,
                                                batch_size=b,
                                                m_steps=50)

        mcd_attributions = integrated_gradients(mcd_loaded_model,
                                                baseline=baseline,
                                                test_sample=sample_rbp,
                                                target_class_idx=target_class_idx,
                                                batch_size=b,
                                                m_steps=50)

        ls_bnn_entropies.append(entropy(np.array(bnn_attributions)))
        ls_mcd_entropies.append(entropy(np.array(mcd_attributions)))

    df_compare_entropies = pd.DataFrame()
    df_compare_entropies['bnn_entropy'] = pd.Series(ls_bnn_entropies)
    df_compare_entropies['mcd_entropy'] = pd.Series(ls_mcd_entropies)
        
#     elif xai_method == 'deep_shap':
# #         bnn_shap_values = get_shap(bnn_loaded_model, x_train, x_test)
#         for i in range(mcd_shape_values.shape[0]):
#
#             ls_bnn_entropies.append(entropy(np.array(mcd_shap_values)))
#             ls_mcd_entropies.append(entropy(np.array(mcd_shap_values)))
            
#         df_compare_entropies = pd.DataFrame()
#         df_compare_entropies['bnn_entropy'] = pd.Series(ls_bnn_entropies)
#         df_compare_entropies['mcd_entropy'] = pd.Series(ls_mcd_entropies)
#     else:
    print(df_compare_entropies.head())
    fig1 = plt.figure()
    sns.scatterplot(data=df_compare_entropies, x='bnn_entropy', y='mcd_entropy', alpha=0.2)
#     plt.plot([6.5, 6.5], [7.3, 7.3], linewidth=2)
    run["entropy_bnn_vs_mcd_test_scatter"].upload(neptune.types.File.as_image(fig1))
    plt.show()
    plt.close("all")

    return




def onehot_dinuc_shuffle(s):
    s = np.squeeze(s)
    argmax_vals = "".join([str(x) for x in np.argmax(s, axis=-1)])
    shuffled_argmax_vals = [int(x) for x in traverse_edges(argmax_vals,
                            shuffle_edges(prepare_edges(argmax_vals)))]
    to_return = np.zeros_like(s)
    to_return[list(range(len(s))), shuffled_argmax_vals] = 1
    return to_return
