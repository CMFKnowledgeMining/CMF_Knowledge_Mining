import pandas as pd

from cmftools import *
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import warnings
import math

warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from sklearn import metrics
import gc
from scipy import stats
from sklearn.metrics import silhouette_score
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib


def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()

def load_dt_more_vars(target, cmf_thresh, qual_rating=-5, manual_path=False, non_tuning=False):
    if target == "intersection":
        keep_cols = ["cmName", "catname", "subcatname", "intersecType", "areaType", "intersecGeometry",
                     "trafficControl",
                     "crashType", "crashSeverityKABCO", "crashTOD", 'country', 'state', 'qualRating', 'priorCondition',
                     'yearsOfDataFrom', 'yearsOfDataTo']
        feature2 = "intersecType"
        # dt_path = './data/Intersection_data.xlsx'
        dt_path = 'C:/Users/yanlin93/Desktop/CMF_data_0712/Intersection_data.xlsx'

        # dt_path = './data/cmfclearinghouse_intersection.xlsx'
    elif target == "roadway":
        keep_cols = ["cmName", "catname", "subcatname", "roadwayType", "areaType", "crashType", "crashSeverityKABCO",
                     "crashTOD",
                     "roadDivType", "numLanes", 'speedLimit', 'country', 'state', 'qualRating', 'priorCondition',
                     'yearsOfDataFrom', 'yearsOfDataTo']
        feature2 = "crashType"
        # dt_path = './data/roadway_data.xlsx'
        # dt_path = './data/cmfclearinghouse_roadway.xlsx'
        if manual_path is False:
            dt_path = 'C:/Users/yanlin93/Desktop/CMF_data_0712/roadway_data.xlsx'
        elif manual_path is True:
            if non_tuning is True:
                keep_cols = ["cmName", "catname", "subcatname", "treatment", "AfterShoulder", "PriorShoulder", "roadwayType", "numLanes", "areaType",
                             "crashType", "crashSeverityKABCO", "PriorWidth", "AfterWidth", "crashTOD",
                             "roadDivType", 'speedLimit', 'country', 'state', 'qualRating', 'priorCondition',
                             'yearsOfDataFrom', 'yearsOfDataTo']
            dt_path = "C:/Users/yanlin93/Desktop/CMF_data_0712/train_shouderWidth1.xlsx"

    dt = pd.read_excel(dt_path, header=0)
    dt = dt.dropna(subset='accModFactor')
    # plot_pairwise_heatmap(dt, "catname", feature2)
    dt = dt[dt["qualRating"] >= qual_rating]
    dt = dt.drop(dt[dt["accModFactor"] <= cmf_thresh[0]].index)
    dt = dt.drop(dt[dt["accModFactor"] > cmf_thresh[1]].index)
    dt[dt == "Not Specified"] = np.nan
    dt[dt == "Not specified"] = np.nan

    dt = dt.sample(frac=1, random_state=2431)
    dt.reset_index(drop=True)
    dt['index'] = range(dt.shape[0])
    dt = dt.set_index('index')

    cmfs = dt["accModFactor"]
    dt.fillna("Not Specified", inplace=True)

    if non_tuning is True:
        for col in keep_cols[:-3]:
            dt[col] = pd.Categorical(dt[col])

    bc_dt = copy.deepcopy(dt[keep_cols])  # base-condition
    return dt, bc_dt, cmfs, keep_cols


def tune_transformer(train_examples, model_name, max_seq_len=16):
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(0.98)
    # Define the model. Either from scratch of by loading a pre-trained model
    # model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
    model = SentenceTransformer(backbone, device=device)

    # Define your train examples. You need more than just two examples...

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100,
              output_path=f'data/model/' + model_name)
    return model


def get_str_inputs(bc_dt):
    x_ls = []
    for idx, r_i in bc_dt.iterrows():
        tmp1 = str()
        for x in r_i.values:
            tmp1 += str(x) + ','
        x_ls.append(tmp1)
    return x_ls


def calculate_semi_score(txt1_cmfs, txt2_cmfs, loss_string):
    simi_score_ls = []
    delta_cmfs = np.abs(txt1_cmfs - txt2_cmfs)
    flags = np.sign(txt1_cmfs - 1) * np.sign(txt2_cmfs - 1) * delta_cmfs
    if loss_string == "1-0.5delta":
        for delta_cmf in delta_cmfs:
            simi_score_i = 1.0 - 0.5 * delta_cmf
            simi_score_ls.append(simi_score_i)

    elif loss_string == "(1-0.5delta)^2":
        for delta_cmf, flag in zip(delta_cmfs, flags):
            if flag < 0.0:
                # simi_score_i = np.cos(np.pi / 2 * delta_cmf) * math.pow(delta_cmf, 2)
                # simi_score_i = np.cos(np.pi / 2 * delta_cmf)
                simi_score_i = math.pow(1.0 - 0.5 * delta_cmf, 2)
            else:
                simi_score_i = 1.0 - 0.5 * delta_cmf
            simi_score_ls.append(simi_score_i)

    elif loss_string == "1-delta":
        for delta_cmf, flag in zip(delta_cmfs, flags):
            if flag >= 0.0:
                simi_score_i = 1.0 - 0.5 * delta_cmf

            elif -1 <= flag < 0.0:
                # simi_score_i = np.cos(np.pi / 2 * delta_cmf) * math.pow(delta_cmf, 2)
                # simi_score_i = np.cos(np.pi / 2 * delta_cmf)
                simi_score_i = 1.0 - delta_cmf
            else:
                simi_score_i = 0.0
            simi_score_ls.append(simi_score_i)

    elif loss_string == 'cos':
        for delta_cmf, flag in zip(delta_cmfs, flags):
            if -1 <= flag <= 0.0:
                simi_score_i = 1.0 - 0.5 * delta_cmf

            elif -1 <= flag < 0.0:
                simi_score_i = np.cos(np.pi / 2 * delta_cmf) * math.pow(delta_cmf, 2)
            else:
                simi_score_i = np.cos(np.pi / 2 * delta_cmf)
            simi_score_ls.append(simi_score_i)

    return simi_score_ls


def load_samples(trn_input_dt, cmfs, tune_percent, loss_string, random_mode=True):
    transformer_samples = []
    if random_mode is True:
        tune_index = random.sample(list(trn_input_dt.index), round(tune_percent * trn_input_dt.shape[0]))
    else:
        tune_index = list(trn_input_dt.drop_duplicates(subset='cmName').index)

    tune_dt = (trn_input_dt.loc[tune_index]).reset_index(drop=True)
    tune_cmf = (cmfs.loc[tune_index]).reset_index(drop=True)
    tunex_ls = get_str_inputs(tune_dt)

    index_ls = list(tune_dt.index)
    n_sample = len(tunex_ls)
    for idy, x in zip(index_ls, tunex_ls):
        start = idy + 1
        print(idy)
        n_rest = n_sample - start
        txt1_ls = [x] * n_rest
        txt1_cmfs = np.array([tune_cmf[idy]] * n_rest)
        txt2_ls = tunex_ls[start:]
        txt2_cmfs = np.array(tune_cmf[start:])
        # simi_score = list(np.array([1]*n_rest) - np.abs(txt1_cmfs-txt2_cmfs)/2.0)
        simi_score = calculate_semi_score(txt1_cmfs, txt2_cmfs, loss_string)
        transformer_samples.extend([InputExample(texts=[x[0], x[1]], label=float(x[2]))
                                    for x in zip(txt1_ls, txt2_ls, simi_score)])
    return transformer_samples


def plot_res(cmf_preds, cmfs_tst, fig_name=None):
    # plt.scatter(cmf_preds, cmfs_tst, color='red')
    mae_ls = np.abs(cmf_preds - cmfs_tst)
    print('max mae', np.max(mae_ls))
    preds_df = pd.DataFrame.from_dict({'CMF Predictions': cmf_preds, 'CMF True Values': cmfs_tst,
                                       'mae': mae_ls, 'Residual Level': ['a'] * len(mae_ls)})
    preds_df.reset_index()
    range_ls = list(np.arange(0.0, np.max(mae_ls) + 0.1, 0.1))
    old_index = set()
    b = pd.DataFrame(['a'] * preds_df.shape[0], index=preds_df.index, columns=['class'])
    hue_order = []
    for idx, l in enumerate(range_ls):
        if idx > 0:
            selected_index = set((preds_df[(preds_df['mae'] <= range_ls[idx])]).index)
            use_index = selected_index - old_index
            b.loc[list(use_index)] = f'[{str(range_ls[idx - 1])[:3]}, {str(range_ls[idx])[:3]}]'
            hue_order.append(f'[{str(range_ls[idx - 1])[:3]}, {str(range_ls[idx])[:3]}]')
            old_index = selected_index

    preds_df['Residual Level'] = b
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        fig = plt.figure()

        # sns.scatterplot(data=preds_df, x="CMF Predictions", y="CMF True Values", hue=preds_df['Residual Level'], s=20,
        #                 palette=sns.color_palette("BrBG",n_colors=len(range_ls), as_cmap=True))
        print('percentage of residual less than 0.1',
              sum(preds_df['Residual Level'] == '[0.0, 0.1]') / preds_df.shape[0])

        sns.scatterplot(data=preds_df, x="CMF Predictions", y="CMF True Values", hue=preds_df['Residual Level'],
                        hue_order=hue_order, s=20,
                        palette=sns.hls_palette(s=.4, l=.5,
                                                n_colors=pd.unique(preds_df['Residual Level']).shape[0] + 8))
        identity_line = np.linspace(0.0, max(max(cmfs_tst) + 0.2, max(cmf_preds)))
        plt.plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
        plt.legend(loc='lower right', prop={'size': 7}, title='Residual Level')
        plt.ylabel("Reported CMF Values")
        plt.savefig(f'{fig_name}_scatter_aap_revision.pdf', format="pdf", bbox_inches="tight")
        fig.show()
    print('...')
    return preds_df


def calculate_res(cmf_preds, cmfs_tst):
    mae_ls = np.abs(cmf_preds - cmfs_tst)
    preds_df = pd.DataFrame.from_dict({'CMF Predictions': cmf_preds, 'CMF True Values': cmfs_tst,
                                       'mae': mae_ls, 'Residual Level': ['a'] * len(mae_ls)})
    preds_df.reset_index()
    range_ls = list(np.arange(0.0, np.max(mae_ls) + 0.1, 0.1))
    old_index = set()
    b = pd.DataFrame(['a'] * preds_df.shape[0], index=preds_df.index, columns=['class'])
    hue_order = []
    for idx, l in enumerate(range_ls):
        if idx > 0:
            selected_index = set((preds_df[(preds_df['mae'] <= range_ls[idx])]).index)
            use_index = selected_index - old_index
            b.loc[list(use_index)] = f'[{str(range_ls[idx - 1])[:3]}, {str(range_ls[idx])[:3]}]'
            hue_order.append(f'[{str(range_ls[idx - 1])[:3]}, {str(range_ls[idx])[:3]}]')
            old_index = selected_index

    preds_df['Residual Level'] = b
    return preds_df


def evaluate_metric_cross_validation(tst_data):
    mse = metrics.mean_squared_error(tst_data['cmf'], tst_data['cmf_preds'])
    mae = metrics.mean_absolute_error(tst_data['cmf'], tst_data['cmf_preds'])
    r_2 = metrics.r2_score(tst_data['cmf'], tst_data['cmf_preds'])
    print('........................................................................\n')
    print('rmse = ' + str(math.sqrt(mse)))
    print('mae = ' + str(mae))
    print('r_2 = ' + str(r_2))

    tst_eval = copy.deepcopy(tst_data[['cmf', 'cmf_preds']])
    pos_cm = tst_eval[tst_eval['cmf'] > 1]
    neg_cm = tst_eval[tst_eval['cmf'] <= 1]
    pos_num = pos_cm[pos_cm['cmf_preds'] > 1].shape[0]
    neg_num = neg_cm[neg_cm['cmf_preds'] <= 1].shape[0]
    correct_rate = (pos_num + neg_num) / tst_eval.shape[0]
    pos_rate = pos_num / pos_cm.shape[0]
    neg_rate = neg_num / neg_cm.shape[0]

    print('correct rate with 1 =' + str(correct_rate))
    print('pos rate with 1 =' + str(pos_rate))
    print('neg rate with 1 =' + str(neg_rate) + '\n')

    mae_ls = np.abs((tst_data['cmf_preds'] - tst_data['cmf']))
    preds_df = pd.DataFrame.from_dict({'CMF Predictions': tst_data['cmf_preds'], 'CMF True Values': tst_data['cmf'],
                                       'mae': mae_ls, 'Residual Level': ['a'] * len(mae_ls)})
    preds_df.reset_index()
    range_ls = list(np.arange(0.0, np.max(mae_ls) + 0.1, 0.1))
    old_index = set()
    b = pd.DataFrame(['a'] * preds_df.shape[0], index=preds_df.index, columns=['class'])
    hue_order = []
    for idx, l in enumerate(range_ls):
        if idx > 0:
            selected_index = set((preds_df[(preds_df['mae'] <= range_ls[idx])]).index)
            use_index = selected_index - old_index
            b.loc[list(use_index)] = f'[{str(range_ls[idx - 1])[:3]}, {str(range_ls[idx])[:3]}]'
            hue_order.append(f'[{str(range_ls[idx - 1])[:3]}, {str(range_ls[idx])[:3]}]')
            old_index = selected_index

    preds_df['Residual Level'] = b
    fig = plt.figure()
    # sns.scatterplot(data=preds_df, x="CMF Predictions", y="CMF True Values", hue=preds_df['Residual Level'], s=20,
    #                 palette=sns.color_palette("BrBG",n_colors=len(range_ls), as_cmap=True))
    print('percentage of residual <= 0.1 ' + str(sum(preds_df['Residual Level'] == '[0.0, 0.1]') / preds_df.shape[0]))


def distance_to_affinity(distance_matrix, preference=None, damping_factor=0.5):
    sigma_values = np.linspace(0.1, 2.0, num=20)
    best_score = -1
    best_sigma = None

    for sigma in sigma_values:
        affinity_matrix = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
        model = AffinityPropagation(affinity='precomputed')
        model.fit(affinity_matrix)
        labels = model.labels_
        score = silhouette_score(distance_matrix, labels, metric='precomputed')

        if score > best_score:
            best_score = score
            best_sigma = sigma
            best_model = model

    print("Best sigma:", best_sigma)
    print("best_score:", best_score)

    return best_model


def plot_corr(clusters, cms_vec_encode):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    # recovered_encode = (np.concatenate([x_trn_encoded, x_tst_encoded], axis=0))[sum([trn_index, list(tst_index)], []), :]
    X_embedded = TSNE(n_components=2, learning_rate='auto').fit_transform(cms_vec_encode)
    plot_clusters(clusters, X_embedded, method="AffinityPropagation")


def predict_cmf_by_k_neighbor(index_id, dist_mtx, trn_index):
    neighbor_flag = [False] * dist_mtx.shape[0]
    trn_flag = np.array([False] * dist_mtx.shape[0])
    trn_flag[trn_index] = True

    for i in np.argsort(dist_mtx[index_id, :])[:10]:
        neighbor_flag[i] = True

    neighbor_flag = neighbor_flag * trn_flag
    ngbr_cmfs = cmfs[neighbor_flag]
    ngbr_dists = dist_mtx[index_id, neighbor_flag]

    if sum(neighbor_flag) != 0:
        cmf_pred = sum(cmfs.iloc[neighbor_flag] * ngbr_dists) / sum(ngbr_dists)
    else:
        cmf_pred = np.nan

    return cmf_pred


def get_clusters(cms_vec_encode):
    # recovered_encode = (np.concatenate([x_trn_encoded, x_tst_encoded], axis=0))
    # [sum([trn_index, list(tst_index)], []), :]
    # x_cls_dist = metrics.pairwise_distances(recovered_encode.reshape(cms_vec_encode.shape[0], -1))

    x_cls_dist = metrics.pairwise.cosine_distances(cms_vec_encode)

    # affi_mtx = distance_to_affinity(x_cls_dist)
    affi_mtx = metrics.pairwise.cosine_similarity(cms_vec_encode)
    # affi_mtx = np.exp(-x_cls_dist ** 2 / (2 * 0.3 ** 2))
    np.fill_diagonal(affi_mtx, 1.0)

    clusters = AffinityPropagation(affinity="precomputed", preference=-1.0).fit(affi_mtx)
    # clusters = AffinityPropagation(damping=damp).fit(cms_vec_encode)
    # plot_corr(clusters, cms_vec_encode)

    return x_cls_dist, affi_mtx, clusters


def get_group_i_corr(tst_i_df, corr_dict, fold):
    tst_dt_for_corr = tst_i_df.dropna()
    tst_dt_for_corr['dist_over_neighbor'] = (
            tst_dt_for_corr['dists'] / tst_dt_for_corr['count_neighbor']).values

    dists_corr_value, dists_pvalue = stats.spearmanr(tst_dt_for_corr['abs_residual'],
                                                     tst_dt_for_corr['dists'])
    neighbor_corr_value, neighbor_pvalue = stats.spearmanr(tst_dt_for_corr['abs_residual'],
                                                           tst_dt_for_corr['count_neighbor'])
    std_corr_value, std_pvalue = stats.spearmanr(tst_dt_for_corr['abs_residual'],
                                                 tst_dt_for_corr['std_cluster'])
    weighted_corr_value, weighted_pvalue = stats.spearmanr(tst_dt_for_corr['abs_residual'],
                                                           tst_dt_for_corr['dist_over_neighbor'])

    corr_dict['fold'].append(fold)
    corr_dict['dist_corr'].append(dists_corr_value)
    corr_dict['neigh_corr'].append(neighbor_corr_value)
    corr_dict['disper_corr'].append(std_corr_value)
    corr_dict['weighted_corr'].append(weighted_corr_value)

    corr_dict['dist_p'].append(dists_pvalue)
    corr_dict['neigh_p'].append(neighbor_pvalue)
    corr_dict['disper_p'].append(std_pvalue)
    corr_dict['weighted_p'].append(weighted_pvalue)
    return corr_dict


def get_avg_group_statistic(tst_res_df):
    sns.boxplot(x='abs_residual_levels', y='simis', data=tst_res_df,
                order=list(np.sort(pd.unique(tst_res_df['abs_residual_levels']))))
    plt.show()
    tst_corr = tst_res_df[['dists', 'simis', 'fold', 'abs_residual_levels', 'count_neighbor', 'std_cluster',
                           'catname', 'abs_residual']]
    tst_corr.dropna()
    count_num = tst_corr.groupby('abs_residual_levels').count()['dists'].values
    group_df = tst_corr.groupby('abs_residual_levels').mean()
    group_df['count'] = count_num
    # group_df.to_excel(f'./discussion/cv_group_df_{target}.xlsx', index=True)
    return group_df


def dump_model(reg_method, regr):
    filename = f'{target}_{reg_method}_regr.joblib'
    joblib.dump(regr, filename)


def load_model(reg_method):
    filename = f'{target}_{reg_method}_regr.joblib'
    loaded_model = joblib.load(filename)
    return loaded_model


def cross_validation():
    for percentage in percentage_ls:
        if target == 'roadway':
            feat2 = 'roadwayType'
        if target == 'intersection':
            feat2 = 'intersecType'
        with open(fname, 'a+') as f:
            f.write(
                f'................................{regr_ls[0]}...................................................\n')
            f.close()
        # ====================================== load data ======================================
        # dt, input_dt, cmfs, keep_cols = load_clearinghouse(cmf_thresh=[0, 2.0])
        dt, input_dt, cmfs, keep_cols = load_dt_more_vars(target=target, cmf_thresh=[0, 2.0],
                                                          manual_path=manual_path, non_tuning=non_tuning)
        dt, input_dt, cmfs = delete_outliers(dt, cmfs, keep_cols)
        per_fold = round(dt.shape[0] / 5)
        # corr_dict = dict({'fold': [], 'dist_corr': [], 'neigh_corr': [], "weighted_corr":[], 'disper_corr': [],
        #                   'dist_p': [], 'neigh_p': [], 'disper_p': [], "weighted_p":[]})
        tst_res_df, tst_group_df = pd.DataFrame(), pd.DataFrame()
        for fold in range(5):
            tst_index = range(fold * per_fold, min(dt.shape[0], (fold + 1) * per_fold))
            trn_index = list(set(range(dt.shape[0])) - set(tst_index))
            # ====================================== supervised learning ======================================
            model_name = backbone + '_' + loss_func + '_' + target + '_' + str(percentage) + apfix
            d2v_model = SentenceTransformer(f'data/model/' + model_name + '/')
            # device = torch.device(0 if torch.cuda.is_available() else "cpu")
            # d2v_model = SentenceTransformer('all-mpnet-base-v2', device=device)

            with open(fname, 'a+') as f:
                f.write('......................................\n')
                f.write(model_name + ':\n')
                f.close()

            cms_ls = get_str_inputs(input_dt)
            cms_vec_encode = d2v_model.encode(cms_ls)
            pca = PCA(n_components=0.9)
            cms_vec_encode = pca.fit(cms_vec_encode).transform(cms_vec_encode)
            if fold == 0 and (non_tuning is False):
                keep_cols.remove('cmName')
            x_trn_encoded, x_tst_encoded, cmfs_trn, cmfs_tst, encoder = encode_input(dt, cmfs, keep_cols, tst_index,
                                                                                     cms_vec_encode,
                                                                                     non_tuning=non_tuning)
            # ====================================== supervised learning ======================================

            for reg_method in regr_ls:
                if (pre_saved is True) and (shoulder_prediction is True):
                    regr = load_model(reg_method)
                    preds = regr.predict(x_tst_encoded)
                elif pre_saved is False:
                    preds, mse, regr = supervised_regression(x_trn_encoded, cmfs_trn, x_tst_encoded, cmfs_tst,
                                                             method=reg_method)
                tst_res = dt.iloc[tst_index]
                tst_res['cmf_preds'] = preds.reshape(-1, 1)
                preds_df = calculate_res(preds, cmfs_tst)
                tst_res_df = pd.concat([tst_res_df, preds_df], axis=0)

                with open(fname, 'a+') as f:
                    f.write(reg_method + ':\n')
                    mae, rmse, cr, pop = evaluate_metric(tst_res, f)
                    f.close()
                    mae_ls.append(mae)
                    rmse_ls.append(rmse)
                    cr_ls.append(cr)
                    pop_ls.append(pop)

            print(np.mean(mae_ls))
            print(np.mean(rmse_ls))
            print(np.mean(cr_ls))
            print(np.mean(pop_ls))

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            def get_group_statistics(res_df):
                x_cls_dist, affi_mtx, clusters = get_clusters(cms_vec_encode)
                avg_dist_tst_ls, avg_simi_tst_ls, avg_count_tst_ls, std_tst_ls = [], [], [], []
                for idx in tst_index:
                    tst_i_cluster = clusters.labels_[idx]
                    ngbr_flag = (clusters.labels_ == tst_i_cluster)
                    trn_flag = np.array([False] * ngbr_flag.shape[0])
                    trn_flag[trn_index] = True
                    ngbr_flag = trn_flag * ngbr_flag

                    ngbr_dists = x_cls_dist[idx, ngbr_flag]
                    avg_dist_tst_i = np.mean(ngbr_dists)
                    ngbr_simis = affi_mtx[idx, ngbr_flag]
                    avg_dist_simi_i = np.mean(ngbr_simis)
                    std_tst_i = np.std(ngbr_dists)
                    avg_count_tst_i = sum(ngbr_flag)

                    std_tst_ls.append(std_tst_i)
                    avg_dist_tst_ls.append(avg_dist_tst_i)
                    avg_simi_tst_ls.append(avg_dist_simi_i)
                    avg_count_tst_ls.append(avg_count_tst_i)

                tst_i_df = pd.DataFrame({'fold': [fold] * len(cmfs_tst),
                                         'cmf': cmfs_tst,
                                         'cmf_preds': preds,
                                         'abs_residual_levels': preds_df['Residual Level'],
                                         'dists': np.array(avg_dist_tst_ls),
                                         'simis': np.array(avg_simi_tst_ls),
                                         'count_neighbor': np.array(avg_count_tst_ls),
                                         'std_cluster': np.array(std_tst_ls),
                                         'abs_residual': np.abs(cmfs_tst - preds),

                                         })
                tst_i_df = pd.concat([tst_i_df, input_dt.iloc[tst_index]], axis=1)
                res_df = pd.concat([res_df, tst_i_df], axis=0)

                return res_df

        # dump_model(reg_method, regr)

        plot_res(tst_res_df['CMF Predictions'], tst_res_df['CMF True Values'],
                 fig_name='testing1')  # tst_group_df = get_group_statistics(tst_group_df)

    # tst_res_df.to_excel(f'./discussion/cv_test_res_df_{target}.xlsx', index=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # tst_mae_pivot = get_df_pivot(tst_res_df, feature_1="catname", feature_2=feat2, feature_3='MAE')
    # tst_mae_pivot.to_excel(f'./discussion/cv_tst_mae_pivot_{target}.xlsx', index=True)
    # plot_heatmap(tst_mae_pivot, feat2, 'MAE')


def plot_cross_validation():
    tst_dt, group_dt, cluster_dt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if target == 'roadway':
        feat2 = 'roadwayType'
    if target == 'intersection':
        feat2 = 'intersecType'

    for i in range(5):
        dt = pd.read_excel(f'./discussion/cv_tst_df_{target}_{str(i)}.xlsx')
        dt['Fold'] = i + 1
        dt['Facility'] = target.capitalize()
        evaluate_metric_cross_validation(dt)
        group = pd.read_excel(f'./discussion/cv_group_df_{target}_{str(i)}.xlsx')
        cluster = pd.read_excel(f'./discussion/cv_test_res_df_{target}_{str(i)}.xlsx')

        tst_dt = pd.concat([tst_dt, dt], axis=0)
        group_dt = pd.concat([group_dt, group], axis=0)
        cluster_dt = pd.concat([cluster_dt, cluster], axis=0)

    # cluster_corr = cluster_dt.corr()
    from scipy import stats
    for cluster_var in ['dists', 'count_neighbor', 'std_cluster']:
        cluster_dt.dropna(inplace=True)
        print(stats.pearsonr(cluster_dt['abs_residual'], cluster_dt[cluster_var]))
        print(stats.pearsonr(cluster_dt['abs_residual_levels'], cluster_dt[cluster_var]))

    if target == 'intersection':
        tst_dt.drop(tst_dt[tst_dt['index'] == 2244].index, inplace=True)
    tst_mae_pivot = get_df_pivot(tst_dt, feature_1="catname", feature_2=feat2, feature_3='MAE')
    plot_heatmap(tst_mae_pivot, feat2, 'MAE')

    # tst_dt3 = tst_dt[tst_dt['qualRating'] > 2]
    # tst_mae_pivot3 = get_df_pivot(tst_dt3, feature_1="catname", feature_2=feat2, feature_3='MAE')
    # plot_heatmap(tst_mae_pivot3, feat2, 'MAE')

    avg_group = group_dt.groupby('abs_residual_levels').mean()
    avg_group.to_excel(f'./discussion/cv_group_df_{target}_avg.xlsx')
    print('...')
    return tst_dt


def plot_cv_scatters(tst_df):
    tst_df.rename(columns={'cmf_preds': 'CMF prediction', 'cmf': 'CMF true values'}, inplace=True)
    sns.set(font_scale=1.4)
    # with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
    g = sns.FacetGrid(tst_df, col="Fold", row="Facility", margin_titles=True, despine=False)
    hue_order = sorted(tst_df['abs_residual_levels'].unique())
    g.map_dataframe(sns.scatterplot, x="CMF prediction", y="CMF true values",
                    hue="abs_residual_levels",
                    hue_order=hue_order,
                    palette=sns.hls_palette(s=.4, l=.5, n_colors=pd.unique(tst_df['abs_residual_levels']).shape[0] + 8))

    def const_line(*args, **kwargs):
        identity_line = np.linspace(0.0, max(max(tst_df["CMF prediction"]) + 0.2, max(tst_df["CMF true values"])))
        plt.plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)

    g.map(const_line)
    g.add_legend()
    g.figure.subplots_adjust(wspace=0.02, hspace=0.02)
    g.figure.savefig(f'cv_scatter.pdf', format='pdf')
    g.figure.show()


def plot_cv_subgroup(tst_df):
    if target == "roadway":
        row_val = "roadwayType"
    elif target == "intersection":
        row_val = "intersecType"
    g = sns.FacetGrid(tst_df, col="catname", row=row_val)
    g.map_dataframe(sns.histplot, x="total_bill", binwidth=2, binrange=(0, 60))


# for (row_val, col_val), ax in g.axes_dict.items():
#     if row_val == "Lunch" and col_val == "Female":
#         ax.set_facecolor(".95")
#     else:
#         ax.set_facecolor((0, 0, 0, 0))


def plot_subgroup_scatter(group_ls):
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        fig, axs = plt.subplots(nrows=2,
                                ncols=5,
                                layout="constrained",
                                figsize=(10, 5)
                                )
        for group_idx, (gdf, ax) in enumerate(zip(group_ls, axs.flat)):
            cmf_preds = gdf['cmf_preds']
            cmfs_tst = gdf['cmf']
            mae_ls = np.abs(cmf_preds - cmfs_tst)
            preds_df = pd.DataFrame.from_dict({'CMF Predictions': cmf_preds, 'CMF True Values': cmfs_tst,
                                               'mae': mae_ls, 'Residual Level': ['a'] * len(mae_ls)})
            preds_df.reset_index()
            range_ls = list(np.arange(0.0, np.max(mae_ls) + 0.1, 0.1))
            old_index = set()
            b = pd.DataFrame(['a'] * preds_df.shape[0], index=preds_df.index, columns=['class'])
            hue_order = []
            for idx, l in enumerate(range_ls):
                if idx > 0:
                    selected_index = set((preds_df[(preds_df['mae'] <= range_ls[idx])]).index)
                    use_index = selected_index - old_index
                    b.loc[list(use_index)] = f'[{str(range_ls[idx - 1])[:3]}, {str(range_ls[idx])[:3]}]'
                    hue_order.append(f'[{str(range_ls[idx - 1])[:3]}, {str(range_ls[idx])[:3]}]')
                    old_index = selected_index

            preds_df['Residual Level'] = b

            im = sns.scatterplot(data=preds_df, x="CMF Predictions", y="CMF True Values",
                                 hue=preds_df['Residual Level'],
                                 s=30,
                                 hue_order=hue_order,
                                 palette=sns.color_palette("Blues",
                                                           pd.unique(preds_df['Residual Level']).shape[0]).reverse(),
                                 ax=ax,
                                 # legend=False,
                                 )
            ax.legend().set_visible(False)
            if group_idx == 0:
                im.set_xticklabels([])
            elif group_idx in [1, 2, 3, 4]:
                im.set_xticklabels([])
                im.set_yticklabels([])

            elif group_idx in [6, 7, 8, 9]:
                im.set_yticklabels([])

            # ax.tick_params(
            #     axis='x',  # changes apply to the x-axis
            #     which='both',  # both major and minor ticks are affected
            #     bottom=False,  # ticks along the bottom edge are off
            #     top=False,  # ticks along the top edge are off
            #     labelbottom=False)  # labels along the bottom edge are off
            identity_line = np.linspace(0.0, max(max(cmfs_tst) + 0.2, max(cmf_preds)))
            ax.plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=2.0)
            ax.set(xlabel=None, ylabel=None)
            # ax.legend(loc='lower right', prop={'size': 7}, title='Residual Level')

    handles, labels = ax.get_legend_handles_labels()
    # mappable = im.get_children()[0]
    # fig.legend([handles[2], handles[1], handles[0]], [labels[2], labels[1], labels[0]], loc=9, ncol=2)
    fig.supylabel('Subgroup sample size')
    fig.supxlabel('Subgroup MAE')
    fig.legend(labels, loc='lower right', bbox_to_anchor=(1, -0.1), ncol=len(labels), bbox_transform=fig.transFigure)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.show()
    # fig.savefig(f'{target}_samplesize.pdf', format='pdf')
    print('...')


def eval_by_catname(tst_df):
    cat_mae_ls, cat_rmse_ls, cat_cr_ls, cat_pop_ls, tar_ls, cat_ls, count_ls = [], [], [], [], [], [], []
    with open(fname, 'a+') as f:
        for idx, tar_df in tst_df.groupby('Facility'):
            for idy, cat_df in tar_df.groupby('catname'):
                if cat_df.shape[0] != 0:
                    mae, rmse, cr, pop = evaluate_metric(cat_df, f)
                    tar_ls.append(idx)
                    cat_ls.append(idy)
                    count_ls.append(cat_df.shape[0])
                    cat_mae_ls.append(mae)
                    cat_rmse_ls.append(rmse)
                    cat_cr_ls.append(cr)
                    cat_pop_ls.append(pop)

    cat_df = pd.DataFrame({'Facility type': tar_ls,
                           'Countermeasure category': cat_ls,
                           'MAE': cat_mae_ls,
                           'RMSE': cat_rmse_ls,
                           'CR': cat_cr_ls,
                           'PoP': cat_pop_ls,
                           'Sample size': count_ls})
    cat_df.to_excel('eval_by_catname.xlsx', index=False)
    print('...')


def eval_clustering_statistic():
    tst_dt, group_dt, cluster_dt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for i in range(5):
        cms_vec = pd.read_excel(f'./discussion/cms_vec_encode_{target}_{str(i)}.xlsx')
        tst_index = pd.read_excel(f'./discussion/tst_index_{target}_{str(i)}.xlsx')
        pred_df = pd.read_excel(f'./discussion/cv_test_res_df_{target}_{str(i)}.xlsx')

        trn_index = list(set(range(cms_vec.shape[0])) - set(tst_index))
        x_cls_dist = metrics.pairwise_distances(cms_vec)

        affi_mtx = 1 / (x_cls_dist / np.max(x_cls_dist) + 1e-3)
        clusters = AffinityPropagation(affinity="precomputed").fit(affi_mtx)
        avg_dist_tst_ls, avg_count_tst_ls, std_tst_ls = [], [], []
        for idx in tst_index:
            tst_i_cluster = clusters.labels_[idx]
            ngbr_flag = (clusters.labels_ == tst_i_cluster)
            trn_flag = np.array([False] * ngbr_flag.shape[0])
            trn_flag[trn_index] = True
            ngbr_flag = trn_flag * ngbr_flag

            ngbr_dists = x_cls_dist[idx, ngbr_flag]
            avg_dist_tst_i = np.mean(ngbr_dists)
            std_tst_i = np.std(ngbr_dists)
            avg_count_tst_i = sum(ngbr_flag)

            std_tst_ls.append(std_tst_i)
            avg_dist_tst_ls.append(avg_dist_tst_i)
            avg_count_tst_ls.append(avg_count_tst_i)


def main(mode, tuning=True):
    # random.seed(2431)
    percentage_ls = [1.0]
    for percentage in percentage_ls:
        for target in targets:
            if target == 'roadway':
                seed = 112
                random.seed(seed)
                feat2_ls = ['roadwayType', 'areaType', 'crashSeverityKABCO', 'crashType']
                # feat2 = 'roadwayType'
                # feat2 = 'crashSeverityKABCO'
                # feat2 = 'country'
            if target == 'intersection':
                seed = 112
                random.seed(seed)
                # feat2 = 'intersecType'
                # feat2 = 'crashSeverityKABCO'
                # feat2 = 'country'
                feat2_ls = ['intersecType', 'areaType', 'crashSeverityKABCO', 'crashType']

            with open(fname, 'a+') as f:
                f.write('............................................................................\n')
                f.close()
            # ====================================== load data ======================================
            # dt, input_dt, cmfs, keep_cols = load_clearinghouse(cmf_thresh=[0, 2.0])
            dt, input_dt, cmfs, keep_cols = load_dt_more_vars(target=target, cmf_thresh=[0, 2.0])
            dt, input_dt, cmfs = delete_outliers(dt, cmfs, keep_cols)

            train_cols = keep_cols + ['cmName']
            dt_train = list(dt[train_cols].drop_duplicates().index)
            tst_index = random.sample(range(dt.shape[0]), round(0.2 * dt.shape[0]))
            trn_index = list(set(range(dt.shape[0])) - set(tst_index))
            # ====================================== supervised learning ======================================
            if mode == 'train':
                model_name = backbone + '_' + loss_func + '_' + target + '_' + str(percentage) + apfix
                tune_examples = load_samples(input_dt.iloc[trn_index], cmfs.iloc[trn_index],
                                             tune_percent=percentage, loss_string=loss_func, random_mode=True)
                d2v_model = tune_transformer(tune_examples, model_name=model_name, max_seq_len=96)

            elif mode == 'predict':
                if tuning is True:
                    model_name = backbone + '_' + loss_func + '_' + target + '_' + str(percentage) + apfix
                    d2v_model = SentenceTransformer(f'data/model/' + model_name + '/')
                else:
                    model_name = backbone
                    d2v_model = SentenceTransformer(model_name)

            with open(fname, 'a+') as f:
                f.write('......................................\n')
                f.write(model_name + ':\n')
                f.close()

            cms_ls = get_str_inputs(input_dt)
            cms_vec_encode = d2v_model.encode(cms_ls)
            # roadway =4, itersection =6
            # pca = PCA(n_components=0.9)
            pca = PCA(n_components=0.9)
            cms_vec_encode = pca.fit(cms_vec_encode).transform(cms_vec_encode)

            x_trn_encoded = cms_vec_encode[trn_index, :]
            x_tst_encoded = cms_vec_encode[tst_index, :]

            cmfs_trn = cmfs.iloc[trn_index]
            cmfs_tst = cmfs.iloc[tst_index]
            keep_cols.remove('cmName')
            x_trn_encoded, x_tst_encoded, cmfs_trn, cmfs_tst, encoder = encode_input(dt, cmfs, keep_cols, tst_index,
                                                                                     cms_vec_encode)
            # ====================================== apply trained model ======================================
            # ====================================== supervised learning ======================================

            for reg_method in ['RF']:
                preds, mse, regr = supervised_regression(x_trn_encoded, cmfs_trn, x_tst_encoded, cmfs_tst,
                                                         method=reg_method)
                tst_res = dt.iloc[tst_index]
                tst_res['cmf_preds'] = preds.reshape(-1, 1)
                preds_df = plot_res(preds, cmfs_tst, fig_name=model_name)

                with open(fname, 'a+') as f:
                    f.write(reg_method + ':\n')
                    evaluate_metric(tst_res, f)
                    f.write('\n')
                    f.close()

            tst_df = pd.DataFrame({'cmf': cmfs_tst, 'cmf_preds': preds,
                                   'abs_residual_levels': preds_df['Residual Level']})
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            x_cls_dist = metrics.pairwise_distances(cms_vec_encode)
            affi_mtx = 1 / (x_cls_dist / np.max(x_cls_dist) + 1e-3)
            clusters = AffinityPropagation(affinity="precomputed").fit(affi_mtx)
            avg_dist_tst_ls, avg_count_tst_ls, std_tst_ls = [], [], []
            for idx in tst_index:
                tst_i_cluster = clusters.labels_[idx]
                ngbr_flag = (clusters.labels_ == tst_i_cluster)
                trn_flag = np.array([False] * ngbr_flag.shape[0])
                trn_flag[trn_index] = True
                ngbr_flag = trn_flag * ngbr_flag

                ngbr_dists = x_cls_dist[idx, ngbr_flag]
                avg_dist_tst_i = np.mean(ngbr_dists)
                std_tst_i = np.std(ngbr_dists)
                avg_count_tst_i = sum(ngbr_flag)

                std_tst_ls.append(std_tst_i)
                avg_dist_tst_ls.append(avg_dist_tst_i)
                avg_count_tst_ls.append(avg_count_tst_i)

            tst_resdf = pd.DataFrame({'cmf': cmfs_tst, 'cmf_preds': preds,
                                      'abs_residual_levels': preds_df['Residual Level'],
                                      'dists': np.array(avg_dist_tst_ls),
                                      'count_neighbor': np.array(avg_count_tst_ls),
                                      'std_cluster': np.array(std_tst_ls),
                                      'abs_residual': np.abs(cmfs_tst - preds),
                                      'catname': input_dt.iloc[tst_index, :]['catname'],
                                      'subcatname': input_dt.iloc[tst_index, :]['subcatname'],
                                      'cmName': input_dt.iloc[tst_index, :]['cmName'],
                                      })

            # tst_cls_dist = metrics.pairwise_distances(cms_vec_encode)[tst_index]
            # tst_resdf = pd.DataFrame({'cmf': cmfs_tst, 'cmf_preds': preds, 'abs_residuals':preds_df['Residual Level'],
            #                           'dists': np.mean(np.sort(x_cls_dist[:, trn_index])[:, :5], axis=1)})
            count_num = tst_resdf.groupby('abs_residual_levels').count()['dists'].values
            group_df = tst_resdf.groupby('abs_residual_levels').mean()
            group_df['count'] = count_num
            # group_df.to_excel(f'./discussion/group_df_{target}_{str(seed)}.xlsx', index=True)
            print('...')

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            tst_df = pd.concat([tst_df, input_dt.iloc[tst_index]], axis=1)

            tst_mae_pivot = get_df_pivot(tst_df, feature_1="catname", feature_2='intersecType', feature_3='MAE')
            # tst_mae_pivot.to_excel(f'./discussion/tst_mae_pivot_{target}_{str(seed)}.xlsx', index=True)
            plot_heatmap(tst_mae_pivot, 'intersecType', 'MAE')

            with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
                ####################################################################################################
                def analyze_sample_size(feat, axe):
                    # tst_mae_pivot = get_df_pivot(tst_df, feature_1="subcatname", feature_2=feat, feature_3='MAE')
                    # plot_heatmap(tst_mae_pivot, feat2, 'MAE')
                    trn_df = pd.DataFrame({'cmf': cmfs_trn, 'cmf_preds': cmfs_trn})
                    trn_df = pd.concat([trn_df, input_dt.iloc[trn_index]], axis=1)
                    tst_mae_pivot = get_df_pivot(tst_df, feature_1="catname", feature_2=feat, feature_3='MAE')
                    tst_count_pivot = get_df_pivot(tst_df, feature_1="catname", feature_2=feat, feature_3='Count')
                    trn_count_pivot = get_df_pivot(trn_df, feature_1="catname", feature_2=feat, feature_3='Count')
                    ploral_keys = set(trn_count_pivot.keys()) - set(tst_count_pivot.keys())
                    trn_count_pivot.drop(list(ploral_keys), axis=1, inplace=True)
                    ploral_keys = set(tst_count_pivot.keys()) - set(trn_count_pivot.keys())
                    tst_count_pivot.drop(list(ploral_keys), axis=1, inplace=True)
                    tst_mae_pivot.drop(list(ploral_keys), axis=1, inplace=True)
                    trn_count_pivot.drop(index=set(trn_count_pivot.index) - set(tst_count_pivot.index), inplace=True)
                    group_name = feat + ',\n' + 'Countermeasure category)'
                    plot_mae_samplesize(axe, tst_mae_pivot, tst_count_pivot, trn_count_pivot, tst_df, group_name)

                ####################################################################################################
                fig1, axs1 = plt.subplots(nrows=len(feat2_ls),
                                          ncols=1,
                                          # layout="constrained"
                                          figsize=(5, 7)
                                          )
                for feat2, ax in zip(feat2_ls, axs1.flat):
                    # tst_mae_pivot = get_df_pivot(tst_df, feature_1="subcatname", feature_2=feat2, feature_3='MAE')
                    # plot_heatmap(tst_mae_pivot, feat2, 'MAE')
                    # ====================================== evaluate model performance ==============================
                    # test_1(tst_df)
                    analyze_sample_size(feat2, ax)
                handles, labels = ax.get_legend_handles_labels()
                # mappable = im.get_children()[0]
                fig1.legend([handles[2], handles[1], handles[0]], [labels[2], labels[1], labels[0]], loc=9, ncol=2)
                fig1.supylabel('Subgroup sample size')
                fig1.supxlabel('Subgroup MAE')
                fig1.show()
                fig1.savefig(f'{target}_samplesize.pdf', format='pdf')
                # ====================================== evaluate model performance =============================
                fig2, axs2 = plt.subplots(nrows=len(feat2_ls),
                                          ncols=1,
                                          layout="constrained",
                                          figsize=(10, 10)
                                          )

                def analyze_mae(feat, ax_mae, flag=False):
                    tst_mae_pivot = get_df_pivot(tst_df, feature_1=feat, feature_2='abs_residual_levels',
                                                 feature_3='Count')
                    names = ["\n".join(wrap(r, 20, break_long_words=False)) for r in
                             list(tst_mae_pivot.keys())]
                    im = sns.heatmap(tst_mae_pivot, cmap="YlGn", linewidths=.01, linecolor='grey',
                                     annot=True, fmt='.0f', xticklabels=flag, yticklabels=True, ax=ax_mae,
                                     annot_kws={'size': 8},
                                     vmin=0, vmax=300,
                                     cbar=False)
                    # gc.collections[0].set_clim(0, 300)
                    ax_mae.set_xlabel('')
                    return im

                for idx, (feat2, ax) in enumerate(zip(feat2_ls, axs2.flat)):
                    analyze_mae(feat2, ax)
                    if idx == (len(feat2_ls) - 1):
                        im = analyze_mae(feat2, ax, flag=True)

                mappable = im.get_children()[0]
                plt.colorbar(mappable, ax=axs2, orientation='vertical')
                plt.xticks(rotation=90)

                fig2.supxlabel('XLAgg')
                fig2.supylabel('YLAgg')
                fig2.show()

                # fig.text(0.5, 0.04, "Subgroup Test MAE Value", ha='center')
                # fig.text(0.04, 0.5, "Subgroup sample size", va='center', rotation='vertical')
                # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
                # plt.xlabel("common X")
                # plt.ylabel("common Y")

        print('...')


def plot_similarity_MAE_boxplot():
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        tst_res = pd.read_excel(f'discussion/cv_test_res_df_{target}.xlsx')
        # sns.boxplot(x='abs_residual_levels', y='dists', data=tst_res,
        #             order=list(np.sort(pd.unique(tst_res['abs_residual_levels']))))
        quantiles = np.percentile(tst_res['abs_residual'], [10, 20, 30, 40, 50, 60, 70, 80, 90])
        # Assign group labels based on quantile values
        group_labels = np.digitize(tst_res['abs_residual'], quantiles)
        tst_res['quantile'] = group_labels

        # Define custom intervals based on data distribution
        custom_intervals = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 2.0]  # Adjust as needed
        interval_ranges = []
        # Apply custom interval labels
        for i in range(len(custom_intervals) - 1):
            interval_ranges.append(
                '[' + str(round(custom_intervals[i], 2)) + ', ' + str(round(custom_intervals[i + 1], 2)) + ']')
        tst_res['interval'] = pd.cut(tst_res['abs_residual'], bins=custom_intervals,
                                     labels=interval_ranges)

        # sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2)
        sns.boxplot(x="interval", y="dists", data=tst_res, palette="colorblind", width=0.8, whis=2)
        quantile_ranges = ['[0%-10%]', '(10%-20%]', '(20%-30%]', '(30%-40%]', '(40%-50%]', '(50%-60%]', '(60%-70%]',
                           '(70%-80%]', '(80%-90%]', '(90%-100%]']
        # plt.xticks(np.arange(len(quantile_ranges)), quantile_ranges, rotation='vertical', fontsize=12)
        plt.xticks(np.arange(len(interval_ranges)), interval_ranges, rotation='vertical', fontsize=14)
        plt.yticks(fontsize=12)
        plt.xlabel("Absolute residual intervals", fontsize=16)
        plt.ylabel("Average similarity values", fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.title("Variation of average similarity values across Quantile Groups of Absolute Residuals")
        plt.tight_layout()
        # plt.savefig(f'{target}_simi_box.pdf', format='pdf', bbox_inches="tight")
        plt.show()
        print('...')


if __name__ == "__main__":
    # target = "roadway"
    report_gpu()
    fname = r'./result.txt'
    backbone = "all-mpnet-base-v2"
    apfix = '_more_vars'
    loss_func = "cos"
    percentage_ls = [1.0]
    regr_ls = ['MLP1']
    targets = ["roadway", "intersection"]

    shoulder_prediction = True
    pre_saved = True
    manual_path = True
    non_tuning = False

    if shoulder_prediction is True:
        targets = ["roadway"]
        manual_path = True
        # if non_tuning is True:
        #     regr_ls = ['RF1']
    cv_ls = []
    cv_df = pd.DataFrame()

    mae_ls, rmse_ls, cr_ls, pop_ls = [], [], [], []
    for target in targets:
        dt = pd.read_excel("discussion/cv_test_res_df_roadway.xlsx")
        dt = dt[dt['subcatname'] == "Shoulder width "]
        plot_res(dt['cmf_preds'], dt['cmf'], fig_name='general')
        with open(fname, 'a+') as f:
            evaluate_metric(dt, f)
        plot_similarity_MAE_boxplot()
        cross_validation()
        # cv_df = pd.concat([cv_df, plot_cross_validation()], axis=0)
        # main(mode='predict', tuning=True)
        # plot_subgroup_scatter(cv_ls)
        # plot_cv_scatters(cv_df)

        # eval_by_catname(cv_df)

        with open(fname, 'a+') as f:
            avg_mae, avg_rmse, avg_cr, avg_pop = np.nanmean(mae_ls), np.nanmean(rmse_ls), \
                                                 np.nanmean(cr_ls), np.nanmean(pop_ls)

            f.write(f'....................avg_results_{target}...................' + '\n')
            f.write('avg_mae = ' + str(avg_mae) + '\n')
            f.write('avg_rmse = ' + str(avg_rmse) + '\n')
            f.write('avg_cr = ' + str(avg_cr) + '\n')
            f.write('avg_pop = ' + str(avg_pop) + '\n')

            f.write(str(mae_ls))
            f.write(str(rmse_ls))
            f.write(str(cr_ls))
            f.write(str(pop_ls))

            f.write(str(avg_mae) + ", " + str(avg_rmse) + ", " + ", " + str(avg_cr) + ", " + str(avg_pop) + '\n')
            f.close()

    """
        dt = pd.read_excel(f'discussion/cv_test_res_df_{target}.xlsx')
        res_level_dict = dict()
        for (idx, x) in enumerate(list(np.sort(pd.unique(dt['abs_residual_levels'])))):
            res_level_dict[x] = idx
        dt['abs_level_order'] = dt['abs_residual_levels'].map(lambda x: res_level_dict[x], na_action='ignore')
        for val in ['dists', 'std_cluster', 'count_neighbor']:
            dt_corr = dt[[val, 'abs_level_order', 'damping', 'fold']]
            dt_corr.dropna(inplace=True)
            print(stats.pearsonr(dt_corr[dt_corr['damping'] == 0.5]['abs_level_order'],
                                 dt_corr[dt_corr['damping'] == 0.5][val]))
    """
