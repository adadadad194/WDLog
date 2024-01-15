import sys
# 将当前目录和父目录加入导包的路径
sys.path.extend([".", ".."])
from CONSTANTS import *
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.Preprocess import Preprocessor
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score

co_occur = [[3, 5], [4, 5], [6, 7], [6, 11], [6, 12], [7, 11], [7, 12], [8, 9], [14, 24], [17, 32], [19, 20],
                [19, 25], [20, 25], [22, 25], [22, 26], [23, 24], [26, 27], [27, 28], [35, 36], [42, 43]]

event_list = [6, 7, 11, 12, 16, 32, 33]

if __name__ == '__main__':
    # 解析命令行参数
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='HDFS', type=str, help='Target dataset. Default HDFS')
    argparser.add_argument('--mode', default='train', type=str, help='train or test')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--min_cluster_size', type=int, default=100,
                           help="min_cluster_size.")
    argparser.add_argument('--min_samples', type=int, default=100,
                           help="min_samples")
    argparser.add_argument('--reduce_dimension', type=int, default=50,
                           help="Reduce dimentsion for fastICA, to accelerate the HDBSCAN probabilistic label estimation.")
    argparser.add_argument('--threshold', type=float, default=0.5,
                           help="Anomaly threshold for WDLog.")
    args, extra_args = argparser.parse_known_args()

    dataset = args.dataset
    parser = args.parser
    mode = args.mode
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    reduce_dimension = args.reduce_dimension
    threshold = args.threshold

    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    base = os.path.join(PROJECT_ROOT, 'datasets/' + dataset)
    output_model_dir = os.path.join(save_dir, 'models/WDLog/' + dataset + '_' + parser + '/model')
    output_res_dir = os.path.join(save_dir, 'results/WDLog/' + dataset + '_' + parser + '/detect_res')
    prob_label_res_file = os.path.join(save_dir,
                                       'results/WDLog/' + dataset + '_' + parser +
                                       '/prob_label_res/mcs-' + str(min_cluster_size) + '_ms-' + str(min_samples))
    rand_state = os.path.join(save_dir,
                              'results/WDLog/' + dataset + '_' + parser +
                              '/prob_label_res/random_state')
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')

    # Training, Validating and Testing instances.
    # 根据数据集选择编码器
    template_encoder = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    # 数据预处理
    processor = Preprocessor()
    # 使用HDFS数据集，dataset:HDFS,Parser:IBM(DRAIN的一种实现方式),切割比例：613, 使用TF—IDF和word2vec进行词嵌入。
    train, dev, test = processor.process(dataset=dataset, parsing=parser, cut_func=cut_by_613,
                                         template_encoding=template_encoder.present, co_occur=co_occur, event_list=event_list)

    # GBDT二分类，召回率70，准确度90+
    # gb = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=5, subsample=1
    #                                   , min_samples_split=2, min_samples_leaf=1, max_depth=3
    #                                   , init=None, random_state=None, max_features=None
    #                                   , verbose=1, max_leaf_nodes=None, warm_start=False
    #                                   )
    # train_gbdt_input = np.array([x.event_list_nums + x.occur_nums for x in train])
    # train_gbdt_label = np.array([mytag2id[x.label] for x in train])
    # test_gbdt_input = np.array([x.event_list_nums + x.occur_nums for x in test])
    # test_gbdt_label = np.array([mytag2id[x.label] for x in test])
    # print(train_gbdt_input.shape, train_gbdt_label.shape, test_gbdt_input.shape, test_gbdt_label.shape)
    # gb.fit(train_gbdt_input, train_gbdt_label)
    # pred = gb.predict(test_gbdt_input)
    #
    # total_err = 0
    # num = 0
    # for i in range(pred.shape[0]):
    #     print(total_err, pred[i], test_gbdt_label[i])
    #     if test_gbdt_label[i] == 1:
    #         num += 1
    #         err = (pred[i] - test_gbdt_label[i])
    #         total_err += err * err
    # print(total_err / num)
    gb = GradientBoostingClassifier(n_estimators=50,
                                max_depth=6,
                                max_features=0.8, random_state=0, verbose=1, validation_fraction=0.3)

    train_gbdt_input = np.array([x.event_list_nums + x.occur_nums for x in train])
    train_gbdt_label = np.array([mytag2id[x.label] for x in train])
    dev_gbdt_input = np.array([x.event_list_nums + x.occur_nums for x in dev])
    dev_gbdt_label = np.array([mytag2id[x.label] for x in dev])
    print(train_gbdt_input.shape, train_gbdt_label.shape, dev_gbdt_input.shape, dev_gbdt_label.shape)
    gb.fit(train_gbdt_input, train_gbdt_label)

    print("GB AUC - ROC : ", roc_auc_score(dev_gbdt_label, gb.predict(dev_gbdt_input)))
    train_leaf_feature = gb.apply(train_gbdt_input)[:, :, 0]
    dev_leaf_feature = gb.apply(dev_gbdt_input)[:, :, 0]
    enc = OneHotEncoder()
    enc.fit(train_leaf_feature)

    train_gbdt_feature = np.array(enc.transform(train_leaf_feature).toarray())
    dev_gbdt_feature = np.array(enc.transform(dev_leaf_feature).toarray())

    sel = VarianceThreshold(threshold=(.99 * (1 - .99)))
    sel.fit(train_gbdt_feature)
    train_gbdt_feature = sel.transform(train_gbdt_feature)
    dev_gbdt_feature = sel.transform(dev_gbdt_feature)
    # gb_feat_shape = train_gbdt_feature.shape[1]
    # 将GBDT的特征绑定回数据集
    for inst, feat in zip(train, train_gbdt_feature):
        inst.gbdt_feat = feat
    for inst, feat in zip(train, dev_gbdt_feature):
        inst.gbdt_feat = feat