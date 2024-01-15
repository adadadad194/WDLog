import sys
# 将当前目录和父目录加入导包的路径
sys.path.extend([".", ".."])
from CONSTANTS import *
from sklearn.decomposition import FastICA
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from representations.sequences.statistics import Sequential_TF
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.AutoLabeling import Probabilistic_Labeling
from preprocessing.Preprocess import Preprocessor
from module.Optimizer import Optimizer
from module.Common import data_iter, generate_tinsts_binary_label, batch_variable_inst
from models.gru import AttGRUModel
from myutils.Vocab import Vocab
from models.widedeep import WideDeep
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score


# lstm_hiddens = 100
lstm_hiddens = 150
num_layer = 2
batch_size = 100
epochs = 300
estimators_nums = 150
#经过单次出现次数的person筛选出

co_occur = [[3, 5], [4, 5], [6, 7], [6, 11], [6, 12], [7, 11], [7, 12], [8, 9], [14, 24], [17, 32], [19, 20],
                [19, 25], [20, 25], [22, 25], [22, 26], [23, 24], [26, 27], [27, 28], [35, 36], [42, 43]]
bglco_occur = [[4,7],[20,31],[20,32],[20,33],[20,34],[20,35],[20,36],[20,37],[20,38],[20,39],[20,41],[21,22],[23,24],
               [23,25],[23,26],[23,27],[23,28],[24,25],[24,26],[24,27],[24,28],[25,26],[25,27],[25,28],[26,27],[26,28]]
event_list = [6, 7, 11, 12, 16, 32, 33]
bglevent_list = [19,20,29,30,31,32,33,34,35,36,37,38,39,40,41,127,128,129,130,131,132,133,134,135,136,137]
# class WDLog:
#     _logger = logging.getLogger('WDLog')
#     _logger.setLevel(logging.DEBUG)
#     console_handler = logging.StreamHandler(sys.stderr)
#     console_handler.setLevel(logging.DEBUG)
#     console_handler.setFormatter(
#         logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
#     file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'WDLog.log'))
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(
#         logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
#     _logger.addHandler(console_handler)
#     _logger.addHandler(file_handler)
#     _logger.info(
#         'Construct logger for WDLog succeeded, current working directory: %s, logs will be written in %s' %
#         (os.getcwd(), LOG_ROOT))
#
#     @property
#     def logger(self):
#         return WDLog._logger
#
#     def __init__(self, vocab, num_layer, hidden_size, label2id):
#         self.label2id = label2id
#         self.vocab = vocab
#         self.num_layer = num_layer
#         self.hidden_size = hidden_size
#         self.batch_size = 128
#         self.test_batch_size = 1024
#         self.model = AttGRUModel(vocab, self.num_layer, self.hidden_size)
#         if torch.cuda.is_available():
#             self.model = self.model.cuda(device)
#         # 二分类交叉熵
#         self.loss = nn.BCELoss()
#
#     def forward(self, inputs, targets):
#         tag_logits = self.model(inputs)
#         tag_logits = F.softmax(tag_logits, dim=1)
#         loss = self.loss(tag_logits, targets)
#         return loss
#
#     def predict(self, inputs, threshold=None):
#         with torch.no_grad():
#             tag_logits = self.model(inputs)
#             tag_logits = F.softmax(tag_logits)
#         if threshold is not None:
#             probs = tag_logits.detach().cpu().numpy()
#             anomaly_id = self.label2id['Anomalous']
#             pred_tags = np.zeros(probs.shape[0])
#             for i, logits in enumerate(probs):
#                 if logits[anomaly_id] >= threshold:
#                     pred_tags[i] = anomaly_id
#                 else:
#                     pred_tags[i] = 1 - anomaly_id
#
#         else:
#             pred_tags = tag_logits.detach().max(1)[1].cpu()
#         return pred_tags, tag_logits
#
#     def evaluate(self, instances, threshold=0.5):
#         self.logger.info('Start evaluating by threshold %.3f' % threshold)
#         with torch.no_grad():
#             self.model.eval()
#             globalBatchNum = 0
#             TP, TN, FP, FN = 0, 0, 0, 0
#             tag_correct, tag_total = 0, 0
#             for onebatch in data_iter(instances, self.test_batch_size, False):
#                 tinst = generate_tinsts_binary_label(onebatch, vocab, False)
#                 tinst.to_cuda(device)
#                 self.model.eval()
#                 pred_tags, tag_logits = self.predict(tinst.inputs, threshold)
#                 for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, processor.id2tag):
#                     tag_total += 1
#                     if bmatch:
#                         tag_correct += 1
#                         if inst.label == 'Normal':
#                             TN += 1
#                         else:
#                             TP += 1
#                     else:
#                         if inst.label == 'Normal':
#                             FP += 1
#                         else:
#                             FN += 1
#                 globalBatchNum += 1
#             self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
#             if TP + FP != 0:
#                 precision = 100 * TP / (TP + FP)
#                 recall = 100 * TP / (TP + FN)
#                 f = 2 * precision * recall / (precision + recall)
#                 end = time.time()
#                 self.logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f'
#                                  % (TP, (TP + FP), precision, TP, (TP + FN), recall, f))
#             else:
#                 self.logger.info('Precision is 0 and therefore f is 0')
#                 precision, recall, f = 0, 0, 0
#         return precision, recall, f


if __name__ == '__main__':
    # 解析命令行参数
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='BGL', type=str, help='Target dataset. Default HDFS')
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
                                         template_encoding=template_encoder.present, co_occur=bglco_occur, event_list=bglevent_list)
    num_classes = len(processor.train_event2idx)

    # Log sequence representation.
    sequential_encoder = Sequential_TF(processor.embedding)
    # 将日志序号转化为向量
    train_reprs = sequential_encoder.present(train)
    # 将向量绑定在instance实体上，repr属性即为语义向量
    for index, inst in enumerate(train):
        inst.repr = train_reprs[index]
    # dev_reprs = sequential_encoder.present(dev)
    # for index, inst in enumerate(dev):
    #     inst.repr = dev_reprs[index]
    test_reprs = sequential_encoder.present(test)
    for index, inst in enumerate(test):
        inst.repr = test_reprs[index]
    # 开始降维
    # Dimension reduction if specified.
    transformer = None
    if reduce_dimension != -1:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer = FastICA(n_components=reduce_dimension)
        train_reprs = transformer.fit_transform(train_reprs)
        for idx, inst in enumerate(train):
            inst.repr = train_reprs[idx]
        print('Finished at %.2f' % (time.time() - start_time))

    # Probabilistic labeling.
    # Sample normal instances.
    # 提取出训练集中标签为正常的数据
    train_normal = [x for x, inst in enumerate(train) if inst.label == 'Normal']
    normal_ids = train_normal[:int(0.5 * len(train_normal))]
    label_generator = Probabilistic_Labeling(min_samples=min_samples, min_clust_size=min_cluster_size,
                                             res_file=prob_label_res_file, rand_state_file=rand_state)
    # labeled_train带有概率标签的instance集合
    labeled_train = label_generator.auto_label(train, normal_ids)

    # Below is used to test if the loaded result match the original clustering result.
    TP, TN, FP, FN = 0, 0, 0, 0

    for inst in labeled_train:
        if inst.predicted == 'Normal':
            if inst.label == 'Normal':
                TN += 1
            else:
                FN += 1
        else:
            if inst.label == 'Anomalous':
                TP += 1
            else:
                FP += 1
    from myutils.common import get_precision_recall

    print(len(normal_ids))
    print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
    p, r, f = get_precision_recall(TP, TN, FP, FN)
    print('%.4f, %.4f, %.4f' % (p, r, f))

    # Load Embeddings
    vocab = Vocab()
    vocab.load_from_dict(processor.embedding)



    gb = GradientBoostingClassifier(n_estimators=estimators_nums,
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
    gb_feat_shape = train_gbdt_feature.shape[1]
    # 将GBDT的特征绑定回数据集
    for inst, feat in zip(train, train_gbdt_feature):
        inst.gbdt_feat = feat.tolist()
    for inst, feat in zip(dev, dev_gbdt_feature):
        inst.gbdt_feat = feat.tolist()

    WDLog = WideDeep(deep_hidden_size=lstm_hiddens, deep_vocab=vocab, deep_num_layers=num_layer,
                      wide_input_size=len(train[0].event_list_nums) + len(train[0].occur_nums) + len(train[0].gbdt_feat), num_classes=2).cuda(device)

    log = 'layer={}_hidden={}_epoch={}'.format(num_layer, lstm_hiddens, epochs)
    best_model_file = os.path.join(output_model_dir, log + '_best.pt')
    last_model_file = os.path.join(output_model_dir, log + '_last.pt')

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if mode == 'train':
        # Train
        optimizer = Optimizer(filter(lambda p: p.requires_grad, WDLog.dnn_network.parameters()))
        bestClassifier = None
        global_step = 0
        bestF = 0
        batch_num = int(np.ceil(len(labeled_train) / float(batch_size)))

        for epoch in range(epochs):
            WDLog.train()
            start = time.strftime("%H:%M:%S")
            WDLog.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                               (epoch + 1, start, optimizer.lr))
            batch_iter = 0
            correct_num, total_num = 0, 0
            # start batch
            for onebatch in data_iter(labeled_train, batch_size, True):
                WDLog.train()
                # 将一批instance转化成一批Tensor instance
                tinst = generate_tinsts_binary_label(onebatch, vocab)
                tinst.to_cuda(device)
                loss = WDLog.forward(tinst.inputs, tinst.wide_input, tinst.targets)
                loss_value = loss.data.cpu().numpy()
                # 计算梯度
                loss.backward()
                if batch_iter % 100 == 0:
                    WDLog.logger.info("Step:%d, Iter:%d, batch:%d, loss:%.2f" \
                                       % (global_step, epoch, batch_iter, loss_value))
                batch_iter += 1
                # 每10次梯度裁剪和更新一次参数，或者最后一次迭代的时候更新参数。
                if batch_iter % 1 == 0 or batch_iter == batch_num:
                    # 梯度裁剪，缓解梯度爆炸和梯度消失
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, WDLog.parameters()),
                        max_norm=1)
                    # 更新参数
                    optimizer.step()
                    WDLog.zero_grad()
                    global_step += 1
                if dev:
                    # 每500词或者最后一次测试dev,dev用来训练时评估模型，test在训练结束时进行评估。
                    if batch_iter % 500 == 0 or batch_iter == batch_num:
                        WDLog.logger.info('Testing on test set.')
                        _, _, f = WDLog.evaluate(dev)
                        if f > bestF:
                            WDLog.logger.info("Exceed best f: history = %.2f, current = %.2f" % (bestF, f))
                            torch.save(WDLog.state_dict(), best_model_file)
                            bestF = f
            WDLog.logger.info('Training epoch %d finished.' % epoch)
            torch.save(WDLog.state_dict(), last_model_file)

    if os.path.exists(last_model_file):
        WDLog.logger.info('=== Final Model ===')
        WDLog.load_state_dict(torch.load(last_model_file))
        WDLog.evaluate(test, threshold)
    if os.path.exists(best_model_file):
        WDLog.logger.info('=== Best Model ===')
        WDLog.load_state_dict(torch.load(best_model_file))
        WDLog.evaluate(test, threshold)
    WDLog.logger.info('All Finished')
