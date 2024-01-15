import torch

from models.gru import AttGRUModel
from CONSTANTS import *
from module.Common import data_iter, generate_tinsts_binary_label, batch_variable_inst

class WideDeep(nn.Module):
    def __init__(self, deep_hidden_size=None, deep_vocab=None, deep_num_layers=None, wide_input_size=None, num_classes=2):
        super(WideDeep, self).__init__()


        self.dnn_network = AttGRUModel(deep_vocab, deep_num_layers, deep_hidden_size)
        self.vocab = deep_vocab
        # if torch.cuda.is_available():
        #     self.dnn_network = self.dnn_network.cuda(device)

        self.fc = nn.Linear(2 * deep_hidden_size + wide_input_size, num_classes)

        self.logger = self._set_logger()
        # 二分类交叉熵
        self.loss = nn.BCELoss()
        self.id2tag = {0: 'Normal', 1: 'Anomalous'}
        self.label2id = {'Normal': 0, 'Anomalous': 1}
        self.test_batch_size = 1024

    def _set_logger(self):
        # 设置Logger
        WDLogger= logging.getLogger('wdLog')
        WDLogger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'WDLog.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
        WDLogger.addHandler(console_handler)
        WDLogger.addHandler(file_handler)
        WDLogger.info(
            'Construct logger for myLog succeeded, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))
        return WDLogger

    # def forward(self, deep_input):
    def forward(self, deep_input=None, wide_input=None, targets=None):
        # Deep
        deep_out = self.dnn_network(deep_input)
        # wide
        if wide_input is not None:
            output = torch.concat((deep_out, wide_input), dim=1)

        tag_logits = self.fc(output)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits,  targets)
        return loss

    def predict(self, inputs, wide_input, threshold=None):
        with torch.no_grad():
            deep_out = self.dnn_network(inputs)
            tag_logits = torch.concat((deep_out, wide_input), dim=1)
            tag_logits = self.fc(tag_logits)
            tag_logits = F.softmax(tag_logits)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomalous']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id

        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def evaluate(self, instances, threshold=0.5):
        self.logger.info('Start evaluating by threshold %.3f' % threshold)
        with torch.no_grad():
            self.eval()
            globalBatchNum = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            tag_correct, tag_total = 0, 0
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, self.vocab, False)
                tinst.to_cuda(device)
                pred_tags, tag_logits = self.predict(tinst.inputs, tinst.wide_input, threshold)
                for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, self.id2tag):
                    tag_total += 1
                    if bmatch:
                        tag_correct += 1
                        if inst.label == 'Normal':
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == 'Normal':
                            FP += 1
                        else:
                            FN += 1
                globalBatchNum += 1
            self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
            if TP + FP != 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
                end = time.time()
                self.logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f'
                                 % (TP, (TP + FP), precision, TP, (TP + FN), recall, f))
            else:
                self.logger.info('Precision is 0 and therefore f is 0')
                precision, recall, f = 0, 0, 0
        return precision, recall, f

