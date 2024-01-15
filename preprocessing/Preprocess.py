from CONSTANTS import *
from entities.instances import Instance
from preprocessing.dataloader.BGLLoader import BGLLoader
from preprocessing.dataloader.HDFSLoader import HDFSLoader


class Preprocessor:
    def __init__(self):
        # process函数中赋值HDFSLoader
        self.dataloader = None
        # 以下两个都在update_event2idx_mapping中获取
        self.train_event2idx = {}
        self.test_event2idx = {}
        # 以下三个都是dataloader运行结束后，通过update_dicts函数赋值
        self.id2label = {}
        self.label2id = {}
        self.templates = []
        self.embedding = None
        # 已被drain提取好事件的处理好的输入数据 datasets/HDFS/inputs/IBM
        self.base = None
        # HDFS
        self.dataset = None
        # IBM
        self.parsing = None
        self.tag2id = {'Normal': 0, 'Anomalous': 1}
        self.id2tag = {0: 'Normal', 1: 'Anomalous'}
        self.logger = self._set_logger()
        pass

    def _set_logger(self):
        # Dispose Loggers.
        PreprocessorLogger = logging.getLogger('Preprocessor')
        PreprocessorLogger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Preprocessor.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        PreprocessorLogger.addHandler(console_handler)
        PreprocessorLogger.addHandler(file_handler)
        PreprocessorLogger.info(
            'Construct PreprocessorLogger success, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))
        return PreprocessorLogger

    def process(self, dataset, parsing, template_encoding, cut_func, co_occur, event_list):
        # 已被drain提取好事件的处理好的输入数据
        self.base = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + '/inputs/' + parsing)
        self.dataset = dataset
        self.parsing = parsing
        dataloader = None
        parser_config = None
        parser_persistence = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + '/persistences')

        if dataset == 'HDFS':
            # 提取HDFS信息，转化为 Blk:日志序列的格式。
            dataloader = HDFSLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/HDFS/HDFS.log'),
                                    semantic_repr_func=template_encoding)
            # 为drain配置
            parser_config = os.path.join(PROJECT_ROOT, 'conf/HDFS.ini')
        elif dataset == 'BGL' or dataset == 'BGLSample':
            in_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + '/' + dataset + '.log')
            dataset_base = os.path.join(PROJECT_ROOT, 'datasets/' + dataset)
            dataloader = BGLLoader(in_file=in_file, dataset_base=dataset_base,
                                   semantic_repr_func=template_encoding)
            parser_config = os.path.join(PROJECT_ROOT, 'conf/BGL.ini')

        self.dataloader = dataloader

        if parsing == 'IBM':
            # drain解析，将日志中变量部分用掩码替换，将日志归纳为通用的几个模板。
            self.dataloader.parse_by_IBM(config_file=parser_config, persistence_folder=parser_persistence)
        else:
            self.logger.error('Parsing method %s not implemented yet.')
            raise NotImplementedError
        return self._gen_instances(co_occur, event_list, cut_func=cut_func)


    def _gen_instances(self, co_occur, event_list, cut_func=None):
        self.logger.info('Start preprocessing dataset %s by parsing method %s' % (self.dataset, self.parsing))
        instances = []
        if not os.path.exists(self.base):
            os.makedirs(self.base)
        train_file = os.path.join(self.base, 'train')
        dev_file = os.path.join(self.base, 'dev')
        test_file = os.path.join(self.base, 'test')

        self.logger.info('Start generating instances.')
        # Prepare semantic embedding sequences for instances.
        for block in tqdm(self.dataloader.blocks):
            if block in self.dataloader.block2eventseq.keys() and block in self.dataloader.block2label.keys():
                id = block
                label = self.dataloader.block2label[id]
                inst = Instance(id, self.dataloader.block2eventseq[id], label)
                inst.get_len()
                inst.count_event(self.dataloader.cluster_len)
                inst.set_occur(co_occur)
                inst.set_event(event_list)
                instances.append(inst)
            else:
                self.logger.error('Found mismatch block: %s. Please check.' % block)
        self.embedding = self.dataloader.id2embed
        train, dev, test = cut_func(instances)
        # 各个集合中的标签分布情况
        self.label_distribution(train, dev, test)
        # 写入保存分割好的训练集测试集开发集
        self.record_files(train, train_file, dev, dev_file, test, test_file)
        # 将dataloader的结果写入Preprocess类
        self.update_dicts()
        # 利用dataloader得到train, dev, test即可删除，释放内存。
        self.update_event2idx_mapping(train, test)
        del self.dataloader
        gc.collect()
        return train, dev, test

    # 将dataloader的结果写入Preprocess类
    def update_dicts(self):
        self.id2label = self.dataloader.id2label
        self.label2id = self.dataloader.label2id
        self.templates = self.dataloader.templates

    # 写入保存分割好的训练集测试集开发集
    def record_files(self, train, train_file, dev, dev_file, test, test_file, pretrain_source=None):
        with open(train_file, 'w', encoding='utf-8') as writer:
            for instance in train:
                writer.write(str(instance) + '\n')
        if dev:
            with open(dev_file, 'w', encoding='utf-8') as writer:
                for instance in dev:
                    writer.write(str(instance) + '\n')
        with open(test_file, 'w', encoding='utf-8') as writer:
            for instance in test:
                writer.write(str(instance) + '\n')
        if pretrain_source:
            with open(pretrain_source, 'w', encoding='utf-8') as writer:
                for inst in train:
                    writer.write(' '.join([str(x) for x in inst.sequence]) + '\n')

    # 各个集合中的标签分布情况
    def label_distribution(self, train, dev, test):
        train_label_counter = Counter([inst.label for inst in train])
        if dev:
            dev_label_counter = Counter([inst.label for inst in dev])
            self.logger.info('Dev: %d Normal, %d Anomalous instances.', dev_label_counter['Normal'],
                             dev_label_counter['Anomalous'])
        test_label_counter = Counter([inst.label for inst in test])
        self.logger.info('Train: %d Normal, %d Anomalous instances.', train_label_counter['Normal'],
                         train_label_counter['Anomalous'])
        self.logger.info('Test: %d Normal, %d Anomalous instances.', test_label_counter['Normal'],
                         test_label_counter['Anomalous'])

    # pre是train post是test，目的是计算事件的维度并集
    def update_event2idx_mapping(self, pre, post):
        '''
        Calculate unique events in pre & post for event count vector calculation.
        :param pre: pre data, including training set and validation set(if has)
        :param post: post data, mostly testing set
        :return: update mappings in self
        '''
        self.logger.info('Update train instances\' event-idx mapping.')
        # 事件Set之后转化为list从小到大排序
        pre_ordered_events = self._count_events(pre)
        # 事件的总数量
        embed_size = len(pre_ordered_events)
        self.logger.info('Embed size: %d in pre dataset.' % embed_size)
        for idx, event in enumerate(pre_ordered_events):
            self.train_event2idx[event] = idx
        self.logger.info('Update test instances\' event-idx mapping.')
        post_ordered_events = self._count_events(post)
        base = len(pre_ordered_events)
        increment = 0
        for event in post_ordered_events:
            # 如果test集合中出现了train集合中没有的事件
            if event not in pre_ordered_events:
                pre_ordered_events.append(event)
                # 在和Train前面预留相同空间的情况下，后续追加映射，避免位置相同发生映射冲突
                self.test_event2idx[event] = base + increment
                increment += 1
            else:
                # test和train集合统一映射
                self.test_event2idx[event] = self.train_event2idx[event]
        embed_size = len(pre_ordered_events)
        # 嵌入的维度是测试集+训练集的并集
        self.logger.info('Embed size: %d in pre+post dataset.' % embed_size)
        pass

    # 将事件去重后从大到小排序
    def _count_events(self, sequence):
        events = set()
        for inst in sequence:
            for event in inst.sequence:
                events.add(int(event))
        ordered_events = sorted(list(events))
        return ordered_events
