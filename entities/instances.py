import hashlib
from collections import Counter


class Instance:
    def __init__(self, block_id, log_sequence, label):
        self.id = block_id
        self.sequence = log_sequence
        self.label = label
        self.repr = None
        self.predicted = ''
        self.confidence = 0
        self.semantic_emb_seq = []
        self.context_emb_seq = []
        self.semantic_emb = None
        self.encode = None
        self.semantic_repr = []
        self.context_repr = []
        self.seq_len = None
        self.event_nums = None
        self.occur_nums = None
        self.event_list_nums = None
        self.gbdt_feat = None

    def count_event(self, cluster_len):
        # 只使用1~末尾
        self.event_nums = [0]*(cluster_len+1)
        # 0 用来统计seq长度
        self.event_nums[0] = self.seq_len
        for event in self.sequence:
            self.event_nums[event] += 1

    def set_occur(self, co_occur):
        self.occur_nums = [0]*len(co_occur)
        for x in range(len(co_occur)):
            if self.event_nums[co_occur[x][1]] > 0 or self.event_nums[co_occur[x][0]] > 0:
                if self.event_nums[co_occur[x][1]] != self.event_nums[co_occur[x][0]]:
                    equal = 1
                else:
                    equal = 0
            else:
                equal = 0
            self.occur_nums[x] = equal

    def set_event(self, event_list):
        self.event_list_nums = [0]*len(event_list)
        for x in range(len(event_list)):
            self.event_list_nums[x] = self.event_nums[event_list[x]]

    def __str__(self):
        sequence_str = ' '.join([str(x) for x in self.sequence])
        if self.predicted == '':
            return sequence_str + '\n' \
                   + str(self.id) + ',' + self.label + '\n' \
                   + str(self.event_list_nums) + ',' + str(self.seq_len) + '\n' \
                   + str(self.occur_nums) + '\n'
        else:
            return sequence_str + '\n' \
                   + str(self.id) + ',' + self.label + ',' + self.predicted + ',' + str(self.confidence) + '\n' \
                   + str(self.event_list_nums) + ',' + str(self.seq_len) + '\n' \
                   + str(self.occur_nums) + '\n'
        pass

    def __hash__(self):
        return hashlib.md5(str(self).encode('utf-8')).hexdigest()

    @property
    def seq_hash(self):
        return hash(' '.join([str(x) for x in self.sequence]))

    @property
    def event_count(self):
        return Counter(self.sequence)

    def get_len(self):
        self.seq_len = len(self.sequence)
        return self.seq_len

class Log_With_Datetime:
    def __init__(self, idx, label, datetime, message):
        self.id = idx
        self.label = label
        self.time = datetime
        self.message = message


class Log_Time_Step:
    def __init__(self, logs):
        self.logs = logs
        self.sequence = [log.id for log in self.logs]
        self.label = 'Normal'
        for log in self.logs:
            if log.label == 'Anomalous':
                self.label = 'Anomalous'
                break
