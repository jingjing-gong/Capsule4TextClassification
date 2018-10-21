import _pickle as pkl
import numpy as np
from multiprocessing import Queue

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, datapath, batch_size=128, bucket_sz=1000, shuffle=False, sample_balance=False, id2weight=None):

        with open(datapath, 'rb') as fd:
            data = pkl.load(fd)
        '''data==> [(labe, doc),]'''
        example_num = len(data)
        '''shape(example_num)'''
        doc_sz = np.array([len(doc) for _, doc in data], dtype=np.int32)

        if shuffle:
            self.tidx = np.argsort(doc_sz)
        else:
            self.tidx = np.arange(example_num)

        self.num_example = example_num
        self.shuffle = shuffle
        self.bucket_sz = bucket_sz
        self.batch_sz = batch_size
        self.data = data

        self.sample_balance = sample_balance
        self.id2weight = id2weight

    def __iter__(self):
        if self.bucket_sz < self.batch_sz:
            self.bucket_sz = self.batch_sz
        if self.bucket_sz > self.num_example:
            self.bucket_sz = self.num_example
        self.startpoint = 0
        return self

    def __next__(self):
        if self.startpoint >= self.num_example:
            raise StopIteration

        if self.shuffle:
            bucket_start = np.random.randint(0, self.num_example)
            bucket_end = (bucket_start + self.bucket_sz) % self.num_example
            if bucket_end - bucket_start < self.bucket_sz:
                candidate = np.concatenate([self.tidx[bucket_start:], self.tidx[:bucket_end]])
            else:
                candidate = self.tidx[bucket_start: bucket_end]
            candidate_p = None
            if self.sample_balance and self.id2weight:
                candidate_label = [self.data[c][0] for c in candidate]
                candidate_p = np.array([self.id2weight[l] for l in candidate_label])
                candidate_p = candidate_p/np.sum(candidate_p)
            target_idx = np.random.choice(candidate, size=self.batch_sz, p=candidate_p)
        else:
            target_idx = self.tidx[self.startpoint:self.startpoint+self.batch_sz]
        self.startpoint += self.batch_sz

        labels = []
        data_x = []
        for idx in target_idx:
            l, d = self.data[idx]
            labels.append(l)
            data_x.append(d)
        return labels, data_x


def preparedata(dataset: list, q: Queue, max_wd_num: int, class_freq: dict):
    for labels, data_x in dataset:
        example_weight = np.array([class_freq[i] for i in labels])    #(b_sz)
        data_batch, wNum = paddata(data_x, max_wd_num=max_wd_num)
        labels = np.array(labels)
        q.put((data_batch, labels, wNum, example_weight))
    q.put(None)


def paddata(data_x: list, max_wd_num: int):
    '''

    :param data_x: (b_sz, wd_num)
    :param max_wd_num:
    :return: b shape(b_sz, wd_sz), wNum shape(b_sz,)
    '''

    b_sz = len(data_x)

    wd_num = np.array([len(doc) if len(doc) < max_wd_num else max_wd_num for doc in data_x], dtype=np.int32)
    wd_sz = np.max(wd_num)

    b = np.zeros(shape=[b_sz, wd_sz], dtype=np.int32)  # == PAD
    wNum = wd_num
    for i, snt in enumerate(data_x):
        for j, wd in enumerate(snt):
            if j >= wd_sz:
                continue
            b[i, j] = wd


    return b, wNum