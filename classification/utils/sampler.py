from torch.utils.data import Sampler
import numpy as np

class BalancedSampler(Sampler):
    """Samples elements from [0, length) randomly without replacement.

    Parameters
    ----------
    length : int
        Length of the sequence.
    """
    def __init__(self, index_wrt_label):
        self._index_wrt_label = index_wrt_label
        self._label_num = len(self._index_wrt_label.keys())
        self._total = sum([len(self._index_wrt_label[key]) for key in self._index_wrt_label.keys()])

    def __iter__(self):
        indices = np.arange(self._total)
        for key in self._index_wrt_label.keys():
            np.random.shuffle(self._index_wrt_label[key])
        next_wrt_label = [0 for _ in range(self._label_num)]
        for ind in indices:
            key = ind % self._label_num
            pos = next_wrt_label[key]
            yield self._index_wrt_label[key][pos]
            pos += 1
            if pos >= len(self._index_wrt_label[key]):
                pos = 0
            next_wrt_label[key] = pos


    def __len__(self):
        return self._total