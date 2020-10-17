import os
import csv
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

TASK_DICT = {'imdb': 'cl', 'agnews': 'cl', 'mnli': 'et', 'rte': 'et'}
CLASS_DICT = {'imdb': 2, 'agnews': 4, 'mnli': 3, 'rte': 2}

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        for name, submodule in model.named_children():
            if submodule == target:
                if isinstance(model, nn.ModuleList):
                    model[int(name)] = replacement
                elif isinstance(model, nn.Sequential):
                    model[int(name)] = replacement
                else:
                    model.__setattr__(name, replacement)
                return True
            elif len(list(submodule.named_children())) > 0:
                if replace_in(submodule, target, replacement):
                    return True
        return False

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)

def visualize_heatmap(hm, start_idx, end_idx, text_words):
    # hm should be a 2d array
    hm = np.array(hm)
    trim_hm = hm[:, start_idx:end_idx]
    trim_len = end_idx - start_idx
    _, ax = plt.subplots()
    ax.imshow(trim_hm, cmap="viridis")
    ax.set_xticks(range(trim_len))
    ax.set_xticklabels(text_words[start_idx:end_idx])
    ax.set_yticks(range(len(hm)))
    ax.set_yticklabels(['layer{}'.format(i) for i in range(len(hm))])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
    plt.tight_layout()
    plt.show()

# Read data functions
def process_imdb(d):
    def read_imdb_data(subd):
        result = []
        for fn in os.listdir(subd):
            if fn.endswith('.txt'):
                text = open(os.path.join(subd, fn)).read().strip()
                result.append(text)
        return result
    def add_label(l, label):
        labeled = []
        for t in l:
            labeled.append((t, label))
        return labeled
    pos_data = read_imdb_data(os.path.join(d, 'pos'))
    neg_data = read_imdb_data(os.path.join(d, 'neg'))
    pos_labeled = add_label(pos_data, 0)
    neg_labeled = add_label(neg_data, 1)
    total_data = pos_labeled + neg_labeled
    return total_data

def process_agnews(fn):
    data = []
    with open(fn) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            label = int(row[0]) - 1
            content = ' '.join(row[1:])
            data.append((content, label))
    return np.array(data)

def process_mnli(fn):
    res = []
    labels = ["contradiction", "entailment", "neutral"]
    with open(fn) as fo:
        fo.readline()
        for line in fo:
            ele = line.strip().split('\t')
            s1, s2, label = ele[8], ele[9], ele[-1]
            label_idx = labels.index(label)
            res.append((s1, s2, label_idx))
    return res

def process_rte(fn):
    res = []
    with open(fn) as fo:
        fo.readline()
        for line in fo:
            idx, s1, s2, label = line.strip().split('\t')
            if label == 'entailment':
                label_idx = 0
            else:
                label_idx = 1
            res.append((s1, s2, label_idx))
    return res
