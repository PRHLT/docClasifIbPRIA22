from __future__ import print_function
from __future__ import division
import torch
from torch.utils.data import Dataset
import logging
import pytorch_lightning as pl
import numpy as np
from functools import wraps
from time import time

class tDataset(Dataset):

    def __init__(self, data, logger=None, transform=None, sequence=True, lime=False):
        """
        data es el array "data" que proviene de la funcion load que te copio aqui abajo.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform
        self.data = []
        self.ids = []
        self.lime = lime
        for i in range(len(data)):
            self.ids.append(data[i][-1])
            self.data.append(
                (data[i][:-2], data[i][-2]) # ([array de caracts.], clase)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        data = [0,1,2,3,4,5,6]
        bs = 2

        2,6

        get(2)
        get(6)

        """
        data_row = self.data[idx]
        info, labels = data_row
        if self.lime:
            info = torch.tensor(info, dtype=torch.float)
            # labels = torch.tensor(int(labels))
        else:
            info = torch.tensor(info, dtype=torch.float)
        labels = torch.tensor(int(labels))
        sample = {
            "row": info,
            "label": labels,
            "id": self.ids[idx],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


def load_dict_class(path):
    res = {}
    number_to_class = {}
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip()
        legajo, c = line.split(" ")
        c = int(c)
        res[legajo] = c
        number_to_class[c] = legajo
    return res, number_to_class

def load_RWs(opts):
    """
    # Expd-ID		  10k feats LABEL
    :param opts:
    :return:
    """
    def load(path):
        print("Loading {}".format(path))
        f = open(path, "r")
        lines = f.readlines()[1:]#[:100]
        f.close()
        data, labels = [], []
        count_mal = 0
        fnames = []

        for line in lines:
            line = line.strip()
            fname = line.split()[0]
            fnames.append(fname)

            feats = line.split()[1:]
            label = int(feats[-1])
            labels.append(label)
            feats = feats[:-1]
            feats = [float(f) for f in feats[:opts.num_feats]]
            f = np.array(feats)
            # TODO cuidado norm
            f = f / f.sum()
            # mean = f.mean()
            # if mean > 0:
            #     f = (f - mean) / f.std()

            if any(f < 0):
                raise Exception("TFIDF NEGATIVO")
                #count_mal += 1

            f = list(f)
            #f = f / f.sum()
            # print(f.sum())


            data.append(feats)
        # print("Un total de {} legajos no tienen RWs!! -> {}".format(count_mal, labels_mal))

        classes = set()
        for i in range(len(labels)):
            data[i].append(labels[i])
            data[i].append(fnames[i])
            classes.add(labels[i])
        print(classes)
        len_feats = len(data[0]) - 2
        return data, len_feats, len(classes)

    # Class dict
    path_class_dict = opts.class_dict
    class_dict, number_to_class = load_dict_class(path_class_dict)

    path_tr = opts.tr_data
    data_tr, len_feats, classes = load(path_tr)
    if not opts.LOO:
        if opts.do_test:
            path_te = opts.te_data
            data_te, _, _ = load(path_te)
        elif opts.do_prod:
            path_te = opts.prod_data
            data_te, _, _ = load(path_te)
    else:
        data_te = None
    return data_tr, data_te, len_feats, classes, class_dict, number_to_class

def get_groups(p:str, classes:list):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        l, c, ini, fin = line.strip().split(" ")
        if c not in classes:
            continue
        ini, fin = int(ini), int(fin)
        res.append([l, c, ini, fin])
    return res

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[ %r] took: %2.4f sec' % \
          (f.__name__, kw, te-ts))
        return result
    return wrap

# @timing
def search_group(groups:list, npage:int, l:str):
    for lgroup, c, ini, fin in groups:
        if ini <= npage <= fin:
            if l == lgroup:
                return ini, fin
    raise Exception(f'Group for {npage} not found')

# @timing
def search_pages_tfidf(data:list, ini, fin, legajo:str):
    train = []
    for i in data:
        npage = int(i[-1].split("_")[2])
        l_page = i[-1].split("_")[0]
        if ini <= npage <= fin or legajo != l_page:
            continue
        train.append(i)
    return train


class TextDataset(pl.LightningDataModule):

    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None, opts=None, n_test=None, info=None, legajo=""):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        # self.setup(opts)
        self.opts = opts
        
        if opts.LOO:
            self.data_tr_dev, self.data_test, self.len_feats, self.num_classes, self.class_dict, self.number_to_class = info
            self.data_test = [self.data_tr_dev[n_test]]
            if opts.path_file_groups != "":
                groups = get_groups(opts.path_file_groups, opts.classes)
                page_test = int(self.data_test[0][-1].split("_")[2])
                ini, fin = search_group(groups, page_test, legajo)
                self.data_tr_dev = search_pages_tfidf(self.data_tr_dev, ini, fin, legajo)
                print(f'Data: {len(self.data_tr_dev)} samples')
            else:
                self.data_tr_dev = self.data_tr_dev[:n_test] + self.data_tr_dev[n_test+1:]
        else:
            self.data_tr_dev, self.data_test, self.len_feats, self.num_classes, self.class_dict, self.number_to_class = load_RWs(self.opts)
        # print(self.data_tr_dev)
    def setup(self, stage):
        print("-----------------------------------------------")
        
        self.cancerDt_train = tDataset(self.data_tr_dev, transform=None)
        self.cancerDt_val = tDataset(self.data_tr_dev, transform=None)
        self.cancerDt_test = tDataset(self.data_test, transform=None)
        
    def train_dataloader(self):
        trainloader_train = torch.utils.data.DataLoader(self.cancerDt_train, batch_size=self.opts.batch_size, shuffle=True, num_workers=0)
        return trainloader_train
    
    def val_dataloader(self):
        trainloader_train = torch.utils.data.DataLoader(self.cancerDt_train, batch_size=self.opts.batch_size, shuffle=True, num_workers=0)
        return trainloader_train
    
    def test_dataloader(self):
        trainloader_train = torch.utils.data.DataLoader(self.cancerDt_test, batch_size=self.opts.batch_size, shuffle=False, num_workers=0)
        return trainloader_train
    
    def predict_dataloader(self):
        trainloader_train = torch.utils.data.DataLoader(self.cancerDt_test, batch_size=self.opts.batch_size, shuffle=False, num_workers=0)
        return trainloader_train
