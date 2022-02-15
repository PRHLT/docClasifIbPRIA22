from __future__ import print_function
from __future__ import division
import time
import torch.optim as optim
import torch
import torch.nn as nn
import logging, os
import numpy as np
from utils.optparse import Arguments as arguments
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
from data import dataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
import wandb


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def acc_on_nbest(gts, hyps, n=5):
    res = []
    for i, _ in enumerate(gts):
        gt = gts[i]
        hyp = hyps[i]
        best_n = hyp.argsort()[-n:]
        #print(gt, best_n, gt in best_n, "({})".format(n))
        res.append(gt in best_n)
    return res, np.mean(res)*100

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results_eval(results_tests, dir, logger, number_to):
    create_dir(dir)
    fname = os.path.join(dir, "results_eval")
    logger.info("Saving results on {}".format(fname))
    f = open(fname, "w")
    #f.write("ID-testSample ID-Class GT(0|1)  P(Class|testSample)\n")
    for id_x, label, prediction in results_tests:
        #c, pages = id_x.split("_")
        for i, p_hyp in enumerate(prediction):
            gt_01 =  int(i == label)
            c = number_to[i]
            f.write("{} {} {} {}\n".format(id_x, c, gt_01, p_hyp))
    f.close()

def save_results(dataset, tensor, opts):
    outputs = tensor_to_numpy(tensor)
    dir = opts.work_dir
    create_dir(dir)
    fname = os.path.join(dir, "results.txt")
    f = open(fname, "w")
    f.write("Legajo GT(index) Softmax\n")
    ys = [y[1] for y in dataset.data]
    for id_x, label, prediction in zip(dataset.ids, ys, outputs):
        res=""
        for s in prediction:
            res+=" {}".format(str(s))
        f.write("{} {}{}\n".format(id_x, label, res))
    f.close()

def save_results_per_class(results_tests, dir, logger):
    create_dir(dir)
    fname = os.path.join(dir, "results_per_class.txt")
    logger.info("Saving results per class on on {}".format(fname))
    f = open(fname, "w")
    for line in results_tests:
        f.write("{}\n".format(line))
    f.close()

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def to_one_hot(y, n_dims=7):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    y =  Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
    return torch.transpose(torch.transpose(y, 1,3), 2,3)

def prepare():
    """
    Logging and arguments
    :return:
    """

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # --- keep this logger at DEBUG level, until aguments are processed
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- Get Input Arguments
    in_args = arguments(logger)
    opts = in_args.parse()
    # if check_inputs_graph(opts, logger):
    #     logger.critical("Execution aborted due input errors...")
    #     exit(1)

    fh = logging.FileHandler(opts.log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # --- restore ch logger to INFO
    ch.setLevel(logging.INFO)
    return logger, opts

def main():

    logger, opts = prepare()
    print(opts.work_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    torch.set_default_tensor_type("torch.FloatTensor")
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    random.seed(opts.seed)
    os.environ['PYTHONHASHSEED'] = str(opts.seed)
    net = None
    logger.info(opts)
    logger.info("Model: {}".format(opts.model))
    from models import model as models



    logger_csv = CSVLogger(opts.work_dir, name=opts.exp_name)
    
    path_save = os.path.join(opts.work_dir, "checkpoints")
    if not opts.LOO:
        wandb.init()
        wandb.run.name = opts.work_dir
        wandb.run.save()
        wandb.config = {
            "layers": ",".join([str(z) for z in opts.layers]),
            "num_feats": opts.num_feats,
        }
        wandb_logger = WandbLogger(project=opts.exp_name)
        textDataset = dataset.TextDataset(opts=opts)
        net = models.Net(layers=opts.layers, len_feats=textDataset.len_feats, n_classes=textDataset.num_classes, opts=opts)
        if opts.checkpoint_load:
            net = net.load_from_checkpoint(opts.checkpoint_load, layers=opts.layers, len_feats=textDataset.len_feats, n_classes=textDataset.num_classes, opts=opts)
        net.to(device)
        wandb_logger.watch(net)
        trainer = pl.Trainer(min_epochs=opts.epochs, max_epochs=opts.epochs, logger=[logger_csv, wandb_logger], #wandb_logger
                deterministic=True if opts.seed is not None else False,
                default_root_dir=path_save,
            )
        if opts.do_train:
            trainer.fit(net, textDataset)
        results_test = trainer.test(net, textDataset)
        results_test = trainer.predict(net, textDataset)
        results_test = torch.cat(results_test, dim=0)
        # print(results_test)
        save_results(textDataset.cancerDt_test, results_test, opts)
    else:
        n_test, num_exps = 0, 90000
        info = dataset.load_RWs(opts) #data_tr_dev, data_test, len_feats, num_classes, class_dict, number_to_class
        dir = opts.work_dir
        create_dir(dir)
        fname = os.path.join(dir, "results.txt")
        f = open(fname, "w")
        f.write("Legajo GT(index) Softmax\n")
        ys, hyps = [], []
        while(n_test < num_exps):
            textDataset = dataset.TextDataset(opts=opts, n_test=n_test, info=info)
            num_exps = len(textDataset.data_tr_dev) + 1
            n_test += 1
            net = models.Net(layers=opts.layers, len_feats=textDataset.len_feats, n_classes=textDataset.num_classes, opts=opts)
            net.to(device)
            trainer = pl.Trainer(min_epochs=opts.epochs, max_epochs=opts.epochs, logger=[logger_csv], #wandb_logger
                    deterministic=True if opts.seed is not None else False,
                    default_root_dir=path_save,
                )
            trainer.fit(net, textDataset)
            results_test = trainer.test(net, textDataset)
            results_test = trainer.predict(net, textDataset)
            results_test = torch.cat(results_test, dim=0)

            # Save to file
            y = [y[1] for y in textDataset.cancerDt_test.data][0]
            ids = textDataset.cancerDt_test.ids[0]
            results_test = tensor_to_numpy(results_test)[0]
            res=""
            for s in results_test:
                res+=" {}".format(str(s))
            f.write("{} {}{}\n".format(ids, y, res))
            ys.append(y)
            hyps.append(np.argmax(results_test))
            del net
        f.close()
        acc = accuracy_score(ys, hyps)
        logger.info(f'Accuracy: {acc}')
        logger.info(f'Error: {1-acc}')
        

if __name__ == "__main__":
    main()