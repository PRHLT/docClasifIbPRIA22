import os
import numpy as np
from sklearn.metrics import confusion_matrix

def load_dict(path):
    res = {}
    f = open(path, "r")
    for line in f.readlines():
        line = line.strip()
        c, num = line.split(" ")
        num = int(num)
        res[num] = c
    f.close()
    return res

def load_results(path, dict_class):
    f = open(path, "r")
    lines = f.readlines()
    lines = lines[1:]
    f.close()
    res_muestras = []
    res_hyp = []
    res_gt = []
    res = []
    cont = 0
    for line in lines:
        line = line.strip()
        l = line.split(" ")
        muestra, gt = l[0:2]
        gt = int(gt)
        probs = l[2:]
        probs = [float(x) for x in probs]
        hyp = np.argmax(probs)
        gt = dict_class.get(gt)
        hyp = dict_class.get(hyp)
        res.append((muestra,gt,hyp))
        res_muestras.append(muestra)
        res_hyp.append(hyp)
        res_gt.append(gt)
        cont += gt == hyp
    print(cont / len(lines))
    return res, res_muestras, res_hyp, res_gt

def save_file_res(data, path):
    f = open(path, "w")
    f.write("Muestra GT HYP\n")
    for muestra,gt,hyp in data:
        f.write("{} {} {}\n".format(muestra,gt,hyp))
    f.close()

if __name__ == "__main__":
    path_work = "works_tr49_te50_groups_11classes_other/work_128_numFeat2048/"
    path_dict = "work_tr49_te50_groups_11classes_other/tfidf_tr49_classes.txt"
    # path_work = "works_tr49_te50_groups_12classes/work_128_numFeat2048/"
    # path_dict = "work_tr49_te50_groups_12classes/tfidf_tr49_classes.txt"
    # labels = [x.lower() for x in "P,CP,O,A,T,V,R,CEN,DP,D,C,TH,other".split(",")] # P,CP,O,A,T,V,R,CEN,DP,D,C,TH
    labels = [x.lower() for x in "P,CP,O,A,T,V,R,DP,D,C,TH,other".split(",")] # P,CP,O,A,T,V,R,DP,D,C,TH
    # path_work = "works_tr49_te50_groups_5classes/work_128_numFeat2048/"
    # path_dict = "work_tr49_te50_groups_5classes/tfidf_tr49_classes.txt"
    # labels = [x.lower() for x in "P,CP,O,A,T".split(",")] # P,CP,O,A,T,V,R,CEN,DP,D,C,TH
    path = os.path.join(path_work, "results.txt")
    # if not os.path.exists(path_save):
    #     os.mkdir(path_save)
    path_save = os.path.join(path_work, "confmat")
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    # path_save = os.path.join(path_save, path_work)
    # if not os.path.exists(path_save):
    #     os.mkdir(path_save)
    dict_class = load_dict(path_dict)
    res, muestras, hyp, gt = load_results(path, dict_class)
    path_save_res = os.path.join(path_save, "results")
    save_file_res(res, path_save_res)
    # labels = [v for k,v in dict_class.items()]
    # labels.sort()
    confmat = confusion_matrix(gt, hyp, labels=labels)
    print(confmat)
    path_save_confmat = os.path.join(path_save, "confmat")
    f = open(path_save_confmat, "w")
    n_labels = []
    res = []
    for idx, arr in enumerate(confmat):
        arr = " ".join([str(x) for x in arr])
        #c = dict_class[idx]
        c = labels[idx]
        n_labels.append(c)
        res.append("{} {}\n".format(c, arr))
    f.write("{} {}\n".format("-", " ".join(n_labels)))
    for r in res:
        f.write(r)
    f.close()