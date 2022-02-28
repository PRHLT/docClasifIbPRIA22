import numpy as np
from sklearn.metrics import accuracy_score

path_file_groups = "/data/carabela_segmentacion/idxs_JMBD4949/idxs_clasif_per_page/all_classes_noS/groups"
path_results = "/data2/jose/projects/docClasifIbPRIA22/works_LOO_JMBD4949_1page/work_128_numFeat512/results.txt"

def read_results(p:str):
    f = open(p, "r")
    lines = f.readlines()[1:]
    f.close()
    res = {}
    for line in lines:
        pname, gt, *hyps = line.strip().split(" ")
        gt = int(gt)
        hyps = [float(x) for x in hyps]
        page = int(pname.split("_")[1])
        res[page] = [gt,hyps]
    return res

def get_groups(p:str, classes:list):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        c, ini, fin = line.strip().split(" ")
        if c not in classes:
            continue
        ini, fin = int(ini), int(fin)
        res.append([c, ini, fin])
    return res

def voting(results:dict, groups:list):
    res_hyp, res_gt = [], []
    fallos = []
    res_results, y_results = [], []
    for c, ini, fin in groups:
        a = list(results.items())[0][1][1]
        hyps_sum = np.zeros_like(a)
        for i in range(ini, fin+1):
            gt, hyps = results.get(i)
            h = np.argmax(hyps)
            res_results.append(h)
            y_results.append(gt)
            hyps_sum += hyps
        hyp = np.argmax(hyps_sum)
        res_hyp.append(hyp)
        res_gt.append(gt)
        if gt != hyp:
            fallos.append([c,ini,fin, gt, hyps_sum])
    acc = accuracy_score(res_gt, res_hyp)
    acc_results = accuracy_score(y_results, res_results)
    return acc, acc_results, fallos
    

if __name__ == "__main__":
    classes = ["p","cp","o","a","t"]
    results = read_results(path_results)
    groups = get_groups(path_file_groups, classes)
    acc, acc_results, fallos = voting(results, groups)
    print(f'Voting: acc {acc}, Error: {1-acc}')
    print(f'results file: acc {acc_results}, Error: {1-acc_results}')
    for fallo in fallos:
        print(fallo)