import numpy as np
from sklearn.metrics import accuracy_score

path_file_groups = "/data/carabela_segmentacion/JMBD4949_4950_1page_idx/groups"
path_results = "/data2/jose/projects/docClasifIbPRIA22/works_JMBD4949_4950_loo_1page/work_128_numFeat4/results.txt"

def read_results(p:str):
    f = open(p, "r")
    lines = f.readlines()[1:]
    f.close()
    res = {}
    for line in lines:
        pname, gt, *hyps = line.strip().split(" ")
        gt = int(gt)
        hyps = [float(x) for x in hyps]
        page = int(pname.split("_")[2])
        l = pname.split("_")[0]
        res[f'{l}_{page}'] = [gt,hyps]
    return res

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

def voting(results:dict, groups:list):
    res_hyp, res_gt = [], []
    fallos = []
    res_results, y_results = [], []
    for l, c, ini, fin in groups:
        # print(f'{l}, {c}, {ini}, {fin}')
        a = list(results.items())[0][1][1]
        hyps_sum = np.zeros_like(a)
        for i in range(ini, fin+1):
            page = f'{l}_{i}'
            gt, hyps = results.get(page)
            # print(f'{i}  -> {gt} - {hyps}')
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
    # for fallo in fallos:
    #     print(fallo)