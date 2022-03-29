import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split

def read_results(p:str):
    f = open(p, "r")
    lines = f.readlines()[1:]
    f.close()
    res = {}
    for line in lines:
        pname, gt, *hyps = line.strip().split(" ")
        gt = int(gt)
        hyps = [float(x) for x in hyps]
        pages = pname.split("_")[2]
        l = pname.split("_")[0]
        res[f'{l}_{pages}'] = [gt,hyps]
    return res

if __name__ == "__main__":
    path_results = "/data2/jose/projects/docClasifIbPRIA22/works_LOO_JMBD4949_allFiles/work_128,128_numFeat1024/results.txt"
    res = read_results(path_results)
    gts, hyps = [], []
    for pages, (gt,hyp) in res.items():
        gts.append(gt)
        hyps.append(hyp)
    print(gts)
    print(hyps)
    precision, recall, _ = precision_recall_curve(gts, hyps)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()

    # plt.show()
    plt.savefig("a.png")