import numpy as np 

def read_results(p:str):
    f = open(p, "r")
    lines = f.readlines()[1:]
    f.close()
    res = {}
    hyps_, gts_ = [], []
    for line in lines:
        pname, gt, *hyps = line.strip().split(" ")
        gt = int(gt)
        hyps = [float(x) for x in hyps]
        pages = pname.split("_")[2]
        l = pname.split("_")[0]
        res[f'{l}_{pages}'] = [gt,hyps]
        h = np.argmax(hyps)
        hyps_.append(h)
        gts_.append(gt)


    return res, hyps_, gts_

if __name__ == "__main__":
    # path_results = "/data2/jose/projects/docClasifIbPRIA22/works_JMBD4949_loo_1page_LSTM/work_128,128_numFeat1024_128epochs_0.01lrADAM/results.txt"
    tr = "tr49"
    nmb_feats = [2**x for x in range(3,12)]
    layers="128,128"
    print(f"TRAINING WITH {tr}")
    for feats in nmb_feats:
        path_results = f"/data2/jose/projects/docClasifIbPRIA22/works_IMF/work_tr49_{layers}_numFeat{feats}/results.txt"
        res, gts, hyps = read_results(path_results)
        gts = np.array(gts)
        hyps = np.array(hyps)
        acc = (gts == hyps).sum() / hyps.shape[0]
        print(f'{feats} feats Error {1-acc} from {len(hyps)} samples')
    
