import numpy as np 

def read_results(p:str):
    f = open(p, "r")
    lines = f.readlines()[1:]
    f.close()
    res = {}
    hyps_, gts_ = [], []
    pnames = []
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
        pnames.append(pname)


    return res, hyps_, gts_, pnames

if __name__ == "__main__":
    # path_results = "/data2/jose/projects/docClasifIbPRIA22/works_JMBD4949_loo_1page_LSTM/work_128,128_numFeat1024_128epochs_0.01lrADAM/results.txt"
    # tr = "tr49"; te = "te50"
    tr = "tr50"; te = "te49"
    nmb_feats = [2**x for x in range(7,15)]
    layers_list=["0", "128", "128,128"]
    work_dir = f"works_{tr}_{te}_groups_12classes"
    # work_dir = f"works_{tr}_{te}_perPage_LSTMvoting"
    for layers in layers_list:
        print(f"TRAINING WITH {tr} - layers {layers}")
        for feats in nmb_feats:
            path_results = f"/data2/jose/projects/docClasifIbPRIA22/{work_dir}/work_{layers}_numFeat{feats}/results.txt"
            res, hyps, gts, pnames = read_results(path_results)
            gts = np.array(gts)
            hyps = np.array(hyps)
            errors = (gts == hyps).sum()
            acc = errors / hyps.shape[0]
            print(f'{feats} feats Error {(1-acc)*100} from {len(hyps)} samples - [{len(hyps) - errors} errors]')
            # for hyp, gt, pname in zip(hyps, gts, pnames):
            #     if hyp != gt:
            #         print(f"Error in {pname} - hyp {hyp}  gt {gt}")
    
