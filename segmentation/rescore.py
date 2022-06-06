import glob, os, re
import numpy as np
import math

def read_results(paths:list, LOG:bool=False) -> dict:
    res = {}
    min_pag = 1500
    for path in paths:
        # print("Reading results from : ", path)
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        lines = lines[1:]
        for line in lines:
            fname, _, *probs = line.strip().split(" ")
            probs = [float(x) for x in probs]
            ini, fin = fname.split("_")[1].split("-")
            ini, fin = int(ini), int(fin)
            min_pag = min(ini, min_pag)
            prob_max = np.max(probs)
            if LOG:
                prob_max = math.log(prob_max)
            if (ini, fin) in res:
                raise Exception(f"Group {ini}-{fin} duplicated?")
            res[(ini,fin)] = prob_max
    return res, min_pag-1
 
def read_files_segment_IMF(p:str, min_pag:int=0, LOG:bool=False) -> list:
    files = glob.glob(os.path.join(p, "*"))
    res = []
    
    for file in files:
        nbest = int(file.split("/")[-1])
        f = open(file, "r")
        lines = f.readlines()
        f.close()
        lines = [x.strip() for x in lines]
        prob = lines[0]
        segm = lines[2:]
        res_segm = []
        prob = float(prob.split("#P(s|Z)=")[-1])
        for line in segm:
            _, ini, fin, *_ = line.split()
            ini, fin = int(ini)+min_pag, int(fin)+min_pag
            res_segm.append((ini,fin))
        if LOG:
            prob = math.log(prob)
        res.append((nbest, file, prob, res_segm))
    res.sort()
    return res

def read_results_inf(p:str, LOG:bool=True) -> dict:
    res = {}
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    lines = lines[1:]
    for line in lines:
        line = re.sub(' +', ' ', line)
        s = line.strip().split(" ")
        nb_pos, errs, _, err_porc, *_ = s
        # print(nb_pos, errs, err_porc)
        nb_pos, errs, err_porc = int(nb_pos), int(errs), float(err_porc)
        res[nb_pos] = (errs, err_porc)
    return res

def main(pathsegm_IMF:str, paths_results:list, path_results_inf:str):
    LOG = False
    if LOG:
        LOGPROB_str = "LOGPROB_"
    else:
        LOGPROB_str = ""
    results, min_pag = read_results(paths_results, LOG)
    files_segm = read_files_segment_IMF(pathsegm_IMF, min_pag, LOG)
    res_inf =  read_results_inf(path_results_inf)
    # print(f"Groups start at {min_pag}")
    rescored = []
    for nbest, file, prob, segm in files_segm:
        # print(nbest, file)
        prob_classifier = 0 if LOG else 1
        for ini, fin in segm:
            # print(ini, fin)
            prob_segm = results.get((ini,fin), None)
            # print(prob_segm, ini, fin)
            if prob_segm is None:
                raise Exception(f"{ini}-{fin} group not found")
            if LOG:
                prob_classifier += prob_segm
            else:
                prob_classifier *= prob_segm
        # print(nbest, prob, prob_classifier)
        p = prob+prob_classifier if LOG else prob*prob_classifier
        errs, err_porc = res_inf[nbest]
        rescored.append((p, nbest, file, prob, segm, prob_classifier, errs, err_porc))
        # exit()
    rescored.sort()
    print("{: >30} {: >30} {: >30} {: >30} {: >30} {: >30}".format(f"{LOGPROB_str}rscored", f"{LOGPROB_str}probSegm", f"{LOGPROB_str}probText", f"nbest_probSegm", "#Errs", "Err(%)")) 
    for rscore, nbest, file, prob, segm, prob_classifier, errs, err_porc in rescored[::-1]:
        # print(rscore, prob, prob_classifier, nbest)
        print("{: >30} {: >30} {: >30} {: >30} {: >30} {: >30}".format(rscore, prob, prob_classifier, nbest, errs, err_porc))

if __name__ == "__main__":
    path_results_inf = "results.inf"
    pathsegm_IMF = "JMBD4950-NB"
    work_dir = "works_tr49_te50_groups_6classes"
    paths_results = [f"../{work_dir}/work_128_numFeat2048/results.txt", f"../{work_dir}/work_128_numFeat2048/results_prod.txt"]
    main(pathsegm_IMF, paths_results, path_results_inf)
