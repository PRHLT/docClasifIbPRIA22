import os, glob
import numpy as np
import pickle as pkl

def get_groups(p:str, classes:list) -> list:
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

def read_tfidf_file(p:str):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = {}
    for line in lines[1:]:
        word, *tfidf_v = line.strip().split(" ")
        tfidf_v = [float(x) for x in tfidf_v[:-1]]
        res[word] = tfidf_v
    return res

def create_group(l, c, ini, fin, tfidf_file:dict):
    tfidfs = []
    for i in range(ini, fin+1):
        vector_tfidf = tfidf_file.get(f'{l}_page_{i}_{c}.idx')
        tfidfs.append(vector_tfidf)
    return np.array(tfidfs, np.float32)


def main(path_tfidf:str, path_gruos:str, path_save:str, classes:list):
    tfidf_file = read_tfidf_file(path_tfidf)
    res_groups = get_groups(path_gruos, classes)
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    for l, c, ini, fin in res_groups:
        v = create_group( l, c, ini, fin, tfidf_file)
        path_save_f = os.path.join(path_save, f'{l}_pages_{ini}-{fin}_{c}.idx')
        print(l, c, ini, fin, v.shape, path_save_f)
        with open(path_save_f, 'wb') as handle:
            pkl.dump(v, handle, protocol=pkl.HIGHEST_PROTOCOL)
        


if __name__ == "__main__":
    path_groups = "/data/carabela_segmentacion/JMBD4949_1page_idx/groups"
    path_tfidf = "/data2/jose/projects/docClasifIbPRIA22/work_JMBD4949_loo_1page/tfidf_4949_loo.txt"
    path_save = "/data/carabela_segmentacion/JMBD4949_1page_idx/sequence_groups"
    classes = [x.lower() for x in "P,CP,O,A,T".split(",")]
    main(path_tfidf, path_groups, path_save, classes)