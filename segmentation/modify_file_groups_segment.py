import os, glob

def read_files(p:str) -> dict:
    res = {}
    all = glob.glob(os.path.join(p, "*idx"))
    for f in all:
        fname = f.split("/")[-1]
        page = int(fname.split("_")[1])
        c = fname.split("_")[-1].split(".")[0]
        res[page] = c
    return res

def read_file_groups(p:str, ini_page:int=0) -> list:
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        ini, fin, err_type = line.strip().split()
        ini, fin = int(ini)+ini_page, int(fin)+ini_page
        print(ini, fin)
        err_type = f" {err_type}"
        res.append((ini,fin, err_type))
    return res

def main(path_idxs_per_page:str, path_file_group:str, path_file_group_res:str):
    ## Create idxs - train and test
    ini_page = 61
    res_ = read_files(path_idxs_per_page)
    res_groups = read_file_groups(path_file_group, ini_page)
    f_res = open(path_file_group_res, "w")
    for ini, fin, err_type in res_groups:
        res=[]
        for i in range(ini, fin+1):
            c = res_[i]
            res.append(c)
        row = [ini, fin, err_type, "||"]
        res = ''.join([f'{x:5}' for x in res])
        # print(res)
        # exit()
        row.append(res)
        print(row)
        # row = '\n'.join([''.join([f'{x:4}' for x in row])])
        # row = ' '.join([x for x in row])
        r = "{: >5} {: >5} {: >5} {: >5} {: >5}".format(*row)
        f_res.write(f'{r}\n')

    f_res.close()

if __name__ == "__main__":
    path_idxs_per_page = "/data/carabela_segmentacion/idxs_JMBD4950/idxs_clasif_per_page/all_classes_noS"
    path_file_group = "/data2/jose/projects/docClasifIbPRIA22/work_tr49_te50_groups/file_te50_badsegments"
    path_file_group_res = "/data2/jose/projects/docClasifIbPRIA22/work_tr49_te50_groups/file_te50_badsegments_gt"
    main(path_idxs_per_page, path_file_group, path_file_group_res)